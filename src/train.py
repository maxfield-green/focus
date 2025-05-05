"""
Launch with either of the two idioms:

  # ① torchrun (PyTorch ≥ 1.9)
  torchrun --standalone --nproc_per_node=4 train_ddp.py --config config.yaml

  # ② python -m torch.distributed.run (legacy)
  python -m torch.distributed.run --standalone --nproc_per_node=4 train_ddp.py --config config.yaml
"""
import argparse, os, yaml, torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from datetime import datetime

from model import AttentionUNet            # <— your model file
from utils import EarlyStopper, log_tensorboard

# --------------------------------------------------------------------------- #
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# --------------------------------------------------------------------------- #
def main_worker(rank, world_size, cfg):
    if cfg["distributed"]:
        setup(rank, world_size, cfg["backend"])

    # ---------------- data ----------------
    train_set, val_set = build_datasets(cfg)             # <-- implement
    train_samp = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) \
                 if cfg["distributed"] else None
    val_samp   = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, shuffle=False) \
                 if cfg["distributed"] else None

    train_loader = torch.utils.data.DataLoader(train_set,
                     batch_size   = cfg["batch_size"],
                     sampler      = train_samp,
                     shuffle      = (train_samp is None),
                     num_workers  = 4, pin_memory=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_set,
                     batch_size   = cfg["batch_size"],
                     sampler      = val_samp,
                     shuffle      = False,
                     num_workers  = 4, pin_memory=True)

    # ---------------- model ----------------
    model = AttentionUNet(
        channels_in   = cfg["channels_in"],
        channels_out  = cfg["channels_out"],
        width_factors = cfg["width_factors"],
        p_drop        = cfg["dropout"],
    ).cuda(rank)

    if cfg["distributed"] and cfg["sync_batchnorm"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=cfg["find_unused_params"]) if cfg["distributed"] else model

    optimiser  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)
    early_stop = EarlyStopper(patience=cfg["early_stop_patience"])

    # --- TensorBoard: only rank 0 writes --- #
    from torch.utils.tensorboard import SummaryWriter
    writer = None
    if rank == 0:
        log_path = Path(cfg["log_dir"]) / datetime.now().strftime("%Y%m%d‑%H%M%S")
        writer = SummaryWriter(log_dir=log_path)

    # ---------------- loop -----------------
    for epoch in range(cfg["epochs"]):
        if cfg["distributed"]:
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimiser, rank)
        val_loss   = validate(model, val_loader, rank)

        # ---------- reduce across GPUs ----------
        if cfg["distributed"]:
            # broadcast the summed loss and num samples, then average
            metric = torch.tensor([train_loss, val_loss], device=rank)
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            metric /= world_size
            train_loss, val_loss = metric.tolist()

        # ---------- only rank 0 handles side‑effects ----------
        if rank == 0:
            scheduler.step(val_loss)
            log_tensorboard(writer, epoch, train_loss, val_loss)
            print(f"[{epoch:03d}] train {train_loss:.4f} | val {val_loss:.4f}")
            early_stop(val_loss)
            if early_stop.stop: break

    # save weights once
    if rank == 0:
        torch.save(model.module.state_dict() if cfg["distributed"] else model.state_dict(),
                   cfg["model_out"])

    if cfg["distributed"]:
        cleanup()

# --------------------------------------------------------------------------- #
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    world_size = torch.cuda.device_count() if cfg["distributed"] else 1
    if cfg["distributed"]:
        mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        main_worker(rank=0, world_size=1, cfg=cfg)

if __name__ == "__main__":
    cli()
