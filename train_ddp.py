"""Paso 2: entrenamiento distribuido (PyTorch DDP, backend gloo) entre 1..N computadoras.

PC maestra:  python train_ddp.py --rank 0 --world-size 2 --master-addr <IP_maestra>
PC 2:        python train_ddp.py --rank 1 --world-size 2 --master-addr <IP_maestra>
Una sola PC: python train_ddp.py --rank 0 --world-size 1 --master-addr 127.0.0.1
"""
import os

# Las builds de PyTorch para Windows no traen libuv; sin esto falla init_process_group
os.environ.setdefault("USE_LIBUV", "0")

import argparse
import socket
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from common import CLASS_FOLDERS, HandMLP, augment

SEED = 42


def log(rank, msg):
    print(f"[rank {rank} | {socket.gethostname()}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default="landmarks.npz")
    parser.add_argument("--out", default="hand_model.pt")
    args = parser.parse_args()
    rank, world = args.rank, args.world_size

    log(rank, f"conectando con {args.master_addr}:{args.master_port} (world_size={world})...")
    dist.init_process_group("gloo", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                            rank=rank, world_size=world, timeout=timedelta(minutes=5))
    log(rank, "conectado al grupo")

    torch.manual_seed(SEED)
    data = np.load(args.data)
    X, y = data["X"], data["y"]
    # Mismo split en todas las PCs gracias a la semilla fija
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    # Cada rank recibe una parte distinta del train set
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=SEED)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    log(rank, f"train total={len(train_ds)}, mi parte={len(sampler)}, val={len(X_val)}")

    # DDP sincroniza pesos iniciales desde rank 0 y promedia gradientes en cada backward()
    model = DDP(HandMLP())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Validación repartida: cada rank evalúa una parte y se suman los aciertos con all_reduce
    Xv = torch.from_numpy(X_val[rank::world])
    yv = torch.from_numpy(y_val[rank::world])

    start = time.time()
    seen = 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total_loss, batches = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(augment(xb)), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            correct = (model(Xv).argmax(1) == yv).sum()
        stats = torch.tensor([correct.item(), len(yv)], dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if epoch % 5 == 0 or epoch == 1:
            log(rank, f"epoch {epoch:3d}  loss {total_loss / max(batches, 1):.4f}  "
                      f"val_acc(global) {stats[0] / stats[1]:.3f}  muestras vistas por mí {seen}")

    elapsed = time.time() - start
    log(rank, f"entrenamiento terminado en {elapsed:.1f}s")

    if rank == 0:
        with torch.no_grad():
            pred = model.module(torch.from_numpy(X_val)).argmax(1).numpy()
        acc = (pred == y_val).mean()
        torch.save({"state_dict": model.module.state_dict(), "classes": CLASS_FOLDERS}, args.out)
        log(rank, f"val_acc final {acc:.3f} -> modelo guardado en {args.out}")
        print("\nMatriz de confusión (filas=real, columnas=predicho):")
        print("  ".join(c.replace("acorde_", "")[:8].rjust(8) for c in CLASS_FOLDERS))
        print(confusion_matrix(y_val, pred, labels=range(len(CLASS_FOLDERS))))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
