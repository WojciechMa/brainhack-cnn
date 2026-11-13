"""Train the paper's small joint CNN from scratch on Posner cueing, visual search and contextual cueing:
6 classes, no task cue.

Defaults follow the paper's joint-training notebook (see data.py and model.py): Adam lr 1e-3,
batch 16, 30 epochs, noise drawn once per training image (--fresh-noise: new noise every epoch).
--tasks posner trains on Posner only (control); the test set always holds all three tasks.
Progress goes to the console and to <out>/train.log; metrics to MLflow (experiment smallcnn).

    python smallcnn/train.py --seed 0 --out runs/smallcnn/joint_fixed_seed0
    python smallcnn/train.py --seed 0 --fresh-noise --out runs/smallcnn/joint_fresh_seed0
    python smallcnn/train.py --seed 0 --tasks posner --out runs/smallcnn/posner_fixed_seed0
    python smallcnn/train.py --limit 60 --epochs 1          # quick smoke test
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CLASSES, NOISE, STIMULI, TASKS, NoisyStimuli, load, resample
from metrics import predict, summary, task_metrics
from model import DENSE, JointCNN

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("smallcnn")


def per_task(cast):
    """Parse 'posner=0.07,search=0.04' into {'posner': 0.07, 'search': 0.04}."""
    def parse(text: str) -> dict:
        out = {}
        for item in text.split(","):
            task, _, value = item.partition("=")
            if task not in TASKS:
                raise argparse.ArgumentTypeError(f"unknown task {task!r}; tasks: {', '.join(TASKS)}")
            out[task] = cast(value)
        return out
    return parse


def recalibrate_bn(model, dataset, device, seed: int, batch_size: int = 16):
    """Replace the BatchNorm running statistics by their exact average over one pass of the training data.
    With momentum 0.01 the running averages lag behind the weights and shift the decision thresholds."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average
    model.train()
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(seed)):
            model(x.to(device))
    for bn in bns:
        bn.momentum = 0.01
    model.eval()


def setup_logging(out: Path, filename: str = "train.log"):
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(out / filename, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def setup_mlflow(experiment: str):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # a note for coding agents that clutters the log
    try:
        import mlflow
    except ImportError:
        log.info("mlflow not installed: skipping tracking")
        return None
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    return mlflow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=per_task(float), default={},
                    help=f"pixel noise std on the 0-1 scale, default {NOISE}")
    ap.add_argument("--n-per-task", type=per_task(int), default=None,
                    help="training images per task (default: all in the zips)")
    ap.add_argument("--fresh-noise", action="store_true", help="new training noise every epoch")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS), help="tasks to train on")
    ap.add_argument("--limit", type=int, default=None, help="at most N random images per stimulus set")
    ap.add_argument("--log-every", type=int, default=100, help="progress line every N batches")
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--out", type=Path, default=None, help="default: runs/smallcnn/<time>_seed<seed>")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    args.noise = {**NOISE, **args.noise}
    args.out = args.out or ROOT / "runs" / "smallcnn" / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"
    setup_logging(args.out)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'} | output {args.out}")

    images_tr, meta_tr = load("train", args.stimuli, args.noise)
    images_te, meta_te = load("test", args.stimuli, args.noise)
    meta_tr = meta_tr[meta_tr.task.isin(args.tasks)]
    if args.limit:
        pick = lambda m: m.sample(frac=1, random_state=args.seed).groupby("set").head(args.limit).sort_index()
        meta_tr, meta_te = pick(meta_tr), pick(meta_te)
    if args.n_per_task:
        meta_tr = resample(meta_tr, {**meta_tr.task.value_counts().to_dict(), **args.n_per_task}, args.seed)
    log.info("train images per task: " + ", ".join(f"{t} {n}" for t, n in meta_tr.task.value_counts().items())
             + f" | test images: {len(meta_te)}")

    train = NoisyStimuli(images_tr, meta_tr, args.seed, stream=0, fresh_noise=args.fresh_noise)
    test = NoisyStimuli(images_te, meta_te, args.seed, stream=1)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))

    model = JointCNN(len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    mlflow = None if args.no_mlflow else setup_mlflow("smallcnn")
    if mlflow:
        mlflow.start_run(run_name=args.out.name)
        mlflow.log_params({**{k: str(v) for k, v in vars(args).items()}, "n_train": len(train), "n_test": len(test)})

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, loss_sum, n_correct = time.time(), 0, 0.0, 0
        fired = torch.zeros(model.layers[DENSE][0].out_features, dtype=torch.bool, device=device)

        def track(module, inputs, out):
            fired.logical_or_((out > 0).any(0))

        hook = model.layers[DENSE].register_forward_hook(track)
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            seen += len(y)
            loss_sum += loss.item() * len(y)
            n_correct += (logits.argmax(1) == y).sum().item()
            if step % args.log_every == 0:
                log.info(f"epoch {epoch}/{args.epochs} batch {step}/{len(loader)} | loss {loss_sum / seen:.4f}"
                         f" acc {n_correct / seen:.3f} | {time.time() - t0:.0f} s")
        hook.remove()
        logits_te = predict(model, test, device)
        m = task_metrics(logits_te.argmax(1), meta_te)
        m.update(train_loss=loss_sum / seen, train_acc=n_correct / seen, epoch_seconds=time.time() - t0,
                 live_dense_units=fired.sum().item())
        log.info(f"epoch {epoch}/{args.epochs} done in {m['epoch_seconds']:.0f} s | train loss {m['train_loss']:.4f}"
                 f" acc {m['train_acc']:.3f} | Dense units that fired: {m['live_dense_units']} | test:\n{summary(m)}")
        if mlflow:
            mlflow.log_metrics(m, step=epoch)

    recalibrate_bn(model, train, device, args.seed)
    logits_te = predict(model, test, device)
    log.info(f"after recomputing the BatchNorm statistics | test:\n{summary(task_metrics(logits_te.argmax(1), meta_te))}")
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "bn_recalibrated": True,
                "args": {k: str(v) for k, v in vars(args).items()}}, args.out / "model.pt")
    meta_te.assign(pred=logits_te.argmax(1), **{f"logit_{c}": logits_te[:, i] for i, c in enumerate(CLASSES)}) \
        .to_csv(args.out / "test_predictions.csv", index_label="row")
    log.info(f"saved {args.out / 'model.pt'} and test_predictions.csv")
    if mlflow:
        mlflow.log_artifact(str(args.out / "train.log"))
        mlflow.log_artifact(str(args.out / "test_predictions.csv"))
        mlflow.set_tag("model_path", str(args.out / "model.pt"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
