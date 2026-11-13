"""Recompute the BatchNorm statistics of a model trained before train.py did so; rewrites model.pt and
test_predictions.csv and deletes the run's now stale analyses. No-op if already recalibrated.

    python smallcnn/recalibrate.py --run runs/smallcnn/joint_fresh_seed0
"""
import argparse
import ast
import logging
import shutil
from pathlib import Path

import torch

from data import CLASSES, NoisyStimuli, load
from metrics import predict, summary, task_metrics
from model import JointCNN
from train import log, recalibrate_bn, setup_logging

ANALYSES = ("lesion_within", "layers", "neuron_cueing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    ck = torch.load(args.run / "model.pt", map_location="cpu")
    if ck.get("bn_recalibrated"):
        return
    setup_logging(args.run)
    a = ck["args"]
    seed, tasks, fresh = int(a["seed"]), ast.literal_eval(a["tasks"]), a["fresh_noise"] == "True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointCNN(len(CLASSES)).to(device)
    model.load_state_dict(ck["state_dict"])
    images_tr, meta_tr = load("train")
    images_te, meta_te = load("test")
    meta_tr = meta_tr[meta_tr.task.isin(tasks)]
    recalibrate_bn(model, NoisyStimuli(images_tr, meta_tr, seed, stream=0, fresh_noise=fresh), device, seed)
    logits_te = predict(model, NoisyStimuli(images_te, meta_te, seed, stream=1), device)
    log.info(f"recalibrate.py: after recomputing the BatchNorm statistics | test:\n"
             f"{summary(task_metrics(logits_te.argmax(1), meta_te))}")
    torch.save({**ck, "state_dict": model.state_dict(), "bn_recalibrated": True}, args.run / "model.pt")
    meta_te.assign(pred=logits_te.argmax(1), **{f"logit_{c}": logits_te[:, i] for i, c in enumerate(CLASSES)}) \
        .to_csv(args.run / "test_predictions.csv", index_label="row")
    for d in ANALYSES:
        shutil.rmtree(args.run / d, ignore_errors=True)
    log.info(f"saved {args.run / 'model.pt'} and test_predictions.csv; removed stale {', '.join(ANALYSES)}")


if __name__ == "__main__":
    main()
