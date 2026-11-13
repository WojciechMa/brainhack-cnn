"""Lesions at four depths of a trained small joint CNN: which layers carry each task?

    python smallcnn/layer_lesions.py --run runs/smallcnn/joint_fresh_seed0

Depths: the outputs of the three conv layers (12, 14 and 18 channels, all trained) and the Dense
layer (600 units). Same split-half design as
lesion.py: units are ranked on one half of the test trials and lesioned on the other half.

Ranking: each unit's first-order contribution to its task's logit game (mean z on target-present
minus target-absent trials, right minus left tilt for contextual cueing; z = the task's logit
difference), i.e. activation x gradient summed over positions, the Taylor term used to prune channels
(Molchanov et al. 2017). The readout after the Dense layer is affine, so there it equals lesion.py's
exact logit-game Shapley value; in the conv layers it is an approximation, which the held-out
lesions test.

Lesions: the top k % of each task's units against k % random live units (--n-random draws), set to
zero or to their mean map on the ranking half (--modes). Conv 1-2 lesions need a full forward pass
each; conv 3 and Dense lesions start from cached conv-3 outputs. Scored with
metrics.task_metrics and metrics.separation (AUC, threshold-refit accuracy).
Writes <run>/layers/ (or --out): ranking.csv, lesions.csv, layers.log (appends), summary.txt, layer_curves.png.
"""
import argparse
import ast
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Subset

from data import CLASSES, STIMULI, TASKS, NoisyStimuli, load
from lesion import DECISIONS, MEASURES, split_half
from metrics import separation, task_metrics
from model import DENSE, JointCNN
from train import setup_logging

DEPTHS = {"conv1": 0, "conv2": 1, "conv3": 2, "dense": DENSE}  # lesioned output: model.layers[i]
CACHE = DEPTHS["conv3"]
log = logging.getLogger("smallcnn.layers")


def forward(model, x, start: int = 0, stop: int | None = None, lesion=None):
    """model.layers[start:stop] on NCHW (or flat) x; lesion = (i, keep, fill): output of layer i -> output * keep + fill."""
    for i in range(start, len(model.layers) if stop is None else stop):
        x = model.layers[i](x)
        if lesion is not None and i == lesion[0]:
            x = x * lesion[1] + lesion[2]
    return x


def rank(model, dataset, meta: pd.DataFrame, device, batch_size: int):
    """Per depth: activation x gradient of each task's logit game, summed over positions and trials
    (tasks, units); mean output over trials (the mean-ablation map); live units (active on some trial)."""
    label, task = meta.label.to_numpy(), meta.task.map({t: i for i, t in enumerate(TASKS)}).to_numpy()
    pos, weight = label % 2 == 1, np.zeros(len(meta))
    for i in range(len(TASKS)):
        for side, sign in ((pos, 1), (~pos, -1)):
            weight[(task == i) & side] = sign / ((task == i) & side).sum()
    names = list(DEPTHS)
    phi, total, live = dict.fromkeys(names, 0), dict.fromkeys(names, 0), dict.fromkeys(names, False)
    model.requires_grad_(False)
    for b, (x, y) in enumerate(DataLoader(dataset, batch_size=batch_size)):
        idx = slice(b * batch_size, b * batch_size + len(x))
        with torch.no_grad():
            h = forward(model, x.to(device), 0, DEPTHS[names[0]] + 1)
        acts, prev = {names[0]: h.requires_grad_()}, DEPTHS[names[0]]
        for name in names[1:]:
            h = forward(model, h, prev + 1, DEPTHS[name] + 1)
            h.retain_grad()
            acts[name], prev = h, DEPTHS[name]
        L = forward(model, h, prev + 1)
        y, rows = y.to(device), torch.arange(len(y), device=device)
        z = L[rows, y - y % 2 + 1] - L[rows, y - y % 2]
        (torch.as_tensor(weight[idx], device=device, dtype=z.dtype) * z).sum().backward()
        onehot = torch.as_tensor(np.eye(len(TASKS))[task[idx]], device=device, dtype=torch.float32)
        with torch.no_grad():  # else phi would keep every batch's graph alive
            for name, a in acts.items():
                contrib = a.grad * a
                phi[name] = phi[name] + onehot.T @ (contrib.sum((2, 3)) if contrib.dim() == 4 else contrib)
                total[name] = total[name] + a.sum(0)
                live[name] = live[name] | ((a > 0).flatten(2).any(-1) if a.dim() == 4 else a > 0).any(0)
        del acts, h, L, z
    return ({n: phi[n].double().cpu().numpy() for n in names}, {n: total[n] / len(meta) for n in names},
            {n: live[n].cpu().numpy() for n in names})


def plot_curves(lesions: pd.DataFrame, intact: dict, modes, depths, path: Path):
    fig = Figure(figsize=(4.2 * len(TASKS), 3.2 * len(MEASURES) * len(modes)), layout="constrained")
    axes = fig.subplots(len(MEASURES) * len(modes), len(TASKS), squeeze=False)
    for r, (mode, measure) in enumerate((m, k) for m in modes for k in MEASURES):
        for c, task in enumerate(TASKS):
            ax, col = axes[r, c], f"{task}_{measure}"
            for i, name in enumerate(depths):
                sel = lesions[(lesions.depth == name) & (lesions.lesion_mode == mode)]
                top = sel[(sel.selection == "top") & (sel.ranking == task)].set_index("fraction")[col]
                ax.plot(top.index, top, f"C{i}-", marker="o", ms=3, label=f"{name}: top units of the task")
                rnd = sel[sel.selection == "random"].groupby("fraction")[col].mean()
                ax.plot(rnd.index, rnd, f"C{i}--", lw=0.8, label=f"{name}: random units")
            ax.axhline(intact[col], color="grey", lw=0.8, ls=":")
            ax.axhline(0.5, color="grey", lw=0.8, ls="--")
            ax.set(xscale="log", title=f"{task}: {MEASURES[measure]} ({mode} lesions)", xlabel="fraction of units lesioned")
            ax.minorticks_off()
            ax.set_xticks(rnd.index, [f"{f:g}" for f in rnd.index])
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder with model.pt")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--n-random", type=int, default=3, help="random lesions per depth and fraction")
    ap.add_argument("--modes", nargs="+", choices=("zero", "mean"), default=["zero"])
    ap.add_argument("--depths", nargs="+", choices=DEPTHS, default=list(DEPTHS))
    ap.add_argument("--decision", choices=DECISIONS, default="within",
                    help="within: each trial's own task's two classes only; strict: 6-way argmax")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="same seed as lesion.py gives the same halves")
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--out", type=Path, help="output folder (default: <run>/layers)")
    args = ap.parse_args()
    out = args.out or args.run / "layers"
    setup_logging(out, "layers.log")
    rng_split, rng_lesion = np.random.default_rng((args.seed, 0)), np.random.default_rng((args.seed, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=True)
    model = JointCNN(len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    images, meta = load("test", args.stimuli, ast.literal_eval(ckpt["args"]["noise"]))
    test = NoisyStimuli(images, meta, int(ckpt["args"]["seed"]), stream=1)  # = train.py's test set
    meta = meta.reset_index(names="image")
    half = split_half(meta, rng_split)
    idx_a, idx_b = np.flatnonzero(half), np.flatnonzero(~half)
    meta_a, meta_b = meta.iloc[idx_a].reset_index(drop=True), meta.iloc[idx_b].reset_index(drop=True)

    t0 = time.time()
    phi, mean, live = rank(model, Subset(test, idx_a), meta_a, device, args.batch_size)
    n_units = {name: len(p[0]) for name, p in phi.items()}
    log.info(f"ranking half {len(idx_a)} trials: activation x gradient in {time.time() - t0:.0f} s; units (live): "
             + ", ".join(f"{n} {n_units[n]} ({live[n].sum()})" for n in DEPTHS))
    pd.concat([pd.DataFrame({"depth": n, "unit": np.arange(n_units[n]), "live": live[n],
                             "mean_activation": mean[n].flatten(1).mean(1).cpu().numpy() if mean[n].dim() > 1 else mean[n].cpu().numpy(),
                             **{t: phi[n][i] for i, t in enumerate(TASKS)}}) for n in DEPTHS]).to_csv(out / "ranking.csv", index=False)

    t0 = time.time()
    images_b = torch.stack([test[i][0] for i in idx_b])

    def batches(t):
        return (t[s:s + args.batch_size] for s in range(0, len(t), args.batch_size))

    with torch.no_grad():
        cache = torch.cat([forward(model, x.to(device), 0, CACHE + 1).cpu() for x in batches(images_b)])
        check = np.abs(model(images_b[:8].to(device)).cpu().numpy()
                       - forward(model, cache[:8].to(device), CACHE + 1).cpu().numpy()).max()
    log.info(f"lesion half {len(idx_b)} trials: conv-3 outputs cached ({cache.numel() * 4 / 2**30:.1f} GiB) in"
             f" {time.time() - t0:.0f} s; cached path vs model logits: max diff {check:.1e}")

    @torch.no_grad()
    def logits(lesion=None) -> np.ndarray:
        if lesion is not None and lesion[0] < CACHE:
            out_ = [forward(model, x.to(device), 0, lesion=lesion).cpu() for x in batches(images_b)]
        else:
            out_ = []
            for x in batches(cache):
                x = x.to(device)
                if lesion is not None and lesion[0] == CACHE:
                    x = x * lesion[1] + lesion[2]
                out_.append(forward(model, x, CACHE + 1, lesion=lesion if lesion is not None and lesion[0] > CACHE else None).cpu())
        return torch.cat(out_).double().numpy()

    decide, label_b = DECISIONS[args.decision], meta_b.label.to_numpy()

    def score(L):
        return {**task_metrics(decide(L, label_b), meta_b), **separation(L, meta_b)}

    intact = score(logits())
    rows = [{"depth": "none", "lesion_mode": "none", "selection": "intact", "ranking": "none", "fraction": 0.0,
             "n_units": 0, "draw": 0, **intact}]
    for name in args.depths:
        layer = DEPTHS[name]
        t0, n, pool = time.time(), n_units[name], np.flatnonzero(live[name])
        shape = (1, n, 1, 1) if mean[name].dim() == 3 else (1, n)
        for mode in args.modes:
            for f in args.fractions:
                k = max(1, round(f * n))
                sets = [(task, "top", 0, np.argsort(-phi[name][i])[:k]) for i, task in enumerate(TASKS)]
                sets += [("random", "random", d, rng_lesion.choice(pool, min(k, len(pool)), replace=False))
                         for d in range(args.n_random)]
                for ranking, selection, draw, units in sets:
                    keep = torch.ones(n, device=device)
                    keep[torch.as_tensor(units, device=device)] = 0
                    keep = keep.view(shape)
                    fill = 0.0 if mode == "zero" else (1 - keep) * mean[name][None]
                    rows.append({"depth": name, "lesion_mode": mode, "selection": selection, "ranking": ranking,
                                 "fraction": f, "n_units": len(units), "draw": draw, **score(logits((layer, keep, fill)))})
        log.info(f"{name}: lesions scored in {time.time() - t0:.0f} s")
    lesions = pd.DataFrame(rows)
    lesions.to_csv(out / "lesions.csv", index=False)

    report = []

    def note(text):
        log.info(text)
        report.append(text)

    def share(p, f):
        return np.sort(p)[::-1][:max(1, round(f * len(p)))].sum() / p.sum()

    note("share of each task's logit game carried by its top units (above 100 % when other units work against it):\n"
         + "\n".join(f"  {name:7s} {task:8s} top 5 % {share(phi[name][i], 0.05):6.0%} | top 20 % {share(phi[name][i], 0.2):6.0%}"
                     for name in DEPTHS for i, task in enumerate(TASKS)))
    intact_text = ", ".join(f"{t} acc {intact[f'{t}_acc']:.3f} AUC {intact[f'{t}_auc']:.3f}" for t in TASKS)
    for mode in args.modes:
        sel = lesions[lesions.lesion_mode == mode]
        for f in sorted(set(args.fractions) & {0.05, 0.2}):
            at_f = sel[sel.fraction == f]
            table = {}
            for name in args.depths:
                d = at_f[at_f.depth == name]
                rnd = d[d.selection == "random"]
                table[name] = {f"{t} {m} {s}": (d[(d.selection == "top") & (d.ranking == t)].iloc[0][f"{t}_{m}"] if s == "top"
                                                else rnd[f"{t}_{m}"].mean())
                               for t in TASKS for m in ("acc", "auc") for s in ("top", "random")}
            note(f"{mode} lesions of {f:.0%} of the units at each depth: each task's own top units vs random"
                 f" (lesion half; intact: {intact_text}):\n{pd.DataFrame(table).round(3).to_string()}")
            for name in args.depths:
                d = at_f[at_f.depth == name]
                rnd = d[d.selection == "random"]
                matrix = pd.DataFrame({f"top of {src}": {f"{dst} {m}": d[(d.selection == "top") & (d.ranking == src)].iloc[0][f"{dst}_{m}"]
                                                         - rnd[f"{dst}_{m}"].mean() for dst in TASKS for m in MEASURES}
                                       for src in TASKS}).T
                note(f"{name}, cross-task matrix, {mode} lesions of {f:.0%}: change vs random lesions"
                     f" (rows: whose top units, columns: task measured)\n{matrix.round(3).to_string()}")

    plot_curves(lesions, intact, args.modes, args.depths, out / "layer_curves.png")
    (out / "summary.txt").write_text(f"{args.run.name}, decision {args.decision}, depths {args.depths}, modes {args.modes},"
                                     f" n_random {args.n_random}, seed {args.seed}\n\n" + "\n\n".join(report) + "\n", encoding="utf-8")
    log.info(f"saved {out / 'ranking.csv'}, lesions.csv, summary.txt, layer_curves.png")


if __name__ == "__main__":
    main()
