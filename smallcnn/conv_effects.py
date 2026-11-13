"""Are the three attention effects carried by separate channels in the conv layers too?

    python smallcnn/conv_effects.py --runs runs/smallcnn/joint_fresh_v2_seed{0..9} --out runs/smallcnn/conv_effects_v2

The conv layers have few channels (12, 14, 18), so every live channel is lesioned on its own ("single"):
the change of each measure against the intact network on the same trials, separately on the two halves
of lesion.py's split. Measures (threshold-free, so a lesion that only shifts the logits changes nothing):
the three effects in AUC (cueing valid - invalid, set size 1 - 8 items, context old - new layouts) and
the three tasks' AUC. drop = intact - lesioned, positive = the lesion weakens the measure.
1. Profiles: Spearman correlation across a depth's live channels of the drops in two measures (mean of
   the halves). Ceiling = sqrt of the two split-half reliabilities (half A vs half B across channels).
2. Same top channel: does the channel with the largest drop in one effect also have the largest drop in
   another (chance = 1 / live channels)?
3. Group lesions: each measure's top k channels (largest drops on half A, k = 20 % of the depth's
   channels) lesioned together and scored on half B, against --n-random sets of k random live channels.
Ablation (--modes): zero, or mean = the channel's mean map over all test trials (zeroing also removes
a channel's mean activity, a large shift for channels that respond to the grey background).
Writes <run>/conv_effects/{single,group}.csv and, with --out, summary.txt and the combined tables.
"""
import argparse
import ast
import logging
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from data import CLASSES, STIMULI, NoisyStimuli, load
from layer_lesions import forward
from lesion import split_half
from metrics import separation
from model import JointCNN
from train import setup_logging

DEPTHS = {"conv1": 0, "conv2": 1, "conv3": 2}
MEASURES = {"cueing": "posner_cue_effect_auc", "set size": "search_set_size_effect_auc", "context": "context_effect_auc",
            "posner task": "posner_auc", "search task": "search_auc", "context task": "context_auc"}
EFFECTS = ("cueing", "set size", "context")
OWN_TASK = {"cueing": "posner task", "set size": "search task", "context": "context task"}
log = logging.getLogger("smallcnn.conv_effects")


def run_one(run: Path, args, device) -> dict[str, pd.DataFrame]:
    out = run / "conv_effects"
    out.mkdir(exist_ok=True)
    rng_split, rng = np.random.default_rng((args.seed, 0)), np.random.default_rng((args.seed, 3))
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model = JointCNN(len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    images, meta = load("test", args.stimuli, ast.literal_eval(ckpt["args"]["noise"]))
    test = NoisyStimuli(images, meta, int(ckpt["args"]["seed"]), stream=1)  # = train.py's test set
    meta = meta.reset_index(names="image")
    half = split_half(meta, rng_split)  # = lesion.py's and layer_lesions.py's halves
    t0 = time.time()
    X = torch.stack([test[i][0] for i in range(len(test))]).to(device)
    batches = lambda: (X[s:s + args.batch_size] for s in range(0, len(X), args.batch_size))

    with torch.no_grad():
        mean, live = dict.fromkeys(DEPTHS, 0), dict.fromkeys(DEPTHS, False)
        for x in batches():
            for name, i in DEPTHS.items():
                x = forward(model, x, i, i + 1)  # depths are consecutive layers
                mean[name] = mean[name] + x.sum(0) / len(X)
                live[name] = live[name] | (x > 0).flatten(2).any(-1).any(0)
        live = {n: v.cpu().numpy() for n, v in live.items()}

    @torch.no_grad()
    def score(lesion=None) -> dict[str, dict[str, float]]:
        L = torch.cat([forward(model, x, lesion=lesion) for x in batches()]).double().cpu().numpy()
        res = {}
        for h, m in (("a", half), ("b", ~half)):
            s = separation(L[m], meta[m].reset_index(drop=True))
            res[h] = {k: s[c] for k, c in MEASURES.items()}
        return res

    def lesion_of(name, units, mode):
        n = len(live[name])
        keep = torch.ones(n, device=device)
        keep[torch.as_tensor(np.asarray(units), device=device)] = 0
        keep = keep.view(1, n, 1, 1)
        return DEPTHS[name], keep, 0.0 if mode == "zero" else (1 - keep) * mean[name][None]

    intact = score()
    log.info(f"{run.name}: {len(X)} trials on the GPU, mean maps in {time.time() - t0:.0f} s; live channels "
             + ", ".join(f"{n} {live[n].sum()}/{len(live[n])}" for n in DEPTHS)
             + "; intact (half a): " + ", ".join(f"{k} {v:+.3f}" for k, v in intact["a"].items()))

    t0, single = time.time(), []
    for name in DEPTHS:
        for mode in args.modes:
            for c in np.flatnonzero(live[name]):
                s = score(lesion_of(name, [c], mode))
                single += [{"depth": name, "mode": mode, "channel": c, "half": h,
                            **{k: intact[h][k] - s[h][k] for k in MEASURES}} for h in "ab"]
    single = pd.DataFrame(single)
    log.info(f"{run.name}: single-channel lesions in {time.time() - t0:.0f} s")

    t0, group = time.time(), []
    for name in DEPTHS:
        pool = np.flatnonzero(live[name])
        k = max(1, round(args.fraction * len(live[name])))
        for mode in args.modes:
            a = single[(single.depth == name) & (single["mode"] == mode) & (single.half == "a")]
            sets = [(m, "top", 0, a.channel.to_numpy()[np.argsort(-a[m].to_numpy())[:k]]) for m in MEASURES]
            sets += [("random", "random", d, rng.choice(pool, min(k, len(pool)), replace=False)) for d in range(args.n_random)]
            for ranking, selection, draw, units in sets:
                s = score(lesion_of(name, units, mode))["b"]
                group.append({"depth": name, "mode": mode, "ranking": ranking, "selection": selection, "k": len(units),
                              "draw": draw, "channels": " ".join(map(str, units)), **{m: intact["b"][m] - s[m] for m in MEASURES}})
    group = pd.DataFrame(group)
    log.info(f"{run.name}: group lesions in {time.time() - t0:.0f} s")
    del X
    for nm, df in (("single", single), ("group", group)):
        df.insert(0, "run", run.name)
        df.to_csv(out / f"{nm}.csv", index=False)
    return {"single": single, "group": group, "intact": pd.DataFrame([{"run": run.name, "half": h, **intact[h]} for h in "ab"])}


def profiles(single: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, name, mode), d in single.groupby(["run", "depth", "mode"], sort=False):
        a = d[d.half == "a"].set_index("channel")[list(MEASURES)]
        b = d[d.half == "b"].set_index("channel")[list(MEASURES)]
        both = (a + b) / 2
        rel = {m: spearmanr(a[m], b[m])[0] for m in MEASURES}
        pairs = list(combinations(EFFECTS, 2)) + [(e, OWN_TASK[e]) for e in EFFECTS]
        for x, y in pairs:
            rows.append({"run": run, "depth": name, "mode": mode, "a": x, "b": y, "channels": len(both),
                         "rho": spearmanr(both[x], both[y])[0], "reliability_a": rel[x], "reliability_b": rel[y],
                         "ceiling": np.sqrt(max(rel[x], 0) * max(rel[y], 0))})
        for x, y in combinations(EFFECTS, 2):
            rows.append({"run": run, "depth": name, "mode": mode, "a": x, "b": y, "channels": len(both),
                         "same_top": float(both[x].idxmax() == both[y].idxmax()), "chance": 1 / len(both)})
    return pd.DataFrame(rows)


def pts(s: pd.Series) -> str:
    return f"{100 * s.mean():+.1f} ({100 * s.min():+.1f} to {100 * s.max():+.1f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", choices=("zero", "mean"), default=["mean", "zero"])
    ap.add_argument("--fraction", type=float, default=0.2, help="share of a depth's channels in a group lesion")
    ap.add_argument("--n-random", type=int, default=20, help="random channel sets per depth and mode")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="same seed as lesion.py gives the same halves")
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--out", type=Path, help="folder for the tables combined over runs")
    args = ap.parse_args()
    out = args.out or args.runs[0] / "conv_effects"
    setup_logging(out, "conv_effects.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = [run_one(r, args, device) for r in args.runs]
    if not args.out:
        return
    single, group, intact = (pd.concat([r[k] for r in res]) for k in ("single", "group", "intact"))
    prof = profiles(single)
    lines = [f"{len(args.runs)} networks: {', '.join(r.name for r in args.runs)}; modes {args.modes},"
             f" group lesions {args.fraction:.0%} of the channels, {args.n_random} random sets\n",
             "Intact, points of AUC (mean, range over networks and halves):\n"
             + "\n".join(f"  {m:12s} {pts(intact[m])}" for m in MEASURES) + "\n"]

    sg = single.groupby(["run", "depth", "mode", "channel"], sort=False)[list(MEASURES)].mean().reset_index()
    big = sg.groupby(["run", "depth", "mode"], sort=False)[list(MEASURES)].max().groupby(["depth", "mode"], sort=False)
    lines.append("1. Largest single-channel drop per network, points (mean, range over networks; drop = intact - lesioned,"
                 f" mean of the halves):\n{big.agg(pts).to_string()}\n")

    r = prof[prof.rho.notna()].groupby(["depth", "mode", "a", "b"], sort=False).agg(
        rho=("rho", lambda s: f"{100 * s.mean():+.0f} % ({100 * s.min():+.0f} to {100 * s.max():+.0f})"),
        ceiling=("ceiling", lambda s: f"{100 * s.mean():.0f} %"), channels=("channels", "mean"))
    lines.append("2. Channel profiles: Spearman correlation across live channels of the drops in two measures"
                 f" (mean, range over networks; ceiling from the split-half reliabilities):\n{r.to_string()}\n")

    t = prof[prof.same_top.notna()].groupby(["depth", "mode", "a", "b"], sort=False).agg(
        same_top=("same_top", lambda s: f"{int(s.sum())} of {len(s)}"), chance=("chance", lambda s: f"{s.mean():.0%}"))
    lines.append(f"3. Same channel has the largest drop in both effects (networks; chance = 1 / live channels):\n{t.to_string()}\n")

    rows = []
    for (run, name, mode), d in group.groupby(["run", "depth", "mode"], sort=False):
        rnd = d[d.selection == "random"]
        for src in MEASURES:
            top = d[(d.selection == "top") & (d.ranking == src)].iloc[0]
            rows += [{"run": run, "depth": name, "mode": mode, "top of": src, "measure": m,
                      "change": top[m] - rnd[m].mean(), "beats": float((top[m] > rnd[m]).mean() >= 0.95)} for m in MEASURES]
    g = pd.DataFrame(rows)
    mat = g.groupby(["depth", "mode", "top of", "measure"], sort=False)["change"].mean().mul(100).round(1).unstack("measure")
    beats = g.groupby(["depth", "mode", "top of", "measure"], sort=False)["beats"].sum().astype(int).unstack("measure")
    lines.append("4. Group lesions (half B): drop after removing each measure's top channels minus the mean drop after random"
                 f" channels, points, mean over networks (rows: whose top channels; columns: measure):\n{mat.to_string()}\n")
    lines.append(f"   networks where the top channels' drop exceeds 95 % of the random sets:\n{beats.to_string()}\n")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    for nm, df in (("single", single), ("group", g), ("profiles", prof), ("intact", intact)):
        df.to_csv(out / f"{nm}.csv", index=False)
    log.info("\n".join(lines))


if __name__ == "__main__":
    main()
