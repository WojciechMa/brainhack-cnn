"""Is there a shared attention mechanism across the three tasks? Three tests on the Dense layer.

    python smallcnn/shared_attention.py --runs runs/smallcnn/joint_fresh_v2_seed{0..9} --out runs/smallcnn/shared_attention_v2

The three attention effects: Posner cueing (valid vs invalid), search set size (1 vs 8 items),
contextual cueing (old vs new layouts). Posner and search both mark possible target locations with
boxes, so a shared "attend to the box" mechanism is plausible.

1. Overlap: the top 5 % / 20 % of units of each game in lesion_within/shapley.csv (ranking half of
   lesion.py), as used in the lesions. Overlap between two sets against chance = k1 * k2 / live units
   (hypergeometric p). Task games (posner_present, search_present, context_tilt) are included as a
   reference: an effect's units can be compared with the units that solve its own task.
2. Unit profiles: each live unit's sensitivity gain per effect, computed from its activations:
   AUROC(task's two classes) in the easier condition minus the harder one, times the sign of the unit's
   tuning (so a target-inhibitory unit that discriminates better also counts as a gain). Spearman
   correlation across units active on both tasks' trials. Ceiling: split-half reliability of each gain
   (random halves of the trials of every condition cell).
3. Shared direction: the mean activity difference (z-scored units) between the easy and hard condition
   of one task, projected onto the other task's trials: AUROC easy vs hard there, 50 % = not shared.
   Sign fixed in advance: valid ~ 1 item ~ old layout. Null: the same with the labels of the source
   task shuffled within condition cells that keep its other factors (n_perm directions).
   Within-task reference: direction from one half of the trials, AUROC on the other half (noise draws of
   one clean image can fall in both halves, so this is an upper bound).
   Control: AUROC of the summed raw activity of all units (source "total activity"), and the cosine between
   the two effects' directions (0 = unrelated).
   p is one-sided (null >= observed). Caveat: 1 vs 8 items and old vs new layouts are different pictures,
   so the null is wide and not centred on 50 %; compare with null_median, not with 50 %.
Trials: target-present only for Posner and search (the effects are about finding a target); all
context trials. --reps noise draws per test image.
Writes <run>/shared_attention/{overlap,profiles,directions}.csv and, with --out, the combined tables.
"""
import argparse
import ast
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, spearmanr

from data import CLASSES, SET_SIZES, STIMULI, NoisyStimuli, load
from lesion import activations
from model import JointCNN
from neuron_cueing import auroc
from train import setup_logging

EFFECTS = {  # task, condition column, easy value, hard value
    "cueing": ("posner", "validity", "valid", "invalid"),
    "set size": ("search", "set_size", SET_SIZES[0], SET_SIZES[-1]),
    "context": ("context", "config", "old", "new"),
}
POSITIVE = {"posner": 1, "search": 3, "context": 5}  # context: tilt right vs left
GAMES = {"cueing": "logit_posner_validity", "set size": "logit_search_set_size", "context": "logit_context_old_new",
         "posner task": "logit_posner_present", "search task": "logit_search_present", "context task": "logit_context_tilt"}
log = logging.getLogger("smallcnn.shared_attention")


def overlap(shap: pd.DataFrame, live: np.ndarray, fraction: float) -> list[dict]:
    k = round(fraction * len(shap))
    top = {}
    for name, col in GAMES.items():
        phi = shap[col].to_numpy()
        order = [j for j in np.argsort(-phi)[:k] if phi[j] > 0]  # dead units have phi 0
        top[name] = set(order)
    rows = []
    for a, b in combinations(GAMES, 2):
        n, ka, kb = len(live), len(top[a]), len(top[b])
        both = len(top[a] & top[b])
        rows.append({"fraction": fraction, "a": a, "b": b, "k_a": ka, "k_b": kb, "overlap": both,
                     "chance": ka * kb / n, "ratio": both / (ka * kb / n) if ka and kb else np.nan,
                     "p": hypergeom.sf(both - 1, n, ka, kb)})
    return rows


def gains(X: torch.Tensor, meta: pd.DataFrame, rng) -> dict[str, np.ndarray]:
    """Per unit and effect: full estimate and two split-half estimates, shape (3, units)."""
    out = {}
    for effect, (task, col, easy, hard) in EFFECTS.items():
        t = meta.task.to_numpy() == task
        lab = meta.label.to_numpy()
        pos_all = t & (lab == POSITIVE[task])
        neg_all = t & (lab == POSITIVE[task] - 1)
        if task == "posner":  # absent trials have validity "absent": pair them with both conditions
            neg = {c: neg_all for c in (easy, hard)}
        else:
            neg = {c: neg_all & (meta[col].to_numpy() == c) for c in (easy, hard)}
        pos = {c: pos_all & (meta[col].to_numpy() == c) for c in (easy, hard)}
        idx = lambda m: torch.as_tensor(np.flatnonzero(m), device=X.device)
        tuning = torch.sign(auroc(X, idx(pos_all), idx(neg_all)) - 0.5)
        halves = rng.random(len(meta)) < 0.5
        est = []
        for keep in (np.ones(len(meta), bool), halves, ~halves):
            g = auroc(X, idx(pos[easy] & keep), idx(neg[easy] & keep)) - auroc(X, idx(pos[hard] & keep), idx(neg[hard] & keep))
            est.append((g * tuning).cpu().numpy())
        out[effect] = np.stack(est)
    return out


def direction_auc(Z: np.ndarray, d: np.ndarray, easy: np.ndarray, hard: np.ndarray) -> float:
    s = Z @ d
    x, y = s[easy], s[hard]
    return float((np.searchsorted(np.sort(y), x).sum() + np.searchsorted(np.sort(y), x, side="right").sum()) / (2 * len(x) * len(y)))


def directions(A: np.ndarray, meta: pd.DataFrame, n_perm: int, rng) -> list[dict]:
    mu, sd = A.mean(0), A.std(0)
    Z = (A - mu) / np.where(sd > 0, sd, 1.0)
    sel, cells = {}, {}
    for effect, (task, col, easy, hard) in EFFECTS.items():
        m = (meta.task == task).to_numpy()
        if task != "context":
            m = m & (meta.label.to_numpy() == POSITIVE[task])
        c = meta[col].to_numpy()
        sel[effect] = (m & (c == easy), m & (c == hard))
        # shuffles keep the other factors: target side (Posner), label (search, context)
        other = meta.target.to_numpy() if task == "posner" else meta.label.to_numpy()
        cells[effect] = [np.flatnonzero(m & ((c == easy) | (c == hard)) & (other == o)) for o in np.unique(other[m])]
    full = {e: Z[a].mean(0) - Z[b].mean(0) for e, (a, b) in sel.items()}
    rows = [{"source": "total activity", "target": dst, "auc": direction_auc(A, np.ones(A.shape[1]), e_dst, h_dst)}
            for dst, (e_dst, h_dst) in sel.items()]
    for src, (e_src, h_src) in sel.items():
        d = full[src]
        half = rng.random(len(meta)) < 0.5
        d_half = Z[e_src & half].mean(0) - Z[h_src & half].mean(0)
        rows.append({"source": src, "target": src, "auc": direction_auc(Z, d_half, e_src & ~half, h_src & ~half),
                     "null_low": np.nan, "null_median": np.nan, "null_high": np.nan, "p": np.nan})
        null = {dst: [] for dst in sel if dst != src}
        for _ in range(n_perm):
            easy_p = np.zeros(len(meta), bool)
            for idx in cells[src]:
                lab = e_src[idx]
                easy_p[idx] = rng.permutation(lab)
            hard_p = (e_src | h_src) & ~easy_p
            d_p = Z[easy_p].mean(0) - Z[hard_p].mean(0)
            for dst, (e_dst, h_dst) in sel.items():
                if dst != src:
                    null[dst].append(direction_auc(Z, d_p, e_dst, h_dst))
        for dst, (e_dst, h_dst) in sel.items():
            if dst == src:
                continue
            a, nl = direction_auc(Z, d, e_dst, h_dst), np.array(null[dst])
            rows.append({"source": src, "target": dst, "auc": a, "null_low": np.quantile(nl, 0.025),
                         "null_median": np.median(nl), "null_high": np.quantile(nl, 0.975),
                         "p": (1 + (nl >= a).sum()) / (1 + n_perm),  # one-sided: sign fixed in advance
                         "cosine": d @ full[dst] / np.linalg.norm(d) / np.linalg.norm(full[dst])})
    return rows


def run_one(run: Path, args, device) -> dict[str, pd.DataFrame]:
    out = run / "shared_attention"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model = JointCNN(len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    images, meta = load("test", args.stimuli, ast.literal_eval(ckpt["args"]["noise"]))
    meta = pd.concat([meta] * args.reps)  # index = image row; NoisyStimuli keys the noise by position
    A = activations(model, NoisyStimuli(images, meta, int(ckpt["args"]["seed"]), stream=1), device)
    meta = meta.reset_index(drop=True)
    live = np.flatnonzero((A > 0).any(0))
    log.info(f"{run.name}: activations {A.shape}, live units {len(live)}")

    shap = pd.read_csv(run / "lesion_within" / "shapley.csv")
    ov = pd.DataFrame([r for f in (0.05, 0.2) for r in overlap(shap, live, f)])

    X = torch.as_tensor(A, dtype=torch.float32, device=device)
    g = gains(X, meta, rng)
    rows = []
    for a, b in combinations(EFFECTS, 2):
        ta, tb = EFFECTS[a][0], EFFECTS[b][0]
        both = ((A[meta.task.to_numpy() == ta] > 0).any(0) & (A[meta.task.to_numpy() == tb] > 0).any(0))
        rel = {e: spearmanr(g[e][1, both], g[e][2, both])[0] for e in (a, b)}
        rows.append({"a": a, "b": b, "units": int(both.sum()), "rho": spearmanr(g[a][0, both], g[b][0, both])[0],
                     "reliability_a": rel[a], "reliability_b": rel[b],
                     "ceiling": np.sqrt(max(rel[a], 0) * max(rel[b], 0))})
    prof = pd.DataFrame(rows)
    dirs = pd.DataFrame(directions(A, meta, args.n_perm, rng))
    for name, df in (("overlap", ov), ("profiles", prof), ("directions", dirs)):
        df.insert(0, "run", run.name)
        df.to_csv(out / f"{name}.csv", index=False)
    log.info(f"{run.name}:\n{ov[ov.fraction == 0.2].round(3).to_string(index=False)}\n{prof.round(3).to_string(index=False)}"
             f"\n{dirs.round(3).to_string(index=False)}")
    return {"overlap": ov, "profiles": prof, "directions": dirs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=3, help="noise draws per test image")
    ap.add_argument("--n-perm", type=int, default=500, help="shuffled-label directions per source effect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--out", type=Path, help="folder for the tables combined over runs")
    args = ap.parse_args()
    out = args.out or args.runs[0] / "shared_attention"
    setup_logging(out, "shared_attention.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = [run_one(r, args, device) for r in args.runs]
    if not args.out:
        return
    ov, prof, dirs = (pd.concat([r[k] for r in res]) for k in ("overlap", "profiles", "directions"))
    lines = [f"{len(args.runs)} networks: {', '.join(r.name for r in args.runs)}; reps {args.reps}, n_perm {args.n_perm}\n"]
    for f in (0.05, 0.2):
        o = ov[ov.fraction == f].groupby(["a", "b"], sort=False).agg(
            overlap=("overlap", "mean"), chance=("chance", "mean"), ratio=("ratio", "mean"),
            ratio_min=("ratio", "min"), ratio_max=("ratio", "max"), nets_p05=("p", lambda p: f"{(p < 0.05).sum()}/{len(p)}"))
        lines.append(f"1. Overlap of the top {f:.0%} units (mean over networks; ratio = overlap / chance;"
                     f" nets_p05 = networks with more overlap than chance, p < 0.05):\n{o.round(2).to_string()}\n")
    p = prof.groupby(["a", "b"], sort=False).agg(units=("units", "mean"), rho=("rho", "mean"), rho_min=("rho", "min"),
                                                 rho_max=("rho", "max"), ceiling=("ceiling", "mean"))
    lines.append(f"2. Unit profiles: Spearman correlation of the sensitivity gains across units (mean, range over networks;"
                 f" ceiling = sqrt of the two split-half reliabilities):\n{p.round(3).to_string()}\n")
    d = dirs.groupby(["source", "target"], sort=False).agg(
        auc=("auc", "mean"), auc_min=("auc", "min"), auc_max=("auc", "max"),
        null_low=("null_low", "mean"), null_median=("null_median", "mean"), null_high=("null_high", "mean"),
        cosine=("cosine", "mean"),
        nets_p05=("p", lambda q: f"{(q < 0.05).sum()}/{q.notna().sum()}" if q.notna().any() else ""))
    lines.append(f"3. Shared direction: AUROC easy vs hard along the source effect's direction (source = target: split-half,"
                 f" within task; null = shuffled source labels, middle 95 %):\n{d.round(3).to_string()}\n")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    for name, df in (("overlap", ov), ("profiles", prof), ("directions", dirs)):
        df.to_csv(out / f"{name}.csv", index=False)
    log.info("\n".join(lines))


if __name__ == "__main__":
    main()
