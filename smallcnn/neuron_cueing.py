"""Neuronal cueing effect of the 600 Dense units (layer 5) on the Posner trials.

    python smallcnn/neuron_cueing.py --run runs/smallcnn/joint_fresh_seed0

Srivastava et al. (2025, PNAS, Fig. 2B) define a neuron's cueing effect as its target-vs-distractor
AUROC on valid trials minus that on invalid trials. Each AUROC here is computed as metrics.separation
does for the network's decision value: target trials of one validity and target side against
no-target trials with the cue on the same side as theirs, averaged over the two sides. The same
numbers for the decision value z = logit(present) - logit(absent) give the network-level effect
(posner_cue_effect_auc). For a target-inhibitory unit (AUROC below 0.5), better discrimination on
valid trials is a negative difference.

Trials: the 1000 Posner test images, each with --reps noise draws (draw 0 = train.py's test set), in
two sets: all target trials, and without the target images that are pixel-identical to a no-target
image. Those are 55 % of the valid and 62 % of the invalid target images, a difference that alone
lowers the invalid AUROC of every target-tuned unit.
Uncertainty: bootstrap over clean images within each condition cell, with each image's noise draws
kept together (the test set has only 222 distinct clean images); p from the bootstrap standard error,
Benjamini-Hochberg over the live units. Target tuning (all target vs all no-target trials): T+ / T-
when |AUROC - 0.5| >= 0.05 and q < 0.05 (the reference Shapley-perturbation code's criterion), otherwise untuned. The paper's
criterion for counting a neuron is in its SI, which we do not have. All units see the same images,
so their errors are correlated and a shift shared by many units is not evidence by itself: the
population test is the tuned units' mean effect x tuning sign, with the same bootstrap. The link to
the readout is a rank correlation (robust to rarely firing units with very large weights).
No-cue trials (the paper's code, e.g. Posner_neutral.ipynb, cueing = AUROC with the cue at the target
minus AUROC with no cue): the same test images with the cue box painted over (data.read_zip no_cue) and
the same noise draws; target trials of each side against all no-target no-cue trials. The networks never
saw a cue-free image in training. The paper's code counts a unit when |valid - no cue| > 0.05
("paper rule", no test); summary.txt gives that count out of the live and out of the tuned units.
Writes <run>/neuron_cueing/ (or --out): activations_reps<R>.npz and activations_nocue_reps<R>.npz
(caches), units.csv (one row per unit
and trial set; unit "output" = the decision value z), population.csv, summary.txt, neuron_cueing.log
(appends).
"""
import argparse
import ast
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import false_discovery_control, norm, spearmanr

from data import CLASSES, CROP, STIMULI, NoisyStimuli, load, read_zip
from lesion import LAYER, activations, readout
from model import JointCNN
from train import setup_logging

ALPHA, MIN_TUNING, PAPER_RULE = 0.05, 0.05, 0.05
SIDES = (("left", "right"), ("right", "left"))
MEASURES = ("auc_target", "auc_valid", "auc_invalid", "cue_effect", "auc_nocue", "nocue_effect")
CUEING = ("valid > invalid", "valid < invalid")
log = logging.getLogger("smallcnn.neuron_cueing")


def auroc(X: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """AUROC of every column of X: P(pos > neg) + P(tie) / 2 (ties are common: ReLU zeros)."""
    n = X[neg].T.contiguous().sort(dim=1).values
    p = X[pos].T.contiguous()
    count = torch.searchsorted(n, p) + torch.searchsorted(n, p, right=True)
    return count.sum(1, dtype=torch.float64) / (2 * len(pos) * len(neg))


def measures(X: torch.Tensor, cells: dict) -> torch.Tensor:
    """(6, columns) in MEASURES order; cells[validity, side]: target side, or cue side for "absent";
    cells["nocue", side]: target side; cells["nocue_absent"]: no-target trials without a cue."""
    present = torch.cat([cells[v, s] for v in ("valid", "invalid") for s, _ in SIDES])
    absent = torch.cat([cells["absent", s] for s, _ in SIDES])
    valid = sum(auroc(X, cells["valid", s], cells["absent", s]) for s, _ in SIDES) / 2
    invalid = sum(auroc(X, cells["invalid", s], cells["absent", o]) for s, o in SIDES) / 2
    nocue = sum(auroc(X, cells["nocue", s], cells["nocue_absent"]) for s, _ in SIDES) / 2
    return torch.stack([auroc(X, present, absent), valid, invalid, valid - invalid, nocue, valid - nocue])


def analyse(X: torch.Tensor, clusters: dict, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Estimates (6, columns) and bootstrap draws (n_boot, 6, columns); clusters resampled within cells."""
    def to_cells(pick):
        return {k: torch.as_tensor(np.concatenate([cl[i] for i in pick(len(cl))]), device=X.device)
                for k, cl in clusters.items()}
    est = measures(X, to_cells(np.arange))
    boot = torch.stack([measures(X, to_cells(lambda n: rng.integers(n, size=n))) for _ in range(n_boot)])
    return est.cpu().numpy(), boot.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder with model.pt")
    ap.add_argument("--reps", type=int, default=10, help="noise draws per test image (10 = 10,000 trials, as in the paper)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="32 = train.py's (TF32 convolutions round differently at other batch sizes)")
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--out", type=Path, help="output folder (default: <run>/neuron_cueing)")
    args = ap.parse_args()
    out = args.out or args.run / "neuron_cueing"
    setup_logging(out, "neuron_cueing.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=True)
    model = JointCNN(len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    images, meta = load("test", args.stimuli, ast.literal_eval(ckpt["args"]["noise"]))
    posner = meta[meta.task == "posner"]
    assert (posner.index == np.arange(len(posner))).all()  # first in the test split, so draw 0 = train.py's noise
    trials = pd.concat([posner] * args.reps)  # NoisyStimuli keys the noise by position: every copy gets its own
    nocue_images, names = read_zip(args.stimuli / f"{posner.set.iloc[0]}.zip", CROP["posner"], no_cue=True)
    assert names == tuple(posner.file)
    seed = int(ckpt["args"]["seed"])

    def cached(path, imgs):
        if path.exists():
            log.info(f"{args.run.name}: activations from {path}")
            return np.load(path)["A"]
        t0 = time.time()
        A = activations(model, NoisyStimuli(imgs, trials, seed, stream=1), device, args.batch_size)
        np.savez(path, A=A)
        log.info(f"{args.run.name}: layer-{LAYER} activations of {len(posner)} Posner test images x {args.reps} noise draws"
                 f" in {time.time() - t0:.0f} s ({path.name})")
        return A

    A = cached(out / f"activations_reps{args.reps}.npz", images)
    A_nocue = cached(out / f"activations_nocue_reps{args.reps}.npz", nocue_images)  # same noise as the cued trials
    W, b = readout(model)
    w = W[1] - W[0]  # each unit's weight in z
    z = A @ w + b[1] - b[0]
    z_nocue = A_nocue @ w + b[1] - b[0]
    saved = pd.read_csv(args.run / "test_predictions.csv").iloc[:len(posner)]
    diff = np.abs(z[:len(posner)] - (saved.logit_posner_present - saved.logit_posner_absent).to_numpy()).max()
    log.info(f"z of noise draw 0 vs test_predictions.csv: max diff {diff:.1e}")

    image = pd.factorize(pd.Series([images[r].tobytes() for r in posner.index]))[0]  # clean image id
    validity, target, cue = (trials[c].to_numpy() for c in ("validity", "target", "cue"))
    image = np.tile(image, args.reps)
    same = (validity != "absent") & np.isin(image, image[validity == "absent"])
    live = (A > 0).any(0)
    N = len(trials)  # rows N.. of X are the no-cue copies of rows 0..N-1
    X = torch.as_tensor(np.vstack([np.column_stack([A, z]), np.column_stack([A_nocue, z_nocue])]),
                        dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed)

    rows, lines, pop = [], [], []
    for name, keep in (("all", np.ones(len(trials), bool)), ("distinct", ~same)):
        masks = {**{(v, s): keep & (validity == v) & (target == s) for v in ("valid", "invalid") for s, _ in SIDES},
                 **{("absent", s): (validity == "absent") & (cue == s) for s, _ in SIDES}}
        clusters = {k: [np.flatnonzero(m & (image == i)) for i in np.unique(image[m])] for k, m in masks.items()}
        for s, _ in SIDES:
            m = keep & (validity != "absent") & (target == s)
            clusters["nocue", s] = [N + np.flatnonzero(m & (image == i)) for i in np.unique(image[m])]
        m = validity == "absent"
        clusters["nocue_absent"] = [N + np.flatnonzero(m & (image == i)) for i in np.unique(image[m])]
        t0 = time.time()
        est, boot = analyse(X, clusters, args.n_boot, rng)
        log.info(f"{name}: {args.n_boot} bootstrap samples in {time.time() - t0:.0f} s")

        df = pd.DataFrame({"trials": name, "unit": [*range(A.shape[1]), "output"], "live": [*live, True],
                           "weight": [*w, np.nan], "mean_activation": [*A[keep].mean(0), np.nan]})
        se = boot.std(0, ddof=1)
        for i, m in enumerate(MEASURES):
            df[m], df[f"{m}_se"] = est[i], se[i]
        df["ci_low"], df["ci_high"] = np.quantile(boot[:, 3], [0.025, 0.975], axis=0)
        units = df.live & (df.unit != "output")
        for col, centre in (("cue_effect", 0.0), ("nocue_effect", 0.0), ("auc_target", 0.5)):
            with np.errstate(divide="ignore", invalid="ignore"):
                p = 2 * norm.sf(np.abs(df[col] - centre) / df[f"{col}_se"])
            df[f"{col}_p"] = np.nan_to_num(p, nan=1.0)
            df[f"{col}_q"] = np.nan
            df.loc[units, f"{col}_q"] = false_discovery_control(df.loc[units, f"{col}_p"])
        tuned = (df.auc_target_q < ALPHA) & ((df.auc_target - 0.5).abs() >= MIN_TUNING)
        df["tuning"] = np.where(~units, "", np.where(tuned, np.where(df.auc_target > 0.5, "T+", "T-"), "untuned"))
        df["cueing"] = np.where(df.cue_effect_q < ALPHA, np.where(df.cue_effect > 0, *CUEING), "")
        df["paper_rule"] = units & (df.nocue_effect.abs() > PAPER_RULE)
        rows.append(df)

        o = df[df.unit == "output"].iloc[0]
        n = {k: int(sum(len(c) for c in v)) for k, v in clusters.items()}
        lu = df[units]
        sign = np.where(df.tuning == "T+", 1.0, np.where(df.tuning == "T-", -1.0, 0.0))
        pop_boot = boot[:, 3] @ sign / np.abs(sign).sum()
        pop.append({"trials": name, "measure": "tuned units: mean cueing effect x tuning sign",
                    "estimate": est[3] @ sign / np.abs(sign).sum(), "se": pop_boot.std(ddof=1),
                    **dict(zip(("ci_low", "ci_high"), np.quantile(pop_boot, [0.025, 0.975])))})
        pop.append({"trials": name, "measure": "output z: cueing effect", "estimate": o.cue_effect,
                    "se": o.cue_effect_se, "ci_low": o.ci_low, "ci_high": o.ci_high})
        table = lu.groupby("tuning").agg(units=("unit", "size"),
                                         **{c: ("cueing", lambda s, c=c: (s == c).sum()) for c in CUEING},
                                         median_effect=("cue_effect", "median"))
        table.loc["all live"] = [len(lu), *((lu.cueing == c).sum() for c in CUEING), lu.cue_effect.median()]
        tuned_units = lu[lu.tuning != "untuned"]
        lines += [f"--- {name} target trials: valid {n['valid', 'left'] + n['valid', 'right']}, invalid "
                  f"{n['invalid', 'left'] + n['invalid', 'right']}, no-target {n['absent', 'left'] + n['absent', 'right']}"
                  f" ({args.reps} noise draws each); distinct clean images per cell: "
                  + ", ".join(f"{k[0]} {k[1]} {len(c)}" for k, c in clusters.items() if isinstance(k, tuple) and k[0] != "nocue"),
                  f"output z: AUROC valid {o.auc_valid:.3f}, invalid {o.auc_invalid:.3f}, cueing {o.cue_effect:+.3f}"
                  f" [{o.ci_low:+.3f}, {o.ci_high:+.3f}]; no cue {o.auc_nocue:.3f}, valid - no cue {o.nocue_effect:+.3f}"
                  f" (SE {o.nocue_effect_se:.3f})",
                  f"live units {len(lu)}; neuronal cueing at q < {ALPHA} (BH), by target tuning:",
                  table.to_string(float_format=lambda x: f"{x:+.3f}"),
                  f"paper rule (|valid - no cue| > {PAPER_RULE}, no test): {lu.paper_rule.sum()} of {len(lu)} live units"
                  f" ({100 * lu.paper_rule.mean():.0f} %), {tuned_units.paper_rule.sum()} of {len(tuned_units)} tuned"
                  f" ({100 * tuned_units.paper_rule.mean() if len(tuned_units) else 0:.0f} %); valid - no cue at q < {ALPHA}:"
                  f" {(lu.nocue_effect_q < ALPHA).sum()} live units",
                  f"tuned units, mean of cueing effect x tuning sign (+ = better on valid trials): {pop[-2]['estimate']:+.3f}"
                  f" [{pop[-2]['ci_low']:+.3f}, {pop[-2]['ci_high']:+.3f}]",
                  f"Spearman r(cueing effect, weight in z) over live units: {spearmanr(lu.cue_effect, lu.weight).statistic:+.2f}"
                  f" (target AUROC vs weight: {spearmanr(lu.auc_target, lu.weight).statistic:+.2f})", ""]

    pd.concat(rows).to_csv(out / "units.csv", index=False)
    pd.DataFrame(pop).to_csv(out / "population.csv", index=False)
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
