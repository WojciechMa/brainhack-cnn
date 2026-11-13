"""Which kinds of Dense units carry the Posner cueing effect? Classification as in Srivastava et al. (2025, PNAS, Figs. 2B, 3E, 5).

    python smallcnn/cue_units.py --runs runs/smallcnn/joint_fresh_v2_seed{0..9} runs/smallcnn/posner_fresh_v2_seed{0..2} --out runs/smallcnn/cue_units_v2

Per live unit, four AUROCs on the Posner test images (activation caches of neuron_cueing.py, 10 noise draws):
target at the left / right location vs no target, both without the cue box (target sensitivity per location);
cue box at the left / right vs no cue, both on no-target images (cue sensitivity per location; the same
images with the box painted over). A location counts as tuned when |AUROC - 0.5| >= 0.05 and q < 0.05
(cluster bootstrap over clean images, Benjamini-Hochberg over live units); the paper's threshold is in its SI.
Types: T / C = tuned to target / cue at either location; TC = both. For TC units the location pattern:
s = summation (same sign at both locations), o = opponent (opposite signs), 1 = single location, so TsCs,
TsCo, ToCs, ToCo as in the paper's Fig. 3E, plus single-location combinations.
Link to the behaviour: each unit's exact Shapley value for the network's cueing effect in the Posner logit
(lesion_within/shapley.csv, logit_posner_validity), which ranked the units for the lesions. Cueing units =
the top 5 % (30 units), the set whose removal abolished the effect. Same side: the unit prefers the same
location for the target (t_right - t_left) and for the cue (c_right - c_left), tuned or not.
Check without the no-cue images (never seen in training): side preferences from trained-on images only,
c_side = AUROC cue right vs cue left on no-target trials, t_side = AUROC target right vs left on target trials,
averaged over the two cue sides (so validity is balanced). Agreement with c_right - c_left and t_right - t_left,
and the same-side test repeated with them.
Writes <run>/cue_units/units.csv and, with --out, summary.txt and units.csv over runs.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import false_discovery_control, norm

from data import CROP, STIMULI, load, read_zip
from neuron_cueing import ALPHA, MIN_TUNING, auroc
from train import setup_logging

MEASURES = ("t_left", "t_right", "c_left", "c_right", "t_side", "c_side")
TOP = 0.05
log = logging.getLogger("smallcnn.cue_units")


def measures(X: torch.Tensor, cells: dict) -> torch.Tensor:
    t_side = (auroc(X, cells["Tc", "right", "left"], cells["Tc", "left", "left"])
              + auroc(X, cells["Tc", "right", "right"], cells["Tc", "left", "right"])) / 2
    return torch.stack([auroc(X, cells["T", "left"], cells["D"]), auroc(X, cells["T", "right"], cells["D"]),
                        auroc(X, cells["C", "left"], cells["D"]), auroc(X, cells["C", "right"], cells["D"]),
                        t_side, auroc(X, cells["C", "right"], cells["C", "left"])])


def pattern(a: int, b: int) -> str:
    return "s" if a == b != 0 else "o" if a == -b != 0 else "1"


def unit_type(s: np.ndarray) -> tuple[str, str]:
    t, c = s[:2], s[2:]
    tuned = ("T" if t.any() else "") + ("C" if c.any() else "")
    if tuned != "TC":
        return tuned or "untuned", ""
    return "TC", f"T{pattern(*t)}C{pattern(*c)}"


def run_one(run: Path, posner: pd.DataFrame, images: np.ndarray, image: np.ndarray, n_boot: int, rng, device) -> pd.DataFrame:
    A = np.load(run / "neuron_cueing" / "activations_reps10.npz")["A"]
    A0 = np.load(run / "neuron_cueing" / "activations_nocue_reps10.npz")["A"]
    reps = len(A) // len(posner)
    trials = pd.concat([posner] * reps, ignore_index=True)
    img = np.tile(image, reps)
    N = len(trials)
    validity, target, cue = (trials[c].to_numpy() for c in ("validity", "target", "cue"))
    absent = validity == "absent"
    masks = {("T", s): (N, ~absent & (target == s)) for s in ("left", "right")}
    masks |= {("C", s): (0, absent & (cue == s)) for s in ("left", "right")}
    masks["D"] = (N, absent)
    masks |= {("Tc", t, c): (0, ~absent & (target == t) & (cue == c)) for t in ("left", "right") for c in ("left", "right")}
    clusters = {k: [off + np.flatnonzero(m & (img == i)) for i in np.unique(img[m])] for k, (off, m) in masks.items()}
    X = torch.as_tensor(np.vstack([A, A0]), dtype=torch.float32, device=device)

    def to_cells(pick):
        return {k: torch.as_tensor(np.concatenate([cl[i] for i in pick(len(cl))]), device=device) for k, cl in clusters.items()}
    est = measures(X, to_cells(np.arange)).cpu().numpy()
    boot = torch.stack([measures(X, to_cells(lambda n: rng.integers(n, size=n))) for _ in range(n_boot)]).cpu().numpy()
    se = boot.std(0, ddof=1)

    live = (A > 0).any(0) | (A0 > 0).any(0)
    df = pd.DataFrame({"run": run.name, "unit": np.arange(A.shape[1]), "live": live})
    signs = np.zeros((A.shape[1], len(MEASURES)), int)
    for i, m in enumerate(MEASURES):
        df[m], df[f"{m}_se"] = est[i], se[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.nan_to_num(2 * norm.sf(np.abs(est[i] - 0.5) / se[i]), nan=1.0)
        q = np.ones_like(p)
        q[live] = false_discovery_control(p[live])
        tuned = live & (q < ALPHA) & (np.abs(est[i] - 0.5) >= MIN_TUNING)
        signs[:, i] = np.where(tuned, np.sign(est[i] - 0.5), 0).astype(int)
        df[f"{m}_sign"] = signs[:, i]
    df["tuning"], df["type"] = zip(*(unit_type(s) for s in signs[:, :4]))
    df.loc[~live, ["tuning", "type"]] = ["dead", ""]

    shap = pd.read_csv(run / "lesion_within" / "shapley.csv")
    nc = pd.read_csv(run / "neuron_cueing" / "units.csv", dtype={"unit": str})
    nc = nc[(nc.trials == "all") & (nc.unit != "output")]
    df["phi_cueing"] = shap.logit_posner_validity.to_numpy()
    df["neuronal_cue_effect"] = nc.cue_effect.to_numpy()
    k = round(TOP * len(df))
    df["top5"] = False
    df.loc[np.argsort(-df.phi_cueing.to_numpy())[:k], "top5"] = True
    out = run / "cue_units"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "units.csv", index=False)
    log.info(f"{run.name}: live {live.sum()}, tuning {df[live].tuning.value_counts().to_dict()},"
             f" top-5 % types {df[df.top5].type.replace('', 'not TC').value_counts().to_dict()}")
    return df


def pct(s: pd.Series) -> str:
    return f"{s.mean() * 100:.0f} % ({s.min() * 100:.0f}-{s.max() * 100:.0f})"


def summarise(df: pd.DataFrame) -> list[str]:
    df = df[df.live].copy()
    df["group"] = df.run.str.split("_").str[0]
    tc_kind = {"TsCs": "TsCs summation", "ToCo": "ToCo opponent", "TsCo": "TsCo/ToCs mixed", "ToCs": "TsCo/ToCs mixed"}
    df["kind"] = np.where(df.tuning == "TC", df.type.map(tc_kind).fillna("TC single location"),
                          df.tuning.map({"T": "target only", "C": "cue only", "untuned": "untuned"}))
    kinds = ["TsCs summation", "ToCo opponent", "TsCo/ToCs mixed", "TC single location", "target only", "cue only", "untuned"]
    lines = []
    for g, d in df.groupby("group", sort=False):
        size = d.groupby("run").size()
        lines.append(f"===== {g}: {len(size)} networks, live units {size.min()}-{size.max()}")
        tun = {}
        for c in ("t", "c"):
            for s, sym in ((1, "+"), (-1, "-")):
                hit = (d[f"{c}_left_sign"] == s) | (d[f"{c}_right_sign"] == s)
                tun[f"{c.upper()}{sym}"] = hit.groupby(d.run).mean()
        tun["TC"] = (d.tuning == "TC").groupby(d.run).mean()
        lines.append("Tuning, % of live units (mean, range over networks; paper's dense layer, % of all units: "
                     "T+ 24, T- 25, C+ 16, C- 17, TC 29):\n" + ", ".join(f"{k}: {pct(v)}" for k, v in tun.items()))
        rows = []
        for kind in kinds:
            r = {"type": kind}
            for name, sel in (("all live", d), ("top 5 % cueing", d[d.top5])):
                r[name] = pct(sel.groupby("run").kind.apply(lambda s: (s == kind).mean()))
            share = d.groupby("run").apply(lambda u: u.phi_cueing[u.kind == kind].sum() / u.phi_cueing.sum())
            r["share of cue effect"] = pct(share)
            r["mean neuronal cue effect"] = f"{d.neuronal_cue_effect[d.kind == kind].mean() * 100:+.1f} points"
            rows.append(r)
        lines.append("Types (% of units; share of cue effect = the type's summed Shapley value / the network's cue effect"
                     " in the logit, can be negative or above 100 %):\n" + pd.DataFrame(rows).to_string(index=False))
        tc = d[d.tuning == "TC"]
        lines.append("Among TC units only (paper Fig. 3E dense: TsCs 22, TsCo 39, ToCs 3, ToCo 19): " + ", ".join(
            f"{t} {pct(tc.groupby('run').type.apply(lambda s, t=t: (s == t).mean()))}" for t in ("TsCs", "TsCo", "ToCs", "ToCo")))
        d = d.assign(same_side=np.sign(d.t_right - d.t_left) == np.sign(d.c_right - d.c_left))
        for name, sel in (("all live TC units", d[d.tuning == "TC"]), ("top-5 % cueing units", d[d.top5])):
            same = sel.same_side.groupby(sel.run).mean()
            share = sel.groupby("run").apply(lambda u: u.phi_cueing[u.same_side].sum() / u.phi_cueing.sum())
            lines.append(f"{name}: same side preferred by target and cue in {pct(same)} (chance 50 %);"
                         f" their share of these units' summed cue effect {pct(share)}")
        lines.append("Check with trained-on images only (t_side, c_side; no painted-over cue):")
        for c, old in (("c", d.c_right - d.c_left), ("t", d.t_right - d.t_left)):
            new = d[f"{c}_side"] - 0.5
            rho = pd.DataFrame({"o": old, "n": new, "run": d.run}).groupby("run").apply(lambda u: u.o.corr(u.n, method="spearman"))
            both = (d[f"{c}_side_sign"] != 0) & ((d[f"{c}_left_sign"] != 0) | (d[f"{c}_right_sign"] != 0))
            agree = (np.sign(new) == np.sign(old))[both].groupby(d.run[both]).mean()
            tuned = (d[f"{c}_side_sign"] != 0).groupby(d.run).mean()
            lines.append(f"  {'cue' if c == 'c' else 'target'} side: tuned to a side {pct(tuned)} of live units; Spearman with"
                         f" the no-cue-image preference {pct(rho)}; same sign in {pct(agree)} of units tuned in both")
        d = d.assign(same_new=np.sign(d.t_side - 0.5) == np.sign(d.c_side - 0.5))
        for name, sel in (("all live TC units", d[d.tuning == "TC"]), ("top-5 % cueing units", d[d.top5])):
            same = sel.same_new.groupby(sel.run).mean()
            share = sel.groupby("run").apply(lambda u: u.phi_cueing[u.same_new].sum() / u.phi_cueing.sum())
            lines.append(f"  {name}: same side {pct(same)}; their share of these units' summed cue effect {pct(share)}")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    setup_logging(args.out or args.runs[0] / "cue_units", "cue_units.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, meta = load("test", args.stimuli)  # images only: noise is in the cached activations
    posner = meta[meta.task == "posner"]
    _, names = read_zip(args.stimuli / f"{posner.set.iloc[0]}.zip", CROP["posner"], no_cue=True)
    assert names == tuple(posner.file)
    image = pd.factorize(pd.Series([images[r].tobytes() for r in posner.index]))[0]
    rng = np.random.default_rng(args.seed)
    df = pd.concat([run_one(r, posner, images, image, args.n_boot, rng, device) for r in args.runs])
    if args.out:
        df.to_csv(args.out / "units.csv", index=False)
        lines = summarise(df)
        (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
        log.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
