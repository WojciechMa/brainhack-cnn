"""Shapley values and lesions of the 600 Dense units (layer 5) of a trained small joint CNN.

    python smallcnn/lesion.py --run runs/smallcnn/joint_fresh_seed0 --decision within
    python smallcnn/lesion.py --run runs/smallcnn/joint_fresh_seed0 --n-perm 20 --n-random 5   # quick

The Dense layer feeds the linear output layer directly (no batch norm in between).

Question: is a small group of units (say 5 % or 20 %) responsible for most of what the network does?
Two steps on separate halves of the test trials (ranking and testing on the same trials inflates the result):

1. Rank the units by Shapley value (multi-perturbation Shapley analysis, Keinan et al. 2004) on half A,
   with two kinds of value function ("game"):
   - logit games, as in the reference Shapley-perturbation code: a difference between condition means of a logit difference,
     e.g. mean z(valid) - mean z(invalid) with z = logit(present) - logit(absent). Everything after
     the Dense layer is linear, so these games are additive:
     phi_j = (unit j's mean activation difference between the conditions) x (its readout weight).
     Exact without sampling, units cannot interact, and the value a lesioned unit is set to cancels
     out: zeroing and mean ablation give identical values.
   - accuracy games: accuracy of the 6-way decisions per task, i.e. does the network still work.
     With --decision within, a trial is scored on its own task's two classes only, as in
     2-class single-task networks: a lesion that sends trials to another task's classes then does
     not count, which separates lost detection from lost task identity.
     The argmax makes them non-additive (redundant units matter only together), so the values are
     estimated by permutation sampling, separately for zeroing and mean ablation, with the standard
     error over permutations saved next to every value. Shapley values add up exactly to
     accuracy(intact) - accuracy(all units lesioned) in every permutation.
2. Lesion on half B: the top-ranked k % of units, k % random units (control, several draws) and the
   k % with the smallest |phi| (should change little), each with zeroing and mean ablation (mean taken
   on half A). Both controls are drawn from the live units only (active on some trial of half A):
   many ReLU units never fire, and lesioning those changes nothing, which would flatter the top units.
   Every lesion is scored with all task metrics (accuracy, d' and criterion per task, the three
   effects) and with AUC and threshold-refit accuracy (metrics.separation): accuracy alone cannot
   tell lost information from a lesion that pushes all trials to one answer. Rankings from one
   task's game scored on all tasks give the cross-task matrix: shared versus task-specific units.
   The condition matrix shows the AUC per condition (Posner valid / invalid, smallest / largest set
   size, old / new contexts) and so whether a lesion changes an effect or only a task's overall level.

Writes <run>/lesion/ (<run>/lesion_within/ with --decision within): shapley.csv (one row per unit),
lesions.csv (one row per lesion), lesion.log (appends), summary.txt (this run's tables),
lesion_curves.png. Other depths (conv layers): layer_lesions.py.
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
from torch.utils.data import DataLoader

from data import CLASSES, SET_SIZES, STIMULI, TASKS, NoisyStimuli, load
from metrics import separation, task_metrics, within_task
from model import DENSE, JointCNN
from train import setup_logging

LAYER = DENSE
MODES = ("zero", "mean")
MEASURES = {"acc": "accuracy", "acc_refit": "accuracy, threshold refit", "auc": "AUC"}  # chance 0.5 for all
CONDITIONS = {"posner valid": "posner_valid_auc", "posner invalid": "posner_invalid_auc", "cue effect": "posner_cue_effect_auc",
              f"search ss{SET_SIZES[0]}": f"search_ss{SET_SIZES[0]}_auc", f"search ss{SET_SIZES[-1]}": f"search_ss{SET_SIZES[-1]}_auc",
              "set-size effect": "search_set_size_effect_auc",
              "context old": "context_old_auc", "context new": "context_new_auc", "context effect": "context_effect_auc"}
EFFECT_GAMES = ("posner_validity", "search_set_size", "context_old_new")
DECISIONS = {"strict": lambda L, label: L.argmax(1), "within": within_task}
Z = {"posner": (1, 0), "search": (3, 2), "context": (5, 4)}  # z = logit[a] - logit[b] per task
log = logging.getLogger("smallcnn.lesion")


@torch.no_grad()
def activations(model, dataset, device, batch_size: int = 32) -> np.ndarray:
    model.eval()
    return torch.cat([model(x.to(device), upto=LAYER).cpu() for x, _ in DataLoader(dataset, batch_size=batch_size)]).numpy()


def readout(model) -> tuple[np.ndarray, np.ndarray]:
    """The output layer: logits = A @ W.T + b."""
    fc = model.layers[LAYER + 1]
    return fc.weight.detach().double().cpu().numpy(), fc.bias.detach().double().cpu().numpy()


def logit_games(meta: pd.DataFrame) -> dict[str, list[tuple[float, np.ndarray, str]]]:
    """Each game = sum of weight x mean z over a set of trials (mask), z of the given task."""
    lab, v, side = meta.label.to_numpy(), meta.validity.to_numpy(), meta.target.to_numpy()
    ss, cfg = meta.set_size.to_numpy(), meta.config.to_numpy()
    lo, hi = SET_SIZES[0], SET_SIZES[-1]
    P, S, C = TASKS
    return {
        "posner_present": [(1, lab == 1, P), (-1, lab == 0, P)],
        # validity averaged over target sides, as in the reference Shapley-perturbation code
        "posner_validity": [(0.5, (v == "valid") & (side == "left"), P), (-0.5, (v == "invalid") & (side == "left"), P),
                            (0.5, (v == "valid") & (side == "right"), P), (-0.5, (v == "invalid") & (side == "right"), P)],
        "search_present": [(1, lab == 3, S), (-1, lab == 2, S)],
        "search_set_size": [(1, (ss == lo) & (lab == 3), S), (-1, (ss == lo) & (lab == 2), S),
                            (-1, (ss == hi) & (lab == 3), S), (1, (ss == hi) & (lab == 2), S)],
        "context_tilt": [(1, lab == 5, C), (-1, lab == 4, C)],
        "context_old_new": [(1, (cfg == "old") & (lab == 5), C), (-1, (cfg == "old") & (lab == 4), C),
                            (-1, (cfg == "new") & (lab == 5), C), (1, (cfg == "new") & (lab == 4), C)],
    }


def game_value(L: np.ndarray, terms) -> float:
    return sum(w * (L[mask, Z[t][0]] - L[mask, Z[t][1]]).mean() for w, mask, t in terms)


def shapley_logit(A, base, W, terms) -> np.ndarray:
    """Exact Shapley values of an additive game: phi_j = v({j}) - v({})."""
    phi = np.zeros(A.shape[1])
    for w, mask, t in terms:
        phi += w * (A[mask] - base).mean(0) * (W[Z[t][0]] - W[Z[t][1]])
    return phi


def shapley_accuracy(A, base, W, b, labels, groups: dict[str, np.ndarray], n_perm: int, rng, decide=DECISIONS["strict"]):
    """Permutation-sampling Shapley values of accuracy within each trial group.

    Units are switched on one at a time in random order, starting with all of them set to `base`;
    the logits are updated by one unit's contribution per step. Returns mean and standard error
    over permutations, both (n_groups, n_units).
    """
    D = np.asfortranarray(A - base, dtype=np.float64)
    start = base @ W.T + b  # all units lesioned: the same logits on every trial
    G = np.stack([m / m.sum() for m in groups.values()])
    phi = np.empty((n_perm, len(groups), A.shape[1]))
    for p in range(n_perm):
        L = np.tile(start, (len(A), 1))
        prev = G @ (decide(L, labels) == labels)
        for j in rng.permutation(A.shape[1]):
            L += np.outer(D[:, j], W[:, j])
            cur = G @ (decide(L, labels) == labels)
            phi[p, :, j] = cur - prev
            prev = cur
        if (p + 1) % max(1, n_perm // 5) == 0:
            log.info(f"  permutation {p + 1}/{n_perm}")
    return phi.mean(0), phi.std(0, ddof=1) / np.sqrt(n_perm)


def split_half(meta: pd.DataFrame, rng) -> np.ndarray:
    """True = ranking half; random half of every condition cell."""
    half = np.zeros(len(meta), bool)
    for idx in meta.groupby(["set", "label", "validity", "target", "cue"], dropna=False).indices.values():
        half[rng.choice(idx, len(idx) // 2, replace=False)] = True
    return half


def plot_curves(lesions: pd.DataFrame, intact: dict, path: Path):
    fig = Figure(figsize=(4.2 * len(TASKS), 3.2 * len(MEASURES) * len(MODES)), layout="constrained")
    axes = fig.subplots(len(MEASURES) * len(MODES), len(TASKS), squeeze=False)
    for r, (mode, measure) in enumerate((m, k) for m in MODES for k in MEASURES):
        sel = lesions[lesions.lesion_mode == mode]
        for c, task in enumerate(TASKS):
            ax, col = axes[r, c], f"{task}_{measure}"
            rnd = sel[sel.selection == "random"].groupby("fraction")[col].agg(["mean", "std"])
            ax.plot(rnd.index, rnd["mean"], "k-", label="random units")
            ax.fill_between(rnd.index, rnd["mean"] - rnd["std"], rnd["mean"] + rnd["std"], color="k", alpha=0.15)
            for ranking, style, label in ((f"acc_{task}_{mode}", "C3-", "top: own task's accuracy"),
                                          (f"acc_all_{mode}", "C1--", "top: overall accuracy"),
                                          (f"logit_{task}_{'tilt' if task == 'context' else 'present'}", "C0-", "top: logit game")):
                top = sel[(sel.ranking == ranking) & (sel.selection == "top")].set_index("fraction")[col]
                ax.plot(top.index, top, style, marker="o", ms=3, label=label)
            least = sel[(sel.ranking == f"acc_{task}_{mode}") & (sel.selection == "least")].set_index("fraction")[col]
            ax.plot(least.index, least, "C2:", marker="o", ms=3, label="smallest |phi|")
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
    ap.add_argument("--n-perm", type=int, default=200, help="permutations per accuracy game")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--n-random", type=int, default=20, help="random lesions per fraction")
    ap.add_argument("--reps", type=int, default=1, help="noise draws per test image (more trials)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decision", choices=DECISIONS, default="strict",
                    help="strict: 6-way argmax; within: each trial's own task's two classes only")
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    args = ap.parse_args()
    out = args.run / ("lesion" if args.decision == "strict" else f"lesion_{args.decision}")
    setup_logging(out, "lesion.log")
    rng_split, rng_perm, rng_lesion = (np.random.default_rng((args.seed, i)) for i in range(3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=True)
    model = JointCNN(len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    images, meta = load("test", args.stimuli, ast.literal_eval(ckpt["args"]["noise"]))
    meta = pd.concat([meta] * args.reps)  # NoisyStimuli keys the noise by position: every copy gets its own
    test = NoisyStimuli(images, meta, int(ckpt["args"]["seed"]), stream=1)  # reps 1 = train.py's test set

    t0 = time.time()
    A = activations(model, test, device)
    W, b = readout(model)
    with torch.no_grad():
        x = torch.stack([test[i][0] for i in range(32)]).to(device)
        diff = np.abs(model(x).cpu().numpy() - (A[:32] @ W.T + b)).max()
    log.info(f"{args.run.name}: layer {LAYER} activations {A.shape} in {time.time() - t0:.0f} s;"
             f" readout vs model logits: max diff {diff:.1e}; units never active: {(A == 0).all(0).sum()}")

    meta = meta.reset_index(names="image")
    half = split_half(meta, rng_split)
    meta_a, meta_b = meta[half].reset_index(drop=True), meta[~half].reset_index(drop=True)
    A_a, A_b = A[half], A[~half]
    base = {"zero": np.zeros(A.shape[1]), "mean": A_a.mean(0)}
    live = np.flatnonzero((A_a > 0).any(0))
    log.info(f"ranking half {len(meta_a)} trials, lesion half {len(meta_b)} trials; live units {len(live)}")

    shap = pd.DataFrame({"unit": np.arange(A.shape[1]), "mean_activation": base["mean"]})
    L_a = A_a @ W.T + b
    for name, terms in logit_games(meta_a).items():
        phi = shapley_logit(A_a, base["mean"], W, terms)
        shap[f"logit_{name}"] = phi
        total = game_value(L_a, terms) - game_value(np.tile(base["mean"] @ W.T + b, (len(A_a), 1)), terms)
        assert np.isclose(phi.sum(), total, atol=1e-5), (name, phi.sum(), total)  # float32 sums; totals can be ~0

    groups = {"all": np.ones(len(meta_a), bool), **{t: (meta_a.task == t).to_numpy() for t in TASKS}}
    for mode in MODES:
        t0 = time.time()
        log.info(f"accuracy games, {mode} ablation: {args.n_perm} permutations")
        mean, se = shapley_accuracy(A_a, base[mode], W, b, meta_a.label.to_numpy(), groups, args.n_perm, rng_perm,
                                    DECISIONS[args.decision])
        for g, name in enumerate(groups):
            shap[f"acc_{name}_{mode}"], shap[f"acc_{name}_{mode}_se"] = mean[g], se[g]
        log.info(f"  done in {time.time() - t0:.0f} s")
    shap.to_csv(out / "shapley.csv", index=False)

    games = [c for c in shap.columns if c.startswith(("logit_", "acc_")) and not c.endswith("_se")]
    n = A.shape[1]
    report = []

    def note(text):
        log.info(text)
        report.append(text)

    lines = []
    for g in games:
        phi = np.sort(shap[g].to_numpy())[::-1]
        share = [phi[:max(1, round(f * n))].sum() / phi.sum() for f in (0.05, 0.2)]
        lines.append(f"  {g:28s} total {phi.sum():+8.3f} | top 5 % {share[0]:6.0%} | top 20 % {share[1]:6.0%}")
    note("share of each game's total carried by its top units (above 100 % when other units work against it):\n"
         + "\n".join(lines))

    L_b = A_b @ W.T + b
    decide, label_b = DECISIONS[args.decision], meta_b.label.to_numpy()
    intact = {**task_metrics(decide(L_b, label_b), meta_b), **separation(L_b, meta_b)}
    rows = [{"ranking": "none", "lesion_mode": "none", "selection": "intact", "fraction": 0.0, "n_units": 0, "draw": 0, **intact}]

    def lesion(units, mode, **info):
        L = L_b - (A_b[:, units] - base[mode][units]) @ W[:, units].T
        rows.append({**info, "lesion_mode": mode, "n_units": len(units),
                     **task_metrics(decide(L, label_b), meta_b), **separation(L, meta_b)})

    t0 = time.time()
    for g in games:
        phi = shap[g].to_numpy()
        top, least = np.argsort(-phi), live[np.argsort(np.abs(phi[live]))]
        for mode in (MODES if g.startswith("logit_") else [g.rsplit("_", 1)[1]]):
            for f in args.fractions:
                k = max(1, round(f * n))
                lesion(top[:k], mode, ranking=g, selection="top", fraction=f, draw=0)
                lesion(least[:k], mode, ranking=g, selection="least", fraction=f, draw=0)
    for mode in MODES:
        for f in args.fractions:
            for d in range(args.n_random):
                lesion(rng_lesion.choice(live, min(len(live), max(1, round(f * n))), replace=False), mode,
                       ranking="random", selection="random", fraction=f, draw=d)
    lesions = pd.DataFrame(rows)
    lesions.to_csv(out / "lesions.csv", index=False)
    log.info(f"{len(lesions)} lesions scored in {time.time() - t0:.0f} s")

    intact_text = ", ".join(f"{t} acc {intact[f'{t}_acc']:.3f} AUC {intact[f'{t}_auc']:.3f}" for t in TASKS)
    for mode in MODES:
        sel = lesions[lesions.lesion_mode == mode]
        rnd = sel[sel.selection == "random"].groupby("fraction")
        table = {}
        for task in TASKS:
            top = sel[(sel.ranking == f"acc_{task}_{mode}") & (sel.selection == "top")].set_index("fraction")
            for measure in MEASURES:
                table[f"{task} {measure} top"] = top[f"{task}_{measure}"]
                table[f"{task} {measure} random"] = rnd[f"{task}_{measure}"].mean()
            table[f"{task} other-task top"] = top[f"{task}_confusion"]
        note(f"{mode} lesions, top units of each task's accuracy game vs random units (lesion half; intact: {intact_text}):\n"
             + pd.DataFrame(table).round(3).T.to_string())
        for f in sorted(set(args.fractions) & {0.05, 0.2}):
            src_rows = {src: sel[(sel.ranking == f"acc_{src}_{mode}") & (sel.selection == "top") & (sel.fraction == f)].iloc[0]
                        for src in TASKS}
            rnd_f = sel[(sel.selection == "random") & (sel.fraction == f)]
            matrix = pd.DataFrame({f"lesion top of {src}": {f"{dst} {measure}": row[f"{dst}_{measure}"] - rnd_f[f"{dst}_{measure}"].mean()
                                                            for dst in TASKS for measure in MEASURES}
                                   for src, row in src_rows.items()}).T
            note(f"cross-task matrix, {mode} lesions of {f:.0%} of units: change vs random lesions"
                 f" (rows: whose top units, columns: task measured)\n{matrix.round(3).to_string()}")
            cols = list(CONDITIONS.values())
            top_of = {**{f"top of {t} accuracy": f"acc_{t}_{mode}" for t in TASKS},
                      **{f"top of {g} game": f"logit_{g}" for g in EFFECT_GAMES}}
            cond = pd.DataFrame({"intact": pd.Series(intact)[cols], "random": rnd_f[cols].mean(),
                                 **{name: sel[(sel.ranking == r) & (sel.selection == "top") & (sel.fraction == f)].iloc[0][cols]
                                    for name, r in top_of.items()}}).T.astype(float)
            cond.columns = list(CONDITIONS)
            note(f"conditions, {mode} lesions of {f:.0%} of units: AUC per condition, effects as AUC differences"
                 f" (random: mean of {args.n_random} draws, SD of the effects"
                 f" {', '.join(f'{rnd_f[c].std():.3f}' for c in cols if c.endswith('effect_auc'))})\n{cond.round(3).to_string()}")

    plot_curves(lesions, intact, out / "lesion_curves.png")
    (out / "summary.txt").write_text(f"{args.run.name}, decision {args.decision}, n_perm {args.n_perm}, reps {args.reps},"
                                     f" seed {args.seed}\n\n" + "\n\n".join(report) + "\n", encoding="utf-8")
    log.info(f"saved {out / 'shapley.csv'}, lesions.csv, summary.txt, lesion_curves.png")


if __name__ == "__main__":
    main()
