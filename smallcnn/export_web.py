"""Exports one trained JointCNN for the web demo in docs/: conv, batch-norm and output weights in
model/meta.json, the Dense weights as int8 with one float32 scale per block of 54 inputs, example
stimuli from the test zips as PNGs, and a check set (noisy inputs + expected logits) for the JS code.
For the lesion panel: model/lesions.json (unit rankings and lesion results from lesion.py --decision
within, this run and the mean over its seed replicates) and model/trials.bin + trials.json (gzipped
clean test images of the lesion half, for live trials in the browser).

    python smallcnn/export_web.py --run runs/smallcnn/joint_fresh_v2_seed0
"""
import argparse
import ast
import base64
import gzip
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from data import CLASSES, STIMULI, NoisyStimuli, load
from lesion import split_half
from model import JointCNN

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 54
GROUPS = {"cue": "logit_posner_validity", "set_size": "logit_search_set_size", "context": "logit_context_old_new"}
EFFECTS = ["posner_cue_effect_auc", "search_set_size_effect_auc", "context_effect_auc"]
FRACTIONS = [0.05, 0.1, 0.2]
TRIAL_CONDITIONS = {  # name: query on the test meta; Posner absent trials by cue side, the others by target side
    "pvL": "validity == 'valid' and target == 'left'", "pvR": "validity == 'valid' and target == 'right'",
    "piL": "validity == 'invalid' and target == 'left'", "piR": "validity == 'invalid' and target == 'right'",
    "paL": "validity == 'absent' and cue == 'left'", "paR": "validity == 'absent' and cue == 'right'",
    "s1p": "set_size == 1 and label == 3", "s1a": "set_size == 1 and label == 2",
    "s8p": "set_size == 8 and label == 3", "s8a": "set_size == 8 and label == 2",
    "oL": "config == 'old' and label == 4", "oR": "config == 'old' and label == 5",
    "nL": "config == 'new' and label == 4", "nR": "config == 'new' and label == 5",
}
NEW_PER_SIDE = 128  # new layouts carry baked-in noise, so every image is unique and large once compressed;
# fewer layouts make the lesioned context effect depend on the sample (SD 6 points with 32 per side, 2 with 128)
EXAMPLES = [  # (stimulus set, condition, label shown on the site)
    ("Posnerb179_80test", "target == 'left' and cue == 'left'", "Posner: cue left, target left (valid)"),
    ("Posnerb179_80test", "target == 'right' and cue == 'right'", "Posner: cue right, target right (valid)"),
    ("Posnerb179_80test", "target == 'right' and cue == 'left'", "Posner: cue left, target right (invalid)"),
    ("Posnerb179_80test", "target == 'left' and cue == 'right'", "Posner: cue right, target left (invalid)"),
    ("Posnerb179_80test", "target == 'absent' and cue == 'left'", "Posner: cue left, no target"),
    ("Posnerb179_80test", "target == 'absent' and cue == 'right'", "Posner: cue right, no target"),
    ("VSss1angle9testn0", "label == 3", "Search: 1 box, target present"),
    ("VSss1angle9testn0", "label == 2", "Search: 1 box, no target"),
    ("VSss8angle9testn0", "label == 3", "Search: 8 boxes, target present"),
    ("VSss8angle9testn0", "label == 2", "Search: 8 boxes, no target"),
    ("cco75n0", "label == 4", "Contextual cueing: old layout, T tilted left"),
    ("cco75n0", "label == 5", "Contextual cueing: old layout, T tilted right"),
    ("ccn75n5test", "label == 4", "Contextual cueing: new layout, T tilted left"),
    ("ccn75n5test", "label == 5", "Contextual cueing: new layout, T tilted right"),
]


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def lesion_results(lesions: pd.DataFrame) -> dict:
    """Effects intact, after zeroing each group's top units and after random live units (mean of the draws)."""
    z = lesions[lesions.lesion_mode == "zero"]
    out = {"intact": lesions[lesions.selection == "intact"].iloc[0][EFFECTS].tolist(), "top": {}, "random": {}}
    for f in FRACTIONS:
        at = z[np.isclose(z.fraction, f)]
        for g, ranking in GROUPS.items():
            out["top"].setdefault(g, {})[str(f)] = at[(at.ranking == ranking) & (at.selection == "top")].iloc[0][EFFECTS].tolist()
        out["random"][str(f)] = at[at.selection == "random"][EFFECTS].mean().tolist()
    return out


def mean_results(results: list[dict]) -> dict:
    def avg(get):
        return np.mean([get(r) for r in results], axis=0).round(5).tolist()
    return {"intact": avg(lambda r: r["intact"]),
            "top": {g: {str(f): avg(lambda r: r["top"][g][str(f)]) for f in FRACTIONS} for g in GROUPS},
            "random": {str(f): avg(lambda r: r["random"][str(f)]) for f in FRACTIONS}}


def export_lesions(run: Path, out: Path, images: np.ndarray, meta: pd.DataFrame):
    run = run.resolve()
    shap = pd.read_csv(run / "lesion_within" / "shapley.csv")
    replicates = sorted(p for p in run.parent.glob(re.sub(r"seed\d+$", "seed*", run.name))
                        if (p / "lesion_within" / "lesions.csv").exists())
    per_run = [lesion_results(pd.read_csv(p / "lesion_within" / "lesions.csv")) for p in replicates]
    this = per_run[replicates.index(run)]
    doc = {
        "effects": EFFECTS, "fractions": FRACTIONS, "n_units": len(shap),
        "live": np.flatnonzero(shap.mean_activation.to_numpy() > 0).tolist(),
        "rank": {g: np.argsort(-shap[col].to_numpy(), kind="stable").tolist() for g, col in GROUPS.items()},
        "this": this, "mean": mean_results(per_run), "n_runs": len(per_run),
    }
    (out / "model" / "lesions.json").write_text(json.dumps(doc), encoding="utf-8")

    # lesion.py's split: rankings came from half A, so live trials use half B only
    half = split_half(meta.reset_index(names="image"), np.random.default_rng((0, 0)))
    b = meta[~half]
    rng = np.random.default_rng(1)
    tiles, tile_of, trials = [], {}, []
    for c, (name, query) in enumerate(TRIAL_CONDITIONS.items()):
        rows = b.query(query)
        if name in ("nL", "nR"):
            rows = rows.loc[rng.choice(rows.index, NEW_PER_SIDE, replace=False)]
        for i, row in rows.iterrows():
            key = images[i].tobytes()
            if key not in tile_of:
                tile_of[key] = len(tiles)
                tiles.append(images[i])
            trials.append([tile_of[key], c, round(float(row.noise_sd), 2)])
    (out / "model" / "trials.bin").write_bytes(gzip.compress(np.stack(tiles).tobytes(), 9))
    (out / "model" / "trials.json").write_text(json.dumps({"conditions": list(TRIAL_CONDITIONS), "n_tiles": len(tiles),
                                                          "trials": trials}), encoding="utf-8")
    print(f"lesions: {len(per_run)} runs ({', '.join(p.name for p in replicates)}); live trials: {len(trials)} trials, "
          f"{len(tiles)} distinct images")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=True)
    model = JointCNN(len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    noise = ast.literal_eval(ckpt["args"]["noise"])
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}

    W = sd["layers.5.0.weight"]
    q = W.reshape(W.shape[0], -1, BLOCK)
    scales = np.abs(q).max(2) / 127
    q = np.round(q / scales[..., None]).astype(np.int8)
    W8 = (q * scales[..., None]).reshape(W.shape).astype(np.float32)

    def logits_int8(flat: np.ndarray) -> np.ndarray:
        r = np.maximum(flat @ W8.T + sd["layers.5.0.bias"], 0)
        return r @ sd["layers.6.weight"].T + sd["layers.6.bias"]

    @torch.no_grad()
    def features(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        flat = model(x.to(dev), upto=4)
        return flat.cpu().numpy(), model.layers[6](model.layers[5](flat)).cpu().numpy()

    # agreement of the int8 copy with the original on the full test set (train.py's test noise)
    images, meta = load("test", args.stimuli, noise)
    test = NoisyStimuli(images, meta, int(ckpt["args"]["seed"]), stream=1)
    same, worst = 0, 0.0
    for s in range(0, len(test), 256):
        x = torch.stack([test[i][0] for i in range(s, min(s + 256, len(test)))])
        flat, ref = features(x)
        z = logits_int8(flat)
        same += int((z.argmax(1) == ref.argmax(1)).sum())
        worst = max(worst, float(np.abs(z - ref).max()))
    print(f"int8 copy: same answer on {same} of {len(test)} test images, max |logit diff| {worst:.3f}")

    (args.out / "model").mkdir(parents=True, exist_ok=True)
    (args.out / "stimuli").mkdir(parents=True, exist_ok=True)
    # input-major, so the browser can skip the zero inputs: q[i, j], scales[block, j]
    np.ascontiguousarray(q.reshape(W.shape).T).tofile(args.out / "model" / "dense_q.bin")
    np.ascontiguousarray(scales.T, np.float32).tofile(args.out / "model" / "dense_s.bin")
    bn_scale = sd["layers.3.weight"] / np.sqrt(sd["layers.3.running_var"] + model.layers[3].eps)
    conv = {}
    for name, i in (("conv1", 0), ("conv2", 1), ("conv3", 2)):
        m = model.layers[i][0]
        conv[name] = {"in": m.in_channels, "out": m.out_channels, "k": m.kernel_size[0], "stride": m.stride[0],
                      "w": b64(sd[f"layers.{i}.0.weight"]), "b": b64(sd[f"layers.{i}.0.bias"])}
    meta_json = {
        "source": args.run.name, "classes": CLASSES, "noise": noise, "size": 224, "conv": conv,
        "bn": {"scale": b64(bn_scale), "shift": b64(sd["layers.3.bias"] - sd["layers.3.running_mean"] * bn_scale)},
        "dense": {"in": W.shape[1], "out": W.shape[0], "block": BLOCK, "bias": b64(sd["layers.5.0.bias"]),
                  "weights": "dense_q.bin", "scales": "dense_s.bin", "layout": "input-major"},
        "out": {"w": b64(sd["layers.6.weight"]), "b": b64(sd["layers.6.bias"])},
        "agreement": {"same": same, "n": len(test), "max_logit_diff": round(worst, 4)},
    }
    (args.out / "model" / "meta.json").write_text(json.dumps(meta_json), encoding="utf-8")

    rng = np.random.default_rng(0)
    examples, check = [], []
    for k, (set_name, cond, label) in enumerate(EXAMPLES):
        rows = meta[meta.set == set_name].query(cond)
        row = rows.loc[rng.choice(rows.index)]
        name = f"{k:02d}_{set_name}_{row.file}.png"
        img = images[row.name]
        cv2.imwrite(str(args.out / "stimuli" / name), img)
        examples.append({"file": name, "label": label, "task": row.task, "class": CLASSES[row.label],
                         "noise_sd": round(float(row.noise_sd), 2), "source": f"{set_name}.zip/{row.file}.png"})
        x = np.clip(np.round(img + rng.standard_normal(img.shape) * row.noise_sd), 0, 255).astype(np.uint8)
        flat, ref = features(torch.from_numpy(x.astype(np.float32) / 255)[None, None])
        check.append({"file": name, "input": base64.b64encode(x.tobytes()).decode(),
                      "logits_torch": ref[0].round(5).tolist(), "logits_int8": logits_int8(flat)[0].round(5).tolist()})
    (args.out / "stimuli" / "examples.json").write_text(json.dumps(examples, indent=1), encoding="utf-8")
    (args.out / "model" / "check.json").write_text(json.dumps(check), encoding="utf-8")
    export_lesions(args.run, args.out, images, meta)
    print(f"wrote {args.out}: {len(examples)} example stimuli; model files "
          + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted((args.out / 'model').iterdir())))


if __name__ == "__main__":
    main()
