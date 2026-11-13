"""Stimuli of Srivastava, Yang & Eckstein (2024) for one network trained on all three tasks.

Follows the data setup of the paper's joint-training notebook (One_Network_All_Three_Tasks5VS.ipynb):
Posner 2000, visual search 4 x 1200, contextual cueing 1120 old + 1120 new training images,
Gaussian pixel noise per task, 6 classes, no task cue. Images come as the notebook's small CNN takes
them: (1, 224, 224) float32 in 0-1.
Deviations: visual search uses the angle-9 zips (the notebook read angle-8 folders, not in the
Example Stimuli) and is tested on its separate test zips (the notebook re-noised the training images).

    python smallcnn/data.py        # summary of both splits
"""
import logging
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
STIMULI = (ROOT / "input_docs" / "source_code" / "sc_Emergent Human-Like Covert Attention in Feedforward "
           "Convolutional Neural Networks" / "Example Stimuli")
SIZE = 224
TASKS = ("posner", "search", "context")
CLASSES = ("posner_absent", "posner_present", "search_absent", "search_present",
           "context_tilt_left", "context_tilt_right")
NOISE = {"posner": 0.07, "search": 0.04, "context": 0.09}  # std on the 0-1 scale, from the notebook
SET_SIZES = (1, 2, 4, 8)
CROP = {"posner": (73, 427), "search": (100, 400), "context": None}  # columns kept before resizing
CUE_BOX, BACKGROUND = 179, 128  # Posner grey levels: the box outline is the only pixels at 179
log = logging.getLogger("smallcnn.data")

SPLITS = {
    "train": [("posner", "Posnerb179_80"), *[("search", f"VSss{s}angle9n0") for s in SET_SIZES],
              ("context", "cco75n0"), ("context", "ccn75n0")],
    # old contextual cueing test = the old training layouts with new noise (the effect needs seen layouts)
    "test": [("posner", "Posnerb179_80test"), *[("search", f"VSss{s}angle9testn0") for s in SET_SIZES],
             ("context", "cco75n0"), ("context", "ccn75n5test")],
}


@lru_cache(maxsize=None)
def read_zip(path: Path, crop: tuple[int, int] | None, no_cue: bool = False) -> tuple[np.ndarray, tuple[str, ...]]:
    """Grey uint8 images (N, 224, 224), sorted by file name, and their names without extension.
    no_cue: Posner images with the cue box painted over in the background grey."""
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".png"))
        images = np.empty((len(names), SIZE, SIZE), np.uint8)
        for i, n in enumerate(names):
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
            if no_cue:
                img[img == CUE_BOX] = BACKGROUND
            if crop:
                img = img[:, crop[0]:crop[1]]
            images[i] = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    return images, tuple(Path(n).stem for n in names)


def parse_names(task: str, set_name: str, names: tuple[str, ...]) -> pd.DataFrame:
    """Label (0-5) and trial conditions from the file names."""
    parts = [n.split("_") for n in names]
    if task == "posner":  # target_{absent|left|right}_box_{left|right}{index}
        target = [p[1] for p in parts]
        cue = [p[3].rstrip("0123456789") for p in parts]
        return pd.DataFrame({
            "label": [0 if t == "absent" else 1 for t in target],
            "target": target, "cue": cue,
            "validity": ["absent" if t == "absent" else "valid" if t == c else "invalid" for t, c in zip(target, cue)],
        })
    if task == "search":  # tc{k}_{index} = target present, ntc{k}_{index} = absent
        return pd.DataFrame({
            "label": [2 if n.startswith("n") else 3 for n in names],
            "set_size": int(set_name[4]),
        })
    # cue_combloc_{combloc}_gt_{1 = T tilted left | 2 = right}_{target location}_{index}
    return pd.DataFrame({
        "label": [3 + int(p[4]) for p in parts],
        "config": "old" if set_name.startswith("cco") else "new",
        "combloc": [int(p[2]) for p in parts],
        "target_loc": [int(p[5]) for p in parts],
    })


def load(split: str, stimuli: Path = STIMULI, noise: dict[str, float] = NOISE) -> tuple[np.ndarray, pd.DataFrame]:
    """Clean images of one split and one metadata row per image; meta.index = row in the images array."""
    images, metas = [], []
    for task, set_name in SPLITS[split]:
        t0 = time.time()
        imgs, names = read_zip(stimuli / f"{set_name}.zip", CROP[task])
        meta = parse_names(task, set_name, names)
        sd = noise[task] * 255
        baked = imgs[:, :6, :6].std()  # the corner holds no stimulus elements, only noise baked into the file
        meta.insert(0, "noise_sd", np.sqrt(max(sd ** 2 - baked ** 2, 0.0)))
        meta.insert(0, "file", names)
        meta.insert(0, "set", set_name)
        meta.insert(0, "task", task)
        images.append(imgs)
        metas.append(meta)
        log.info(f"{split} {set_name}: {len(imgs)} images, noise in file {baked:.1f}, added {meta.noise_sd[0]:.1f}"
                 f" grey levels | {time.time() - t0:.1f} s")
        if baked > sd:
            log.warning(f"{set_name}: noise in the file ({baked:.1f}) exceeds the target {sd:.1f}; none added")
    return np.concatenate(images), pd.concat(metas, ignore_index=True)


def resample(meta: pd.DataFrame, n_per_task: dict[str, int], seed: int) -> pd.DataFrame:
    """n random images per task; drawn with replacement only when n exceeds what the zips hold."""
    rng = np.random.default_rng(seed)
    picks = []
    for task, n in n_per_task.items():
        rows = meta.index[meta.task == task].to_numpy()
        if n > len(rows):
            log.warning(f"{task}: {n} images requested, {len(rows)} available; drawing with replacement")
        picks.append(rng.choice(rows, n, replace=n > len(rows)))
    return meta.loc[np.concatenate(picks)]


class NoisyStimuli(Dataset):
    """Adds Gaussian pixel noise on the fly. By default every item gets the same noise in every epoch,
    as in the paper (noise added once); fresh_noise=True draws new noise each time."""

    def __init__(self, images: np.ndarray, meta: pd.DataFrame, seed: int, stream: int, fresh_noise: bool = False):
        self.images = images
        self.rows = meta.index.to_numpy()
        self.labels = torch.tensor(meta.label.to_numpy())
        self.sd = meta.noise_sd.to_numpy(np.float32)
        self.key = (seed, stream)  # different streams keep train and test noise apart for the same seed
        self.rng = np.random.default_rng(self.key) if fresh_noise else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = self.rng or np.random.default_rng((*self.key, i))
        x = self.images[self.rows[i]].astype(np.float32)
        x += rng.standard_normal(x.shape, dtype=np.float32) * self.sd[i]
        np.clip(x, 0, 255, out=x)
        return torch.from_numpy(x / 255)[None], self.labels[i]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for split in SPLITS:
        images, meta = load(split)
        print(f"--- {split}: {images.shape}, {images.nbytes / 2**20:.0f} MiB")
        summary = meta.groupby(["task", "set"], sort=False).agg(
            n=("label", "size"), labels=("label", lambda s: dict(s.value_counts().sort_index())),
            noise_sd=("noise_sd", "first"))
        print(summary.to_string())
