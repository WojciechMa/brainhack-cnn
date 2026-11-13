# One small CNN, three attention tasks

A small convolutional network trained from scratch on three classic tasks of visuospatial attention (Posner cueing, visual search and contextual cueing), with no cue telling it which task it is doing. Shapley values and artificial lesions then show which units carry each attention effect.

The architecture and stimuli come from Srivastava, Yang & Eckstein (2024), *Emergent human-like covert attention in feedforward convolutional neural networks* (Current Biology): Conv 12 / 14 / 18 → BatchNorm → Dense(600) → 6 outputs.

**Try it in the browser:** [docs/index.html](docs/index.html), served by GitHub Pages at <https://wojciechma.github.io/brainhack-cnn/>. You can pick one of the paper's images, draw your own or upload one, then watch the network's answer layer by layer. You can also switch off groups of Dense units and measure the effects live.

## Results

These are from 10 networks: learning rate 0.0003, search noise 0.12 and new noise every epoch, plus 3 Posner-only networks as a control. Effects are differences in AUC, in percentage points. AUC is the chance that the network rates a random target image above a random no-target image. Full tables: [smallcnn/comparison.ipynb](smallcnn/comparison.ipynb).

1. **One network solves all three tasks:** accuracy Posner 78 %, search 81 %, contextual cueing 89 %.
2. **All three attention effects appear in 10 out of 10 networks:** Posner cueing +10 points (hit rate 81.5 % valid vs 59 % invalid), set size +13 points (1 vs 8 items), contextual cueing +13 points. Joint training does not weaken cueing (Posner-only: +11 points).
3. **Neuronal cueing is rare:** the rule from Srivastava et al. (2025, PNAS) finds it in 1–5 % of live Dense units (their single-task network: 56 %). At the output a valid cue helps by only about 3 points over no cue; most of the effect is the cost of an invalid cue.
4. **The Dense units are task-specific:** removing one task's top units hurts only that task. Removing the top 20 % of units for the cueing or the set-size effect removes that effect on average; contextual cueing survives every 20 % lesion (spread over many units).
5. **No shared attention module:** the cueing and set-size effects use different units (top 5 % share 2.8 units, 2.1 by chance), unrelated unit profiles and nearly unrelated activity directions, and removing one effect's units leaves the other intact.
6. **The cueing effect runs through units tuned to target and cue at the same location** (56 % of the effect; target-only units 28 %). The paper's location-summation and opponent types carry 14 %. The result holds when tuning is measured on trained-on images only.
7. **The conv layers have no effect-specific channels:** an effect's own top channels weaken it no more than random channels, and the channels that matter most for any task weaken all three effects together. The effects separate only in the Dense layer.
8. **Training settings matter:** learning rate 0.001 left only 75–121 of 600 Dense units live; fixed noise (the paper's setup) memorises the training images and shows no contextual cueing.

Caveats:
- **Cueing-unit lesions vary between networks:** in 8 of the 10 networks the cue effect ends within 3 points of zero, but one keeps +6.9 points and one reverses to −8.2.
- **Small control group:** there are only 3 Posner-only networks.
- **Neuron rule:** the paper's exact rule for counting cueing neurons is in its supplement, which we do not have.
- **Different network:** our network differs from the paper's single-task Posner network (stride 2 vs 3, Dense 600 vs 256).
- **Frozen VGG-16:** an earlier attempt on a frozen VGG-16 showed no reliable Posner cueing.

## Setup

Python 3.10+ (tested on 3.12). The commands are for Windows PowerShell; on macOS or Linux, activate with `source .venv/bin/activate`. The CUDA build of PyTorch needs an NVIDIA driver 570 or newer. It ships its own CUDA runtime, so you don't need a separate CUDA install. If you have no NVIDIA GPU, skip the first `pip install`.

```powershell
git clone https://github.com/WojciechMa/brainhack-cnn.git; cd brainhack-cnn
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Stimuli** are not in the repo (3.6 GB). They are the "Example Stimuli" zips from the paper's code release. Put them in `input_docs/source_code/sc_Emergent Human-Like Covert Attention in Feedforward Convolutional Neural Networks/Example Stimuli/` or pass `--stimuli <folder>`. Needed: `Posnerb179_80`, `Posnerb179_80test`, `VSss{1,2,4,8}angle9n0`, `VSss{1,2,4,8}angle9testn0`, `cco75n0`, `ccn75n0`, `ccn75n5test`.

## Run

```powershell
python smallcnn/data.py                          # checks that the stimuli load
python smallcnn/train.py --limit 60 --epochs 1   # smoke test
pwsh smallcnn/run_all.ps1 -Tasks joint -Noise fresh -Tag _v2 -TrainArgs "--lr","3e-4","--noise","search=0.12" -Seeds (0..9)
pwsh smallcnn/run_all.ps1 -Tasks posner -Noise fresh -Tag _v2 -TrainArgs "--lr","3e-4"   # control, seeds 0-2
$joint = 0..9 | ForEach-Object { "runs/smallcnn/joint_fresh_v2_seed$_" }; $posner = 0..2 | ForEach-Object { "runs/smallcnn/posner_fresh_v2_seed$_" }
python smallcnn/shared_attention.py --runs $joint --reps 3 --n-perm 500 --out runs/smallcnn/shared_attention_v2
python smallcnn/cue_units.py --runs ($joint + $posner) --out runs/smallcnn/cue_units_v2
python smallcnn/conv_effects.py --runs $joint --out runs/smallcnn/conv_effects_v2   # about 4.5 min per network
```

- **Training time:** one network trains in about 27 min (30 epochs) on a 6 GB laptop GPU, and `model.pt` is 120 MB.
- **`run_all.ps1`:** trains each network and runs the per-network analyses. Without options it trains the first groups (joint and Posner-only, fixed and fresh noise, learning rate 0.001, seeds 0–2), which the notebook also compares.
- **Notebook:** execute [smallcnn/comparison.ipynb](smallcnn/comparison.ipynb) from `smallcnn/`.
- **Outputs:** runs are written to `runs/` (git-ignored). Metrics also go to MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

| | |
|---|---|
| [smallcnn/model.py](smallcnn/model.py) | `JointCNN`, the paper's architecture |
| [smallcnn/data.py](smallcnn/data.py), [smallcnn/metrics.py](smallcnn/metrics.py), [smallcnn/train.py](smallcnn/train.py) | stimuli, effects and training; `--tasks posner` trains the Posner-only control |
| [smallcnn/lesion.py](smallcnn/lesion.py), [smallcnn/layer_lesions.py](smallcnn/layer_lesions.py), [smallcnn/neuron_cueing.py](smallcnn/neuron_cueing.py) | Shapley values and lesions of the Dense units, lesions per depth, neuronal cueing per unit |
| [smallcnn/shared_attention.py](smallcnn/shared_attention.py) | do the three effects share units, unit profiles or an activity direction? |
| [smallcnn/cue_units.py](smallcnn/cue_units.py) | the paper's unit types (target / cue tuning per location) and their share of the cueing effect |
| [smallcnn/conv_effects.py](smallcnn/conv_effects.py) | single-channel and group lesions of the conv layers, scored on the three effects |
| [smallcnn/run_all.ps1](smallcnn/run_all.ps1) | trains a group of networks and runs the per-network analyses |
| [smallcnn/comparison.ipynb](smallcnn/comparison.ipynb) | results across networks, with a reading under every table |
| [smallcnn/export_web.py](smallcnn/export_web.py), [docs/](docs) | the web page and its model files |

### Web page

```powershell
python smallcnn/export_web.py --run runs/smallcnn/joint_fresh_v2_seed0   # writes docs/model and docs/stimuli (needs the stimuli and the lesion.py --decision within outputs of the seed runs)
python -m http.server 8765 -d docs                                        # then open http://localhost:8765 (opening the file directly does not work)
```

- **Dense weights:** stored in 8 bits (blocks of 54 inputs share a scale), 34 MB in total. The 8-bit copy gives the same answer as the original on 8038 of 8040 test images. Adding `?check` to the URL runs a self-test against PyTorch.
- **Live trials:** they use test images from the half not used for ranking the units. The images take 10 MB and load only when you click "Run trials".

## Credits

- **Architecture, stimuli and tasks:** Srivastava, Yang & Eckstein (2024), *Current Biology*, and its code release.
- **Neuronal cueing measure:** Srivastava et al. (2025), *PNAS*.
- **Multi-perturbation Shapley analysis:** Keinan et al. (2004).

The example images on the web page are taken from the paper's stimuli.
