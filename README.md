<h1 align="center">Neural Collapse-Guided Plasticity Regularization for Prompt-based Few-Shot Class-Incremental Learning</h1>

<p align="center">
  <b>Geonwoo Im</b><sup>1</sup>, <b><a href="https://scholar.google.com/citations?user=V6HVW-QAAAAJ&hl=ko&oi=ao">Sang Min Yoon</a></b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>HCI Lab, College of Computer Science, Kookmin University, Seoul, Korea
</p>

<p align="center">
  <img src="assets/main_figure.png" width="90%">
</p>

---

## Installation

```bash
# 1. Create the environment and install PyTorch (CUDA 12.1)
conda create -n ncp python=3.10 -y
conda activate ncp
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Install the remaining dependencies
pip install -r requirements.txt
```

---

## Data

Datasets and few-shot session splits follow [SEC-Prompt](https://github.com/yeyeyeye33/SEC-Prompt). Lay them out under `./data/`:

```
data/
├── cub/                         # CUB-200 (ImageFolder layout)
│   ├── train/<class>/*.jpg
│   └── test/<class>/*.jpg
├── imagenet-r/                  # ImageNet-R (ImageFolder layout)
│   ├── train/<class>/*.jpg
│   └── test/<class>/*.jpg
└── index_list/                  # copy from SEC-Prompt: data/index_list/
    ├── cub/session_*.txt
    ├── imagenet-r/session_*.txt
    └── cifar100/session_*.txt
```

- **CIFAR-100** is downloaded automatically by torchvision into `./data/`.
- **CUB-200 / ImageNet-R**: download the prepared archives linked in the SEC-Prompt README and rename the extracted folders to `cub/` and `imagenet-r/` if needed.
- `index_list/` fixes the few-shot samples of every incremental session; copy it from the SEC-Prompt repository.

| Dataset | Config folder | Split |
|---------|---------------|-------|
| CUB-200 | `configs/cub/` | 100 base / 10 sessions × 10 classes |
| CIFAR-100 | `configs/cifar100/` | 60 base / 8 sessions × 5 classes |
| ImageNet-R | `configs/imagenet_r/` | 100 base / 10 sessions × 10 classes |

---

## Training and Evaluation

NC-Plasticity adds a variability floor to the SEC-Prompt base-session objective,

$$\mathcal{L}_{\text{plastic}} = [\tau - V(\mathcal{B})]_+ ,$$

while the ViT backbone stays frozen and only the prompts are trained. Incremental sessions are unchanged. The defaults (`tau_var: 0.5`, `lambda_plastic: 1.0`) are set in the configs.

Training and evaluation run together: every session is evaluated right after it is learned.

```bash
# NC-Plasticity
python main.py --config configs/cub/nc_plasticity.json

# SEC-Prompt baseline
python main.py --config configs/cub/baseline.json
```

`main.py` runs every seed listed in the config's `seed` field (default: the 15 paired seeds). To run a single seed, set e.g. `"seed": [0]` in the config.

### Paired 15-seed protocol

```bash
python run_multiseed.py --config configs/cub/nc_plasticity.json
python run_multiseed.py --config configs/cub/baseline.json
```

`run_multiseed.py` runs the same seeds and additionally prints the final `mean ± std` of the average accuracy (the `=== FINAL | ... ===` line at the end of `logs/repeat_<prefix>.log`). Use the `cifar100/` and `imagenet_r/` configs for the other datasets.

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{im2026ncplasticity,
  title     = {Neural Collapse-Guided Plasticity Regularization for Prompt-based Few-Shot Class-Incremental Learning},
  author    = {Im, Geonwoo and Yoon, Sang Min},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

---

## Acknowledgements

This codebase is built on [SEC-Prompt](https://github.com/yeyeyeye33/SEC-Prompt), which in turn builds on [FSCIL-ASP](https://github.com/DawnLIU35/FSCIL-ASP) and [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt). The Gram-Schmidt prompt initialization is adapted from [pytorch-gram-schmidt](https://github.com/legendongary/pytorch-gram-schmidt).
