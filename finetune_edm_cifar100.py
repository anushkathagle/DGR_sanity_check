"""
Fine-tune NVIDIA's pretrained CIFAR-10 EDM diffusion model on CIFAR-100,
following the MimicDiffusion (https://github.com/psky1111/MimicDiffusion)
recipe of fine-tuning the *unconditional* EDM checkpoint on a different
32x32 dataset (avoids the class-count mismatch you'd hit trying to
fine-tune the conditional checkpoint: it was trained with a 10-way label
embedding, CIFAR-100 has 100 classes).

Meant to be pasted directly into a Colab cell -- it uses `!` / `%cd`
IPython magics, so it is NOT valid to run as `python finetune_edm_cifar100.py`
from a plain shell; copy its body into one or more notebook cells instead.
Set USE_DRIVE=True below to persist everything (repo clone, dataset,
checkpoints, training runs) to Google Drive instead of the ephemeral Colab
disk; mount Drive yourself first with
`from google.colab import drive; drive.mount('/content/drive')`.

Corrects the following bugs found in a first draft of this setup:
  - `--cfg` and `--kimg` are not real NVlabs/edm train.py options (those
    names belong to the edm2 / StyleGAN3 CLIs). The real duration flag is
    `--duration`, in *millions* of images.
  - The pretrained checkpoints are `.pkl`, not `.pt`.
  - `--resume` restores a full training state (optimizer + tick counters)
    written by a previous run of this same script; it cannot load
    NVIDIA's released inference checkpoint. Loading pretrained weights
    for fine-tuning is what `--transfer` is for.
  - train.py's ImageFolderDataset never reads labels out of filenames.
    It only picks up a dataset.json, or a label per top-level subfolder.
    A flat dump of PNGs (as in the first draft) produces an unlabeled
    dataset, which trips the `--cond=True requires labels` assertion and,
    even where cond isn't requested, silently loses the label metadata.
    Images must go through dataset_tool.py into per-class subfolders (or
    a zip) before train.py will accept them.
"""

import os
import torchvision

# ── Configuration ────────────────────────────────────────────────────────

USE_DRIVE = False  # set True to read/write everything under Google Drive
drive_base_path = '/content/drive/MyDrive/edm-cifar100'  # only used if USE_DRIVE

BASE = drive_base_path if USE_DRIVE else '/content'
os.makedirs(BASE, exist_ok=True)

COND = False  # unconditional fine-tuning (recommended, see module docstring).
# Setting COND=True is possible but the label-embedding weights won't
# transfer (shape mismatch: 10 classes -> 100), so --transfer effectively
# reinitializes that layer at random; only the backbone benefits from the
# pretrained weights in that case.

DURATION_MIMG = 10   # fine-tuning budget, in millions of images (the paper's
                      # from-scratch CIFAR-10 run used 200; fine-tuning needs
                      # nowhere near that)
BATCH = 128           # lower than the paper's default of 512 to fit a single
                      # Colab GPU; reduce --tick/--snap proportionally if you
                      # raise DURATION_MIMG a lot

# ── 1. Clone the NVlabs/edm repository ─────────────────────────────────────
print("Cloning NVlabs/edm repository...")
edm_dir = os.path.join(BASE, 'edm')
if not os.path.isdir(edm_dir):
    !git clone https://github.com/NVlabs/edm.git {edm_dir}

%cd {edm_dir}

# ── 2. Install dependencies ─────────────────────────────────────────────
print("Installing dependencies...")
!pip install -r requirements.txt

# ── 3. Prepare the CIFAR-100 dataset ────────────────────────────────────
# dataset_tool.py only derives labels from a dataset.json or from
# top-level subfolder names -- NOT from filenames -- so we save each
# image into a per-class subfolder (train/<label>/*.png), then convert
# that folder into the zip archive train.py actually expects.
print("Preparing CIFAR-100 dataset...")
raw_dir = os.path.join(BASE, 'datasets', 'cifar100-raw', 'train')
os.makedirs(raw_dir, exist_ok=True)

trainset = torchvision.datasets.CIFAR100(
    root=os.path.join(BASE, 'data_temp'), train=True, download=True,
)
for i, (img, label) in enumerate(trainset):
    class_dir = os.path.join(raw_dir, f'{label:03d}')
    os.makedirs(class_dir, exist_ok=True)
    img.save(os.path.join(class_dir, f'{i:05d}.png'))
print(f"CIFAR-100 training images saved to {raw_dir}")

dataset_zip = os.path.join(BASE, 'datasets', 'cifar100-32x32.zip')
print(f"Converting to EDM dataset format ({dataset_zip})...")
!python dataset_tool.py --source={raw_dir} --dest={dataset_zip} --resolution=32x32

# ── 4. Download the pretrained EDM CIFAR-10 checkpoint ─────────────────
# NVIDIA's released checkpoints are .pkl (inference-ready network
# pickles), not .pt.
print("Downloading pretrained EDM CIFAR-10 checkpoint...")
checkpoints_dir = os.path.join(BASE, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)
ckpt_name = 'edm-cifar10-32x32-cond-vp.pkl' if COND else 'edm-cifar10-32x32-uncond-vp.pkl'
ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
!wget -nc https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/{ckpt_name} -P {checkpoints_dir}

# ── 5. Fine-tune the model ──────────────────────────────────────────────
outdir = os.path.join(BASE, 'training-runs-cifar100')
print("Starting fine-tuning...")
print(f"  cond={COND}, duration={DURATION_MIMG}Mimg, batch={BATCH}")
print(f"  transferring weights from {ckpt_path}")

# --transfer loads pretrained network weights for fine-tuning (tolerant of
# shape mismatches). --duration is in millions of images, not kimg.
# --tick/--snap are scaled down from the (50, 50) defaults so a short
# fine-tuning run still produces a handful of checkpoints instead of one
# at the very end.
train_cmd = (
    f'python train.py '
    f'--outdir={outdir} '
    f'--data={dataset_zip} '
    f'--cond={"1" if COND else "0"} '
    f'--transfer={ckpt_path} '
    f'--duration={DURATION_MIMG} '
    f'--batch={BATCH} '
    f'--tick=10 '
    f'--snap=10 '
    f'--dump=10 '
    f'--metrics=none'
)
!{train_cmd}

print("\nFine-tuning finished. Network snapshots (.pkl) are under:")
print(f"  {outdir}")
