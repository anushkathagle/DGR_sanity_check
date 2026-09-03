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

USE_DRIVE=True below persists everything that's worth keeping (repo clone,
prepared dataset zip, downloaded checkpoint, training runs/snapshots) to
Google Drive instead of the ephemeral Colab disk. Drive is mounted for
you below -- IMPORTANT: nothing under BASE is created until *after* that
mount succeeds, because creating '/content/drive/...' locally before Drive
is mounted there makes Colab's drive.mount() refuse to mount ("Mountpoint
must not already contain files"), since it now finds a non-empty local
directory sitting at the mount point instead of an empty one.
The raw 50k-PNG CIFAR-100 dump used only as scratch input to dataset_tool.py
is deliberately kept on local disk even with USE_DRIVE=True -- writing tens
of thousands of individual small files through Drive's FUSE mount is slow
(can take a long time / time out), and that folder is fully disposable
(regenerated from torchvision in under a minute).

If a session dies mid-training, re-running the fine-tuning cell will pick
up the latest training-state-*.pt under outdir on Drive automatically and
--resume from there instead of restarting from the pretrained checkpoint.

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

import glob
import os
import torchvision

# ── Configuration ────────────────────────────────────────────────────────

USE_DRIVE = True  # read/write everything under Google Drive
drive_base_path = '/content/drive/MyDrive/FineTunedCheckpoint/edm-cifar100'  # only used if USE_DRIVE
# Paths below are quoted before being passed to shell magics, so spaces are
# tolerated -- but note this path is unrelated to where the notebook file
# itself lives; it's just where this script writes its own working files.

BASE = drive_base_path if USE_DRIVE else '/content'

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

# ── Mount Google Drive (must happen before BASE is created below) ──────────
if USE_DRIVE:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
    else:
        print("Google Drive is already mounted at /content/drive")

os.makedirs(BASE, exist_ok=True)

# ── 1. Clone the NVlabs/edm repository ─────────────────────────────────────
print("Cloning NVlabs/edm repository...")
edm_dir = os.path.join(BASE, 'edm')
if not os.path.isdir(edm_dir):
    !git clone https://github.com/NVlabs/edm.git "{edm_dir}"

%cd "{edm_dir}"

# ── 2. Install dependencies ─────────────────────────────────────────────
# The repo ships environment.yml (a conda spec), not a pip requirements.txt
# -- `pip install -r environment.yml` fails trying to parse YAML as
# requirements. Colab already has a CUDA-matched torch/numpy/pillow/scipy,
# so we deliberately don't force environment.yml's pinned torch==1.12.1
# (that would fight the preinstalled CUDA build for no benefit); just
# pip-install the packages Colab doesn't already provide.
print("Installing dependencies...")
!pip install "numpy>=1.20" "click>=8.0" "pillow>=8.3.1" "scipy>=1.7.1" psutil requests tqdm imageio "imageio-ffmpeg>=0.4.3" pyspng

# ── 3. Prepare the CIFAR-100 dataset ────────────────────────────────────
# dataset_tool.py only derives labels from a dataset.json or from
# top-level subfolder names -- NOT from filenames -- so we save each
# image into a per-class subfolder (train/<label>/*.png), then convert
# that folder into the zip archive train.py actually expects.
print("Preparing CIFAR-100 dataset...")
# Deliberately local, not under BASE: this is a disposable ~50k-small-file
# scratch dump that's only consumed by dataset_tool.py below. Writing it to
# Drive would be extremely slow; it costs nothing to regenerate locally.
local_scratch = '/content/cifar100-scratch'
raw_dir = os.path.join(local_scratch, 'train')
os.makedirs(raw_dir, exist_ok=True)

dataset_zip = os.path.join(BASE, 'datasets', 'cifar100-32x32.zip')
os.makedirs(os.path.dirname(dataset_zip), exist_ok=True)

if os.path.isfile(dataset_zip):
    # Already prepared and persisted (e.g. from a previous session on Drive)
    print(f"Found existing prepared dataset at {dataset_zip}, skipping re-prep.")
else:
    trainset = torchvision.datasets.CIFAR100(
        root=os.path.join(local_scratch, 'data_temp'), train=True, download=True,
    )
    for i, (img, label) in enumerate(trainset):
        class_dir = os.path.join(raw_dir, f'{label:03d}')
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f'{i:05d}.png'))
    print(f"CIFAR-100 training images saved to {raw_dir}")

    print(f"Converting to EDM dataset format ({dataset_zip})...")
    !python dataset_tool.py --source="{raw_dir}" --dest="{dataset_zip}" --resolution=32x32

# ── 4. Download the pretrained EDM CIFAR-10 checkpoint ─────────────────
# NVIDIA's released checkpoints are .pkl (inference-ready network
# pickles), not .pt.
print("Downloading pretrained EDM CIFAR-10 checkpoint...")
checkpoints_dir = os.path.join(BASE, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)
ckpt_name = 'edm-cifar10-32x32-cond-vp.pkl' if COND else 'edm-cifar10-32x32-uncond-vp.pkl'
ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
!wget -nc https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/{ckpt_name} -P "{checkpoints_dir}"

# ── 5. Fine-tune the model ──────────────────────────────────────────────
outdir = os.path.join(BASE, 'training-runs-cifar100')
os.makedirs(outdir, exist_ok=True)

# If a previous run under outdir got interrupted (Colab disconnect, runtime
# recycle, etc.), --dump periodically wrote a full training-state file
# (optimizer + step count, not just weights) there. Pick the latest one and
# --resume from it instead of --transfer-ing the pretrained checkpoint again
# -- that's what actually avoids losing progress on session termination.
resume_candidates = sorted(glob.glob(os.path.join(outdir, '*', 'training-state-*.pt')))
resume_path = resume_candidates[-1] if resume_candidates else None

if resume_path:
    print(f"Found existing training state, resuming from {resume_path}")
    weight_arg = f'--resume="{resume_path}"'
else:
    print(f"No existing training state found, transferring pretrained weights from {ckpt_path}")
    weight_arg = f'--transfer="{ckpt_path}"'

print("Starting fine-tuning...")
print(f"  cond={COND}, duration={DURATION_MIMG}Mimg, batch={BATCH}")

# --duration is in millions of images, not kimg. --tick/--snap are scaled
# down from the (50, 50) defaults so a short fine-tuning run still produces
# a handful of checkpoints instead of one at the very end.
train_cmd = (
    f'python train.py '
    f'--outdir="{outdir}" '
    f'--data="{dataset_zip}" '
    f'--cond={"1" if COND else "0"} '
    f'{weight_arg} '
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
