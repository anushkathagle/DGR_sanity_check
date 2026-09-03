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

USE_DRIVE=True below persists everything that's worth keeping (prepared
dataset zip, downloaded checkpoint, training runs/snapshots) to Google
Drive instead of the ephemeral Colab disk. Drive is mounted for you below
-- IMPORTANT: nothing under BASE is created until *after* that mount
succeeds, because creating '/content/drive/...' locally before Drive is
mounted there makes Colab's drive.mount() refuse to mount ("Mountpoint
must not already contain files"), since it now finds a non-empty local
directory sitting at the mount point instead of an empty one.

Two things are deliberately kept on local disk even with USE_DRIVE=True:
- The cloned edm repo itself: we `%cd` into it, so if it lived on Drive,
  any Drive hiccup during a multi-hour training run (mount drop, network
  blip) would take the shell's own cwd down with it -- "getcwd: cannot
  access parent directories: Transport endpoint is not connected" -- not
  just a file read. Re-cloning is a few seconds' work each session and
  gains nothing from persistence.
- The raw 50k-PNG CIFAR-100 dump used only as scratch input to
  dataset_tool.py -- writing tens of thousands of individual small files
  through Drive's FUSE mount is slow (can take a long time / time out),
  and that folder is fully disposable (regenerated from torchvision in
  under a minute).

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
BATCH = 128           # effective/logical batch size (affects training dynamics,
                      # e.g. the EMA halflife schedule) -- lower than the paper's
                      # default of 512, but this alone does NOT bound GPU memory;
                      # reduce --tick/--snap proportionally if you raise
                      # DURATION_MIMG a lot

TICK_KIMG = 10   # images (in thousands) per tick -- just a progress/logging unit
SNAP_TICKS = 2   # write a network-snapshot-*.pkl every SNAP_TICKS * TICK_KIMG
                 # images (currently 20k)
DUMP_TICKS = 2   # write a training-state-*.pt (what --resume needs) every
                 # DUMP_TICKS * TICK_KIMG images (currently 20k). This is your
                 # worst-case progress loss on a disconnect -- tightened from
                 # the (10, 10) defaults (100k-image loss window) after losing
                 # a real run's progress to a disconnect that landed before the
                 # first checkpoint ever got written. Lower further (e.g. 1) if
                 # disconnects keep happening well inside this window; each
                 # dump costs some Drive I/O time and space, so don't go
                 # extreme without reason
BATCH_GPU = 32        # actual per-step minibatch size; train.py accumulates
                      # gradients over BATCH // BATCH_GPU steps to reach BATCH,
                      # so this is what actually controls peak GPU memory. A
                      # single-GPU Colab run with no --batch-gpu processes all
                      # of BATCH in one forward/backward pass, which OOMs on a
                      # ~15GB GPU (T4) at BATCH=128 for this architecture; drop
                      # this further (16, 8, ...) if you still hit OOM

# ── Mount Google Drive (must happen before BASE is created below) ──────────
# os.path.ismount() only checks that /content/drive is a distinct mount
# point -- it stays True even if the underlying Drive FUSE connection has
# gone stale (e.g. after a network hiccup or long idle period), which then
# fails every file access with "[Errno 107] Transport endpoint is not
# connected". Actually probe it with a real read and force a remount if
# that fails, instead of trusting ismount() alone.
if USE_DRIVE:
    from google.colab import drive
    needs_mount = True
    if os.path.ismount('/content/drive'):
        try:
            os.listdir('/content/drive/MyDrive')
            needs_mount = False
            print("Google Drive is already mounted and responsive at /content/drive")
        except OSError:
            print("Google Drive mount is stale (not responding) -- remounting...")
    if needs_mount:
        drive.mount('/content/drive', force_remount=True)

os.makedirs(BASE, exist_ok=True)

# ── 1. Clone the NVlabs/edm repository ─────────────────────────────────────
# Deliberately local, not under BASE, even with USE_DRIVE=True: this is
# where we `%cd`, so if it lived on Drive, any Drive hiccup during training
# (mount drop, network blip) would take down the shell's own cwd along with
# it -- "getcwd: cannot access parent directories: Transport endpoint is
# not connected" -- not just a file read. The clone itself is a few
# seconds' work and gains nothing from persistence; only the
# dataset/checkpoint/training-run *data* below actually needs Drive.
print("Cloning NVlabs/edm repository...")
edm_dir = '/content/edm'
if not os.path.isdir(edm_dir):
    !git clone https://github.com/NVlabs/edm.git "{edm_dir}"

%cd "{edm_dir}"

# EDM (2022) predates several current-PyTorch/single-GPU-Colab realities.
# Patch the affected lines in the cloned repo; str.replace is a no-op on
# repeat runs (e.g. if edm_dir is being reused from a previous session on
# Drive), so this is safe to run every time regardless of whether the repo
# was just cloned or already existed.
def _patch_file(rel_path, old, new, description):
    path = os.path.join(edm_dir, rel_path)
    with open(path) as f:
        src = f.read()
    if old not in src:
        print(f"WARNING: expected text not found in {rel_path}, skipping patch: {description}")
        return
    patched = src.replace(old, new)
    if patched != src:
        with open(path, 'w') as f:
            f.write(patched)
        print(f"Patched {rel_path}: {description}")

# 1) torch.utils.data.Sampler's __init__ used to take a `data_source` arg;
# modern PyTorch's no longer does, so InfiniteSampler's
# `super().__init__(dataset)` raises "TypeError: object.__init__() takes
# exactly one argument".
_patch_file(
    'torch_utils/misc.py',
    'super().__init__(dataset)', 'super().__init__()',
    "InfiniteSampler for modern PyTorch's Sampler.__init__()",
)

# 2) dist.init() hardcodes backend='nccl' on any non-Windows OS, even
# though we only ever run a single process (no torchrun/multi-GPU) where
# gloo works identically. This avoids "RuntimeError: Distributed package
# doesn't have NCCL built in" on any torch build without NCCL compiled in
# (e.g. a CPU-only build, if the runtime doesn't actually have a GPU
# attached -- check with !nvidia-smi if you hit this).
_patch_file(
    'torch_utils/distributed.py',
    "backend = 'gloo' if os.name == 'nt' else 'nccl'",
    "backend = 'gloo' if os.name == 'nt' or not torch.distributed.is_nccl_available() else 'nccl'",
    "fall back to gloo backend when NCCL isn't compiled in",
)

# 3) PyTorch 2.6 changed torch.load()'s default from weights_only=False to
# weights_only=True. --resume's training-state-*.pt contains pickled
# custom objects (e.g. torch_utils.persistence._reconstruct_persistent_obj)
# that trip the new default's unpickling restriction, raising
# "Weights only load failed ... WeightsUnpickler error: Unsupported
# global". This is EDM's own file (written by this same script's earlier
# run), not an untrusted download, so weights_only=False is safe here.
_patch_file(
    'training/training_loop.py',
    "data = torch.load(resume_state_dump, map_location=torch.device('cpu'))",
    "data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)",
    "load training-state checkpoints with weights_only=False for PyTorch 2.6+",
)

# 4) train.py's Logger duplicates *every single print()* to a log.txt under
# run_dir, flushing on every write since should_flush=True. Since run_dir
# is under outdir (on Drive when USE_DRIVE=True), this means any Drive
# hiccup -- at any point during the whole run, not just at checkpoint
# writes -- crashes the process immediately with
# "OSError: [Errno 107] Transport endpoint is not connected", as seen
# crashing on the very first print (the network summary table) before any
# training even started. Point the log file at local disk instead; it's
# just a human-readable log, not something that needs Drive persistence
# (checkpoints/snapshots are still written to Drive separately, on their
# own controlled schedule).
_patch_file(
    'train.py',
    "os.path.join(c.run_dir, 'log.txt')",
    "f'/content/edm-log-{os.path.basename(c.run_dir)}.txt'",
    "write the stdout log locally instead of through the Drive mount",
)

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
print(f"  cond={COND}, duration={DURATION_MIMG}Mimg, batch={BATCH}, batch_gpu={BATCH_GPU}")

# --duration is in millions of images, not kimg. There's no --metrics flag
# on this train.py -- FID/metric computation lives in the separate fid.py
# script, run manually against a snapshot after training.
train_cmd = (
    f'python train.py '
    f'--outdir="{outdir}" '
    f'--data="{dataset_zip}" '
    f'--cond={"1" if COND else "0"} '
    f'{weight_arg} '
    f'--duration={DURATION_MIMG} '
    f'--batch={BATCH} '
    f'--batch-gpu={BATCH_GPU} '
    f'--tick={TICK_KIMG} '
    f'--snap={SNAP_TICKS} '
    f'--dump={DUMP_TICKS}'
)
!{train_cmd}

# Don't just assume success -- check train.py's actual exit code (IPython
# sets the special _exit_code var after every `!` command) and confirm a
# snapshot really landed on Drive, rather than printing "finished"
# unconditionally regardless of what happened.
snapshots = sorted(glob.glob(os.path.join(outdir, '*', 'network-snapshot-*.pkl')))
if _exit_code != 0:
    print(f"\n*** train.py exited with code {_exit_code} -- it did NOT complete successfully. ***")
    if snapshots:
        print(f"Latest snapshot checkpointed on Drive before the failure: {snapshots[-1]}")
        print("Re-run this cell to resume training from there.")
    else:
        print("No snapshot was checkpointed yet -- re-running this cell will restart from the pretrained checkpoint.")
elif snapshots:
    print(f"\nTraining completed. Final network snapshot saved to Drive:\n  {snapshots[-1]}")
else:
    print("\ntrain.py exited cleanly but no network-snapshot-*.pkl was found under outdir -- this is unexpected, check the run's log.txt.")
