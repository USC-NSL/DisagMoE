#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/USC-NSL/DisagMoE.git"
REPO_DIR="$HOME/DisagMoE"
REPO_TAG="asym"

CONDA_DIR="$HOME/miniconda3"
CONDA_ENV="disag12"
PYTHON_VERSION="3.12"

MINICONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER_PATH="$CONDA_DIR/miniconda.sh"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if ! sudo -n true >/dev/null 2>&1; then
  echo "sudo requires a password on this machine. Configure passwordless sudo for apt installs, then re-run." >&2
  exit 1
fi

echo "[0/8] Checking base prerequisites..."
need_cmd git
need_cmd wget
need_cmd sudo
need_cmd python3

echo "[0/8] Cloning DisagMoE and checking out ${REPO_TAG}..."
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --tags
git checkout "$REPO_TAG"

echo "[1/8] Installing Miniconda to ${CONDA_DIR}..."
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  mkdir -p "$CONDA_DIR"
  wget "$MINICONDA_INSTALLER_URL" -O "$MINICONDA_INSTALLER_PATH"
  bash "$MINICONDA_INSTALLER_PATH" -b -u -p "$CONDA_DIR"
  rm -f "$MINICONDA_INSTALLER_PATH"
fi

echo "[1/8] Initializing conda for this script session..."
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "[1/8] Accepting Anaconda Terms of Service (required for conda channels)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "[2/8] Creating conda env ${CONDA_ENV} with Python ${PYTHON_VERSION}..."
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" "python=${PYTHON_VERSION}"
fi
conda activate "$CONDA_ENV"

echo "[2/8] Verifying Makefile uses --no-build-isolation for editable install..."
if [ ! -f "$REPO_DIR/Makefile" ]; then
  echo "Missing $REPO_DIR/Makefile; expected make pip to exist." >&2
  exit 1
fi
if ! grep -Eq '^[[:space:]]*pip[[:space:]]+install[[:space:]]+-e[[:space:]]+\.[[:space:]]+--no-build-isolation([[:space:]]|$)' "$REPO_DIR/Makefile"; then
  echo "Makefile 'pip' target must run: pip install -e . --no-build-isolation" >&2
  echo "Please update the repo Makefile (we no longer patch it in this script)." >&2
  exit 1
fi

echo "[3/8] Installing apt dependencies + python requirements (vllm==0.8.2, etc.)..."
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  linux-headers-$(uname -r) \
  libzmq3-dev \
  libcereal-dev \
  libucx-dev

git submodule update --init --recursive

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[4/8] Installing MLNX_OFED (required for nvidia_peermem / GPU-Direct RDMA)..."
if ofed_info -s >/dev/null 2>&1; then
  echo "  MLNX_OFED already installed: $(ofed_info -s)"
else
  OFED_VER="24.10-1.1.4.0"
  OFED_DIR="MLNX_OFED_LINUX-${OFED_VER}-ubuntu24.04-x86_64"
  OFED_TGZ="${OFED_DIR}.tgz"
  OFED_URL="https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VER}/${OFED_TGZ}"
  cd /tmp
  if [ ! -f "$OFED_TGZ" ]; then
    wget -q "$OFED_URL" -O "$OFED_TGZ"
  fi
  tar xzf "$OFED_TGZ"
  cd "$OFED_DIR"
  sudo ./mlnxofedinstall --add-kernel-support --without-fw-update --force < /dev/null
  sudo /etc/init.d/openibd restart
  cd "$REPO_DIR"
fi

echo "[4/8] Building nvidia_peermem against MLNX_OFED RDMA stack..."
NVIDIA_VER=$(modinfo nvidia 2>/dev/null | awk '/^version:/{print $2}')
if [ -n "$NVIDIA_VER" ]; then
  sudo dkms build "nvidia/${NVIDIA_VER}" -k "$(uname -r)" --force || true
  sudo dkms install "nvidia/${NVIDIA_VER}" -k "$(uname -r)" --force || true
fi

echo "[4/8] Loading nvidia_peermem (with PeerMappingOverride)..."
PEERMEM_CONF="/etc/modprobe.d/nvidia-peermem.conf"
if [ ! -f "$PEERMEM_CONF" ] || ! grep -q PeerMappingOverride "$PEERMEM_CONF"; then
  echo 'options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee "$PEERMEM_CONF" >/dev/null
fi
if ! lsmod | grep -q nvidia_peermem; then
  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset gdrdrv 2>/dev/null || true
  sudo rmmod nvidia 2>/dev/null || true
  sudo modprobe nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"
  sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset
  sudo modprobe nvidia_peermem
fi
lsmod | grep nvidia_peermem && echo "  nvidia_peermem loaded OK" || echo "  WARNING: nvidia_peermem failed to load"

echo "[4/8] Installing NCCL packages..."
sudo apt-get install -y libnccl2 libnccl-dev

echo "[5/8] Installing and loading GDRCopy kernel module..."
sudo apt-get install -y flex bison

GDRCOPY_DIR="$HOME/gdrcopy"
if [ ! -d "$GDRCOPY_DIR/.git" ]; then
  git clone https://github.com/NVIDIA/gdrcopy.git "$GDRCOPY_DIR"
fi
cd "$GDRCOPY_DIR"
if [ ! -f "/usr/local/gdrcopy/lib/libgdrapi.so" ]; then
  make
  sudo make prefix=/usr/local/gdrcopy install
  sudo ldconfig
fi

if lsmod | awk '$1=="gdrdrv" {found=1} END {exit found?0:1}'; then
  echo "[5/8] gdrdrv module already loaded; skipping insmod"
else
  sudo bash ./insmod.sh
fi

echo "[5/8] Ensuring /usr/local/gdrcopy/lib is in dynamic linker path..."
GDR_LD_CONF="/etc/ld.so.conf.d/gdrcopy.conf"
if [ ! -f "$GDR_LD_CONF" ] || ! grep -qx "/usr/local/gdrcopy/lib" "$GDR_LD_CONF"; then
  echo "/usr/local/gdrcopy/lib" | sudo tee "$GDR_LD_CONF" >/dev/null
fi
sudo ldconfig

echo "[5/8] Ensuring linker can find -lgdrapi at build time..."
if [ -f /usr/local/gdrcopy/lib/libgdrapi.so ]; then
  if [ ! -f /usr/local/lib/libgdrapi.so ]; then
    sudo ln -s /usr/local/gdrcopy/lib/libgdrapi.so /usr/local/lib/libgdrapi.so
  fi
  if [ -f /usr/local/gdrcopy/lib/libgdrapi.so.2 ] && [ ! -f /usr/local/lib/libgdrapi.so.2 ]; then
    sudo ln -s /usr/local/gdrcopy/lib/libgdrapi.so.2 /usr/local/lib/libgdrapi.so.2
  fi
  sudo ldconfig
fi

echo "[5/8] Ensuring runtime can load conda/torch shared libs..."
CONDA_ENV_PREFIX="$CONDA_DIR/envs/$CONDA_ENV"
CONDA_LD_CONF="/etc/ld.so.conf.d/conda-${CONDA_ENV}.conf"
sudo tee "$CONDA_LD_CONF" >/dev/null <<EOF
${CONDA_ENV_PREFIX}/lib
${CONDA_ENV_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/torch/lib
EOF
sudo ldconfig

echo "[6/8] Re-installing python requirements (as specified)..."
cd "$REPO_DIR"
python -m pip install -r ./requirements.txt

echo "[6/8] Verifying vLLM version is 0.8.2..."
python - <<'PY'
import importlib.metadata as m

ver = m.version("vllm")
if ver != "0.8.2":
    raise SystemExit(f"Expected vllm==0.8.2, found vllm=={ver}")
print("vllm==0.8.2 OK")
PY

echo "[7/8] Applying vLLM 0.8.2 patch to installed vllm..."
python -c "import os, site; paths=[p for p in site.getsitepackages() if os.path.isdir(os.path.join(p, 'vllm'))]; print(paths[0] if paths else site.getsitepackages()[0])" > /tmp/disag_site_dir.txt
SITE_DIR="$(cat /tmp/disag_site_dir.txt)"

if [ ! -d "$SITE_DIR/.git" ]; then
  echo "Initializing git repo inside site-packages for patching..."
  (cd "$SITE_DIR" && git init -q && git add -A >/dev/null 2>&1 || true)
fi

echo "Applying patch in $SITE_DIR ..."
(cd "$SITE_DIR" && {
  if git apply --check "$REPO_DIR/patches/vllm_0.8.2.patch" >/dev/null 2>&1; then
    git apply "$REPO_DIR/patches/vllm_0.8.2.patch"
  elif git apply -R --check "$REPO_DIR/patches/vllm_0.8.2.patch" >/dev/null 2>&1; then
    echo "Patch already applied; skipping."
  else
    echo "Patch cannot be applied cleanly. Ensure vllm==0.8.2 is installed, then retry." >&2
    exit 1
  fi
})

echo "[8/8] Building and installing DisagMoE (make pip)..."
cd "$REPO_DIR"

echo "[8/8] Verifying build-time Python deps are importable..."
python -c "import pybind11; import torch" >/dev/null

echo "[8/8] Verifying pip build isolation is disabled in Makefile..."
if grep -qF -- "--no-build-isolation" "$REPO_DIR/Makefile"; then
  true
else
  echo "Expected --no-build-isolation in $REPO_DIR/Makefile pip target." >&2
  exit 1
fi

echo "[8/8] Exporting build-time linker search paths (GDRCopy)..."
export LIBRARY_PATH="/usr/local/gdrcopy/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/gdrcopy/lib:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L/usr/local/gdrcopy/lib ${LDFLAGS:-}"

make pip

echo "Done. DisagMoE installed into conda env: $CONDA_ENV"
