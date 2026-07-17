#!/usr/bin/env bash
# Download + extract the KWS corpora into $KWS_DATA_ROOT.
#
#   AISHELL-1 (OpenSLR 33)  ~15 GB  — base acoustic training (Apache-2.0, ship-safe)
#   MUSAN     (OpenSLR 17)  ~11 GB  — additive-noise augmentation
#   RIRS      (OpenSLR 28)  ~ 5 GB  — room-impulse reverberation augmentation
#
# Usage:
#   KWS_DATA_ROOT=/teamspace/lightning_storage/kws-mandarin scripts/download_data.sh all
#   scripts/download_data.sh aishell           # just one
#   scripts/download_data.sh aishell musan     # a subset
#
# Idempotent: downloads resume (wget -c) and extraction is skipped if the marker exists.
set -euo pipefail

ROOT="${KWS_DATA_ROOT:-/teamspace/lightning_storage/kws-mandarin}"
DL="$ROOT/downloads"
CORP="$ROOT/corpora"
NPROC="$(nproc)"
mkdir -p "$DL" "$CORP"

echo ">> KWS_DATA_ROOT = $ROOT   (cpus: $NPROC)"

fetch() {  # url dest
  local url="$1" dest="$2"
  if [[ -f "$dest.done" ]]; then echo ">> have $(basename "$dest"), skip download"; return; fi
  echo ">> downloading $(basename "$dest")"
  wget -c -O "$dest" "$url"
  touch "$dest.done"
}

download_aishell() {
  local out="$CORP/aishell1"
  if [[ -f "$out/.extracted" ]]; then echo ">> aishell1 already extracted"; return; fi
  mkdir -p "$out"
  fetch "https://www.openslr.org/resources/33/data_aishell.tgz" "$DL/data_aishell.tgz"
  echo ">> extracting data_aishell.tgz"
  tar -xzf "$DL/data_aishell.tgz" -C "$out"
  # AISHELL-1 quirk: wavs are packed as per-speaker tarballs under wav/. Extract in parallel.
  echo ">> extracting per-speaker wav tarballs ($NPROC-way)"
  find "$out/data_aishell/wav" -name '*.tar.gz' -print0 \
    | xargs -0 -P "$NPROC" -I{} tar -xzf {} -C "$out/data_aishell/wav"
  touch "$out/.extracted"
  echo ">> aishell1 ready at $out/data_aishell"
}

download_musan() {
  local out="$CORP/musan"
  if [[ -f "$out/.extracted" ]]; then echo ">> musan already extracted"; return; fi
  mkdir -p "$out"
  fetch "https://www.openslr.org/resources/17/musan.tar.gz" "$DL/musan.tar.gz"
  echo ">> extracting musan.tar.gz"
  tar -xzf "$DL/musan.tar.gz" -C "$out"
  touch "$out/.extracted"
  echo ">> musan ready at $out/musan"
}

download_rirs() {
  local out="$CORP/rirs"
  if [[ -f "$out/.extracted" ]]; then echo ">> rirs already extracted"; return; fi
  mkdir -p "$out"
  fetch "https://www.openslr.org/resources/28/rirs_noises.zip" "$DL/rirs_noises.zip"
  echo ">> extracting rirs_noises.zip"
  unzip -q -o "$DL/rirs_noises.zip" -d "$out"
  touch "$out/.extracted"
  echo ">> rirs ready at $out/RIRS_NOISES"
}

targets=("$@")
if [[ ${#targets[@]} -eq 0 || " ${targets[*]} " == *" all "* ]]; then
  targets=(aishell musan rirs)
fi
for t in "${targets[@]}"; do
  case "$t" in
    aishell) download_aishell ;;
    musan)   download_musan ;;
    rirs)    download_rirs ;;
    all)     ;;  # handled above
    *) echo "unknown target: $t (expected: aishell musan rirs all)" >&2; exit 1 ;;
  esac
done
echo ">> done."
