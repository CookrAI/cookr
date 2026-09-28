#!/usr/bin/env bash
# One-shot box setup on RunPod / Vast (pytorch template, 1x 4090 or better).
# Run locally first:  tar czf /tmp/cookr-train.tgz -C ~/Can/cookr/data train
# then scp it to /workspace/, then run this on the box.
set -euo pipefail
cd /workspace
[ -d ai-toolkit ] || git clone --depth 1 https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip install -q -r requirements.txt
mkdir -p /workspace/cookr/data
[ -d /workspace/cookr/data/train ] || tar xzf /workspace/cookr-train.tgz -C /workspace/cookr/data
# training data uses symlinks for meme repeats; materialise them on the box
find /workspace/cookr/data/train -type l -exec sh -c 'cp --remove-destination "$(readlink "$1")" "$1"' _ {} \;
export HF_TOKEN="${HF_TOKEN:-}"
CFG="${1:-/workspace/cookr/train/zimage_lora.yaml}"
echo "training with $CFG"
python run.py "$CFG"
