#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json}"
export CONFIG_PATH

exec "${ROOT_DIR}/scripts/run_cifar10_retina_bilateral.sh" "$@"
