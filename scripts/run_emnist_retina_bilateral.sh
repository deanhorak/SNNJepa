#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/emnist_retina_bilateral_experimental.sonata.json}"
TRAIN_IMAGES="${TRAIN_IMAGES:-${ROOT_DIR}/data/EMNIST/emnist-letters-train-images-idx3-ubyte}"
TRAIN_LABELS="${TRAIN_LABELS:-${ROOT_DIR}/data/EMNIST/emnist-letters-train-labels-idx1-ubyte}"
TEST_IMAGES="${TEST_IMAGES:-${ROOT_DIR}/data/EMNIST/emnist-letters-test-images-idx3-ubyte}"
TEST_LABELS="${TEST_LABELS:-${ROOT_DIR}/data/EMNIST/emnist-letters-test-labels-idx1-ubyte}"

EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-3200}"
TEST_LIMIT="${TEST_LIMIT:-5200}"
SEED="${SEED:-42}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
LOG_PATH="${LOG_PATH:-${BUILD_DIR}/emnist_retina_bilateral_baseline.log}"

CMD=(
  "${BUILD_DIR}/experiments/retina_classification"
  --config "${CONFIG_PATH}"
  --train-images "${TRAIN_IMAGES}"
  --train-labels "${TRAIN_LABELS}"
  --test-images "${TEST_IMAGES}"
  --test-labels "${TEST_LABELS}"
  --examples-per-class "${EXAMPLES_PER_CLASS}"
  --test-limit "${TEST_LIMIT}"
  --seed "${SEED}"
)

if [[ "${TIMEOUT_SECONDS}" != "0" ]]; then
  CMD=(timeout "${TIMEOUT_SECONDS}" "${CMD[@]}")
fi

mkdir -p "${BUILD_DIR}"
cd "${ROOT_DIR}"
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} "${CMD[@]}" | tee "${LOG_PATH}"
