#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
DEFAULT_DATA_DIR="${ROOT_DIR}/data/MNIST"
ALT_DATA_DIR="$(cd "${ROOT_DIR}/.." && pwd)/data/MNIST/raw"

if [[ -f "${DEFAULT_DATA_DIR}/train-images-idx3-ubyte" ]]; then
  DATA_DIR="${DEFAULT_DATA_DIR}"
elif [[ -f "${ALT_DATA_DIR}/train-images-idx3-ubyte" ]]; then
  DATA_DIR="${ALT_DATA_DIR}"
else
  DATA_DIR="${DEFAULT_DATA_DIR}"
fi

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/mnist_retina_bilateral_experimental.sonata.json}"
TRAIN_IMAGES="${TRAIN_IMAGES:-${DATA_DIR}/train-images-idx3-ubyte}"
TRAIN_LABELS="${TRAIN_LABELS:-${DATA_DIR}/train-labels-idx1-ubyte}"
TEST_IMAGES="${TEST_IMAGES:-${DATA_DIR}/t10k-images-idx3-ubyte}"
TEST_LABELS="${TEST_LABELS:-${DATA_DIR}/t10k-labels-idx1-ubyte}"

EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-0}"
TEST_LIMIT="${TEST_LIMIT:-0}"
SEED="${SEED:-42}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
LOG_PATH="${LOG_PATH:-${BUILD_DIR}/mnist_retina_bilateral_baseline.log}"

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
