#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
DATA_DIR=""
for candidate in \
  "${ROOT_DIR}/data/cifar-10-batches-bin" \
  "${ROOT_DIR}/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin" \
  "$(cd "${ROOT_DIR}/.." && pwd)/data/cifar-10-batches-bin" \
  "$(cd "${ROOT_DIR}/.." && pwd)/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin"
do
  if [[ -f "${candidate}/test_batch.bin" ]]; then
    DATA_DIR="${candidate}"
    break
  fi
done

if [[ -z "${DATA_DIR}" ]]; then
  DATA_DIR="${ROOT_DIR}/data/CIFAR_10/cifar-10-binary/cifar-10-batches-bin"
fi

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/cifar10_retina_bilateral_experimental.sonata.json}"
TRAIN_IMAGES="${TRAIN_IMAGES:-${DATA_DIR}}"
TRAIN_LABELS="${TRAIN_LABELS:-${DATA_DIR}}"
TEST_IMAGES="${TEST_IMAGES:-${DATA_DIR}/test_batch.bin}"
TEST_LABELS="${TEST_LABELS:-${DATA_DIR}/test_batch.bin}"

EXAMPLES_PER_CLASS="${EXAMPLES_PER_CLASS:-0}"
TEST_LIMIT="${TEST_LIMIT:-0}"
SEED="${SEED:-42}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
LOG_PATH="${LOG_PATH:-${BUILD_DIR}/cifar10_retina_bilateral_baseline.log}"

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

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

if [[ "${TIMEOUT_SECONDS}" != "0" ]]; then
  CMD=(timeout "${TIMEOUT_SECONDS}" "${CMD[@]}")
fi

mkdir -p "${BUILD_DIR}"
cd "${ROOT_DIR}"
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} "${CMD[@]}" | tee "${LOG_PATH}"
