# Baselines

## Included Baselines

Only the current baseline paths are included.

### MNIST

- Config: `configs/mnist_retina_bilateral_experimental.sonata.json`
- Script: `scripts/run_mnist_retina_bilateral.sh`
- Input format: IDX files

### EMNIST

- Config: `configs/emnist_retina_bilateral_experimental.sonata.json`
- Script: `scripts/run_emnist_retina_bilateral.sh`
- Input format: EMNIST IDX files

### CIFAR-10

- Config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
- Scripts:
  - `scripts/run_cifar10_retina_bilateral.sh`
  - `scripts/run_cifar10_retina_bilateral_natural.sh`
- Input format: CIFAR-10 binary batches

## Excluded Surfaces

The following were intentionally excluded from `SNNJepa`:
- historical EMNIST one-off training binaries
- old SONATA-only experiment binaries
- visualization demos unrelated to the retained baseline
- closed CIFAR follow-up configs and runner scripts
- local flow-audit outputs and benchmark logs

## Baseline Role In JEPA Work

These baselines are retained for two reasons:
- they provide a working regression surface while JEPA machinery is introduced
- they give three dataset regimes with different difficulty levels:
  - MNIST for fast iteration
  - EMNIST for larger class count
  - CIFAR-10 for natural-image stress testing
