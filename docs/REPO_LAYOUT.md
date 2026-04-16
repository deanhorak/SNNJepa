# Repository Layout

## Top Level

- `CMakeLists.txt`: top-level build entry
- `configs/`: retained MNIST, EMNIST, and CIFAR-10 baseline configs
- `docs/`: repo-specific documentation for the trimmed workspace
- `experiments/`: the retained benchmark executable
- `include/`: public headers
- `scripts/`: only the retained baseline runner scripts
- `src/`: core runtime implementation
- `third_party/`: optional visualization dependencies retained from the original runtime surface
- `shaders/`: visualization shader assets retained with the runtime

## Key Files

- `experiments/retina_classification.cpp`
- `src/NetworkPropagator.cpp`
- `src/NeuralObjectFactory.cpp`
- `src/adapters/RetinaAdapter.cpp`
- `configs/mnist_retina_bilateral_experimental.sonata.json`
- `configs/emnist_retina_bilateral_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
