# Repository Layout

## Top Level

- `CMakeLists.txt`: top-level build entry
- `configs/`: retained MNIST, EMNIST, and CIFAR-10 baseline configs
- `docs/`: repo-specific documentation for the trimmed workspace
- `experiments/`: the retained benchmark executable
- `include/`: public framework tree
- `scripts/`: only the retained baseline runner scripts
- `include/snnfw/`: core runtime headers and implementation
- `third_party/`: optional visualization dependencies retained from the original runtime surface
- `shaders/`: visualization shader assets retained with the runtime

## Key Files

- `experiments/retina_classification.cpp`
- `include/snnfw/NetworkPropagator.cpp`
- `include/snnfw/NeuralObjectFactory.cpp`
- `include/snnfw/adapters/RetinaAdapter.cpp`
- `include/snnfw/jepa/JepaStateExtractor.cpp`
- `include/snnfw/jepa/JepaMaskSampler.cpp`
- `include/snnfw/jepa/JepaTrainer.cpp`
- `configs/mnist_retina_bilateral_experimental.sonata.json`
- `configs/emnist_retina_bilateral_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`

## JEPA Module

The JEPA-specific implementation surface lives under `include/snnfw/jepa/`:

- `JepaConfig.h`: JEPA modes, trainer knobs, masking controls, and target-mode selection
- `JepaSample.h`: latent tap, masked-view, and export data structures
- `JepaLoss.h`: JEPA loss helpers
- `JepaStateExtractor.*`: stage-1 tap extraction and JSON export
- `JepaMaskSampler.*`: visible/hidden masking and no-leakage validation
- `JepaTrainer.*`: context-encoder, predictor, EMA-target training, and probe embedding support
