# SNNFW Adapters

## Overview

Adapters provide a standardized interface for connecting spiking neural networks to external data sources (sensory input) and actuators (motor output). They encapsulate domain-specific processing and enable reusable, composable components.

## Architecture

```
                    External World
                          |
        +-----------------+-----------------+
        |                                   |
   Sensory Adapters                  Motor Adapters
   (Input/Encoding)                  (Output/Decoding)
        |                                   |
        v                                   ^
   Spike Trains                        Spike Trains
        |                                   |
        +------------> SNN Core <-----------+
                   (Processing)
```

## Class Hierarchy

```
BaseAdapter (abstract)
├── SensoryAdapter (abstract)
│   ├── RetinaAdapter (vision)
│   ├── AudioAdapter (hearing)
│   ├── TactileAdapter (touch)
│   └── ProprioceptiveAdapter (body position)
│
└── MotorAdapter (abstract)
    ├── DisplayAdapter (visualization)
    ├── ServoAdapter (motor control)
    ├── RobotArmAdapter (manipulation)
    └── SpeechAdapter (vocalization)
```

## Core Concepts

### Sensory Adapters (Input)

**Purpose**: Convert external data into spike trains

**Key Methods**:
- `processData(DataSample)` - Process input and generate spikes
- `extractFeatures(DataSample)` - Extract relevant features
- `encodeFeatures(FeatureVector)` - Encode features as spikes
- `getNeurons()` - Get sensory neuron population
- `getActivationPattern()` - Get current activation state

**Encoding Strategies**:
- **Rate Coding**: Feature intensity → spike timing (stronger = earlier)
- **Temporal Coding**: Feature value → precise spike timing
- **Population Coding**: Feature value → population activity pattern
- **Phase Coding**: Feature value → spike phase relative to oscillation

### Motor Adapters (Output)

**Purpose**: Convert spike trains into external actions

**Key Methods**:
- `decodeActivity(SpikeActivity)` - Decode spikes into commands
- `executeCommand(MotorCommand)` - Execute action in environment
- `processNeurons(neurons, time)` - Process motor neuron activity
- `getChannelCount()` - Get number of output channels
- `getCurrentCommand()` - Get current command state

**Decoding Strategies**:
- **Rate Decoding**: Spike rate → command intensity
- **Population Vector**: Population activity → direction/magnitude
- **Temporal Pattern**: Spike timing → precise control signals
- **Winner-Take-All**: Most active neuron → discrete action selection

## Available Adapters

### RetinaAdapter (Sensory)

**Purpose**: Process visual input (images)

**Features**:
- Spatial grid decomposition (receptive fields)
- Edge detection with multiple orientations
- Rate coding (intensity → spike timing)
- Mimics retina and V1 cortex processing

**Configuration**:
```cpp
BaseAdapter::Config config;
config.type = "retina";
config.name = "left_eye";
config.setIntParam("grid_size", 7);           // 7×7 spatial grid
config.setIntParam("num_orientations", 8);    // 8 edge orientations
config.setDoubleParam("edge_threshold", 0.15); // Min edge strength
config.temporalWindow = 100.0;                 // Spike duration (ms)

auto retina = std::make_shared<RetinaAdapter>(config);
retina->initialize();
```

**Usage**:
```cpp
// Process image
SensoryAdapter::DataSample sample;
sample.rawData = imagePixels; // 28×28 grayscale
auto spikes = retina->processData(sample);

// Get activation pattern
auto activations = retina->getActivationPattern();
```

**Applications**:
- Digit recognition (MNIST)
- Object detection
- Visual tracking
- Scene understanding

---

### AudioAdapter (Sensory)

**Purpose**: Process audio input (sound)

**Features**:
- Frequency decomposition (FFT/filterbank)
- Mel-scale frequency mapping
- Rate or temporal encoding
- Mimics cochlear processing

**Configuration**:
```cpp
BaseAdapter::Config config;
config.type = "audio";
config.name = "left_ear";
config.setIntParam("sample_rate", 44100);     // 44.1 kHz
config.setIntParam("num_channels", 32);       // 32 frequency bands
config.setDoubleParam("min_frequency", 20.0); // 20 Hz
config.setDoubleParam("max_frequency", 20000.0); // 20 kHz

auto audio = std::make_shared<AudioAdapter>(config);
audio->initialize();
```

**Applications**:
- Speech recognition
- Music analysis
- Sound localization
- Acoustic event detection

---

### DisplayAdapter (Motor)

**Purpose**: Visualize neural activity

**Features**:
- Spike raster plots
- Firing rate heatmaps
- Population activity vectors
- ASCII art visualization

**Configuration**:
```cpp
BaseAdapter::Config config;
config.type = "display";
config.name = "activity_monitor";
config.setIntParam("display_width", 80);
config.setIntParam("display_height", 24);
config.setStringParam("display_mode", "heatmap");

auto display = std::make_shared<DisplayAdapter>(config);
display->initialize();
```

**Usage**:
```cpp
// Process motor neurons
display->processNeurons(motorNeurons, currentTime);

// Get display buffer
std::string visualization = display->getDisplayBuffer();
std::cout << visualization << std::endl;
```

**Applications**:
- Network activity monitoring
- Debugging and visualization
- Real-time feedback
- Educational demonstrations

---

### InterneuronReceiverAdapter / InterneuronTransmitterAdapter (Network I/O)

**Purpose**: Exchange spike activity between separate SNNFrame instances over TCP/IP.

**Features**:
- Transmit spike-active channels from one process/host to another
- Receive spike frames and expose them as a sensory spike pattern
- Works across machines (TCP sockets)

**Protocol** (line-based):
```
SNNFW_SPIKES v1 <window_start_ms> <window_end_ms> <index_csv_or_->
```

**Factory types**:
- Receiver (sensory): `interneuron_rx`
- Transmitter (motor): `interneuron_tx`

**Configuration example**:
```cpp
BaseAdapter::Config rxCfg;
rxCfg.type = "interneuron_rx";
rxCfg.name = "rx";
rxCfg.setStringParam("bind_host", "0.0.0.0");
rxCfg.setIntParam("bind_port", 5000);
rxCfg.setIntParam("neuron_count", 256);

BaseAdapter::Config txCfg;
txCfg.type = "interneuron_tx";
txCfg.name = "tx";
txCfg.setStringParam("remote_host", "192.168.1.20");
txCfg.setIntParam("remote_port", 5000);
txCfg.setDoubleParam("update_interval_ms", 10.0);
```

**Declarative JSON (SONATA `snnframe.adapters` or native `adapters`)**:
```json
[
  {
    "name": "rx_bridge",
    "type": "interneuron_rx",
    "role": "sensory",
    "bind_to": "input",
    "temporal_window_ms": 10.0,
    "int_params": { "bind_port": 5000, "neuron_count": 784 },
    "string_params": { "bind_host": "0.0.0.0" }
  },
  {
    "name": "tx_bridge",
    "type": "interneuron_tx",
    "role": "motor",
    "bind_to": "output",
    "temporal_window_ms": 10.0,
    "int_params": { "remote_port": 5050 },
    "double_params": { "update_interval_ms": 10.0 },
    "string_params": { "remote_host": "127.0.0.1" }
  }
]
```

---

## Creating Custom Adapters

### Step 1: Define Adapter Class

```cpp
#include "snnfw/adapters/SensoryAdapter.h"

class MyCustomAdapter : public SensoryAdapter {
public:
    explicit MyCustomAdapter(const Config& config);
    
    bool initialize() override;
    SpikePattern processData(const DataSample& data) override;
    FeatureVector extractFeatures(const DataSample& data) override;
    SpikePattern encodeFeatures(const FeatureVector& features) override;
    
    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const override;
    std::vector<double> getActivationPattern() const override;
    size_t getNeuronCount() const override;
    size_t getFeatureDimension() const override;
    void clearNeuronStates() override;
    
private:
    std::vector<std::shared_ptr<Neuron>> neurons_;
    // ... custom fields
};
```

### Step 2: Implement Core Methods

```cpp
MyCustomAdapter::MyCustomAdapter(const Config& config)
    : SensoryAdapter(config) {
    // Load configuration parameters
    int myParam = getIntParam("my_param", 10);
}

bool MyCustomAdapter::initialize() {
    if (!SensoryAdapter::initialize()) {
        return false;
    }
    
    // Create neuron population
    int neuronCount = getIntParam("neuron_count", 100);
    for (int i = 0; i < neuronCount; ++i) {
        auto neuron = std::make_shared<Neuron>(200.0, 0.7, 100, i);
        neurons_.push_back(neuron);
    }
    
    return true;
}

SpikePattern MyCustomAdapter::processData(const DataSample& data) {
    auto features = extractFeatures(data);
    return encodeFeatures(features);
}
```

### Step 3: Register with Factory

```cpp
#include "snnfw/adapters/AdapterFactory.h"

// In your .cpp file
REGISTER_SENSORY_ADAPTER(MyCustomAdapter, "my_custom");
```

### Step 4: Use Your Adapter

```cpp
BaseAdapter::Config config;
config.type = "my_custom";
config.name = "my_adapter_instance";

auto& factory = AdapterFactory::getInstance();
auto adapter = factory.createSensoryAdapter(config);
adapter->initialize();
```

---

## Adapter Factory

The `AdapterFactory` provides centralized adapter creation and management:

```cpp
#include "snnfw/adapters/AdapterFactory.h"

// Get factory instance
auto& factory = AdapterFactory::getInstance();

// Register custom adapter
factory.registerSensoryAdapter("my_type", 
    [](const BaseAdapter::Config& cfg) {
        return std::make_shared<MyAdapter>(cfg);
    });

// Create adapter
BaseAdapter::Config config;
config.type = "retina";
auto adapter = factory.createSensoryAdapter(config);

// Check available types
auto types = factory.getSensoryAdapterTypes();
for (const auto& type : types) {
    std::cout << "Available: " << type << std::endl;
}
```

---

## Configuration

Adapters support flexible configuration through the `BaseAdapter::Config` structure:

```cpp
BaseAdapter::Config config;

// Basic properties
config.name = "my_adapter";
config.type = "retina";
config.temporalWindow = 100.0;

// Type-specific parameters
config.setIntParam("grid_size", 7);
config.setDoubleParam("threshold", 0.15);
config.setStringParam("mode", "edge_detection");

// Retrieve parameters
int gridSize = config.getIntParam("grid_size", 5); // default: 5
double threshold = config.getDoubleParam("threshold", 0.1);
std::string mode = config.getStringParam("mode", "default");
```

---

## Best Practices

### 1. Encapsulation
- Keep domain-specific processing inside adapters
- Don't leak implementation details to the SNN core
- Provide clean, minimal interfaces

### 2. Configurability
- Make all parameters configurable
- Provide sensible defaults
- Document parameter ranges and effects

### 3. Reusability
- Design adapters to be general-purpose
- Avoid hardcoding assumptions
- Support multiple instances

### 4. Performance
- Minimize memory allocations
- Cache frequently used data
- Use efficient algorithms

### 5. Testing
- Test adapters independently
- Provide example data
- Verify spike encoding/decoding

---

## Examples

See the `examples/adapters/` directory for complete examples:
- `retina_example.cpp` - Image processing with RetinaAdapter
- `audio_example.cpp` - Sound processing with AudioAdapter
- `display_example.cpp` - Visualization with DisplayAdapter
- `custom_adapter_example.cpp` - Creating custom adapters

---

## See Also

- [Configuration System Documentation](../../../docs/CONFIGURATION_SYSTEM.md)
- [MNIST Experiments](../../../MNIST_EXPERIMENTS.md)
- [Main README](../../../README.md)
