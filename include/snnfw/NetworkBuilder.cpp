#include "snnfw/NetworkBuilder.h"
#include "snnfw/Logger.h"
#include <stdexcept>
#include <sstream>

namespace snnfw {

NetworkBuilder::NetworkBuilder(NeuralObjectFactory& factory, 
                               Datastore& datastore,
                               bool autoValidate)
    : factory_(factory),
      datastore_(datastore),
      autoValidate_(autoValidate),
      autoPersist_(true),
      defaultWindowSizeMs_(50.0),
      defaultSimilarityThreshold_(0.85),
      defaultMaxReferencePatterns_(100),
      brain_(nullptr),
      currentHemisphere_(nullptr),
      currentLobe_(nullptr),
      currentRegion_(nullptr),
      currentNucleus_(nullptr),
      currentColumn_(nullptr),
      currentLayer_(nullptr) {
    SNNFW_DEBUG("NetworkBuilder created");
}

NetworkBuilder& NetworkBuilder::createBrain(const std::string& name) {
    brain_ = factory_.createBrain();
    if (!name.empty()) {
        brain_->setName(name);
    }
    persistObject(brain_);
    contextStack_.clear();
    contextStack_.push_back(ContextLevel::BRAIN);
    
    SNNFW_INFO("Created brain: {}", name.empty() ? std::to_string(brain_->getId()) : name);
    return *this;
}

NetworkBuilder& NetworkBuilder::addHemisphere(const std::string& name) {
    if (!brain_) {
        throw std::runtime_error("Cannot add hemisphere: no brain created");
    }
    
    currentHemisphere_ = factory_.createHemisphere();
    if (!name.empty()) {
        currentHemisphere_->setName(name);
    }
    brain_->addHemisphere(currentHemisphere_->getId());
    hemispheres_.push_back(currentHemisphere_);
    persistObject(currentHemisphere_);
    persistObject(brain_);  // Update brain with new hemisphere
    pushContext(ContextLevel::HEMISPHERE);
    
    SNNFW_DEBUG("Added hemisphere: {}", name.empty() ? std::to_string(currentHemisphere_->getId()) : name);
    return *this;
}

NetworkBuilder& NetworkBuilder::addLobe(const std::string& name) {
    if (!currentHemisphere_) {
        throw std::runtime_error("Cannot add lobe: no hemisphere context");
    }
    
    currentLobe_ = factory_.createLobe();
    if (!name.empty()) {
        currentLobe_->setName(name);
    }
    currentHemisphere_->addLobe(currentLobe_->getId());
    lobes_.push_back(currentLobe_);
    persistObject(currentLobe_);
    persistObject(currentHemisphere_);
    pushContext(ContextLevel::LOBE);
    
    SNNFW_DEBUG("Added lobe: {}", name.empty() ? std::to_string(currentLobe_->getId()) : name);
    return *this;
}

NetworkBuilder& NetworkBuilder::addRegion(const std::string& name) {
    if (!currentLobe_) {
        throw std::runtime_error("Cannot add region: no lobe context");
    }
    
    currentRegion_ = factory_.createRegion();
    if (!name.empty()) {
        currentRegion_->setName(name);
    }
    currentLobe_->addRegion(currentRegion_->getId());
    regions_.push_back(currentRegion_);
    persistObject(currentRegion_);
    persistObject(currentLobe_);
    pushContext(ContextLevel::REGION);
    
    SNNFW_DEBUG("Added region: {}", name.empty() ? std::to_string(currentRegion_->getId()) : name);
    return *this;
}

NetworkBuilder& NetworkBuilder::addNucleus(const std::string& name) {
    if (!currentRegion_) {
        throw std::runtime_error("Cannot add nucleus: no region context");
    }
    
    currentNucleus_ = factory_.createNucleus();
    if (!name.empty()) {
        currentNucleus_->setName(name);
    }
    currentRegion_->addNucleus(currentNucleus_->getId());
    nuclei_.push_back(currentNucleus_);
    persistObject(currentNucleus_);
    persistObject(currentRegion_);
    pushContext(ContextLevel::NUCLEUS);
    
    SNNFW_DEBUG("Added nucleus: {}", name.empty() ? std::to_string(currentNucleus_->getId()) : name);
    return *this;
}

NetworkBuilder& NetworkBuilder::addColumn(const std::string& name) {
    if (!currentNucleus_) {
        throw std::runtime_error("Cannot add column: no nucleus context");
    }

    currentColumn_ = factory_.createColumn();
    // Note: Column doesn't have setName() method - names are tracked externally if needed
    currentNucleus_->addColumn(currentColumn_->getId());
    columns_.push_back(currentColumn_);
    persistObject(currentColumn_);
    persistObject(currentNucleus_);
    pushContext(ContextLevel::COLUMN);

    SNNFW_DEBUG("Added column: {}", name.empty() ? std::to_string(currentColumn_->getId()) : name);
    return *this;
}

NetworkBuilder& NetworkBuilder::addColumns(size_t count, const std::string& namePrefix) {
    if (!currentNucleus_) {
        throw std::runtime_error("Cannot add columns: no nucleus context");
    }

    for (size_t i = 0; i < count; ++i) {
        auto column = factory_.createColumn();
        // Note: Column doesn't have setName() method - names are tracked externally if needed
        currentNucleus_->addColumn(column->getId());
        columns_.push_back(column);
        persistObject(column);
    }
    persistObject(currentNucleus_);

    // Set the last column as current
    if (!columns_.empty()) {
        currentColumn_ = columns_.back();
        pushContext(ContextLevel::COLUMN);
    }

    SNNFW_INFO("Added {} columns with prefix '{}'", count, namePrefix);
    return *this;
}

NetworkBuilder& NetworkBuilder::addLayer(const std::string& name) {
    if (!currentColumn_) {
        throw std::runtime_error("Cannot add layer: no column context");
    }

    currentLayer_ = factory_.createLayer();
    // Note: Layer doesn't have setName() method - names are tracked externally if needed
    currentColumn_->addLayer(currentLayer_->getId());
    layers_.push_back(currentLayer_);
    persistObject(currentLayer_);
    persistObject(currentColumn_);
    pushContext(ContextLevel::LAYER);

    SNNFW_DEBUG("Added layer: {}", name.empty() ? std::to_string(currentLayer_->getId()) : name);
    return *this;
}

NetworkBuilder& NetworkBuilder::addLayers(size_t count, const std::string& namePrefix) {
    if (!currentColumn_) {
        throw std::runtime_error("Cannot add layers: no column context");
    }

    for (size_t i = 0; i < count; ++i) {
        auto layer = factory_.createLayer();
        // Note: Layer doesn't have setName() method - names are tracked externally if needed
        currentColumn_->addLayer(layer->getId());
        layers_.push_back(layer);
        persistObject(layer);
    }
    persistObject(currentColumn_);

    // Set the last layer as current
    if (!layers_.empty()) {
        currentLayer_ = layers_.back();
        pushContext(ContextLevel::LAYER);
    }

    SNNFW_INFO("Added {} layers with prefix '{}'", count, namePrefix);
    return *this;
}

NetworkBuilder& NetworkBuilder::addCluster(size_t neuronCount,
                                           double windowSizeMs,
                                           double similarityThreshold,
                                           size_t maxReferencePatterns) {
    if (!currentLayer_) {
        throw std::runtime_error("Cannot add cluster: no layer context");
    }

    auto cluster = factory_.createCluster();

    // Use default values if 0 is passed
    double actualWindowSize = (windowSizeMs == 0.0) ? defaultWindowSizeMs_ : windowSizeMs;
    double actualThreshold = (similarityThreshold == 0.0) ? defaultSimilarityThreshold_ : similarityThreshold;
    size_t actualMaxPatterns = (maxReferencePatterns == 0) ? defaultMaxReferencePatterns_ : maxReferencePatterns;

    // Create neurons for this cluster
    for (size_t i = 0; i < neuronCount; ++i) {
        auto neuron = factory_.createNeuron(actualWindowSize, actualThreshold, actualMaxPatterns);
        cluster->addNeuron(neuron->getId());  // addNeuron expects ID, not object
        neurons_.push_back(neuron);
        persistObject(neuron);
    }

    currentLayer_->addCluster(cluster->getId());
    clusters_.push_back(cluster);
    persistObject(cluster);
    persistObject(currentLayer_);

    SNNFW_DEBUG("Added cluster with {} neurons", neuronCount);
    return *this;
}

NetworkBuilder& NetworkBuilder::addClusters(size_t clusterCount,
                                            size_t neuronsPerCluster,
                                            double windowSizeMs,
                                            double similarityThreshold,
                                            size_t maxReferencePatterns) {
    if (!currentLayer_) {
        throw std::runtime_error("Cannot add clusters: no layer context");
    }
    
    for (size_t i = 0; i < clusterCount; ++i) {
        addCluster(neuronsPerCluster, windowSizeMs, similarityThreshold, maxReferencePatterns);
    }
    
    SNNFW_INFO("Added {} clusters with {} neurons each", clusterCount, neuronsPerCluster);
    return *this;
}

NetworkBuilder& NetworkBuilder::up() {
    popContext();
    return *this;
}

NetworkBuilder& NetworkBuilder::toRoot() {
    while (contextStack_.size() > 1) {
        popContext();
    }
    return *this;
}

std::shared_ptr<Brain> NetworkBuilder::build() {
    if (!brain_) {
        throw std::runtime_error("Cannot build: no brain created");
    }

    if (autoValidate_) {
        NetworkValidator validator;
        auto result = validator.validateNetwork(brain_->getId(), datastore_);

        if (!result.isValid) {
            std::ostringstream oss;
            oss << "Network validation failed with " << result.errors.size() << " errors:\n";
            for (const auto& error : result.errors) {
                oss << "  - " << error.message << "\n";
            }
            SNNFW_ERROR("{}", oss.str());
            throw std::runtime_error(oss.str());
        }

        SNNFW_INFO("Network validation passed");
    }

    SNNFW_INFO("Built brain with {} neurons in {} clusters", neurons_.size(), clusters_.size());
    return brain_;
}

NetworkBuilder& NetworkBuilder::setAutoPersist(bool autoPersist) {
    autoPersist_ = autoPersist;
    return *this;
}

NetworkBuilder& NetworkBuilder::setNeuronParams(double windowSizeMs,
                                                double similarityThreshold,
                                                size_t maxReferencePatterns) {
    defaultWindowSizeMs_ = windowSizeMs;
    defaultSimilarityThreshold_ = similarityThreshold;
    defaultMaxReferencePatterns_ = maxReferencePatterns;
    return *this;
}

void NetworkBuilder::persistObject(const std::shared_ptr<NeuralObject>& obj) {
    if (autoPersist_ && obj) {
        datastore_.put(obj);
        datastore_.markDirty(obj->getId());
    }
}

void NetworkBuilder::pushContext(ContextLevel level) {
    contextStack_.push_back(level);
}

void NetworkBuilder::popContext() {
    if (contextStack_.empty()) {
        throw std::runtime_error("Cannot navigate up: already at root");
    }
    
    ContextLevel level = contextStack_.back();
    contextStack_.pop_back();
    
    // Clear the current context at this level
    switch (level) {
        case ContextLevel::LAYER:
            currentLayer_ = nullptr;
            break;
        case ContextLevel::COLUMN:
            currentColumn_ = nullptr;
            break;
        case ContextLevel::NUCLEUS:
            currentNucleus_ = nullptr;
            break;
        case ContextLevel::REGION:
            currentRegion_ = nullptr;
            break;
        case ContextLevel::LOBE:
            currentLobe_ = nullptr;
            break;
        case ContextLevel::HEMISPHERE:
            currentHemisphere_ = nullptr;
            break;
        case ContextLevel::BRAIN:
            // Can't go above brain
            break;
    }
}

} // namespace snnfw

