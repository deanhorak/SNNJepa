#include "snnfw/Dendrite.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <algorithm>

using json = nlohmann::json;

namespace snnfw {

void Dendrite::addSynapse(uint64_t synapseId) {
    // Check if synapse is already connected
    auto it = std::find(synapseIds.begin(), synapseIds.end(), synapseId);
    if (it == synapseIds.end()) {
        synapseIds.push_back(synapseId);
        SNNFW_DEBUG("Dendrite {}: Added synapse {} (total: {})", 
                    getId(), synapseId, synapseIds.size());
    } else {
        SNNFW_WARN("Dendrite {}: Synapse {} already connected", getId(), synapseId);
    }
}

bool Dendrite::removeSynapse(uint64_t synapseId) {
    auto it = std::find(synapseIds.begin(), synapseIds.end(), synapseId);
    if (it != synapseIds.end()) {
        synapseIds.erase(it);
        SNNFW_DEBUG("Dendrite {}: Removed synapse {} (remaining: {})", 
                    getId(), synapseId, synapseIds.size());
        return true;
    }
    SNNFW_WARN("Dendrite {}: Synapse {} not found for removal", getId(), synapseId);
    return false;
}

void Dendrite::receiveSpike(const std::shared_ptr<ActionPotential>& actionPotential) {
    if (!actionPotential) {
        SNNFW_ERROR("Dendrite {}: Received null action potential", getId());
        return;
    }

    receivedSpikeCount_.fetch_add(1, std::memory_order_relaxed);

    SNNFW_TRACE("Dendrite {} (Neuron {}): Received spike from synapse {} at time {:.3f}ms (amplitude: {:.3f})",
                getId(),
                targetNeuronId,
                actionPotential->getSynapseId(),
                actionPotential->getScheduledTime(),
                actionPotential->getAmplitude());

    // Deliver spike to target neuron via NetworkPropagator
    auto propagator = networkPropagator_.lock();
    if (propagator) {
        propagator->deliverSpikeToNeuron(
            targetNeuronId,
            actionPotential->getSynapseId(),
            actionPotential->getScheduledTime(),
            actionPotential->getAmplitude(),
            actionPotential->getDispatchTime()
        );
    } else {
        // Fallback: just log if no propagator is set (for backward compatibility)
        SNNFW_DEBUG("Dendrite {}: No NetworkPropagator set, spike not delivered to neuron {}",
                   getId(), targetNeuronId);
    }
}

size_t Dendrite::getReceivedSpikeCount() const {
    return receivedSpikeCount_.load(std::memory_order_relaxed);
}

void Dendrite::resetReceivedSpikeCount() {
    receivedSpikeCount_.store(0, std::memory_order_relaxed);
}

std::string Dendrite::toJson() const {
    json j;
    j["type"] = "Dendrite";
    j["id"] = getId();
    j["targetNeuronId"] = targetNeuronId;
    j["synapseIds"] = synapseIds;
    return j.dump();
}

bool Dendrite::fromJson(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);

        if (j["type"] != "Dendrite") {
            SNNFW_ERROR("Invalid type in JSON: expected 'Dendrite', got '{}'", j["type"].get<std::string>());
            return false;
        }

        // Deserialize ID from base class
        setId(j["id"]);

        targetNeuronId = j["targetNeuronId"];
        synapseIds = j["synapseIds"].get<std::vector<uint64_t>>();

        return true;
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to deserialize Dendrite from JSON: {}", e.what());
        return false;
    }
}

} // namespace snnfw
