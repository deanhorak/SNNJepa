#include "snnfw/STDPLearning.h"
#include "snnfw/Synapse.h"
#include "snnfw/Logger.h"
#include <cmath>
#include <algorithm>

namespace snnfw {

STDPLearning::STDPLearning(double aPlus, double aMinus, 
                           double tauPlus, double tauMinus,
                           double wMin, double wMax)
    : aPlus(aPlus),
      aMinus(aMinus),
      tauPlus(tauPlus),
      tauMinus(tauMinus),
      wMin(wMin),
      wMax(wMax),
      numPotentiations(0),
      numDepressions(0),
      totalWeightChange(0.0),
      totalWeightChanges(0) {
}

void STDPLearning::recordPreSpike(uint64_t synapseId, double time) {
    preSpikeHistory[synapseId] = time;
}

void STDPLearning::recordPostSpike(uint64_t neuronId,
                                   double time,
                                   const std::vector<uint64_t>& synapseIds,
                                   std::map<uint64_t, std::shared_ptr<Synapse>>& synapses) {
    // For each synapse connecting to this neuron
    for (uint64_t synapseId : synapseIds) {
        // Check if we have a pre-synaptic spike for this synapse
        auto it = preSpikeHistory.find(synapseId);
        if (it == preSpikeHistory.end()) {
            continue;  // No pre-spike recorded yet
        }
        
        double preTime = it->second;
        double deltaT = time - preTime;  // t_post - t_pre
        
        // Calculate weight change
        double dw = calculateWeightChange(deltaT);
        
        // Find the synapse and update its weight
        auto synapseIt = synapses.find(synapseId);
        if (synapseIt != synapses.end()) {
            auto& synapse = synapseIt->second;
            double oldWeight = synapse->getWeight();
            double newWeight = clampWeight(oldWeight + dw);
            synapse->setWeight(newWeight);
            
            // Update statistics
            if (dw > 0) {
                numPotentiations++;
            } else if (dw < 0) {
                numDepressions++;
            }
            totalWeightChange += std::abs(dw);
            totalWeightChanges++;
            
            SNNFW_TRACE("STDP: Synapse {} weight: {:.4f} -> {:.4f} (Δt={:.2f}ms, Δw={:.6f})",
                       synapseId, oldWeight, newWeight, deltaT, dw);
        }
    }
}

double STDPLearning::calculateWeightChange(double deltaT) const {
    if (deltaT > 0) {
        // Pre before post: potentiation
        return aPlus * std::exp(-deltaT / tauPlus);
    } else if (deltaT < 0) {
        // Post before pre: depression
        return -aMinus * std::exp(deltaT / tauMinus);
    } else {
        // Simultaneous spikes: no change
        return 0.0;
    }
}

void STDPLearning::clearHistory() {
    preSpikeHistory.clear();
}

} // namespace snnfw

