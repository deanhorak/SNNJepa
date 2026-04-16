#ifndef SNNFW_STDP_LEARNING_H
#define SNNFW_STDP_LEARNING_H

#include <cstdint>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

namespace snnfw {

// Forward declarations
class Synapse;

/**
 * @brief Spike-Timing-Dependent Plasticity (STDP) learning rule
 * 
 * Implements the classical STDP learning rule where synaptic weights
 * are modified based on the relative timing of pre- and post-synaptic spikes.
 * 
 * Rule:
 * - If pre-synaptic spike occurs before post-synaptic spike (Δt > 0):
 *   Weight increases (potentiation): Δw = A+ * exp(-Δt / τ+)
 * 
 * - If pre-synaptic spike occurs after post-synaptic spike (Δt < 0):
 *   Weight decreases (depression): Δw = -A- * exp(Δt / τ-)
 * 
 * Where:
 * - Δt = t_post - t_pre (time difference in ms)
 * - A+, A- are learning rate parameters
 * - τ+, τ- are time constants (ms)
 * 
 * Reference:
 * - Bi, G. Q., & Poo, M. M. (1998). "Synaptic modifications in cultured
 *   hippocampal neurons: dependence on spike timing, synaptic strength,
 *   and postsynaptic cell type." Journal of Neuroscience.
 */
class STDPLearning {
public:
    /**
     * @brief Constructor with default parameters
     * @param aPlus Learning rate for potentiation (default: 0.01)
     * @param aMinus Learning rate for depression (default: 0.012)
     * @param tauPlus Time constant for potentiation in ms (default: 20.0)
     * @param tauMinus Time constant for depression in ms (default: 20.0)
     * @param wMin Minimum synaptic weight (default: 0.0)
     * @param wMax Maximum synaptic weight (default: 1.0)
     */
    STDPLearning(double aPlus = 0.01,
                 double aMinus = 0.012,
                 double tauPlus = 20.0,
                 double tauMinus = 20.0,
                 double wMin = 0.0,
                 double wMax = 1.0);
    
    /**
     * @brief Record a pre-synaptic spike
     * @param synapseId ID of the synapse
     * @param time Time of the spike (ms)
     */
    void recordPreSpike(uint64_t synapseId, double time);
    
    /**
     * @brief Record a post-synaptic spike and update weights
     * @param neuronId ID of the post-synaptic neuron
     * @param time Time of the spike (ms)
     * @param synapseIds IDs of all synapses connecting to this neuron
     * @param synapses Map of synapse ID to synapse pointer for weight updates
     */
    void recordPostSpike(uint64_t neuronId,
                        double time,
                        const std::vector<uint64_t>& synapseIds,
                        std::map<uint64_t, std::shared_ptr<Synapse>>& synapses);
    
    /**
     * @brief Calculate weight change for given time difference
     * @param deltaT Time difference (t_post - t_pre) in ms
     * @return Weight change (can be positive or negative)
     */
    double calculateWeightChange(double deltaT) const;
    
    /**
     * @brief Clear spike history (for new training epoch)
     */
    void clearHistory();
    
    /**
     * @brief Set learning rates
     */
    void setLearningRates(double aPlus, double aMinus) {
        this->aPlus = aPlus;
        this->aMinus = aMinus;
    }
    
    /**
     * @brief Set time constants
     */
    void setTimeConstants(double tauPlus, double tauMinus) {
        this->tauPlus = tauPlus;
        this->tauMinus = tauMinus;
    }
    
    /**
     * @brief Set weight bounds
     */
    void setWeightBounds(double wMin, double wMax) {
        this->wMin = wMin;
        this->wMax = wMax;
    }
    
    /**
     * @brief Get statistics
     */
    void getStats(size_t& numPotentiations, size_t& numDepressions, 
                  double& avgWeightChange) const {
        numPotentiations = this->numPotentiations;
        numDepressions = this->numDepressions;
        avgWeightChange = (totalWeightChanges > 0) ? 
            (totalWeightChange / totalWeightChanges) : 0.0;
    }
    
    /**
     * @brief Reset statistics
     */
    void resetStats() {
        numPotentiations = 0;
        numDepressions = 0;
        totalWeightChange = 0.0;
        totalWeightChanges = 0;
    }
    
private:
    // STDP parameters
    double aPlus;      ///< Learning rate for potentiation
    double aMinus;     ///< Learning rate for depression
    double tauPlus;    ///< Time constant for potentiation (ms)
    double tauMinus;   ///< Time constant for depression (ms)
    double wMin;       ///< Minimum weight
    double wMax;       ///< Maximum weight
    
    // Spike history (synapse ID -> last spike time)
    std::map<uint64_t, double> preSpikeHistory;
    
    // Statistics
    size_t numPotentiations;
    size_t numDepressions;
    double totalWeightChange;
    size_t totalWeightChanges;
    
    /**
     * @brief Clamp weight to bounds
     */
    double clampWeight(double weight) const {
        if (weight < wMin) return wMin;
        if (weight > wMax) return wMax;
        return weight;
    }
};

} // namespace snnfw

#endif // SNNFW_STDP_LEARNING_H

