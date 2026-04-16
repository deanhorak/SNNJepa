#ifndef SNNFW_NETWORK_PROPAGATOR_H
#define SNNFW_NETWORK_PROPAGATOR_H

#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Synapse.h"
#include "snnfw/Dendrite.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/SpikeAcknowledgment.h"
#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <limits>

namespace snnfw {

/**
 * @brief NetworkPropagator manages forward propagation of spikes through a multi-layer network
 *
 * This class coordinates the propagation of action potentials through explicit synaptic
 * connections in a biologically plausible manner. It maintains registries of all neural
 * objects and provides methods for:
 * - Registering neurons, axons, synapses, and dendrites
 * - Propagating spikes from source neurons through their axons and synapses
 * - Delivering spikes to target neurons via dendrites
 * - Computing layer activations based on spike patterns
 *
 * Architecture:
 * - Maintains registries for all neural objects (neurons, axons, synapses, dendrites)
 * - Uses SpikeProcessor for temporal spike delivery with delays
 * - Supports layer-by-layer forward propagation
 * - Thread-safe for concurrent access
 *
 * Biological Motivation:
 * In biological neural networks, action potentials propagate from the soma through
 * the axon, across synapses (with delays and weight modulation), and into dendrites
 * of downstream neurons. This class simulates that process explicitly.
 */
class NetworkPropagator {
public:
    struct StdpUpdateStats {
        uint64_t total = 0;
        uint64_t ltp = 0;
        uint64_t ltd = 0;
        uint64_t reward = 0;
    };

    enum class SynapseGroup {
        Unknown = 0,
        InputToL4,
        L4ToL5,
        L5ToOutput,
    };

    struct StdpGroupStats {
        uint64_t ltp = 0;
        uint64_t ltd = 0;
    };

    struct StdpGroupStatsSet {
        StdpGroupStats inputToL4;
        StdpGroupStats l4ToL5;
        StdpGroupStats l5ToOutput;
        StdpGroupStats other;
    };

    struct StdpTimingStats {
        uint64_t preBeforePost = 0;
        uint64_t postBeforePre = 0;
        uint64_t nearZero = 0;
    };

    struct StdpTimingGroupStats {
        StdpTimingStats inputToL4;
        StdpTimingStats l4ToL5;
        StdpTimingStats l5ToOutput;
        StdpTimingStats other;
    };

    struct NeuronStdpEligibilityStats {
        uint64_t totalUpdates = 0;
        uint64_t ltpUpdates = 0;
        uint64_t ltdUpdates = 0;
        double ltpMagnitude = 0.0;
        double ltdMagnitude = 0.0;

        double score(double ltdPenalty = 1.0) const {
            return ltpMagnitude - (ltdPenalty * ltdMagnitude);
        }
    };

    struct DeliveryStats {
        uint64_t total = 0;
        uint64_t l5ToOutput = 0;
        double l5ToOutputDeltaSum = 0.0;
        double l5ToOutputDeltaMin = std::numeric_limits<double>::infinity();
        double l5ToOutputDeltaMax = -std::numeric_limits<double>::infinity();
    };

    struct ScheduleStats {
        uint64_t inputToL4 = 0;
        uint64_t other = 0;
    };

    /**
     * @brief Constructor
     * @param spikeProcessor Shared pointer to the spike processor for temporal delivery
     */
    explicit NetworkPropagator(std::shared_ptr<SpikeProcessor> spikeProcessor);

    /**
     * @brief Register a neuron with the propagator
     * @param neuron Shared pointer to the neuron
     */
    void registerNeuron(const std::shared_ptr<Neuron>& neuron);

    /**
     * @brief Register an axon with the propagator
     * @param axon Shared pointer to the axon
     */
    void registerAxon(const std::shared_ptr<Axon>& axon);

    /**
     * @brief Register a synapse with the propagator
     * @param synapse Shared pointer to the synapse
     */
    void registerSynapse(const std::shared_ptr<Synapse>& synapse);

    /**
     * @brief Register a dendrite with the propagator
     * @param dendrite Shared pointer to the dendrite
     */
    void registerDendrite(const std::shared_ptr<Dendrite>& dendrite);

    /**
     * @brief Fire a neuron and propagate spikes through its axon
     * 
     * When a neuron fires, this method:
     * 1. Gets the neuron's axon
     * 2. For each synapse connected to the axon:
     *    - Creates an ActionPotential with appropriate delay and weight
     *    - Schedules it for delivery via SpikeProcessor
     *
     * @param neuronId ID of the neuron that is firing
     * @param firingTime Time when the neuron fires (in ms)
     * @return Number of spikes scheduled, or -1 if neuron/axon not found
     */
    int fireNeuron(uint64_t neuronId, double firingTime);

    /**
     * @brief Deliver a spike to a target neuron via dendrite
     *
     * This is called by the dendrite when it receives an action potential.
     * It inserts the spike into the target neuron's spike buffer and records
     * the incoming spike for STDP.
     *
     * @param neuronId ID of the target neuron
     * @param synapseId ID of the synapse that delivered the spike
     * @param spikeTime Time of the spike (in ms)
     * @param amplitude Amplitude of the spike (modulated by synaptic weight)
     * @param dispatchTime Time when the spike was originally dispatched (in ms)
     * @return true if delivered successfully, false if neuron not found
     */
    bool deliverSpikeToNeuron(uint64_t neuronId, uint64_t synapseId, double spikeTime, double amplitude, double dispatchTime = 0.0);

    /**
     * @brief Send an acknowledgment from a postsynaptic neuron to update a synapse
     *
     * This implements the feedback mechanism for STDP. When a neuron fires, it sends
     * acknowledgments to all synapses that contributed spikes within the temporal window.
     * The synapse uses the timing information to adjust its weight.
     *
     * @param acknowledgment Shared pointer to the spike acknowledgment
     */
    void sendAcknowledgment(const std::shared_ptr<SpikeAcknowledgment>& acknowledgment);

    /**
     * @brief Apply STDP weight update to a synapse
     *
     * Uses the classic STDP learning rule:
     * - If Δt > 0 (pre before post): LTP (strengthen synapse)
     * - If Δt < 0 (post before pre): LTD (weaken synapse)
     * - Magnitude decreases exponentially with |Δt|
     *
     * @param synapseId ID of the synapse to update
     * @param timeDifference Δt = t_post - t_pre (in ms)
     * @return true if synapse was updated, false if not found
     */
    bool applySTDP(uint64_t synapseId, double timeDifference);

    /**
     * @brief Get a neuron by ID
     * @param neuronId ID of the neuron
     * @return Shared pointer to the neuron, or nullptr if not found
     */
    std::shared_ptr<Neuron> getNeuron(uint64_t neuronId) const;

    /**
     * @brief Get an axon by ID
     * @param axonId ID of the axon
     * @return Shared pointer to the axon, or nullptr if not found
     */
    std::shared_ptr<Axon> getAxon(uint64_t axonId) const;

    /**
     * @brief Get a synapse by ID
     * @param synapseId ID of the synapse
     * @return Shared pointer to the synapse, or nullptr if not found
     */
    std::shared_ptr<Synapse> getSynapse(uint64_t synapseId) const;

    /**
     * @brief Get a dendrite by ID
     * @param dendriteId ID of the dendrite
     * @return Shared pointer to the dendrite, or nullptr if not found
     */
    std::shared_ptr<Dendrite> getDendrite(uint64_t dendriteId) const;

    /**
     * @brief Compute activation vector for a layer of neurons
     * 
     * For each neuron in the layer, computes the best similarity between
     * its current spike pattern and its learned reference patterns.
     *
     * @param neuronIds Vector of neuron IDs in the layer
     * @return Vector of activation values (one per neuron)
     */
    std::vector<double> computeLayerActivation(const std::vector<uint64_t>& neuronIds) const;

    /**
     * @brief Clear all spike buffers in registered neurons
     */
    void clearAllSpikes();

    /**
     * @brief Get the number of registered neurons
     * @return Number of neurons in the registry
     */
    size_t getNeuronCount() const;

    /**
     * @brief Get the number of registered synapses
     * @return Number of synapses in the registry
     */
    size_t getSynapseCount() const;

    /**
     * @brief Set STDP learning parameters
     * @param aPlus LTP amplitude (default: 0.01)
     * @param aMinus LTD amplitude (default: 0.012)
     * @param tauPlus LTP time constant in ms (default: 20.0)
     * @param tauMinus LTD time constant in ms (default: 20.0)
     */
    void setSTDPParameters(double aPlus, double aMinus, double tauPlus, double tauMinus);
    void setStdpLtdScale(double scale);
    void setStdpLtdWindowMs(double windowMs);

    /**
     * @brief Apply reward-modulated STDP to synapses targeting a specific neuron
     * @param neuronId ID of the target neuron
     * @param rewardFactor Reward multiplier (1.0 = normal STDP, >1.0 = enhanced learning, <1.0 = reduced learning)
     *
     * This strengthens synapses that contributed to the neuron's activation when reward is positive.
     * Used for supervised learning where correct classifications receive reward.
     */
    void applyRewardModulatedSTDP(uint64_t neuronId, double rewardFactor);

    /**
     * @brief Set activity monitor for automatic spike recording
     * @param monitor Pointer to activity monitor (nullptr to disable)
     */
    void setActivityMonitor(class ActivityMonitor* monitor);

    /**
     * @brief Set recording manager for direct spike recording (bypasses ActivityMonitor)
     * @param manager Pointer to recording manager (nullptr to disable)
     */
    void setRecordingManager(class RecordingManager* manager);

    /**
     * @brief Reset STDP update counters
     */
    void resetStdpUpdateStats();

    /**
     * @brief Get current STDP update counters
     * @return STDP update counts since last reset
     */
    StdpUpdateStats getStdpUpdateStats() const;

    /**
     * @brief Register a synapse group for per-group STDP accounting
     * @param synapseId ID of the synapse
     * @param group Synapse group classification
     */
    void registerSynapseGroup(uint64_t synapseId, SynapseGroup group);

    /**
     * @brief Reset per-group STDP update counters
     */
    void resetStdpGroupStats();

    /**
     * @brief Get current per-group STDP update counters
     * @return Per-group STDP update counts since last reset
     */
    StdpGroupStatsSet getStdpGroupStats() const;

    /**
     * @brief Reset per-group STDP timing counters
     */
    void resetStdpTimingStats();

    /**
     * @brief Get current per-group STDP timing counters
     * @return Per-group STDP timing counts since last reset
     */
    StdpTimingGroupStats getStdpTimingStats() const;

    /**
     * @brief Reset per-neuron STDP eligibility accumulators.
     *
     * Call at the beginning of each training sample to track eligibility
     * induced by that sample's spikes only.
     */
    void resetNeuronStdpEligibility();

    /**
     * @brief Get accumulated STDP eligibility stats for one neuron.
     * @param neuronId Target neuron ID
     * @return Eligibility stats (all zeros if neuron has no updates)
     */
    NeuronStdpEligibilityStats getNeuronStdpEligibility(uint64_t neuronId) const;

    /**
     * @brief Enable trace-based STDP (pre/post traces) instead of acknowledgment timing
     * @param enabled When true, use nearest-neighbor pre/post traces for STDP
     */
    void setTraceStdpEnabled(bool enabled);

    /**
     * @brief Enable or disable all STDP learning
     * @param enabled When false, no weight updates occur (useful for inference)
     */
    void setStdpEnabled(bool enabled);

    /**
     * @brief Check if STDP learning is currently enabled
     * @return true if STDP is enabled, false otherwise
     */
    bool isStdpEnabled() const;

    /**
     * @brief Reset delivery statistics counters
     */
    void resetDeliveryStats();

    /**
     * @brief Get current delivery statistics
     * @return Delivery stats since last reset
     */
    DeliveryStats getDeliveryStats() const;

    /**
     * @brief Reset schedule statistics counters
     */
    void resetScheduleStats();

    /**
     * @brief Get current schedule statistics
     * @return Schedule stats since last reset
     */
    ScheduleStats getScheduleStats() const;

private:
    void accumulateNeuronStdpEligibility(uint64_t neuronId, double timeDifference);

    // Spike processor for temporal delivery
    std::shared_ptr<SpikeProcessor> spikeProcessor_;

    // Registries for neural objects
    std::map<uint64_t, std::shared_ptr<Neuron>> neuronRegistry_;
    std::map<uint64_t, std::shared_ptr<Axon>> axonRegistry_;
    std::map<uint64_t, std::shared_ptr<Synapse>> synapseRegistry_;
    std::map<uint64_t, std::shared_ptr<Dendrite>> dendriteRegistry_;

    // Reverse index: dendrite ID -> list of synapses targeting that dendrite
    // This enables O(1) lookup of synapses by dendrite instead of O(n) iteration
    std::map<uint64_t, std::vector<std::shared_ptr<Synapse>>> dendriteToSynapsesIndex_;

    // Mutexes for thread-safe access
    mutable std::mutex neuronMutex_;
    mutable std::mutex axonMutex_;
    mutable std::mutex synapseMutex_;
    mutable std::mutex dendriteMutex_;

    // STDP learning parameters
    double stdpAPlus_;      ///< LTP amplitude (default: 0.01)
    double stdpAMinus_;     ///< LTD amplitude (default: 0.012)
    double stdpTauPlus_;    ///< LTP time constant in ms (default: 20.0)
    double stdpTauMinus_;   ///< LTD time constant in ms (default: 20.0)
    double stdpLtdScale_;   ///< Additional LTD scaling factor
    double stdpLtdWindowMs_; ///< Max |Δt| window for LTD in trace STDP

    // Activity monitoring (optional)
    class ActivityMonitor* activityMonitor_;  ///< Optional activity monitor for recording
    class RecordingManager* recordingManager_;  ///< Optional recording manager for direct recording

    // STDP update counters (thread-safe)
    std::atomic<uint64_t> stdpUpdatesTotal_{0};
    std::atomic<uint64_t> stdpUpdatesLtp_{0};
    std::atomic<uint64_t> stdpUpdatesLtd_{0};
    std::atomic<uint64_t> stdpUpdatesReward_{0};

    // Per-group STDP update counters (thread-safe)
    std::atomic<uint64_t> stdpInputToL4Ltp_{0};
    std::atomic<uint64_t> stdpInputToL4Ltd_{0};
    std::atomic<uint64_t> stdpL4ToL5Ltp_{0};
    std::atomic<uint64_t> stdpL4ToL5Ltd_{0};
    std::atomic<uint64_t> stdpL5ToOutputLtp_{0};
    std::atomic<uint64_t> stdpL5ToOutputLtd_{0};
    std::atomic<uint64_t> stdpOtherLtp_{0};
    std::atomic<uint64_t> stdpOtherLtd_{0};

    // Per-group STDP timing counters (thread-safe)
    std::atomic<uint64_t> stdpInputToL4PreBeforePost_{0};
    std::atomic<uint64_t> stdpInputToL4PostBeforePre_{0};
    std::atomic<uint64_t> stdpInputToL4NearZero_{0};
    std::atomic<uint64_t> stdpL4ToL5PreBeforePost_{0};
    std::atomic<uint64_t> stdpL4ToL5PostBeforePre_{0};
    std::atomic<uint64_t> stdpL4ToL5NearZero_{0};
    std::atomic<uint64_t> stdpL5ToOutputPreBeforePost_{0};
    std::atomic<uint64_t> stdpL5ToOutputPostBeforePre_{0};
    std::atomic<uint64_t> stdpL5ToOutputNearZero_{0};
    std::atomic<uint64_t> stdpOtherPreBeforePost_{0};
    std::atomic<uint64_t> stdpOtherPostBeforePre_{0};
    std::atomic<uint64_t> stdpOtherNearZero_{0};

    bool traceStdpEnabled_ = false;
    std::atomic<bool> stdpEnabled_{true};  ///< Master switch for all STDP learning

    // Last presynaptic firing time per synapse for trace-based STDP (thread-safe)
    std::unordered_map<uint64_t, double> synapseLastPreTime_;
    mutable std::mutex synapseLastPreMutex_;

    // Last postsynaptic firing time per synapse for pre-triggered LTD (thread-safe)
    std::unordered_map<uint64_t, double> synapseLastPostTime_;
    mutable std::mutex synapseLastPostMutex_;

    std::unordered_map<uint64_t, SynapseGroup> synapseGroupMap_;
    mutable std::mutex synapseGroupMutex_;

    // Per-neuron STDP eligibility accumulators (thread-safe)
    std::unordered_map<uint64_t, NeuronStdpEligibilityStats> neuronStdpEligibility_;
    mutable std::mutex neuronStdpEligibilityMutex_;

    // Delivery stats (thread-safe)
    std::atomic<uint64_t> deliveryTotal_{0};
    std::atomic<uint64_t> deliveryL5ToOutput_{0};
    double deliveryL5ToOutputDeltaSum_ = 0.0;
    double deliveryL5ToOutputDeltaMin_ = std::numeric_limits<double>::infinity();
    double deliveryL5ToOutputDeltaMax_ = -std::numeric_limits<double>::infinity();
    mutable std::mutex deliveryStatsMutex_;

    std::atomic<uint64_t> scheduledInputToL4_{0};
    std::atomic<uint64_t> scheduledOther_{0};
};

} // namespace snnfw

#endif // SNNFW_NETWORK_PROPAGATOR_H
