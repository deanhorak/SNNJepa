#ifndef SNNFW_MOTOR_ADAPTER_H
#define SNNFW_MOTOR_ADAPTER_H

#include "snnfw/adapters/BaseAdapter.h"
#include "snnfw/Neuron.h"
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

namespace snnfw {
namespace adapters {

/**
 * @brief Base class for motor (output) adapters
 *
 * Motor adapters convert spike trains from the neural network into external actions.
 * They implement the action step of the sense-process-act cycle.
 *
 * Key Responsibilities:
 * - Monitor a population of motor neurons
 * - Decode spike patterns into motor commands
 * - Execute actions in the external environment
 * - Provide feedback for closed-loop control
 * - Handle temporal integration and smoothing
 *
 * Decoding Strategies:
 * - Rate Decoding: Spike rate → command intensity
 * - Population Vector: Population activity → direction/magnitude
 * - Temporal Pattern: Spike timing → precise control signals
 * - Winner-Take-All: Most active neuron → discrete action selection
 *
 * Example Implementations:
 * - ServoAdapter: Spike patterns → servo motor angles
 * - DisplayAdapter: Spike patterns → visual display
 * - RobotArmAdapter: Spike patterns → joint commands
 * - SpeechAdapter: Spike patterns → speech synthesis
 */
class MotorAdapter : public BaseAdapter {
public:
    /**
     * @brief Motor command structure
     */
    struct MotorCommand {
        std::vector<double> values;        ///< Command values
        std::vector<std::string> channels; ///< Command channels/actuators
        double timestamp;                   ///< Command timestamp (ms)
        std::map<std::string, double> metadata; ///< Additional metadata
    };

    /**
     * @brief Spike activity from motor neurons
     */
    struct SpikeActivity {
        std::vector<std::vector<double>> spikeTimes; ///< Spike times per neuron
        std::vector<double> firingRates;             ///< Firing rates per neuron
        double windowStart;                           ///< Activity window start (ms)
        double windowEnd;                             ///< Activity window end (ms)
    };

    /**
     * @brief Action callback function type
     * @param command Motor command to execute
     * @return true if action was successful
     */
    using ActionCallback = std::function<bool(const MotorCommand&)>;

    /**
     * @brief Constructor
     * @param config Adapter configuration
     */
    explicit MotorAdapter(const Config& config) 
        : BaseAdapter(config) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~MotorAdapter() = default;

    /**
     * @brief Decode spike activity into motor commands
     * @param activity Spike activity from motor neurons
     * @return Motor command decoded from the activity
     */
    virtual MotorCommand decodeActivity(const SpikeActivity& activity) = 0;

    /**
     * @brief Execute a motor command
     * @param command Motor command to execute
     * @return true if successful, false otherwise
     */
    virtual bool executeCommand(const MotorCommand& command) = 0;

    /**
     * @brief Process motor neuron activity and execute actions
     * @param neurons Motor neuron population
     * @param currentTime Current simulation time (ms)
     * @return true if action was executed, false otherwise
     */
    virtual bool processNeurons(const std::vector<std::shared_ptr<Neuron>>& neurons,
                                double currentTime) = 0;

    /**
     * @brief Register a callback for action execution
     * @param callback Function to call when executing actions
     */
    virtual void registerActionCallback(ActionCallback callback) {
        actionCallback_ = callback;
    }

    /**
     * @brief Get the number of motor channels/actuators
     */
    virtual size_t getChannelCount() const = 0;

    /**
     * @brief Get current motor command state
     */
    virtual MotorCommand getCurrentCommand() const = 0;

    /**
     * @brief Set motor command directly (for testing/debugging)
     * @param command Motor command to set
     */
    virtual void setCommand(const MotorCommand& command) = 0;

    /**
     * @brief Reset adapter state
     */
    void reset() override {
        // Reset to neutral/default command
        MotorCommand neutral;
        neutral.timestamp = 0.0;
        setCommand(neutral);
    }

    /**
     * @brief Get adapter statistics
     */
    std::map<std::string, double> getStatistics() const override {
        std::map<std::string, double> stats;
        stats["channel_count"] = static_cast<double>(getChannelCount());
        auto currentCmd = getCurrentCommand();
        stats["last_command_time"] = currentCmd.timestamp;
        return stats;
    }

protected:
    ActionCallback actionCallback_; ///< Callback for action execution

    /**
     * @brief Helper: Decode spike rate to command value
     * @param spikeRate Spike rate (Hz)
     * @param minRate Minimum rate for mapping
     * @param maxRate Maximum rate for mapping
     * @param minValue Minimum output value
     * @param maxValue Maximum output value
     * @return Decoded command value
     */
    double rateToValue(double spikeRate, double minRate, double maxRate,
                       double minValue, double maxValue) const {
        if (spikeRate <= minRate) return minValue;
        if (spikeRate >= maxRate) return maxValue;
        
        double normalized = (spikeRate - minRate) / (maxRate - minRate);
        return minValue + normalized * (maxValue - minValue);
    }

    /**
     * @brief Helper: Calculate firing rate from spike times
     * @param spikeTimes Vector of spike times (ms)
     * @param windowStart Window start time (ms)
     * @param windowEnd Window end time (ms)
     * @return Firing rate (Hz)
     */
    double calculateFiringRate(const std::vector<double>& spikeTimes,
                               double windowStart, double windowEnd) const {
        if (windowEnd <= windowStart) return 0.0;
        
        size_t spikeCount = 0;
        for (double t : spikeTimes) {
            if (t >= windowStart && t < windowEnd) {
                spikeCount++;
            }
        }
        
        double windowDuration = (windowEnd - windowStart) / 1000.0; // Convert to seconds
        return static_cast<double>(spikeCount) / windowDuration;
    }

    /**
     * @brief Helper: Population vector decoding
     * @param firingRates Firing rates of neurons
     * @param preferredDirections Preferred direction of each neuron (radians)
     * @return Decoded direction (radians)
     */
    double populationVectorDecoding(const std::vector<double>& firingRates,
                                    const std::vector<double>& preferredDirections) const {
        if (firingRates.size() != preferredDirections.size()) {
            return 0.0;
        }
        
        double sumX = 0.0;
        double sumY = 0.0;
        
        for (size_t i = 0; i < firingRates.size(); ++i) {
            sumX += firingRates[i] * std::cos(preferredDirections[i]);
            sumY += firingRates[i] * std::sin(preferredDirections[i]);
        }
        
        return std::atan2(sumY, sumX);
    }
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_MOTOR_ADAPTER_H

