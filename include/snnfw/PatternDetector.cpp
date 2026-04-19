#include "snnfw/PatternDetector.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace snnfw {

PatternDetector::PatternDetector()
    : lastDetectionTime_(0) {
}

PatternDetector::~PatternDetector() {
}

void PatternDetector::setConfig(const PatternDetectorConfig& config) {
    config_ = config;
}

void PatternDetector::recordSpike(uint64_t neuronId, uint64_t timestamp, const glm::vec3& position) {
    PatternSpikeEvent spike(neuronId, timestamp, position);
    spikeHistory_.push_back(spike);
    neuronSpikes_[neuronId].push_back(spike);
    neuronPositions_[neuronId] = position;
}

void PatternDetector::clearHistory() {
    spikeHistory_.clear();
    neuronSpikes_.clear();
    neuronPositions_.clear();
}

void PatternDetector::clearPatterns() {
    synchronyEvents_.clear();
    oscillations_.clear();
    waveEvents_.clear();
    burstEvents_.clear();
}

void PatternDetector::update(uint64_t currentTime) {
    // Cleanup old spikes
    cleanupOldSpikes(currentTime);
    
    // Run detection algorithms
    if (config_.detectSynchrony) {
        detectSynchrony(currentTime);
    }
    
    if (config_.detectOscillations) {
        detectOscillations(currentTime);
    }
    
    if (config_.detectWaves) {
        detectWaves(currentTime);
    }
    
    if (config_.detectBursts) {
        detectBursts(currentTime);
    }
    
    lastDetectionTime_ = currentTime;
}

void PatternDetector::cleanupOldSpikes(uint64_t currentTime) {
    // Remove spikes older than history duration
    uint64_t cutoffTime = currentTime > config_.historyDuration ? 
                         currentTime - config_.historyDuration : 0;
    
    // Remove from global history
    spikeHistory_.erase(
        std::remove_if(spikeHistory_.begin(), spikeHistory_.end(),
            [cutoffTime](const PatternSpikeEvent& spike) {
                return spike.timestamp < cutoffTime;
            }),
        spikeHistory_.end()
    );

    // Remove from per-neuron history
    for (auto& pair : neuronSpikes_) {
        auto& spikes = pair.second;
        spikes.erase(
            std::remove_if(spikes.begin(), spikes.end(),
                [cutoffTime](const PatternSpikeEvent& spike) {
                    return spike.timestamp < cutoffTime;
                }),
            spikes.end()
        );
    }
}

std::vector<uint64_t> PatternDetector::getSpikesInWindow(uint64_t startTime, uint64_t endTime) const {
    std::vector<uint64_t> timestamps;
    for (const auto& spike : spikeHistory_) {
        if (spike.timestamp >= startTime && spike.timestamp <= endTime) {
            timestamps.push_back(spike.timestamp);
        }
    }
    return timestamps;
}

std::vector<PatternSpikeEvent> PatternDetector::getNeuronSpikes(uint64_t neuronId,
                                                         uint64_t startTime,
                                                         uint64_t endTime) const {
    std::vector<PatternSpikeEvent> spikes;
    auto it = neuronSpikes_.find(neuronId);
    if (it != neuronSpikes_.end()) {
        for (const auto& spike : it->second) {
            if (spike.timestamp >= startTime && spike.timestamp <= endTime) {
                spikes.push_back(spike);
            }
        }
    }
    return spikes;
}

glm::vec3 PatternDetector::getNeuronPosition(uint64_t neuronId) const {
    auto it = neuronPositions_.find(neuronId);
    if (it != neuronPositions_.end()) {
        return it->second;
    }
    return glm::vec3(0.0f);
}

float PatternDetector::calculateCoherence(const std::vector<uint64_t>& timestamps, 
                                         float timeWindow) const {
    if (timestamps.size() < 2) return 0.0f;
    
    // Calculate variance of timestamps
    double mean = 0.0;
    for (uint64_t t : timestamps) {
        mean += static_cast<double>(t);
    }
    mean /= timestamps.size();
    
    double variance = 0.0;
    for (uint64_t t : timestamps) {
        double diff = static_cast<double>(t) - mean;
        variance += diff * diff;
    }
    variance /= timestamps.size();
    
    // Coherence is inversely related to variance
    // Normalize by time window
    float stddev = std::sqrt(static_cast<float>(variance));
    float coherence = 1.0f - std::min(1.0f, stddev / timeWindow);
    
    return coherence;
}

float PatternDetector::calculateFrequency(const std::vector<uint64_t>& timestamps) const {
    if (timestamps.size() < 2) return 0.0f;
    
    // Calculate inter-spike intervals
    std::vector<uint64_t> intervals;
    for (size_t i = 1; i < timestamps.size(); ++i) {
        intervals.push_back(timestamps[i] - timestamps[i-1]);
    }
    
    // Calculate mean interval
    double meanInterval = 0.0;
    for (uint64_t interval : intervals) {
        meanInterval += static_cast<double>(interval);
    }
    meanInterval /= intervals.size();
    
    // Convert to frequency (Hz)
    // meanInterval is in milliseconds
    if (meanInterval > 0.0) {
        return static_cast<float>(1000.0 / meanInterval);
    }
    
    return 0.0f;
}

bool PatternDetector::detectOscillationInSpikes(const std::vector<uint64_t>& timestamps,
                                               float& frequency, float& confidence) const {
    if (timestamps.size() < 3) return false;
    
    // Calculate frequency
    frequency = calculateFrequency(timestamps);
    
    // Check if frequency is in valid range
    if (frequency < config_.oscillationMinFrequency || 
        frequency > config_.oscillationMaxFrequency) {
        return false;
    }
    
    // Calculate regularity (confidence)
    // More regular = higher confidence
    std::vector<uint64_t> intervals;
    for (size_t i = 1; i < timestamps.size(); ++i) {
        intervals.push_back(timestamps[i] - timestamps[i-1]);
    }
    
    // Calculate coefficient of variation
    double mean = 0.0;
    for (uint64_t interval : intervals) {
        mean += static_cast<double>(interval);
    }
    mean /= intervals.size();
    
    double variance = 0.0;
    for (uint64_t interval : intervals) {
        double diff = static_cast<double>(interval) - mean;
        variance += diff * diff;
    }
    variance /= intervals.size();
    
    float cv = std::sqrt(static_cast<float>(variance)) / static_cast<float>(mean);
    
    // Confidence is inversely related to CV
    // CV < 0.5 is considered regular
    confidence = 1.0f - std::min(1.0f, cv / 0.5f);
    
    return confidence >= config_.oscillationConfidenceThreshold;
}

void PatternDetector::detectSynchrony(uint64_t currentTime) {
    // Look for groups of neurons firing within a small time window
    uint64_t windowStart = currentTime > static_cast<uint64_t>(config_.synchronyTimeWindow) ?
                          currentTime - static_cast<uint64_t>(config_.synchronyTimeWindow) : 0;
    
    // Group spikes by time bins
    std::unordered_map<uint64_t, std::vector<uint64_t>> timeBins;
    float binSize = config_.synchronyTimeWindow / 10.0f;  // 10 bins per window
    
    for (const auto& spike : spikeHistory_) {
        if (spike.timestamp >= windowStart && spike.timestamp <= currentTime) {
            uint64_t bin = static_cast<uint64_t>(spike.timestamp / binSize);
            timeBins[bin].push_back(spike.neuronId);
        }
    }
    
    // Find bins with enough neurons
    for (const auto& pair : timeBins) {
        const auto& neurons = pair.second;
        if (static_cast<int>(neurons.size()) >= config_.synchronyMinNeurons) {
            // Get timestamps for these neurons
            std::vector<uint64_t> timestamps;
            for (uint64_t neuronId : neurons) {
                auto spikes = getNeuronSpikes(neuronId, windowStart, currentTime);
                if (!spikes.empty()) {
                    timestamps.push_back(spikes.back().timestamp);
                }
            }
            
            // Calculate coherence
            float coherence = calculateCoherence(timestamps, config_.synchronyTimeWindow);
            
            if (coherence >= config_.synchronyThreshold) {
                // Create synchrony event
                SynchronyEvent event;
                event.timestamp = currentTime;
                event.neuronIds = neurons;
                event.coherence = coherence;
                event.timeWindow = config_.synchronyTimeWindow;
                
                synchronyEvents_.push_back(event);
            }
        }
    }
}

void PatternDetector::detectOscillations(uint64_t currentTime) {
    // Analyze each neuron's spike train for oscillations
    uint64_t windowStart = currentTime > static_cast<uint64_t>(config_.oscillationMinDuration) ?
                          currentTime - static_cast<uint64_t>(config_.oscillationMinDuration) : 0;
    
    for (const auto& pair : neuronSpikes_) {
        uint64_t neuronId = pair.first;
        const auto& spikes = pair.second;
        
        // Get timestamps in window
        std::vector<uint64_t> timestamps;
        for (const auto& spike : spikes) {
            if (spike.timestamp >= windowStart && spike.timestamp <= currentTime) {
                timestamps.push_back(spike.timestamp);
            }
        }
        
        // Detect oscillation
        float frequency, confidence;
        if (detectOscillationInSpikes(timestamps, frequency, confidence)) {
            // Create oscillation pattern
            OscillationPattern pattern;
            pattern.neuronIds.push_back(neuronId);
            pattern.frequency = frequency;
            pattern.amplitude = 1.0f;  // Could calculate from spike rate variation
            pattern.phase = 0.0f;      // Could calculate from timing
            pattern.confidence = confidence;
            pattern.startTime = windowStart;
            pattern.endTime = currentTime;
            
            oscillations_.push_back(pattern);
        }
    }
}

void PatternDetector::detectWaves(uint64_t currentTime) {
    // Look for spatially organized sequences of spikes
    uint64_t windowStart = currentTime > static_cast<uint64_t>(config_.waveTimeWindow) ?
                          currentTime - static_cast<uint64_t>(config_.waveTimeWindow) : 0;

    // Get all spikes in window, sorted by time
    std::vector<PatternSpikeEvent> windowSpikes;
    for (const auto& spike : spikeHistory_) {
        if (spike.timestamp >= windowStart && spike.timestamp <= currentTime) {
            windowSpikes.push_back(spike);
        }
    }

    if (windowSpikes.size() < static_cast<size_t>(config_.waveMinNeurons)) {
        return;
    }

    std::sort(windowSpikes.begin(), windowSpikes.end(),
        [](const PatternSpikeEvent& a, const PatternSpikeEvent& b) {
            return a.timestamp < b.timestamp;
        });
    
    // Try to fit a wave starting from each spike
    for (size_t i = 0; i < windowSpikes.size(); ++i) {
        const auto& origin = windowSpikes[i];
        
        std::vector<uint64_t> waveNeurons;
        std::vector<uint64_t> waveTimes;
        std::vector<float> distances;
        
        waveNeurons.push_back(origin.neuronId);
        waveTimes.push_back(origin.timestamp);
        distances.push_back(0.0f);
        
        // Find subsequent spikes that could be part of wave
        for (size_t j = i + 1; j < windowSpikes.size(); ++j) {
            const auto& spike = windowSpikes[j];
            
            float distance = glm::length(spike.position - origin.position);
            uint64_t timeDiff = spike.timestamp - origin.timestamp;
            
            if (timeDiff > 0) {
                float speed = distance / static_cast<float>(timeDiff);
                
                if (speed >= config_.waveMinSpeed && speed <= config_.waveMaxSpeed) {
                    waveNeurons.push_back(spike.neuronId);
                    waveTimes.push_back(spike.timestamp);
                    distances.push_back(distance);
                }
            }
        }
        
        // Check if we have enough neurons for a wave
        if (static_cast<int>(waveNeurons.size()) >= config_.waveMinNeurons) {
            // Calculate wave direction and speed
            glm::vec3 direction(0.0f);
            float totalSpeed = 0.0f;
            int count = 0;
            
            for (size_t j = 1; j < waveNeurons.size(); ++j) {
                glm::vec3 pos1 = getNeuronPosition(waveNeurons[j-1]);
                glm::vec3 pos2 = getNeuronPosition(waveNeurons[j]);
                glm::vec3 dir = pos2 - pos1;
                float dist = glm::length(dir);
                
                if (dist > 0.0f) {
                    direction += glm::normalize(dir);
                    uint64_t timeDiff = waveTimes[j] - waveTimes[j-1];
                    if (timeDiff > 0) {
                        totalSpeed += dist / static_cast<float>(timeDiff);
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                direction = glm::normalize(direction);
                float avgSpeed = totalSpeed / count;
                
                // Calculate coherence
                float coherence = calculateCoherence(waveTimes, config_.waveTimeWindow);
                
                // Create wave event
                WaveEvent event;
                event.timestamp = origin.timestamp;
                event.origin = origin.position;
                event.direction = direction;
                event.speed = avgSpeed;
                event.neuronIds = waveNeurons;
                event.firingTimes = waveTimes;
                event.coherence = coherence;
                
                waveEvents_.push_back(event);
            }
        }
    }
}

void PatternDetector::detectBursts(uint64_t currentTime) {
    // Detect bursts in each neuron's spike train
    uint64_t windowStart = currentTime > static_cast<uint64_t>(config_.burstTimeWindow) ?
                          currentTime - static_cast<uint64_t>(config_.burstTimeWindow) : 0;
    
    for (const auto& pair : neuronSpikes_) {
        uint64_t neuronId = pair.first;
        const auto& spikes = pair.second;

        // Get spikes in window
        std::vector<PatternSpikeEvent> windowSpikes;
        for (const auto& spike : spikes) {
            if (spike.timestamp >= windowStart && spike.timestamp <= currentTime) {
                windowSpikes.push_back(spike);
            }
        }
        
        if (windowSpikes.size() < static_cast<size_t>(config_.burstMinSpikes)) {
            continue;
        }
        
        // Check if spikes form a burst
        std::vector<uint64_t> timestamps;
        for (const auto& spike : windowSpikes) {
            timestamps.push_back(spike.timestamp);
        }
        
        float frequency = calculateFrequency(timestamps);
        
        if (frequency >= config_.burstMinFrequency) {
            // Create burst event
            BurstEvent event;
            event.neuronId = neuronId;
            event.startTime = windowSpikes.front().timestamp;
            event.endTime = windowSpikes.back().timestamp;
            event.spikeCount = static_cast<int>(windowSpikes.size());
            event.frequency = frequency;
            
            burstEvents_.push_back(event);
        }
    }
}

} // namespace snnfw

