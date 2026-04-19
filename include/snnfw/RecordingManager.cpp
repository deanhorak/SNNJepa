#include "snnfw/RecordingManager.h"
#include "snnfw/ActivityVisualizer.h"
#include <fstream>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <set>
#include <iostream>

namespace snnfw {

RecordingManager::RecordingManager(ActivityVisualizer& visualizer)
    : visualizer_(&visualizer)
    , recording_(false)
    , recordingStartTime_(0)
    , streamingMode_(false)
    , wasStreamedToFile_(false)
    , streamingFile_(nullptr)
    , streamedSpikeCount_(0)
    , playbackIndex_(0)
    , streamingPlayback_(false)
    , playbackFile_(nullptr)
    , playbackFileOffset_(0)
    , totalSpikesInFile_(0)
{
}

RecordingManager::RecordingManager()
    : visualizer_(nullptr)
    , recording_(false)
    , recordingStartTime_(0)
    , streamingMode_(false)
    , wasStreamedToFile_(false)
    , streamingFile_(nullptr)
    , streamedSpikeCount_(0)
    , playbackIndex_(0)
    , streamingPlayback_(false)
    , playbackFile_(nullptr)
    , playbackFileOffset_(0)
    , totalSpikesInFile_(0)
{
}

RecordingManager::~RecordingManager() {
    if (streamingFile_) {
        if (streamingFile_->is_open()) {
            updateFileHeader();
            streamingFile_->close();
        }
        delete streamingFile_;
    }
    if (playbackFile_) {
        if (playbackFile_->is_open()) {
            playbackFile_->close();
        }
        delete playbackFile_;
    }
}

void RecordingManager::startRecording(bool streamToFile, const std::string& filename) {
    if (recording_) return;

    recording_ = true;
    recordingStartTime_ = 0;  // Will be set on first spike
    streamingMode_ = streamToFile;
    wasStreamedToFile_ = streamToFile;  // Track for later
    streamedSpikeCount_ = 0;

    if (streamingMode_) {
        // Streaming mode - open file and write header
        streamingFilename_ = filename;
        streamingFile_ = new std::ofstream(filename, std::ios::binary);

        if (!streamingFile_->is_open()) {
            std::cerr << "ERROR: RecordingManager: Failed to open recording file for writing: " << filename << std::endl;
            std::cerr << "ERROR: RecordingManager: Check that the directory exists and is writable" << std::endl;
            recording_ = false;
            streamingMode_ = false;
            delete streamingFile_;
            streamingFile_ = nullptr;
            return;
        }

        std::cout << "RecordingManager: Opened recording file for streaming: " << filename << std::endl;
        writeFileHeader();
    } else {
        // Memory mode - clear spike vector
        spikes_.clear();
    }

    metadata_.name = "Recording";
    metadata_.startTime = 0;
    metadata_.endTime = 0;
    metadata_.duration = 0;
    metadata_.spikeCount = 0;
    metadata_.neuronCount = 0;
    metadata_.timestamp = getCurrentTimestamp();
}

void RecordingManager::stopRecording() {
    if (!recording_) return;

    recording_ = false;

    if (streamingMode_ && streamingFile_) {
        // Update header with final metadata and close file
        updateFileHeader();
        streamingFile_->close();
        delete streamingFile_;
        streamingFile_ = nullptr;
        streamingMode_ = false;
    } else {
        updateMetadata();
    }
}

void RecordingManager::recordSpike(const RecordedSpike& spike) {
    if (!recording_) return;

    // Set start time on first spike
    if (streamingMode_) {
        if (streamedSpikeCount_ == 0) {
            recordingStartTime_ = spike.timestamp;
            metadata_.startTime = spike.timestamp;
        }

        // Write spike directly to file
        streamingFile_->write(reinterpret_cast<const char*>(&spike.timestamp), sizeof(spike.timestamp));
        streamingFile_->write(reinterpret_cast<const char*>(&spike.sourceNeuronId), sizeof(spike.sourceNeuronId));
        streamingFile_->write(reinterpret_cast<const char*>(&spike.targetNeuronId), sizeof(spike.targetNeuronId));
        streamingFile_->write(reinterpret_cast<const char*>(&spike.synapseId), sizeof(spike.synapseId));

        streamedSpikeCount_++;
        metadata_.endTime = spike.timestamp;
        metadata_.spikeCount = streamedSpikeCount_;
    } else {
        // Memory mode
        if (spikes_.empty()) {
            recordingStartTime_ = spike.timestamp;
            metadata_.startTime = spike.timestamp;
        }

        spikes_.push_back(spike);
        metadata_.endTime = spike.timestamp;
    }
}

bool RecordingManager::saveRecording(const std::string& filename) {
    // If we were in streaming mode, the file is already written
    // Just return success (even if no spikes were recorded)
    if (wasStreamedToFile_) {
        // File was already written in streaming mode
        return true;
    }

    if (spikes_.empty()) {
        return false;
    }

    updateMetadata();
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write magic number
    const char magic[4] = {'S', 'N', 'N', 'R'};
    file.write(magic, 4);
    
    // Write version
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write metadata as JSON-like string
    std::stringstream metaStream;
    metaStream << "{"
               << "\"name\":\"" << metadata_.name << "\","
               << "\"startTime\":" << metadata_.startTime << ","
               << "\"endTime\":" << metadata_.endTime << ","
               << "\"duration\":" << metadata_.duration << ","
               << "\"spikeCount\":" << metadata_.spikeCount << ","
               << "\"neuronCount\":" << metadata_.neuronCount << ","
               << "\"timestamp\":\"" << metadata_.timestamp << "\""
               << "}";
    
    std::string metaStr = metaStream.str();
    uint32_t metaLength = static_cast<uint32_t>(metaStr.length());
    file.write(reinterpret_cast<const char*>(&metaLength), sizeof(metaLength));
    file.write(metaStr.c_str(), metaLength);

    // Write spike count
    uint64_t spikeCount = spikes_.size();
    file.write(reinterpret_cast<const char*>(&spikeCount), sizeof(spikeCount));

    // Write spikes
    for (const auto& spike : spikes_) {
        file.write(reinterpret_cast<const char*>(&spike.timestamp), sizeof(spike.timestamp));
        file.write(reinterpret_cast<const char*>(&spike.sourceNeuronId), sizeof(spike.sourceNeuronId));
        file.write(reinterpret_cast<const char*>(&spike.targetNeuronId), sizeof(spike.targetNeuronId));
        file.write(reinterpret_cast<const char*>(&spike.synapseId), sizeof(spike.synapseId));
    }
    
    file.close();
    return true;
}

bool RecordingManager::loadRecording(const std::string& filename, bool streamingPlayback) {
    streamingPlayback_ = streamingPlayback;
    playbackFilename_ = filename;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read and verify magic number
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "SNNR", 4) != 0) {
        return false;
    }

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        return false;  // Unsupported version
    }

    // Read metadata
    uint32_t metaLength;
    file.read(reinterpret_cast<char*>(&metaLength), sizeof(metaLength));

    std::vector<char> metaBuffer(metaLength + 1);
    file.read(metaBuffer.data(), metaLength);
    metaBuffer[metaLength] = '\0';

    // Parse metadata (simple parsing, not full JSON)
    std::string metaStr(metaBuffer.data());
    // For now, just extract the values we need
    // In production, use a proper JSON parser

    // Read spike count
    uint64_t spikeCount;
    file.read(reinterpret_cast<char*>(&spikeCount), sizeof(spikeCount));

    totalSpikesInFile_ = spikeCount;

    if (streamingPlayback_) {
        // Streaming mode: don't load spikes into memory, just remember file offset
        playbackFileOffset_ = file.tellg();  // Save position where spike data starts
        file.close();

        // Clear any existing spikes from memory
        spikes_.clear();

        // Open file for streaming playback
        if (playbackFile_) {
            if (playbackFile_->is_open()) {
                playbackFile_->close();
            }
            delete playbackFile_;
        }
        playbackFile_ = new std::ifstream(filename, std::ios::binary);
        if (!playbackFile_->is_open()) {
            return false;
        }
        // Seek to spike data start
        playbackFile_->seekg(playbackFileOffset_);
    } else {
        // Legacy mode: load all spikes into memory
        spikes_.clear();
        spikes_.reserve(spikeCount);

        for (uint64_t i = 0; i < spikeCount; ++i) {
            RecordedSpike spike;
            file.read(reinterpret_cast<char*>(&spike.timestamp), sizeof(spike.timestamp));
            file.read(reinterpret_cast<char*>(&spike.sourceNeuronId), sizeof(spike.sourceNeuronId));
            file.read(reinterpret_cast<char*>(&spike.targetNeuronId), sizeof(spike.targetNeuronId));
            file.read(reinterpret_cast<char*>(&spike.synapseId), sizeof(spike.synapseId));
            spikes_.push_back(spike);
        }

        file.close();
    }
    
    // Update metadata
    if (streamingPlayback_) {
        // For streaming mode, we need to scan the file to get metadata
        // For now, set basic metadata from file
        metadata_.spikeCount = totalSpikesInFile_;

        // Read first and last spike to get time range
        if (totalSpikesInFile_ > 0 && playbackFile_ && playbackFile_->is_open()) {
            // Read first spike timestamp
            std::streampos savedPos = playbackFile_->tellg();
            playbackFile_->seekg(playbackFileOffset_);
            uint64_t firstTimestamp;
            playbackFile_->read(reinterpret_cast<char*>(&firstTimestamp), sizeof(firstTimestamp));
            metadata_.startTime = firstTimestamp;

            // Read last spike timestamp
            const size_t spikeSize = sizeof(uint64_t) * 4;  // timestamp + 3 IDs
            playbackFile_->seekg(playbackFileOffset_ + (totalSpikesInFile_ - 1) * spikeSize);
            uint64_t lastTimestamp;
            playbackFile_->read(reinterpret_cast<char*>(&lastTimestamp), sizeof(lastTimestamp));
            metadata_.endTime = lastTimestamp;
            metadata_.duration = metadata_.endTime - metadata_.startTime;

            // Restore file position
            playbackFile_->seekg(savedPos);
        }
    } else {
        updateMetadata();
    }

    // Reset playback state
    playbackState_.startTime = metadata_.startTime;
    playbackState_.endTime = metadata_.endTime;
    playbackState_.currentTime = metadata_.startTime;
    playbackIndex_ = 0;

    return true;
}

void RecordingManager::play() {
    // Allow playback in streaming mode even if spikes_ is empty
    if (!streamingPlayback_ && spikes_.empty()) return;
    if (streamingPlayback_ && (!playbackFile_ || !playbackFile_->is_open())) return;

    playbackState_.playing = true;
    playbackState_.paused = false;
}

void RecordingManager::pause() {
    playbackState_.paused = true;
}

void RecordingManager::stop() {
    playbackState_.playing = false;
    playbackState_.paused = false;
    playbackState_.currentTime = playbackState_.startTime;
    playbackIndex_ = 0;
}

void RecordingManager::setSpeed(float speed) {
    playbackState_.speed = std::max(0.1f, std::min(10.0f, speed));
}

void RecordingManager::setLooping(bool loop) {
    playbackState_.looping = loop;
}

void RecordingManager::seek(uint64_t time) {
    playbackState_.currentTime = std::max(playbackState_.startTime,
                                          std::min(playbackState_.endTime, time));

    // Find corresponding index in recording
    playbackIndex_ = 0;
    for (size_t i = 0; i < spikes_.size(); ++i) {
        if (spikes_[i].timestamp >= playbackState_.currentTime) {
            playbackIndex_ = i;
            break;
        }
    }
}

void RecordingManager::update(uint64_t deltaTime) {
    if (!playbackState_.playing || playbackState_.paused) {
        return;
    }

    // Check if we have data to play
    if (!streamingPlayback_ && spikes_.empty()) {
        return;
    }
    if (streamingPlayback_ && (!playbackFile_ || !playbackFile_->is_open())) {
        return;
    }

    // Advance playback time
    uint64_t scaledDelta = static_cast<uint64_t>(deltaTime * playbackState_.speed);
    playbackState_.currentTime += scaledDelta;

    if (streamingPlayback_) {
        // Streaming mode: read spikes from file as needed
        RecordedSpike spike;
        const size_t spikeSize = sizeof(spike.timestamp) + sizeof(spike.sourceNeuronId) +
                                sizeof(spike.targetNeuronId) + sizeof(spike.synapseId);

        while (playbackIndex_ < totalSpikesInFile_ && playbackFile_->good()) {
            // Peek at the next spike's timestamp
            uint64_t nextTimestamp;
            std::streampos currentPos = playbackFile_->tellg();
            playbackFile_->read(reinterpret_cast<char*>(&nextTimestamp), sizeof(nextTimestamp));

            if (!playbackFile_->good() || nextTimestamp > playbackState_.currentTime) {
                // Rewind to before we read the timestamp
                playbackFile_->seekg(currentPos);
                break;
            }

            // Read the rest of the spike
            spike.timestamp = nextTimestamp;
            playbackFile_->read(reinterpret_cast<char*>(&spike.sourceNeuronId), sizeof(spike.sourceNeuronId));
            playbackFile_->read(reinterpret_cast<char*>(&spike.targetNeuronId), sizeof(spike.targetNeuronId));
            playbackFile_->read(reinterpret_cast<char*>(&spike.synapseId), sizeof(spike.synapseId));

            if (!playbackFile_->good()) {
                break;
            }

            // Call visualizer if available
            if (visualizer_) {
                visualizer_->recordSpike(spike.sourceNeuronId, spike.targetNeuronId,
                                       spike.synapseId, spike.timestamp);
            }

            // Call playback callback if set
            if (playbackCallback_) {
                playbackCallback_(spike.sourceNeuronId, spike.targetNeuronId,
                                spike.synapseId, spike.timestamp);
            }

            playbackIndex_++;
        }
    } else {
        // Legacy mode: play spikes from memory
        while (playbackIndex_ < spikes_.size() &&
               spikes_[playbackIndex_].timestamp <= playbackState_.currentTime) {

            const auto& spike = spikes_[playbackIndex_];

            // Call visualizer if available
            if (visualizer_) {
                visualizer_->recordSpike(spike.sourceNeuronId, spike.targetNeuronId,
                                       spike.synapseId, spike.timestamp);
            }

            // Call playback callback if set
            if (playbackCallback_) {
                playbackCallback_(spike.sourceNeuronId, spike.targetNeuronId,
                                spike.synapseId, spike.timestamp);
            }

            playbackIndex_++;
        }
    }

    // Check if we've reached the end
    if (playbackState_.currentTime >= playbackState_.endTime) {
        if (playbackState_.looping) {
            // Loop back to start
            playbackState_.currentTime = playbackState_.startTime;
            playbackIndex_ = 0;
            if (streamingPlayback_ && playbackFile_) {
                playbackFile_->clear();  // Clear EOF flag
                playbackFile_->seekg(playbackFileOffset_);  // Seek back to start of spike data
            }
        } else {
            // Stop playback
            stop();
        }
    }
}

void RecordingManager::clearRecording() {
    spikes_.clear();
    playbackIndex_ = 0;
    playbackState_ = PlaybackState();
    metadata_ = RecordingMetadata();
}

void RecordingManager::setPlaybackCallback(PlaybackSpikeCallback callback) {
    playbackCallback_ = callback;
}

void RecordingManager::updateMetadata() {
    metadata_.spikeCount = spikes_.size();

    if (!spikes_.empty()) {
        metadata_.startTime = spikes_.front().timestamp;
        metadata_.endTime = spikes_.back().timestamp;
        metadata_.duration = metadata_.endTime - metadata_.startTime;

        // Count unique neurons
        std::set<uint64_t> uniqueNeurons;
        for (const auto& spike : spikes_) {
            uniqueNeurons.insert(spike.sourceNeuronId);
            uniqueNeurons.insert(spike.targetNeuronId);
        }
        metadata_.neuronCount = uniqueNeurons.size();
    }
}

std::string RecordingManager::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

void RecordingManager::writeFileHeader() {
    if (!streamingFile_ || !streamingFile_->is_open()) return;

    // Write magic number
    const char magic[4] = {'S', 'N', 'N', 'R'};
    streamingFile_->write(magic, 4);

    // Write version
    uint32_t version = 1;
    streamingFile_->write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write fixed-size metadata (512 bytes) with padding to allow in-place updates
    std::stringstream metaStream;
    metaStream << "{"
               << "\"name\":\"" << metadata_.name << "\","
               << "\"startTime\":" << metadata_.startTime << ","
               << "\"endTime\":" << metadata_.endTime << ","
               << "\"duration\":" << metadata_.duration << ","
               << "\"spikeCount\":" << metadata_.spikeCount << ","
               << "\"neuronCount\":" << metadata_.neuronCount << ","
               << "\"timestamp\":\"" << metadata_.timestamp << "\""
               << "}";

    std::string metaStr = metaStream.str();

    // Pad to fixed size (512 bytes)
    const uint32_t FIXED_META_SIZE = 512;
    if (metaStr.length() > FIXED_META_SIZE - 1) {
        metaStr = metaStr.substr(0, FIXED_META_SIZE - 1);
    }
    metaStr.resize(FIXED_META_SIZE, ' ');  // Pad with spaces

    uint32_t metaLength = FIXED_META_SIZE;
    streamingFile_->write(reinterpret_cast<const char*>(&metaLength), sizeof(metaLength));
    streamingFile_->write(metaStr.c_str(), metaLength);

    // Write placeholder spike count (will be updated when recording stops)
    uint64_t spikeCount = 0;
    streamingFile_->write(reinterpret_cast<const char*>(&spikeCount), sizeof(spikeCount));

    // Flush to ensure header is written
    streamingFile_->flush();
}

void RecordingManager::updateFileHeader() {
    if (!streamingFile_ || !streamingFile_->is_open()) return;

    // Update metadata
    metadata_.duration = metadata_.endTime - metadata_.startTime;
    metadata_.spikeCount = streamedSpikeCount_;

    // Seek to metadata content position (after magic, version, and metadata length)
    streamingFile_->seekp(4 + sizeof(uint32_t) + sizeof(uint32_t), std::ios::beg);

    // Write updated metadata (must fit in fixed size)
    std::stringstream metaStream;
    metaStream << "{"
               << "\"name\":\"" << metadata_.name << "\","
               << "\"startTime\":" << metadata_.startTime << ","
               << "\"endTime\":" << metadata_.endTime << ","
               << "\"duration\":" << metadata_.duration << ","
               << "\"spikeCount\":" << metadata_.spikeCount << ","
               << "\"neuronCount\":" << metadata_.neuronCount << ","
               << "\"timestamp\":\"" << metadata_.timestamp << "\""
               << "}";

    std::string metaStr = metaStream.str();

    // Pad to fixed size (512 bytes) - must match writeFileHeader
    const uint32_t FIXED_META_SIZE = 512;
    if (metaStr.length() > FIXED_META_SIZE - 1) {
        metaStr = metaStr.substr(0, FIXED_META_SIZE - 1);
    }
    metaStr.resize(FIXED_META_SIZE, ' ');  // Pad with spaces

    streamingFile_->write(metaStr.c_str(), FIXED_META_SIZE);

    // Seek to spike count position and update it
    // Position is: magic(4) + version(4) + metaLength(4) + metadata(512)
    streamingFile_->seekp(4 + sizeof(uint32_t) + sizeof(uint32_t) + FIXED_META_SIZE, std::ios::beg);
    streamingFile_->write(reinterpret_cast<const char*>(&streamedSpikeCount_), sizeof(streamedSpikeCount_));

    // Flush to ensure updates are written
    streamingFile_->flush();
}

} // namespace snnfw

