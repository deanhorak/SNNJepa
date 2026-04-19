#include "snnfw/PlaybackControls.h"
#include "snnfw/RecordingManager.h"
#include <imgui.h>
#include <cstdio>

namespace snnfw {

PlaybackControls::PlaybackControls(RecordingManager* recordingManager)
    : recordingManager_(recordingManager)
    , visible_(true)
    , posX_(10.0f)
    , posY_(10.0f)
    , width_(400.0f)
    , height_(250.0f)
{
}

void PlaybackControls::render() {
    if (!visible_ || !recordingManager_) {
        return;
    }
    
    ImGui::SetNextWindowPos(ImVec2(posX_, posY_), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(width_, height_), ImGuiCond_FirstUseEver);
    
    if (ImGui::Begin("Playback Controls", &visible_)) {
        renderPlaybackButtons();
        ImGui::Separator();
        renderSpeedControl();
        ImGui::Separator();
        renderTimeline();
        ImGui::Separator();
        renderSingleStepButtons();
        ImGui::Separator();
        renderMetadata();
    }
    ImGui::End();
}

void PlaybackControls::setPosition(float x, float y) {
    posX_ = x;
    posY_ = y;
}

void PlaybackControls::setSize(float width, float height) {
    width_ = width;
    height_ = height;
}

void PlaybackControls::setVisible(bool show) {
    visible_ = show;
}

void PlaybackControls::renderPlaybackButtons() {
    const auto& state = recordingManager_->getPlaybackState();
    
    // Play/Pause button
    if (state.playing && !state.paused) {
        if (ImGui::Button("Pause", ImVec2(80, 0))) {
            recordingManager_->pause();
        }
    } else {
        if (ImGui::Button("Play", ImVec2(80, 0))) {
            recordingManager_->play();
        }
    }
    
    ImGui::SameLine();
    
    // Stop button
    if (ImGui::Button("Stop", ImVec2(80, 0))) {
        recordingManager_->stop();
    }
    
    ImGui::SameLine();
    
    // Loop checkbox
    bool looping = state.looping;
    if (ImGui::Checkbox("Loop", &looping)) {
        recordingManager_->setLooping(looping);
    }
}

void PlaybackControls::renderSpeedControl() {
    const auto& state = recordingManager_->getPlaybackState();
    
    float speed = state.speed;
    ImGui::Text("Playback Speed:");
    ImGui::SameLine();
    
    // Preset speed buttons
    if (ImGui::Button("0.1x")) recordingManager_->setSpeed(0.1f);
    ImGui::SameLine();
    if (ImGui::Button("0.5x")) recordingManager_->setSpeed(0.5f);
    ImGui::SameLine();
    if (ImGui::Button("1x")) recordingManager_->setSpeed(1.0f);
    ImGui::SameLine();
    if (ImGui::Button("2x")) recordingManager_->setSpeed(2.0f);
    ImGui::SameLine();
    if (ImGui::Button("5x")) recordingManager_->setSpeed(5.0f);
    ImGui::SameLine();
    if (ImGui::Button("10x")) recordingManager_->setSpeed(10.0f);
    
    // Speed slider
    if (ImGui::SliderFloat("##speed", &speed, 0.1f, 10.0f, "%.1fx")) {
        recordingManager_->setSpeed(speed);
    }
}

void PlaybackControls::renderTimeline() {
    const auto& state = recordingManager_->getPlaybackState();
    
    // Timeline scrubber
    ImGui::Text("Timeline:");
    
    uint64_t currentTime = state.currentTime;
    uint64_t startTime = state.startTime;
    uint64_t endTime = state.endTime;
    
    if (ImGui::SliderScalar("##timeline", ImGuiDataType_U64, &currentTime, 
                            &startTime, &endTime, "")) {
        recordingManager_->seek(currentTime);
    }
    
    // Time display
    char currentBuf[32], totalBuf[32];
    formatTime(currentTime - startTime, currentBuf, sizeof(currentBuf));
    formatTime(endTime - startTime, totalBuf, sizeof(totalBuf));
    
    ImGui::Text("Time: %s / %s", currentBuf, totalBuf);
}

void PlaybackControls::renderSingleStepButtons() {
    const auto& state = recordingManager_->getPlaybackState();
    auto seekClamped = [&](int64_t deltaMs) {
        if (state.endTime <= state.startTime) return;
        int64_t target = static_cast<int64_t>(state.currentTime) + deltaMs;
        if (target < static_cast<int64_t>(state.startTime)) target = static_cast<int64_t>(state.startTime);
        if (target > static_cast<int64_t>(state.endTime)) target = static_cast<int64_t>(state.endTime);
        recordingManager_->seek(static_cast<uint64_t>(target));
    };

    ImGui::Text("Seek:");
    if (ImGui::Button("|< Start")) {
        recordingManager_->seek(state.startTime);
    }
    ImGui::SameLine();
    if (ImGui::Button("<< 10s")) {
        seekClamped(-10000);
    }
    ImGui::SameLine();
    if (ImGui::Button("< 1s")) {
        seekClamped(-1000);
    }
    ImGui::SameLine();
    if (ImGui::Button("1s >")) {
        seekClamped(1000);
    }
    ImGui::SameLine();
    if (ImGui::Button("10s >>")) {
        seekClamped(10000);
    }
    ImGui::SameLine();
    if (ImGui::Button("End >|")) {
        recordingManager_->seek(state.endTime);
    }
}

void PlaybackControls::renderMetadata() {
    const auto& metadata = recordingManager_->getMetadata();
    
    ImGui::Text("Recording Info:");
    ImGui::Indent();
    
    if (!metadata.name.empty()) {
        ImGui::Text("Name: %s", metadata.name.c_str());
    }
    
    char durationBuf[32];
    formatTime(metadata.duration, durationBuf, sizeof(durationBuf));
    ImGui::Text("Duration: %s", durationBuf);
    
    ImGui::Text("Spikes: %zu", metadata.spikeCount);
    ImGui::Text("Neurons: %zu", metadata.neuronCount);
    
    if (!metadata.timestamp.empty()) {
        ImGui::Text("Recorded: %s", metadata.timestamp.c_str());
    }
    
    ImGui::Unindent();
}

const char* PlaybackControls::formatTime(uint64_t milliseconds, char* buffer, size_t bufferSize) {
    uint64_t ms = milliseconds % 1000;
    uint64_t seconds = (milliseconds / 1000) % 60;
    uint64_t minutes = (milliseconds / 60000) % 60;
    uint64_t hours = milliseconds / 3600000;
    
    if (hours > 0) {
        snprintf(buffer, bufferSize, "%02llu:%02llu:%02llu.%03llu", hours, minutes, seconds, ms);
    } else {
        snprintf(buffer, bufferSize, "%02llu:%02llu.%03llu", minutes, seconds, ms);
    }
    
    return buffer;
}

} // namespace snnfw
