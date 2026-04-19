#include "snnfw/adapters/DisplayAdapter.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace snnfw {
namespace adapters {

DisplayAdapter::DisplayAdapter(const BaseAdapter::Config& config)
    : MotorAdapter(config)
    , displayWidth_(config.getIntParam("display_width", 80))
    , displayHeight_(config.getIntParam("display_height", 24))
    , updateInterval_(config.getDoubleParam("update_interval", 100.0))
    , displayMode_(DisplayMode::ASCII)
    , lastUpdateTime_(0.0)
    , currentCommand_()
    , displayBuffer_()
{
    std::string mode = config.getStringParam("display_mode", "ASCII");
    if (mode == "RASTER") displayMode_ = DisplayMode::RASTER;
    else if (mode == "HEATMAP") displayMode_ = DisplayMode::HEATMAP;
    else if (mode == "VECTOR") displayMode_ = DisplayMode::VECTOR;
}

bool DisplayAdapter::initialize() {
    displayBuffer_.clear();
    lastUpdateTime_ = 0.0;
    return true;
}

MotorAdapter::MotorCommand DisplayAdapter::decodeActivity(const SpikeActivity& activity) {
    MotorCommand command;
    command.timestamp = activity.windowEnd;
    
    // Generate display based on mode
    switch (displayMode_) {
        case DisplayMode::RASTER:
            displayBuffer_ = generateRasterPlot(activity);
            break;
        case DisplayMode::HEATMAP:
            displayBuffer_ = generateHeatmap(activity);
            break;
        case DisplayMode::VECTOR:
            displayBuffer_ = generateVectorDisplay(activity);
            break;
        case DisplayMode::ASCII:
        default:
            displayBuffer_ = generateASCII(activity);
            break;
    }
    
    // Store firing rates as command values
    command.values = activity.firingRates;
    for (size_t i = 0; i < activity.firingRates.size(); ++i) {
        command.channels.push_back("neuron_" + std::to_string(i));
    }
    
    return command;
}

bool DisplayAdapter::executeCommand(const MotorCommand& command) {
    if (command.values.empty()) {
        return false;
    }
    
    lastUpdateTime_ = command.timestamp;
    currentCommand_ = command;
    return true;
}

bool DisplayAdapter::processNeurons(const std::vector<std::shared_ptr<Neuron>>& neurons, double currentTime) {
    if (currentTime - lastUpdateTime_ < updateInterval_) {
        return false;
    }
    
    auto activity = extractActivity(neurons, currentTime);
    auto command = decodeActivity(activity);
    return executeCommand(command);
}



std::string DisplayAdapter::generateRasterPlot(const SpikeActivity& activity) {
    std::ostringstream oss;
    oss << "+" << std::string(displayWidth_, '-') << "+\n";
    int neuronsToShow = std::min(displayHeight_, static_cast<int>(activity.firingRates.size()));
    for (int i = 0; i < neuronsToShow; ++i) {
        oss << "|";
        double rate = activity.firingRates[i];
        int barLength = static_cast<int>(rate * displayWidth_);
        for (int j = 0; j < displayWidth_; ++j) {
            oss << (j < barLength ? "█" : " ");
        }
        oss << "|\n";
    }
    oss << "+" << std::string(displayWidth_, '-') << "+\n";
    return oss.str();
}

std::string DisplayAdapter::generateHeatmap(const SpikeActivity& activity) {
    std::ostringstream oss;
    double maxRate = 0.0;
    for (double rate : activity.firingRates) maxRate = std::max(maxRate, rate);
    oss << "+" << std::string(displayWidth_, '-') << "+\n";
    int neuronsToShow = std::min(displayHeight_, static_cast<int>(activity.firingRates.size()));
    for (int i = 0; i < neuronsToShow; ++i) {
        oss << "|";
        double normalizedRate = (maxRate > 0.0) ? (activity.firingRates[i] / maxRate) : 0.0;
        char heatChar = ' ';
        if (normalizedRate > 0.9) heatChar = '@';
        else if (normalizedRate > 0.7) heatChar = '#';
        else if (normalizedRate > 0.5) heatChar = '%';
        else if (normalizedRate > 0.3) heatChar = '*';
        else if (normalizedRate > 0.1) heatChar = '.';
        for (int j = 0; j < displayWidth_; ++j) oss << heatChar;
        oss << "|\n";
    }
    oss << "+" << std::string(displayWidth_, '-') << "+\n";
    return oss.str();
}

std::string DisplayAdapter::generateVectorDisplay(const SpikeActivity& activity) {
    std::ostringstream oss;
    std::vector<double> angles(activity.firingRates.size());
    for (size_t i = 0; i < angles.size(); ++i) {
        angles[i] = (2.0 * M_PI * i) / angles.size();
    }
    double decodedAngle = populationVectorDecoding(activity.firingRates, angles);

    // Calculate magnitude from firing rates
    double magnitude = 0.0;
    for (double rate : activity.firingRates) {
        magnitude += rate;
    }
    magnitude /= activity.firingRates.size();

    // Convert angle to degrees
    double angleDegrees = decodedAngle * 180.0 / M_PI;

    // Calculate X and Y components
    double x = magnitude * std::cos(decodedAngle);
    double y = magnitude * std::sin(decodedAngle);

    oss << "Population Vector:\n";
    oss << "  X: " << std::fixed << std::setprecision(3) << x << "\n";
    oss << "  Y: " << std::fixed << std::setprecision(3) << y << "\n";
    oss << "  Magnitude: " << magnitude << "\n";
    oss << "  Angle: " << angleDegrees << "°\n";
    return oss.str();
}

std::string DisplayAdapter::generateASCII(const SpikeActivity& activity) {
    const char* chars = " .:-=+*#%@";
    const int numChars = 10;
    double maxRate = 0.0;
    for (double rate : activity.firingRates) maxRate = std::max(maxRate, rate);
    std::ostringstream oss;
    oss << "+" << std::string(displayWidth_, '-') << "+\n";
    int neuronsToShow = std::min(displayHeight_, static_cast<int>(activity.firingRates.size()));
    for (int i = 0; i < neuronsToShow; ++i) {
        oss << "|";
        double normalizedRate = (maxRate > 0.0) ? (activity.firingRates[i] / maxRate) : 0.0;
        int charIdx = static_cast<int>(std::clamp(normalizedRate, 0.0, 1.0) * (numChars - 1));
        for (int j = 0; j < displayWidth_; ++j) oss << chars[charIdx];
        oss << "|\n";
    }
    oss << "+" << std::string(displayWidth_, '-') << "+\n";
    return oss.str();
}

DisplayAdapter::SpikeActivity DisplayAdapter::extractActivity(
    const std::vector<std::shared_ptr<Neuron>>& neurons,
    double currentTime) {
    SpikeActivity activity;
    activity.windowStart = currentTime - updateInterval_;
    activity.windowEnd = currentTime;
    activity.firingRates.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        // Extract spike times from neuron
        std::vector<double> spikeTimes = neurons[i]->getSpikes();
        activity.firingRates[i] = calculateFiringRate(spikeTimes, activity.windowStart, activity.windowEnd);
    }
    return activity;
}

} // namespace adapters
} // namespace snnfw
