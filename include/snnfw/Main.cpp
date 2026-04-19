#include "snnfw/Neuron.h"
#include "snnfw/Logger.h"
#include <spdlog/spdlog.h>

int main() {
    // Initialize the logger with INFO level
    snnfw::Logger::getInstance().initialize("snnfw.log", spdlog::level::info);

    SNNFW_INFO("=== SNNFW Application Started ===");

    snnfw::Neuron neuron(50.0, 0.95); // 50ms window, 0.95 similarity threshold

    neuron.insertSpike(10.0);
    neuron.insertSpike(20.0);
    neuron.insertSpike(30.0);

    neuron.learnCurrentPattern();  // Store pattern

    neuron.insertSpike(80.0); // Old spikes aged out
    neuron.insertSpike(90.0);
    neuron.insertSpike(100.0);

    neuron.learnCurrentPattern(); // Add or blend depending on limit

    neuron.printSpikes();
    neuron.printReferencePatterns();

    SNNFW_INFO("=== SNNFW Application Finished ===");

    return 0;
}
