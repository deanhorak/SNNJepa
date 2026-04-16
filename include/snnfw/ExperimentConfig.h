#ifndef SNNFW_EXPERIMENT_CONFIG_H
#define SNNFW_EXPERIMENT_CONFIG_H

#include <string>
#include <filesystem>
#include <stdexcept>

namespace snnfw {

/**
 * @class ExperimentConfig
 * @brief Configuration for experiment-specific datastore management
 * 
 * This class manages experiment-specific settings including datastore paths,
 * allowing different experiments to maintain separate persistent storage.
 * 
 * Example usage:
 * @code
 * ExperimentConfig config("my_experiment");
 * std::string dbPath = config.getDatastorePath();
 * Datastore datastore(dbPath, 1000000);
 * @endcode
 */
class ExperimentConfig {
public:
    /**
     * @brief Construct a new ExperimentConfig object
     * @param experimentName Name of the experiment (used for datastore path)
     * @param baseDir Base directory for all experiments (default: "./experiments")
     */
    explicit ExperimentConfig(const std::string& experimentName, 
                             const std::string& baseDir = "./experiments")
        : name(experimentName), baseDirectory(baseDir) {
        
        if (experimentName.empty()) {
            throw std::invalid_argument("Experiment name cannot be empty");
        }
        
        // Validate experiment name (no path separators)
        if (experimentName.find('/') != std::string::npos || 
            experimentName.find('\\') != std::string::npos) {
            throw std::invalid_argument("Experiment name cannot contain path separators");
        }
        
        // Construct the experiment directory path
        experimentDir = std::filesystem::path(baseDirectory) / experimentName;
        
        // Construct the datastore path
        datastorePath = experimentDir / "datastore";
    }
    
    /**
     * @brief Get the experiment name
     * @return Experiment name
     */
    const std::string& getName() const {
        return name;
    }
    
    /**
     * @brief Get the base directory for all experiments
     * @return Base directory path
     */
    const std::string& getBaseDirectory() const {
        return baseDirectory;
    }
    
    /**
     * @brief Get the experiment directory path
     * @return Experiment directory path
     */
    std::string getExperimentDirectory() const {
        return experimentDir.string();
    }
    
    /**
     * @brief Get the datastore path for this experiment
     * @return Datastore path
     */
    std::string getDatastorePath() const {
        return datastorePath.string();
    }
    
    /**
     * @brief Create the experiment directory structure if it doesn't exist
     * @return true if directories were created or already exist, false on error
     */
    bool createDirectories() const {
        try {
            std::filesystem::create_directories(experimentDir);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }
    
    /**
     * @brief Check if the experiment directory exists
     * @return true if directory exists, false otherwise
     */
    bool exists() const {
        return std::filesystem::exists(experimentDir);
    }
    
    /**
     * @brief Delete the experiment directory and all its contents
     * @warning This will permanently delete all data for this experiment!
     * @return true if deletion was successful, false otherwise
     */
    bool deleteExperiment() const {
        try {
            if (exists()) {
                std::filesystem::remove_all(experimentDir);
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }
    
    /**
     * @brief Get the size of the experiment directory in bytes
     * @return Size in bytes, or 0 if directory doesn't exist
     */
    size_t getExperimentSize() const {
        if (!exists()) {
            return 0;
        }
        
        size_t totalSize = 0;
        try {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(experimentDir)) {
                if (entry.is_regular_file()) {
                    totalSize += entry.file_size();
                }
            }
        } catch (const std::exception&) {
            return 0;
        }
        
        return totalSize;
    }
    
    /**
     * @brief Get a custom path within the experiment directory
     * @param subpath Subpath within the experiment directory
     * @return Full path to the custom location
     */
    std::string getCustomPath(const std::string& subpath) const {
        return (experimentDir / subpath).string();
    }
    
    /**
     * @brief Set a custom base directory (must be called before using other methods)
     * @param baseDir New base directory
     */
    void setBaseDirectory(const std::string& baseDir) {
        baseDirectory = baseDir;
        experimentDir = std::filesystem::path(baseDirectory) / name;
        datastorePath = experimentDir / "datastore";
    }

private:
    std::string name;                      ///< Experiment name
    std::string baseDirectory;             ///< Base directory for all experiments
    std::filesystem::path experimentDir;   ///< Full path to experiment directory
    std::filesystem::path datastorePath;   ///< Full path to datastore
};

} // namespace snnfw

#endif // SNNFW_EXPERIMENT_CONFIG_H

