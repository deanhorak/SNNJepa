#ifndef SNNFW_MAIN_H
#define SNNFW_MAIN_H

namespace snnfw {

/**
 * @brief Main application class for the spiking neural network framework
 * 
 * This class encapsulates the main application logic and serves as the
 * entry point for the SNNFW program. It handles initialization, execution,
 * and cleanup of the framework.
 */
class Main {
public:
    /**
     * @brief Default constructor
     */
    Main();
    
    /**
     * @brief Constructor with command line arguments
     * @param argc Number of command line arguments
     * @param argv Array of command line argument strings
     */
    Main(int argc, char* argv[]);
    
    /**
     * @brief Destructor
     */
    ~Main();
    
    /**
     * @brief Copy constructor (deleted - Main should not be copied)
     */
    Main(const Main& other) = delete;
    
    /**
     * @brief Assignment operator (deleted - Main should not be copied)
     */
    Main& operator=(const Main& other) = delete;
    
    /**
     * @brief Move constructor
     */
    Main(Main&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    Main& operator=(Main&& other) noexcept;
    
    /**
     * @brief Run the main application
     * @return Exit code (0 for success, non-zero for error)
     */
    int run();
    
    /**
     * @brief Initialize the application
     * @return true if initialization successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Execute the main application logic
     * @return true if execution successful, false otherwise
     */
    bool execute();
    
    /**
     * @brief Clean up resources and shutdown
     */
    void shutdown();

private:
    int argc_;           ///< Number of command line arguments
    char** argv_;        ///< Command line arguments
    bool initialized_;   ///< Flag indicating if application is initialized
    
    /**
     * @brief Parse command line arguments
     */
    void parseArguments();
    
    /**
     * @brief Display help information
     */
    void showHelp() const;
    
    /**
     * @brief Display version information
     */
    void showVersion() const;
};

} // namespace snnfw

#endif // SNNFW_MAIN_H
