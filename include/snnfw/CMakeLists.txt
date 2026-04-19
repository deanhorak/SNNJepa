# Core framework library sources
set(LIB_SOURCES
    Neuron.cpp
    BinaryPattern.cpp
    Cluster.cpp
    Logger.cpp
    ThreadPool.cpp
    Axon.cpp
    Dendrite.cpp
    Synapse.cpp
    SpikeProcessor.cpp
    NetworkPropagator.cpp
    NeuralObjectFactory.cpp
    Datastore.cpp
    MNISTLoader.cpp
    EMNISTLoader.cpp
    domain/VisualDomainAdapter.cpp
    STDPLearning.cpp
    HyperparameterOptimizer.cpp
    ConfigOptimizer.cpp
    adapters/RetinaAdapter.cpp
    adapters/EMNISTAdapter.cpp
    adapters/InterneuronAdapters.cpp
    adapters/DisplayAdapter.cpp
    adapters/AudioAdapter.cpp
    encoding/RateEncoder.cpp
    encoding/TemporalEncoder.cpp
    encoding/PopulationEncoder.cpp
    encoding/EncodingStrategyFactory.cpp
    features/SobelOperator.cpp
    features/GaborOperator.cpp
    features/QuadratureGaborOperator.cpp
    features/OrientationEnergyOperator.cpp
    features/DoGOperator.cpp
    features/EdgeOperatorFactory.cpp
    classification/MajorityVoting.cpp
    classification/HierarchicalKNN.cpp
    classification/WeightedDistance.cpp
    classification/WeightedSimilarity.cpp
    classification/ClassificationStrategyFactory.cpp
    learning/AppendStrategy.cpp
    learning/ReplaceWorstStrategy.cpp
    learning/MergeSimilarStrategy.cpp
    learning/HybridStrategy.cpp
    learning/PatternUpdateStrategyFactory.cpp
    NetworkInspector.cpp
    ActivityMonitor.cpp
    MemoryManager.cpp
    PerformanceProfiler.cpp
    NetworkValidator.cpp
    NetworkBuilder.cpp
    ConnectivityPattern.cpp
    ConnectivityBuilder.cpp
    ModelImporter.cpp
    Camera.cpp
    ShaderManager.cpp
    GeometryRenderer.cpp
    VisualizationManager.cpp
    NetworkDataAdapter.cpp
    LayoutEngine.cpp
    NetworkGraphRenderer.cpp
    ActivityVisualizer.cpp
    SpikeRenderer.cpp
    RecordingManager.cpp
    PlaybackControls.cpp
    RasterPlotRenderer.cpp
    InteractionManager.cpp
    PatternDetector.cpp
    ActivityHistogram.cpp
    NetworkStructureExporter.cpp
    declarative/NetworkIR.cpp
    declarative/NativeJSONParser.cpp
    declarative/SONATAParser.cpp
    declarative/NeuroMLParser.cpp
    declarative/HOCParser.cpp
    declarative/NetworkConstructor.cpp
    declarative/DeclarativeLoader.cpp
    experiment/SpikeEncoder.cpp
    experiment/CompetitionManager.cpp
    experiment/KNNClassifier.cpp
    experiment/SupervisedTeacher.cpp
    experiment/TrainingPipeline.cpp
    experiment/ExperimentRunner.cpp
    jepa/JepaMaskSampler.cpp
    jepa/JepaStateExtractor.cpp
    jepa/JepaTrainer.cpp
)

# Create library
add_library(snnfw STATIC ${LIB_SOURCES})

# Link dependencies
target_link_libraries(snnfw PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    sonata::sonata_shared
    HighFive
    curl
)

if(ROCKSDB_FOUND)
    message(STATUS "Linking RocksDB to snnfw")
    if(TARGET RocksDB::rocksdb)
        message(STATUS "Using RocksDB::rocksdb target")
        target_link_libraries(snnfw PUBLIC RocksDB::rocksdb)
    elseif(ROCKSDB_LIBRARIES)
        message(STATUS "Using ROCKSDB_LIBRARIES: ${ROCKSDB_LIBRARIES}")
        target_link_libraries(snnfw PUBLIC ${ROCKSDB_LIBRARIES})
        target_include_directories(snnfw PUBLIC ${ROCKSDB_INCLUDE_DIRS})
    else()
        message(WARNING "ROCKSDB_FOUND is TRUE but no target or libraries found!")
    endif()
else()
    message(WARNING "ROCKSDB_FOUND is FALSE")
endif()

# Visualization libraries
if(TARGET glfw AND TARGET glad AND TARGET imgui)
    target_link_libraries(snnfw PUBLIC glfw glad imgui)
    message(STATUS "Visualization support ENABLED")
else()
    message(STATUS "Visualization support DISABLED (missing dependencies)")
endif()

# Include directories
target_include_directories(snnfw PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/glm
)
