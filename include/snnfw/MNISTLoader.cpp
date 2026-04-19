#include "snnfw/MNISTLoader.h"
#include "snnfw/Logger.h"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace snnfw {

uint32_t MNISTLoader::readInt32(std::ifstream& file) {
    uint32_t value = 0;
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    
    // Convert from big-endian
    value = (static_cast<uint32_t>(bytes[0]) << 24) |
            (static_cast<uint32_t>(bytes[1]) << 16) |
            (static_cast<uint32_t>(bytes[2]) << 8) |
            (static_cast<uint32_t>(bytes[3]));
    
    return value;
}

bool MNISTLoader::loadImages(const std::string& filename, size_t maxImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        SNNFW_ERROR("Failed to open image file: {}", filename);
        return false;
    }
    
    // Read header
    uint32_t magicNumber = readInt32(file);
    if (magicNumber != 0x00000803) {
        SNNFW_ERROR("Invalid magic number in image file: 0x{:08X} (expected 0x00000803)", magicNumber);
        return false;
    }
    
    uint32_t numImages = readInt32(file);
    uint32_t numRows = readInt32(file);
    uint32_t numCols = readInt32(file);
    
    SNNFW_INFO("MNIST Images: {} images, {}x{} pixels", numImages, numRows, numCols);
    
    // Limit number of images if requested
    if (maxImages > 0 && maxImages < numImages) {
        numImages = maxImages;
        SNNFW_INFO("Loading only first {} images", maxImages);
    }
    
    // Reserve space
    images.resize(numImages);
    
    // Read image data
    for (size_t i = 0; i < numImages; ++i) {
        images[i].rows = numRows;
        images[i].cols = numCols;
        images[i].pixels.resize(numRows * numCols);
        
        file.read(reinterpret_cast<char*>(images[i].pixels.data()), numRows * numCols);
        
        if (!file) {
            SNNFW_ERROR("Failed to read image {}", i);
            return false;
        }
    }
    
    SNNFW_INFO("Successfully loaded {} images", numImages);
    return true;
}

bool MNISTLoader::loadLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        SNNFW_ERROR("Failed to open label file: {}", filename);
        return false;
    }

    // Read header
    uint32_t magicNumber = readInt32(file);
    if (magicNumber != 0x00000801) {
        SNNFW_ERROR("Invalid magic number in label file: 0x{:08X} (expected 0x00000801)", magicNumber);
        return false;
    }

    uint32_t numLabels = readInt32(file);

    SNNFW_INFO("MNIST Labels: {} labels in file", numLabels);

    // Only read as many labels as we have images
    size_t labelsToRead = images.size();

    if (labelsToRead > numLabels) {
        SNNFW_ERROR("Requested {} labels but file only contains {}",
                    labelsToRead, numLabels);
        return false;
    }

    // Read labels
    for (size_t i = 0; i < labelsToRead; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);

        if (!file) {
            SNNFW_ERROR("Failed to read label {}", i);
            return false;
        }

        images[i].label = label;
    }

    SNNFW_INFO("Successfully loaded {} labels", labelsToRead);
    return true;
}

bool MNISTLoader::load(const std::string& imageFile, 
                       const std::string& labelFile,
                       size_t maxImages) {
    images.clear();
    
    if (!loadImages(imageFile, maxImages)) {
        return false;
    }
    
    if (!loadLabels(labelFile)) {
        return false;
    }
    
    return true;
}

void MNISTLoader::printImage(const Image& img, int threshold) {
    std::cout << "Label: " << static_cast<int>(img.label) << std::endl;
    std::cout << std::string(img.cols + 2, '-') << std::endl;
    
    for (int row = 0; row < img.rows; ++row) {
        std::cout << "|";
        for (int col = 0; col < img.cols; ++col) {
            uint8_t pixel = img.getPixel(row, col);
            if (pixel < threshold / 4) {
                std::cout << " ";
            } else if (pixel < threshold / 2) {
                std::cout << ".";
            } else if (pixel < threshold) {
                std::cout << "+";
            } else {
                std::cout << "#";
            }
        }
        std::cout << "|" << std::endl;
    }
    
    std::cout << std::string(img.cols + 2, '-') << std::endl;
}

} // namespace snnfw

