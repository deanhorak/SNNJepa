#include "snnfw/EMNISTLoader.h"
#include "snnfw/Logger.h"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace snnfw {

EMNISTLoader::EMNISTLoader(Variant variant) : variant(variant) {}

uint32_t EMNISTLoader::readInt32(std::ifstream& file) {
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

std::vector<uint8_t> EMNISTLoader::transposeAndFlip(const std::vector<uint8_t>& pixels, int rows, int cols) {
    std::vector<uint8_t> result(rows * cols);
    
    // Transpose: (row, col) -> (col, row)
    // Then flip horizontally: col -> (cols - 1 - col)
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int srcIdx = row * cols + col;
            int newRow = col;
            int newCol = (cols - 1 - row);  // Flip horizontally during transpose
            int dstIdx = newRow * cols + newCol;
            result[dstIdx] = pixels[srcIdx];
        }
    }
    
    return result;
}

bool EMNISTLoader::loadImages(const std::string& filename, size_t maxImages, bool applyTransform) {
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
    
    SNNFW_INFO("EMNIST {} Images: {} images, {}x{} pixels", 
               getVariantName(), numImages, numRows, numCols);
    
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
        
        // Read raw pixel data
        std::vector<uint8_t> rawPixels(numRows * numCols);
        file.read(reinterpret_cast<char*>(rawPixels.data()), numRows * numCols);
        
        if (!file) {
            SNNFW_ERROR("Failed to read image {}", i);
            return false;
        }
        
        // Apply transpose and flip transformation if requested
        if (applyTransform) {
            images[i].pixels = transposeAndFlip(rawPixels, numRows, numCols);
        } else {
            images[i].pixels = rawPixels;
        }
    }
    
    SNNFW_INFO("Successfully loaded {} images (transform: {})", 
               numImages, applyTransform ? "applied" : "skipped");
    return true;
}

bool EMNISTLoader::loadLabels(const std::string& filename) {
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

    SNNFW_INFO("EMNIST {} Labels: {} labels in file", getVariantName(), numLabels);

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

bool EMNISTLoader::load(const std::string& imageFile, 
                        const std::string& labelFile,
                        size_t maxImages,
                        bool applyTransform) {
    // Load images first
    if (!loadImages(imageFile, maxImages, applyTransform)) {
        return false;
    }
    
    // Then load labels
    if (!loadLabels(labelFile)) {
        return false;
    }
    
    SNNFW_INFO("EMNIST {} dataset loaded: {} images", getVariantName(), images.size());
    return true;
}

int EMNISTLoader::getNumClasses() const {
    switch (variant) {
        case Variant::LETTERS:   return 26;  // A-Z
        case Variant::DIGITS:    return 10;  // 0-9
        case Variant::BALANCED:  return 47;  // Balanced 47 classes
        case Variant::BYCLASS:   return 62;  // 0-9, A-Z, a-z
        case Variant::BYMERGE:   return 47;  // Merged similar characters
        default:                 return 26;
    }
}

std::string EMNISTLoader::getVariantName() const {
    switch (variant) {
        case Variant::LETTERS:   return "Letters";
        case Variant::DIGITS:    return "Digits";
        case Variant::BALANCED:  return "Balanced";
        case Variant::BYCLASS:   return "ByClass";
        case Variant::BYMERGE:   return "ByMerge";
        default:                 return "Unknown";
    }
}

void EMNISTLoader::printImage(const Image& img, int threshold) {
    std::cout << "Label: " << static_cast<int>(img.label) 
              << " ('" << img.getCharLabel() << "')" << std::endl;
    
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            uint8_t pixel = img.getPixel(row, col);
            if (pixel > threshold) {
                std::cout << "█";
            } else if (pixel > threshold / 2) {
                std::cout << "▓";
            } else if (pixel > threshold / 4) {
                std::cout << "░";
            } else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}

} // namespace snnfw

