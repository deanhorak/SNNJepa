#ifndef SNNFW_MNIST_LOADER_H
#define SNNFW_MNIST_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

namespace snnfw {

/**
 * @brief MNIST dataset loader for reading IDX format files
 *
 * Loads MNIST handwritten digit images and labels from the standard
 * IDX file format used by the MNIST database.
 *
 * Dataset Information:
 * - Training set: 60,000 images (train-images-idx3-ubyte, train-labels-idx1-ubyte)
 * - Test set: 10,000 images (t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte)
 * - Image size: 28×28 pixels, grayscale (0-255)
 * - Labels: Digits 0-9
 *
 * IDX File Format:
 * - Images: magic number (0x00000803), count, rows, cols, pixel data
 * - Labels: magic number (0x00000801), count, label data
 * - All integers are big-endian
 *
 * Usage in MNIST Experiments:
 * - mnist_optimized.cpp uses 5000 training images per digit (50,000 total)
 * - Full test set of 10,000 images for evaluation
 * - Achieves 81.20% accuracy with spike-based pattern matching
 *
 * Reference: http://yann.lecun.com/exdb/mnist/
 */
class MNISTLoader {
public:
    /**
     * @brief Structure representing a single MNIST image
     */
    struct Image {
        std::vector<uint8_t> pixels;  ///< Pixel values (0-255)
        uint8_t label;                ///< Digit label (0-9)
        int rows;                     ///< Image height (28)
        int cols;                     ///< Image width (28)
        
        /**
         * @brief Get pixel value at (row, col)
         */
        uint8_t getPixel(int row, int col) const {
            return pixels[row * cols + col];
        }
        
        /**
         * @brief Get normalized pixel value (0.0 to 1.0)
         */
        double getNormalizedPixel(int row, int col) const {
            return static_cast<double>(getPixel(row, col)) / 255.0;
        }
    };
    
    /**
     * @brief Load MNIST dataset from files
     * @param imageFile Path to image file (e.g., train-images-idx3-ubyte)
     * @param labelFile Path to label file (e.g., train-labels-idx1-ubyte)
     * @param maxImages Maximum number of images to load (0 = all)
     * @return true if successful, false otherwise
     */
    bool load(const std::string& imageFile, 
              const std::string& labelFile,
              size_t maxImages = 0);
    
    /**
     * @brief Get the number of loaded images
     */
    size_t size() const { return images.size(); }
    
    /**
     * @brief Get image at index
     */
    const Image& getImage(size_t index) const { return images[index]; }
    
    /**
     * @brief Get all images
     */
    const std::vector<Image>& getImages() const { return images; }
    
    /**
     * @brief Print image as ASCII art (for debugging)
     */
    static void printImage(const Image& img, int threshold = 128);
    
private:
    std::vector<Image> images;
    
    /**
     * @brief Read 32-bit big-endian integer
     */
    static uint32_t readInt32(std::ifstream& file);
    
    /**
     * @brief Load image data from IDX file
     */
    bool loadImages(const std::string& filename, size_t maxImages);
    
    /**
     * @brief Load label data from IDX file
     */
    bool loadLabels(const std::string& filename);
};

} // namespace snnfw

#endif // SNNFW_MNIST_LOADER_H

