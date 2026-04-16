#ifndef SNNFW_EMNIST_LOADER_H
#define SNNFW_EMNIST_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

namespace snnfw {

/**
 * @brief EMNIST dataset loader for reading IDX format files
 *
 * Loads EMNIST (Extended MNIST) handwritten character images and labels.
 * Supports multiple EMNIST variants: Letters, Digits, Balanced, ByClass, ByMerge.
 *
 * Dataset Information (Letters variant):
 * - Training set: 124,800 images
 * - Test set: 20,800 images
 * - Image size: 28×28 pixels, grayscale (0-255)
 * - Labels: 26 classes (A-Z), encoded as 1-26
 * - Character mapping: 1=A, 2=B, ..., 26=Z
 *
 * Important EMNIST Quirks:
 * - Images are stored rotated 90° and need to be transposed
 * - Images are mirrored and need to be flipped horizontally
 * - Labels start at 1 (not 0 like MNIST)
 *
 * IDX File Format (same as MNIST):
 * - Images: magic number (0x00000803), count, rows, cols, pixel data
 * - Labels: magic number (0x00000801), count, label data
 * - All integers are big-endian
 *
 * Reference: https://www.nist.gov/itl/products-and-services/emnist-dataset
 */
class EMNISTLoader {
public:
    /**
     * @brief EMNIST dataset variant
     */
    enum class Variant {
        LETTERS,    ///< 26 classes (A-Z), 124,800 train / 20,800 test
        DIGITS,     ///< 10 classes (0-9), same as MNIST
        BALANCED,   ///< 47 classes (balanced distribution)
        BYCLASS,    ///< 62 classes (digits + upper + lower case)
        BYMERGE     ///< 47 classes (merged similar characters)
    };

    /**
     * @brief Structure representing a single EMNIST image
     */
    struct Image {
        std::vector<uint8_t> pixels;  ///< Pixel values (0-255), already transposed and flipped
        uint8_t label;                ///< Character label (1-26 for letters, 0-9 for digits)
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
        
        /**
         * @brief Get character label as ASCII (for letters: 'A'-'Z')
         */
        char getCharLabel() const {
            if (label >= 1 && label <= 26) {
                return 'A' + (label - 1);
            }
            return '?';
        }
    };
    
    /**
     * @brief Constructor
     * @param variant EMNIST dataset variant to load
     */
    explicit EMNISTLoader(Variant variant = Variant::LETTERS);
    
    /**
     * @brief Load EMNIST dataset from files
     * @param imageFile Path to image file (e.g., emnist-letters-train-images-idx3-ubyte)
     * @param labelFile Path to label file (e.g., emnist-letters-train-labels-idx1-ubyte)
     * @param maxImages Maximum number of images to load (0 = all)
     * @param applyTransform Apply transpose and flip transformation (default: true)
     * @return true if successful, false otherwise
     */
    bool load(const std::string& imageFile, 
              const std::string& labelFile,
              size_t maxImages = 0,
              bool applyTransform = true);
    
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
     * @brief Get number of classes for current variant
     */
    int getNumClasses() const;
    
    /**
     * @brief Get variant name as string
     */
    std::string getVariantName() const;
    
    /**
     * @brief Print image as ASCII art (for debugging)
     */
    static void printImage(const Image& img, int threshold = 128);
    
private:
    Variant variant;
    std::vector<Image> images;
    
    /**
     * @brief Read 32-bit big-endian integer
     */
    static uint32_t readInt32(std::ifstream& file);
    
    /**
     * @brief Load image data from IDX file
     */
    bool loadImages(const std::string& filename, size_t maxImages, bool applyTransform);
    
    /**
     * @brief Load label data from IDX file
     */
    bool loadLabels(const std::string& filename);
    
    /**
     * @brief Transpose and flip image (EMNIST correction)
     * @param pixels Original pixel data (row-major)
     * @param rows Image height
     * @param cols Image width
     * @return Transformed pixel data
     */
    static std::vector<uint8_t> transposeAndFlip(const std::vector<uint8_t>& pixels, int rows, int cols);
};

} // namespace snnfw

#endif // SNNFW_EMNIST_LOADER_H

