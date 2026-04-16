#ifndef SNNFW_DOMAIN_VISUALDOMAINADAPTER_H
#define SNNFW_DOMAIN_VISUALDOMAINADAPTER_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace snnfw::domain {

struct VisualStimulus {
    std::vector<uint8_t> pixels;
    int label = -1;
    int rows = 0;
    int cols = 0;
    int channels = 1;
    double timestamp = 0.0;

    uint8_t getPixel(int row, int col, int channel = 0) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols ||
            channel < 0 || channel >= std::max(1, channels)) {
            return 0;
        }
        const size_t idx =
            (static_cast<size_t>(row * cols + col) * static_cast<size_t>(std::max(1, channels))) +
            static_cast<size_t>(channel);
        if (idx >= pixels.size()) {
            return 0;
        }
        return pixels[idx];
    }

    double getNormalizedPixel(int row, int col, int channel) const {
        return static_cast<double>(getPixel(row, col, channel)) / 255.0;
    }

    double getNormalizedPixel(int row, int col) const {
        if (channels <= 1) {
            return static_cast<double>(getPixel(row, col, 0)) / 255.0;
        }

        const double red = getNormalizedPixel(row, col, 0);
        const double green = getNormalizedPixel(row, col, 1);
        const double blue = getNormalizedPixel(row, col, 2);
        return 0.299 * red + 0.587 * green + 0.114 * blue;
    }
};

struct VisualDomainConfig {
    std::string source = "emnist";
    std::string variant = "letters";
    bool applyTransform = true;
};

class VisualDomainAdapter {
public:
    virtual ~VisualDomainAdapter() = default;

    virtual bool load(const std::string& imageFile,
                      const std::string& labelFile,
                      size_t maxImages = 0) = 0;
    virtual size_t size() const = 0;
    virtual const VisualStimulus& getStimulus(size_t index) const = 0;
    virtual int numClasses() const = 0;
    virtual std::vector<std::string> classNames() const = 0;
    virtual std::string domainName() const = 0;
};

std::unique_ptr<VisualDomainAdapter> createVisualDomainAdapter(const VisualDomainConfig& config);

} // namespace snnfw::domain

#endif // SNNFW_DOMAIN_VISUALDOMAINADAPTER_H
