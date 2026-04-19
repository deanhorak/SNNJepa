#include "snnfw/domain/VisualDomainAdapter.h"

#include "snnfw/EMNISTLoader.h"
#include "snnfw/MNISTLoader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace snnfw::domain {

namespace {

namespace fs = std::filesystem;

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> makeDigitClassNames(int count) {
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(std::max(0, count)));
    for (int i = 0; i < count; ++i) {
        names.push_back(std::to_string(i));
    }
    return names;
}

std::vector<std::string> makeLetterClassNames() {
    std::vector<std::string> names;
    names.reserve(26);
    for (char c = 'A'; c <= 'Z'; ++c) {
        names.emplace_back(1, c);
    }
    return names;
}

std::vector<std::string> makeCifar10ClassNames() {
    return {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
}

uint8_t rgbToGrayscale(uint8_t red, uint8_t green, uint8_t blue) {
    const double luminance =
        0.299 * static_cast<double>(red) +
        0.587 * static_cast<double>(green) +
        0.114 * static_cast<double>(blue);
    return static_cast<uint8_t>(std::clamp(luminance, 0.0, 255.0));
}
std::vector<fs::path> collectCifar10BatchFiles(const std::string& imageFile) {
    const fs::path path(imageFile);
    if (fs::is_regular_file(path)) {
        return {path};
    }
    if (!fs::is_directory(path)) {
        throw std::runtime_error("CIFAR-10 path is neither a file nor a directory: " + imageFile);
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("data_batch_", 0) == 0 && entry.path().extension() == ".bin") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("No CIFAR-10 training batch files found in: " + imageFile);
    }
    return files;
}

class EMNISTDomainAdapter final : public VisualDomainAdapter {
public:
    explicit EMNISTDomainAdapter(const VisualDomainConfig& config)
        : variant_(parseVariant(config.variant)),
          applyTransform_(config.applyTransform),
          loader_(variant_) {}

    bool load(const std::string& imageFile,
              const std::string& labelFile,
              size_t maxImages = 0) override {
        if (!loader_.load(imageFile, labelFile, maxImages, applyTransform_)) {
            return false;
        }

        stimuli_.clear();
        stimuli_.reserve(loader_.size());
        for (size_t i = 0; i < loader_.size(); ++i) {
            const auto& image = loader_.getImage(i);
            VisualStimulus stimulus;
            stimulus.pixels = image.pixels;
            stimulus.rows = image.rows;
            stimulus.cols = image.cols;
            stimulus.channels = 1;
            stimulus.timestamp = static_cast<double>(i);
            stimulus.label = normalizeLabel(image.label);
            stimuli_.push_back(std::move(stimulus));
        }
        return true;
    }

    size_t size() const override {
        return stimuli_.size();
    }

    const VisualStimulus& getStimulus(size_t index) const override {
        return stimuli_[index];
    }

    int numClasses() const override {
        return loader_.getNumClasses();
    }

    std::vector<std::string> classNames() const override {
        switch (variant_) {
            case EMNISTLoader::Variant::LETTERS:
                return makeLetterClassNames();
            case EMNISTLoader::Variant::DIGITS:
                return makeDigitClassNames(10);
            case EMNISTLoader::Variant::BALANCED:
            case EMNISTLoader::Variant::BYMERGE:
                return makeDigitClassNames(loader_.getNumClasses());
            case EMNISTLoader::Variant::BYCLASS:
                return makeDigitClassNames(loader_.getNumClasses());
            default:
                return makeDigitClassNames(loader_.getNumClasses());
        }
    }

    std::string domainName() const override {
        return "EMNIST " + loader_.getVariantName();
    }

private:
    static EMNISTLoader::Variant parseVariant(const std::string& variant) {
        const std::string lowered = toLower(variant);
        if (lowered.empty() || lowered == "letters") {
            return EMNISTLoader::Variant::LETTERS;
        }
        if (lowered == "digits") {
            return EMNISTLoader::Variant::DIGITS;
        }
        if (lowered == "balanced") {
            return EMNISTLoader::Variant::BALANCED;
        }
        if (lowered == "byclass") {
            return EMNISTLoader::Variant::BYCLASS;
        }
        if (lowered == "bymerge") {
            return EMNISTLoader::Variant::BYMERGE;
        }
        throw std::runtime_error("Unsupported EMNIST variant: " + variant);
    }

    int normalizeLabel(uint8_t rawLabel) const {
        if (variant_ == EMNISTLoader::Variant::LETTERS) {
            return static_cast<int>(rawLabel) - 1;
        }
        return static_cast<int>(rawLabel);
    }

    EMNISTLoader::Variant variant_;
    bool applyTransform_ = true;
    EMNISTLoader loader_;
    std::vector<VisualStimulus> stimuli_;
};

class MNISTDomainAdapter final : public VisualDomainAdapter {
public:
    explicit MNISTDomainAdapter(const VisualDomainConfig&) {}

    bool load(const std::string& imageFile,
              const std::string& labelFile,
              size_t maxImages = 0) override {
        if (!loader_.load(imageFile, labelFile, maxImages)) {
            return false;
        }

        stimuli_.clear();
        stimuli_.reserve(loader_.size());
        for (size_t i = 0; i < loader_.size(); ++i) {
            const auto& image = loader_.getImage(i);
            VisualStimulus stimulus;
            stimulus.pixels = image.pixels;
            stimulus.rows = image.rows;
            stimulus.cols = image.cols;
            stimulus.channels = 1;
            stimulus.timestamp = static_cast<double>(i);
            stimulus.label = static_cast<int>(image.label);
            stimuli_.push_back(std::move(stimulus));
        }
        return true;
    }

    size_t size() const override {
        return stimuli_.size();
    }

    const VisualStimulus& getStimulus(size_t index) const override {
        return stimuli_[index];
    }

    int numClasses() const override {
        return 10;
    }

    std::vector<std::string> classNames() const override {
        return makeDigitClassNames(10);
    }

    std::string domainName() const override {
        return "MNIST Digits";
    }

private:
    MNISTLoader loader_;
    std::vector<VisualStimulus> stimuli_;
};

class CIFAR10DomainAdapter final : public VisualDomainAdapter {
public:
    explicit CIFAR10DomainAdapter(const VisualDomainConfig&) {}

    bool load(const std::string& imageFile,
              const std::string& labelFile,
              size_t maxImages = 0) override {
        (void)labelFile;

        std::vector<fs::path> files;
        try {
            files = collectFiles(imageFile);
        } catch (const std::exception&) {
            return false;
        }

        stimuli_.clear();
        const size_t reserveCount = maxImages > 0 ? maxImages : files.size() * 10000ULL;
        stimuli_.reserve(reserveCount);

        for (const auto& file : files) {
            if (!loadBatchFile(file, maxImages)) {
                return false;
            }
            if (maxImages > 0 && stimuli_.size() >= maxImages) {
                break;
            }
        }
        return !stimuli_.empty();
    }

    size_t size() const override {
        return stimuli_.size();
    }

    const VisualStimulus& getStimulus(size_t index) const override {
        return stimuli_[index];
    }

    int numClasses() const override {
        return 10;
    }

    std::vector<std::string> classNames() const override {
        return makeCifar10ClassNames();
    }

    std::string domainName() const override {
        return "CIFAR-10";
    }

private:
    static std::vector<fs::path> collectFiles(const std::string& imageFile) {
        const fs::path path(imageFile);
        if (fs::is_regular_file(path)) {
            return {path};
        }
        return collectCifar10BatchFiles(imageFile);
    }

    bool loadBatchFile(const fs::path& file, size_t maxImages) {
        constexpr size_t kImageSize = 32 * 32;
        constexpr size_t kRecordSize = 1 + 3 * kImageSize;

        std::ifstream input(file, std::ios::binary);
        if (!input.is_open()) {
            return false;
        }

        std::array<uint8_t, kRecordSize> record{};
        while (input.read(reinterpret_cast<char*>(record.data()),
                          static_cast<std::streamsize>(record.size()))) {
            if (maxImages > 0 && stimuli_.size() >= maxImages) {
                break;
            }

            VisualStimulus stimulus;
            stimulus.label = static_cast<int>(record[0]);
            stimulus.rows = 32;
            stimulus.cols = 32;
            stimulus.channels = 3;
            stimulus.timestamp = static_cast<double>(stimuli_.size());
            stimulus.pixels.resize(kImageSize * 3);

            const uint8_t* red = record.data() + 1;
            const uint8_t* green = red + kImageSize;
            const uint8_t* blue = green + kImageSize;
            for (size_t i = 0; i < kImageSize; ++i) {
                const size_t idx = i * 3;
                stimulus.pixels[idx] = red[i];
                stimulus.pixels[idx + 1] = green[i];
                stimulus.pixels[idx + 2] = blue[i];
            }
            stimuli_.push_back(std::move(stimulus));
        }

        return input.eof() || static_cast<bool>(input);
    }

    std::vector<VisualStimulus> stimuli_;
};

} // namespace

std::unique_ptr<VisualDomainAdapter> createVisualDomainAdapter(const VisualDomainConfig& config) {
    const std::string source = toLower(config.source);
    if (source.empty() || source == "emnist") {
        return std::make_unique<EMNISTDomainAdapter>(config);
    }
    if (source == "mnist") {
        return std::make_unique<MNISTDomainAdapter>(config);
    }
    if (source == "cifar10" || source == "cifar-10") {
        return std::make_unique<CIFAR10DomainAdapter>(config);
    }
    throw std::runtime_error("Unsupported visual domain source: " + config.source);
}

} // namespace snnfw::domain
