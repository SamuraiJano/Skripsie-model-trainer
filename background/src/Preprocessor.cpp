#include "Preprocessor.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include <opencv2/dnn.hpp>

namespace {
using Clock = std::chrono::high_resolution_clock;

// Small shared helpers ------------------------------------------------------
double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}
}

// Model type parsing --------------------------------------------------------
std::string modelTypeToString(ModelType model) {
    return model == ModelType::Hsv ? "hsv" : "rgb";
}

ModelType modelTypeFromString(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized == "rgb") {
        return ModelType::Rgb;
    }
    if (normalized == "hsv") {
        return ModelType::Hsv;
    }
    throw std::invalid_argument("model must be 'rgb' or 'hsv'");
}

// Preprocessing steps -------------------------------------------------------
Preprocessor::Preprocessor(int input_size, cv::Rect roi)
    : input_size_(input_size), roi_(roi) {
    if (input_size_ <= 0) {
        throw std::invalid_argument("input_size must be positive");
    }
}

int Preprocessor::inputSize() const {
    return input_size_;
}

cv::Mat Preprocessor::cropFrame(const cv::Mat& frame) const {
    if (roi_.area() <= 0) {
        return frame;
    }
    const cv::Rect frame_rect(0, 0, frame.cols, frame.rows);
    const cv::Rect bounded = roi_ & frame_rect;
    if (bounded.area() <= 0) {
        return frame;
    }
    return frame(bounded);
}

cv::Mat Preprocessor::letterbox(const cv::Mat& image, LetterboxInfo& info) const {
    const int width = image.cols;
    const int height = image.rows;
    info.source_size = image.size();
    info.input_size = cv::Size(input_size_, input_size_);
    info.scale = std::min(static_cast<float>(input_size_) / static_cast<float>(width),
                          static_cast<float>(input_size_) / static_cast<float>(height));

    const int new_width = static_cast<int>(std::round(width * info.scale));
    const int new_height = static_cast<int>(std::round(height * info.scale));
    info.pad_x = (input_size_ - new_width) / 2;
    info.pad_y = (input_size_ - new_height) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, new_width, new_height)));
    return canvas;
}

cv::Mat Preprocessor::toNchwBlob(const cv::Mat& image, bool swap_first_last_channels) {
    cv::Mat blob;
    cv::dnn::blobFromImage(
        image,
        blob,
        1.0 / 255.0,
        cv::Size(),
        cv::Scalar(),
        swap_first_last_channels,
        false,
        CV_32F
    );
    return blob;
}

// Full pipeline -------------------------------------------------------------
PreprocessResult Preprocessor::process(const cv::Mat& bgr_frame, ModelType model) const {
    if (bgr_frame.empty()) {
        throw std::invalid_argument("Cannot preprocess an empty frame");
    }

    const auto start = Clock::now();
    PreprocessResult result;
    result.source_frame = cropFrame(bgr_frame);

    const cv::Mat letterboxed = letterbox(result.source_frame, result.letterbox);

    if (model == ModelType::Hsv) {
        const auto hsv_start = Clock::now();
        cv::Mat hsv;
        cv::cvtColor(letterboxed, hsv, cv::COLOR_BGR2HSV);
        const auto hsv_end = Clock::now();
        result.hsv_ms = elapsedMs(hsv_start, hsv_end);
        // Ultralytics loads image files through its normal BGR-to-RGB path.
        // Mirror that channel flip here so HSV ONNX inference matches HSV training.
        result.tensor = toNchwBlob(hsv, true);
    } else {
        result.tensor = toNchwBlob(letterboxed, true);
    }

    const auto end = Clock::now();
    result.preprocess_ms = std::max(0.0, elapsedMs(start, end) - result.hsv_ms);
    return result;
}
