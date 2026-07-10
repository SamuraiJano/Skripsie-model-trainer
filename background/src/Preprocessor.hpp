#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

enum class ModelType {
    Rgb = 0,
    Hsv = 1,
};

std::string modelTypeToString(ModelType model);
ModelType modelTypeFromString(const std::string& value);

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    cv::Size input_size;
    cv::Size source_size;
};

struct PreprocessResult {
    cv::Mat tensor;
    cv::Mat source_frame;
    LetterboxInfo letterbox;
    double preprocess_ms = 0.0;
    double hsv_ms = 0.0;
};

class Preprocessor {
public:
    explicit Preprocessor(int input_size = 640, cv::Rect roi = {});

    PreprocessResult process(const cv::Mat& bgr_frame, ModelType model) const;
    int inputSize() const;

private:
    cv::Mat cropFrame(const cv::Mat& frame) const;
    cv::Mat letterbox(const cv::Mat& image, LetterboxInfo& info) const;
    static cv::Mat toNchwBlob(const cv::Mat& image, bool swap_first_last_channels);

    int input_size_;
    cv::Rect roi_;
};
