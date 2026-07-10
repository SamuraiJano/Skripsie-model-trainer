#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "Preprocessor.hpp"

struct Detection {
    int class_id = 0;
    std::string class_name;
    float confidence = 0.0f;
    cv::Rect2f box;
    cv::Mat mask;
    std::vector<float> mask_coeffs;
};

struct TimingInfo {
    double fps = 0.0;
    double preprocess_ms = 0.0;
    double hsv_ms = 0.0;
    double inference_ms = 0.0;
    double total_ms = 0.0;
};

struct InferenceResult {
    std::vector<Detection> detections;
    cv::Mat annotated;
    double inference_ms = 0.0;
};

class YoloSegONNX {
public:
    YoloSegONNX(
        const std::filesystem::path& rgb_model,
        const std::filesystem::path& hsv_model,
        int input_size,
        float conf_threshold,
        float nms_threshold,
        int cpu_threads = 4
    );

    bool hasModel(ModelType model) const;
    std::string modelMessage(ModelType model) const;
    InferenceResult run(ModelType model, const PreprocessResult& input);

private:
    struct SessionBundle {
        bool loaded = false;
        std::filesystem::path path;
        std::string message;
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
    };

    SessionBundle maybeLoadSession(const std::filesystem::path& model_path);
    SessionBundle loadSession(const std::filesystem::path& model_path);
    SessionBundle& bundleFor(ModelType model);
    const SessionBundle& bundleFor(ModelType model) const;

    std::vector<Detection> decodeDetections(
        const float* data,
        const std::vector<int64_t>& shape,
        const PreprocessResult& input
    ) const;

    void buildMasks(
        std::vector<Detection>& detections,
        const float* proto_data,
        const std::vector<int64_t>& proto_shape,
        const PreprocessResult& input
    ) const;

    std::vector<Detection> applyNms(const std::vector<Detection>& detections) const;
    cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections) const;

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    int input_size_;
    float conf_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
    SessionBundle rgb_;
    SessionBundle hsv_;
};
