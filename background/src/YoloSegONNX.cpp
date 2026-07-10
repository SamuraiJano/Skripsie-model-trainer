#include "YoloSegONNX.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {
using Clock = std::chrono::high_resolution_clock;

// General helpers -----------------------------------------------------------
double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<const char*> asCStrings(const std::vector<std::string>& values) {
    std::vector<const char*> out;
    out.reserve(values.size());
    for (const auto& value : values) {
        out.push_back(value.c_str());
    }
    return out;
}

float clampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

// Prototype mask helpers ----------------------------------------------------
// YOLO segmentation models output a stack of low-resolution prototype masks.
// Each detection stores coefficients that are multiplied by this stack to make
// its instance mask. Keeping prototypes as a 2D matrix lets OpenCV do that
// multiply with its optimized CPU/OpenCL paths.
struct ProtoLayout {
    int channels = 0;
    int height = 0;
    int width = 0;
    bool nhwc = false;
};

ProtoLayout parseProtoLayout(const std::vector<int64_t>& shape) {
    ProtoLayout layout;
    if (shape.size() == 4) {
        const bool looks_nchw = shape[1] <= 512 && shape[1] <= shape[2] && shape[1] <= shape[3];
        if (looks_nchw) {
            layout.channels = static_cast<int>(shape[1]);
            layout.height = static_cast<int>(shape[2]);
            layout.width = static_cast<int>(shape[3]);
            layout.nhwc = false;
        } else {
            layout.height = static_cast<int>(shape[1]);
            layout.width = static_cast<int>(shape[2]);
            layout.channels = static_cast<int>(shape[3]);
            layout.nhwc = true;
        }
    } else if (shape.size() == 3) {
        layout.channels = static_cast<int>(shape[0]);
        layout.height = static_cast<int>(shape[1]);
        layout.width = static_cast<int>(shape[2]);
        layout.nhwc = false;
    }
    if (layout.channels <= 0 || layout.height <= 0 || layout.width <= 0) {
        throw std::runtime_error("Unsupported YOLO mask prototype output shape");
    }
    return layout;
}

cv::Mat flattenPrototypeMasks(const float* data, const ProtoLayout& layout) {
    const int pixels = layout.height * layout.width;
    if (!layout.nhwc) {
        return cv::Mat(layout.channels, pixels, CV_32F, const_cast<float*>(data));
    }

    cv::Mat flat(layout.channels, pixels, CV_32F);
    for (int y = 0; y < layout.height; ++y) {
        for (int x = 0; x < layout.width; ++x) {
            const int pixel = y * layout.width + x;
            const float* source = data + pixel * layout.channels;
            for (int c = 0; c < layout.channels; ++c) {
                flat.at<float>(c, pixel) = source[c];
            }
        }
    }
    return flat;
}

void applySigmoidInPlace(cv::Mat& logits) {
    cv::Mat exp_values;
    cv::multiply(logits, -1.0, exp_values);
    cv::exp(exp_values, exp_values);
    exp_values += 1.0f;
    cv::divide(1.0, exp_values, logits);
}
}

// Session setup -------------------------------------------------------------
YoloSegONNX::YoloSegONNX(
    const fs::path& rgb_model,
    const fs::path& hsv_model,
    int input_size,
    float conf_threshold,
    float nms_threshold,
    int cpu_threads
) : env_(ORT_LOGGING_LEVEL_WARNING, "naartjie-defect-onnx"),
    input_size_(input_size),
    conf_threshold_(conf_threshold),
    nms_threshold_(nms_threshold),
    class_names_({"safe", "minor_defect", "major_defect"}) {
    if (input_size_ <= 0) {
        throw std::invalid_argument("input_size must be positive");
    }

    session_options_.SetIntraOpNumThreads(std::max(1, cpu_threads));
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    rgb_ = maybeLoadSession(rgb_model);
    hsv_ = maybeLoadSession(hsv_model);
}

YoloSegONNX::SessionBundle YoloSegONNX::maybeLoadSession(const fs::path& model_path) {
    SessionBundle bundle;
    bundle.path = model_path;
    if (!fs::exists(model_path)) {
        bundle.message = "Model file not found: " + model_path.string();
        return bundle;
    }
    try {
        bundle = loadSession(model_path);
        bundle.loaded = true;
        bundle.message = "Loaded " + model_path.string();
    } catch (const std::exception& exc) {
        bundle.loaded = false;
        bundle.message = exc.what();
    }
    return bundle;
}

YoloSegONNX::SessionBundle YoloSegONNX::loadSession(const fs::path& model_path) {
    SessionBundle bundle;
    bundle.path = model_path;

#ifdef _WIN32
    const std::wstring wide_path = model_path.wstring();
    bundle.session = std::make_unique<Ort::Session>(env_, wide_path.c_str(), session_options_);
#else
    const std::string narrow_path = model_path.string();
    bundle.session = std::make_unique<Ort::Session>(env_, narrow_path.c_str(), session_options_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = bundle.session->GetInputCount();
    const size_t output_count = bundle.session->GetOutputCount();
    if (input_count < 1 || output_count < 1) {
        throw std::runtime_error("ONNX model must have at least one input and one output: " + model_path.string());
    }

    for (size_t i = 0; i < input_count; ++i) {
        auto name = bundle.session->GetInputNameAllocated(i, allocator);
        bundle.input_names.emplace_back(name.get());
    }
    for (size_t i = 0; i < output_count; ++i) {
        auto name = bundle.session->GetOutputNameAllocated(i, allocator);
        bundle.output_names.emplace_back(name.get());
    }
    return bundle;
}

// Public runtime API --------------------------------------------------------
YoloSegONNX::SessionBundle& YoloSegONNX::bundleFor(ModelType model) {
    return model == ModelType::Hsv ? hsv_ : rgb_;
}

const YoloSegONNX::SessionBundle& YoloSegONNX::bundleFor(ModelType model) const {
    return model == ModelType::Hsv ? hsv_ : rgb_;
}

bool YoloSegONNX::hasModel(ModelType model) const {
    const auto& bundle = bundleFor(model);
    return bundle.loaded && bundle.session != nullptr;
}

std::string YoloSegONNX::modelMessage(ModelType model) const {
    const auto& bundle = bundleFor(model);
    return bundle.message;
}

InferenceResult YoloSegONNX::run(ModelType model, const PreprocessResult& input) {
    auto& bundle = bundleFor(model);
    if (!bundle.loaded || bundle.session == nullptr) {
        throw std::runtime_error(modelTypeToString(model) + " model is not loaded: " + bundle.message);
    }
    auto input_names = asCStrings(bundle.input_names);
    auto output_names = asCStrings(bundle.output_names);

    if (input.tensor.empty() || input.tensor.depth() != CV_32F || !input.tensor.isContinuous()) {
        throw std::runtime_error("Preprocessed tensor must be a continuous float32 OpenCV blob");
    }

    std::array<int64_t, 4> input_shape = {1, 3, input_size_, input_size_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input.tensor.ptr<float>()),
        input.tensor.total(),
        input_shape.data(),
        input_shape.size()
    );

    const auto start = Clock::now();
    auto outputs = bundle.session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &tensor,
        1,
        output_names.data(),
        output_names.size()
    );

    auto det_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> det_shape = det_info.GetShape();
    const float* det_data = outputs[0].GetTensorData<float>();
    auto detections = decodeDetections(det_data, det_shape, input);
    detections = applyNms(detections);

    if (outputs.size() > 1 && !detections.empty()) {
        auto proto_info = outputs[1].GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> proto_shape = proto_info.GetShape();
        const float* proto_data = outputs[1].GetTensorData<float>();
        buildMasks(detections, proto_data, proto_shape, input);
    }

    InferenceResult result;
    result.detections = std::move(detections);
    result.annotated = drawDetections(input.source_frame, result.detections);
    result.inference_ms = elapsedMs(start, Clock::now());
    return result;
}

// Detection decoding --------------------------------------------------------
std::vector<Detection> YoloSegONNX::decodeDetections(
    const float* data,
    const std::vector<int64_t>& shape,
    const PreprocessResult& input
) const {
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Unsupported YOLO detection output shape");
    }

    const int64_t dim_a = shape.size() == 3 ? shape[1] : shape[0];
    const int64_t dim_b = shape.size() == 3 ? shape[2] : shape[1];
    const bool attrs_first = dim_a <= dim_b && dim_a <= 512;
    const int attrs = static_cast<int>(attrs_first ? dim_a : dim_b);
    const int candidates = static_cast<int>(attrs_first ? dim_b : dim_a);
    const int class_count = static_cast<int>(class_names_.size());

    if (attrs < 4 + class_count) {
        throw std::runtime_error("YOLO detection output does not contain expected box and class attributes");
    }

    auto valueAt = [&](int candidate, int attr) -> float {
        if (attrs_first) {
            return data[static_cast<size_t>(attr) * candidates + candidate];
        }
        return data[static_cast<size_t>(candidate) * attrs + attr];
    };

    std::vector<Detection> detections;
    detections.reserve(128);
    for (int i = 0; i < candidates; ++i) {
        int best_class = 0;
        float best_score = 0.0f;
        for (int c = 0; c < class_count; ++c) {
            const float score = valueAt(i, 4 + c);
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        if (best_score < conf_threshold_) {
            continue;
        }

        const float cx = valueAt(i, 0);
        const float cy = valueAt(i, 1);
        const float w = valueAt(i, 2);
        const float h = valueAt(i, 3);
        const auto& lb = input.letterbox;
        const float x1 = (cx - w * 0.5f - static_cast<float>(lb.pad_x)) / lb.scale;
        const float y1 = (cy - h * 0.5f - static_cast<float>(lb.pad_y)) / lb.scale;
        const float x2 = (cx + w * 0.5f - static_cast<float>(lb.pad_x)) / lb.scale;
        const float y2 = (cy + h * 0.5f - static_cast<float>(lb.pad_y)) / lb.scale;

        Detection det;
        det.class_id = best_class;
        det.class_name = class_names_[static_cast<size_t>(best_class)];
        det.confidence = best_score;
        const float max_x = static_cast<float>(lb.source_size.width - 1);
        const float max_y = static_cast<float>(lb.source_size.height - 1);
        const float bx1 = clampFloat(x1, 0.0f, max_x);
        const float by1 = clampFloat(y1, 0.0f, max_y);
        const float bx2 = clampFloat(x2, 0.0f, max_x);
        const float by2 = clampFloat(y2, 0.0f, max_y);
        det.box = cv::Rect2f(bx1, by1, std::max(0.0f, bx2 - bx1), std::max(0.0f, by2 - by1));

        for (int attr = 4 + class_count; attr < attrs; ++attr) {
            det.mask_coeffs.push_back(valueAt(i, attr));
        }
        detections.push_back(std::move(det));
    }
    return detections;
}

// Mask reconstruction -------------------------------------------------------
void YoloSegONNX::buildMasks(
    std::vector<Detection>& detections,
    const float* proto_data,
    const std::vector<int64_t>& proto_shape,
    const PreprocessResult& input
) const {
    const ProtoLayout layout = parseProtoLayout(proto_shape);
    const cv::Mat proto_flat = flattenPrototypeMasks(proto_data, layout);
    for (auto& det : detections) {
        if (static_cast<int>(det.mask_coeffs.size()) < layout.channels) {
            continue;
        }

        cv::Mat coeffs(1, layout.channels, CV_32F, det.mask_coeffs.data());
        cv::Mat low_row = coeffs * proto_flat;
        cv::Mat low = low_row.reshape(1, layout.height).clone();
        applySigmoidInPlace(low);

        cv::Mat full;
        cv::resize(low, full, cv::Size(input_size_, input_size_), 0, 0, cv::INTER_LINEAR);

        cv::Rect valid(
            input.letterbox.pad_x,
            input.letterbox.pad_y,
            input_size_ - 2 * input.letterbox.pad_x,
            input_size_ - 2 * input.letterbox.pad_y
        );
        valid &= cv::Rect(0, 0, full.cols, full.rows);
        if (valid.area() <= 0) {
            continue;
        }

        cv::Mat cropped = full(valid).clone();
        cv::Mat resized;
        cv::resize(cropped, resized, input.letterbox.source_size, 0, 0, cv::INTER_LINEAR);
        cv::threshold(resized, resized, 0.5, 255.0, cv::THRESH_BINARY);
        resized.convertTo(det.mask, CV_8U);

        cv::Rect box_rect(
            static_cast<int>(std::round(det.box.x)),
            static_cast<int>(std::round(det.box.y)),
            static_cast<int>(std::round(det.box.width)),
            static_cast<int>(std::round(det.box.height))
        );
        box_rect &= cv::Rect(0, 0, det.mask.cols, det.mask.rows);
        if (box_rect.area() > 0) {
            cv::Mat box_mask(det.mask.size(), CV_8U, cv::Scalar(0));
            box_mask(box_rect).setTo(255);
            cv::bitwise_and(det.mask, box_mask, det.mask);
        }
    }
}

// Post-processing and drawing ----------------------------------------------
std::vector<Detection> YoloSegONNX::applyNms(const std::vector<Detection>& detections) const {
    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[static_cast<size_t>(a)].confidence > detections[static_cast<size_t>(b)].confidence;
    });

    std::vector<bool> removed(detections.size(), false);
    std::vector<Detection> kept;
    kept.reserve(detections.size());
    for (size_t pos = 0; pos < order.size(); ++pos) {
        const int idx = order[pos];
        if (removed[static_cast<size_t>(idx)]) {
            continue;
        }
        kept.push_back(detections[static_cast<size_t>(idx)]);
        for (size_t next = pos + 1; next < order.size(); ++next) {
            const int other = order[next];
            if (removed[static_cast<size_t>(other)]) {
                continue;
            }
            const auto& a = detections[static_cast<size_t>(idx)];
            const auto& b = detections[static_cast<size_t>(other)];
            if (a.class_id == b.class_id && iou(a.box, b.box) > nms_threshold_) {
                removed[static_cast<size_t>(other)] = true;
            }
        }
    }
    return kept;
}

float YoloSegONNX::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float uni = a.area() + b.area() - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

cv::Mat YoloSegONNX::drawDetections(const cv::Mat& image, const std::vector<Detection>& detections) const {
    cv::Mat annotated = image.clone();
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(70, 210, 70),
        cv::Scalar(40, 180, 255),
        cv::Scalar(50, 70, 235),
    };

    for (const auto& det : detections) {
        const cv::Scalar color = colors[static_cast<size_t>(det.class_id) % colors.size()];
        if (!det.mask.empty()) {
            cv::Mat overlay = annotated.clone();
            overlay.setTo(color, det.mask);
            cv::addWeighted(overlay, 0.35, annotated, 0.65, 0.0, annotated);
        }

        cv::Rect box(
            static_cast<int>(std::round(det.box.x)),
            static_cast<int>(std::round(det.box.y)),
            static_cast<int>(std::round(det.box.width)),
            static_cast<int>(std::round(det.box.height))
        );
        box &= cv::Rect(0, 0, annotated.cols, annotated.rows);
        cv::rectangle(annotated, box, color, 2);

        std::string label = det.class_name + " " + cv::format("%.2f", det.confidence);
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        cv::Rect label_bg(
            box.x,
            std::max(0, box.y - text_size.height - 8),
            text_size.width + 8,
            text_size.height + 8
        );
        label_bg &= cv::Rect(0, 0, annotated.cols, annotated.rows);
        cv::rectangle(annotated, label_bg, color, cv::FILLED);
        cv::putText(
            annotated,
            label,
            cv::Point(label_bg.x + 4, label_bg.y + text_size.height + 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(20, 20, 20),
            1,
            cv::LINE_AA
        );
    }
    return annotated;
}
