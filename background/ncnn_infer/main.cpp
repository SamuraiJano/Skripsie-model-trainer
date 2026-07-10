#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <net.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct Args {
    fs::path model_dir;
    fs::path input_dir;
    fs::path output_dir;
    std::string input_name = "in0";
    std::string det_output = "out0";
    std::string proto_output = "out1";
    std::string channels = "rgb";
    std::vector<std::string> class_names = {"class_1", "class_2", "class_3"};
    float conf = 0.25f;
    float nms_iou = 0.50f;
    int imgsz = 640;
    int source_width = 640;
    int source_height = 480;
    int threads = 4;
};

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct Detection {
    int class_id = 0;
    float conf = 0.0f;
    cv::Rect2f box;
    std::vector<float> mask_coeffs;
    cv::Mat mask;
};

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static Args parse_args(int argc, char** argv) {
    Args args;
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected argument: " + key);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        kv[key.substr(2)] = argv[++i];
    }
    if (kv.count("model")) args.model_dir = kv["model"];
    if (kv.count("input")) args.input_dir = kv["input"];
    if (kv.count("output")) args.output_dir = kv["output"];
    if (kv.count("input-name")) args.input_name = kv["input-name"];
    if (kv.count("det-output")) args.det_output = kv["det-output"];
    if (kv.count("proto-output")) args.proto_output = kv["proto-output"];
    if (kv.count("channels")) args.channels = kv["channels"];
    if (kv.count("classes")) args.class_names = split(kv["classes"], ',');
    if (kv.count("conf")) args.conf = std::stof(kv["conf"]);
    if (kv.count("iou")) args.nms_iou = std::stof(kv["iou"]);
    if (kv.count("imgsz")) args.imgsz = std::stoi(kv["imgsz"]);
    if (kv.count("source-width")) args.source_width = std::stoi(kv["source-width"]);
    if (kv.count("source-height")) args.source_height = std::stoi(kv["source-height"]);
    if (kv.count("threads")) args.threads = std::stoi(kv["threads"]);

    if (args.model_dir.empty() || args.input_dir.empty() || args.output_dir.empty()) {
        throw std::runtime_error("Required: --model <dir> --input <image_dir> --output <output_dir>");
    }
    if (args.channels != "rgb" && args.channels != "rgbhsv") {
        throw std::runtime_error("--channels must be rgb or rgbhsv");
    }
    return args;
}

static std::vector<fs::path> list_images(const fs::path& dir) {
    std::vector<fs::path> files;
    const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"};
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) != exts.end()) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static cv::Mat resize_source_720p(const cv::Mat& image, int width, int height) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    return resized;
}

static cv::Mat letterbox(const cv::Mat& image, int size, LetterboxInfo& info) {
    int w = image.cols;
    int h = image.rows;
    info.scale = std::min(static_cast<float>(size) / w, static_cast<float>(size) / h);
    int new_w = static_cast<int>(std::round(w * info.scale));
    int new_h = static_cast<int>(std::round(h * info.scale));
    info.pad_x = (size - new_w) / 2;
    info.pad_y = (size - new_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, new_w, new_h)));
    return canvas;
}

static ncnn::Mat make_input_tensor(const cv::Mat& bgr_letterboxed, const std::string& channels) {
    cv::Mat rgb;
    cv::cvtColor(bgr_letterboxed, rgb, cv::COLOR_BGR2RGB);
    int c_count = channels == "rgbhsv" ? 6 : 3;
    ncnn::Mat input(bgr_letterboxed.cols, bgr_letterboxed.rows, c_count, static_cast<size_t>(4u));

    std::vector<cv::Mat> planes;
    cv::split(rgb, planes);
    if (channels == "rgbhsv") {
        cv::Mat hsv;
        cv::cvtColor(bgr_letterboxed, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_planes;
        cv::split(hsv, hsv_planes);
        hsv_planes[0].convertTo(hsv_planes[0], CV_32F, 255.0 / 179.0);
        hsv_planes[0].convertTo(hsv_planes[0], CV_8U);
        planes.insert(planes.end(), hsv_planes.begin(), hsv_planes.end());
    }

    for (int c = 0; c < c_count; ++c) {
        float* dst = input.channel(c);
        for (int y = 0; y < bgr_letterboxed.rows; ++y) {
            const uchar* src = planes[c].ptr<uchar>(y);
            for (int x = 0; x < bgr_letterboxed.cols; ++x) {
                dst[y * bgr_letterboxed.cols + x] = static_cast<float>(src[x]) / 255.0f;
            }
        }
    }
    return input;
}

static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

static std::vector<int> nms(const std::vector<Detection>& detections, float threshold) {
    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[a].conf > detections[b].conf;
    });
    std::vector<int> keep;
    std::vector<bool> removed(detections.size(), false);
    for (int idx : order) {
        if (removed[idx]) continue;
        keep.push_back(idx);
        for (int other : order) {
            if (idx == other || removed[other]) continue;
            if (detections[idx].class_id == detections[other].class_id && iou(detections[idx].box, detections[other].box) > threshold) {
                removed[other] = true;
            }
        }
    }
    return keep;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float read_feature(const ncnn::Mat& det, int row, int col, bool transposed) {
    if (!transposed) {
        return det.row(row)[col];
    }
    return det.row(col)[row];
}

static std::vector<Detection> decode_detections(
    const ncnn::Mat& det,
    int class_count,
    int input_size,
    const LetterboxInfo& lb,
    const cv::Size& source_size,
    float conf_threshold
) {
    int rows = det.h;
    int cols = det.w;
    bool transposed = false;

    if (rows > 0 && cols > rows && rows <= 256) {
        transposed = true;
        std::swap(rows, cols);
    }

    std::vector<Detection> detections;
    for (int r = 0; r < rows; ++r) {
        if (cols == 6) {
            float conf = read_feature(det, r, 4, transposed);
            if (conf < conf_threshold) continue;
            int class_id = static_cast<int>(read_feature(det, r, 5, transposed));
            cv::Rect2f box(
                read_feature(det, r, 0, transposed),
                read_feature(det, r, 1, transposed),
                read_feature(det, r, 2, transposed) - read_feature(det, r, 0, transposed),
                read_feature(det, r, 3, transposed) - read_feature(det, r, 1, transposed)
            );
            box.x = (box.x - lb.pad_x) / lb.scale;
            box.y = (box.y - lb.pad_y) / lb.scale;
            box.width /= lb.scale;
            box.height /= lb.scale;
            detections.push_back({class_id, conf, box, {}, cv::Mat()});
            continue;
        }

        if (cols < 4 + class_count) continue;
        int best_class = 0;
        float best_score = 0.0f;
        for (int c = 0; c < class_count; ++c) {
            float score = read_feature(det, r, 4 + c, transposed);
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        if (best_score < conf_threshold) continue;

        float cx = read_feature(det, r, 0, transposed);
        float cy = read_feature(det, r, 1, transposed);
        float w = read_feature(det, r, 2, transposed);
        float h = read_feature(det, r, 3, transposed);
        float x1 = (cx - w * 0.5f - lb.pad_x) / lb.scale;
        float y1 = (cy - h * 0.5f - lb.pad_y) / lb.scale;
        float x2 = (cx + w * 0.5f - lb.pad_x) / lb.scale;
        float y2 = (cy + h * 0.5f - lb.pad_y) / lb.scale;
        x1 = std::clamp(x1, 0.0f, static_cast<float>(source_size.width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(source_size.height - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(source_size.width - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(source_size.height - 1));

        Detection d;
        d.class_id = best_class;
        d.conf = best_score;
        d.box = cv::Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));
        for (int k = 4 + class_count; k < cols; ++k) {
            d.mask_coeffs.push_back(read_feature(det, r, k, transposed));
        }
        detections.push_back(std::move(d));
    }
    return detections;
}

static void build_masks(
    std::vector<Detection>& detections,
    const ncnn::Mat& proto,
    int input_size,
    const LetterboxInfo& lb,
    const cv::Size& source_size
) {
    if (proto.empty() || proto.c <= 0) return;
    int mask_dim = proto.c;
    int mask_w = proto.w;
    int mask_h = proto.h;
    for (auto& det : detections) {
        if (static_cast<int>(det.mask_coeffs.size()) < mask_dim) continue;
        cv::Mat low(mask_h, mask_w, CV_32F, cv::Scalar(0));
        for (int k = 0; k < mask_dim; ++k) {
            const float* channel = proto.channel(k);
            for (int y = 0; y < mask_h; ++y) {
                float* dst = low.ptr<float>(y);
                for (int x = 0; x < mask_w; ++x) {
                    dst[x] += det.mask_coeffs[k] * channel[y * mask_w + x];
                }
            }
        }
        for (int y = 0; y < mask_h; ++y) {
            float* row = low.ptr<float>(y);
            for (int x = 0; x < mask_w; ++x) row[x] = sigmoid(row[x]);
        }
        cv::Mat full;
        cv::resize(low, full, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);
        cv::Rect valid(lb.pad_x, lb.pad_y, input_size - 2 * lb.pad_x, input_size - 2 * lb.pad_y);
        valid &= cv::Rect(0, 0, full.cols, full.rows);
        cv::Mat cropped = full(valid).clone();
        cv::resize(cropped, det.mask, source_size, 0, 0, cv::INTER_LINEAR);
        cv::threshold(det.mask, det.mask, 0.5, 255.0, cv::THRESH_BINARY);
        det.mask.convertTo(det.mask, CV_8U);
    }
}

static void draw_detection(cv::Mat& image, const Detection& det, const std::vector<std::string>& names) {
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(40, 180, 255),
        cv::Scalar(70, 220, 90),
        cv::Scalar(220, 80, 180)
    };
    cv::Scalar color = colors[det.class_id % colors.size()];
    if (!det.mask.empty()) {
        cv::Mat overlay = image.clone();
        overlay.setTo(color, det.mask);
        cv::addWeighted(overlay, 0.35, image, 0.65, 0, image);
    }
    cv::Rect box_i(
        static_cast<int>(std::round(det.box.x)),
        static_cast<int>(std::round(det.box.y)),
        static_cast<int>(std::round(det.box.width)),
        static_cast<int>(std::round(det.box.height))
    );
    box_i &= cv::Rect(0, 0, image.cols, image.rows);
    cv::rectangle(image, box_i, color, 2);
    std::string label = (det.class_id < static_cast<int>(names.size()) ? names[det.class_id] : std::to_string(det.class_id));
    label += " " + std::to_string(det.conf).substr(0, 4);
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::Rect bg(box_i.x, std::max(0, box_i.y - text_size.height - 6), text_size.width + 6, text_size.height + 6);
    cv::rectangle(image, bg, color, cv::FILLED);
    cv::putText(image, label, cv::Point(bg.x + 3, bg.y + text_size.height + 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::create_directories(args.output_dir);

        fs::path param_path;
        fs::path bin_path;
        for (const auto& entry : fs::directory_iterator(args.model_dir)) {
            if (entry.path().extension() == ".param") param_path = entry.path();
            if (entry.path().extension() == ".bin") bin_path = entry.path();
        }
        if (param_path.empty() || bin_path.empty()) {
            throw std::runtime_error("Could not find .param and .bin files in " + args.model_dir.string());
        }

        ncnn::Net net;
        net.opt.num_threads = std::max(1, args.threads);
        net.opt.use_vulkan_compute = false;
        net.opt.use_packing_layout = true;
        net.opt.use_winograd_convolution = true;
        net.opt.use_sgemm_convolution = true;
        net.opt.use_int8_inference = true;
        net.opt.use_int8_storage = true;
        if (net.load_param(param_path.string().c_str()) != 0) throw std::runtime_error("Failed to load param file");
        if (net.load_model(bin_path.string().c_str()) != 0) throw std::runtime_error("Failed to load bin file");

        std::ofstream csv(args.output_dir / "detections.csv");
        csv << "image,class_id,class_name,confidence,x1,y1,x2,y2,has_mask,latency_ms\n";

        auto files = list_images(args.input_dir);
        if (files.empty()) throw std::runtime_error("No images found in " + args.input_dir.string());

        for (const auto& image_path : files) {
            cv::Mat raw = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (raw.empty()) {
                std::cerr << "Skipping unreadable image: " << image_path << "\n";
                continue;
            }
            cv::Mat source = resize_source_720p(raw, args.source_width, args.source_height);
            LetterboxInfo lb;
            cv::Mat lb_image = letterbox(source, args.imgsz, lb);
            ncnn::Mat input = make_input_tensor(lb_image, args.channels);

            auto t0 = std::chrono::high_resolution_clock::now();
            ncnn::Extractor ex = net.create_extractor();
            ex.input(args.input_name.c_str(), input);
            ncnn::Mat det_out;
            if (ex.extract(args.det_output.c_str(), det_out) != 0) {
                throw std::runtime_error("Failed to extract detection output blob: " + args.det_output);
            }
            ncnn::Mat proto_out;
            ex.extract(args.proto_output.c_str(), proto_out);

            auto detections = decode_detections(det_out, static_cast<int>(args.class_names.size()), args.imgsz, lb, source.size(), args.conf);
            auto keep = nms(detections, args.nms_iou);
            std::vector<Detection> kept;
            kept.reserve(keep.size());
            for (int idx : keep) kept.push_back(std::move(detections[idx]));
            build_masks(kept, proto_out, args.imgsz, lb, source.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            cv::Mat annotated = source.clone();
            for (const auto& d : kept) {
                draw_detection(annotated, d, args.class_names);
                std::string class_name = d.class_id < static_cast<int>(args.class_names.size()) ? args.class_names[d.class_id] : std::to_string(d.class_id);
                csv << image_path.filename().string() << ","
                    << d.class_id << ","
                    << class_name << ","
                    << d.conf << ","
                    << d.box.x << ","
                    << d.box.y << ","
                    << d.box.x + d.box.width << ","
                    << d.box.y + d.box.height << ","
                    << (!d.mask.empty()) << ","
                    << latency_ms << "\n";
            }
            cv::imwrite((args.output_dir / image_path.filename()).string(), annotated);
            std::cout << image_path.filename().string() << ": " << kept.size() << " detections, " << latency_ms << " ms\n";
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
