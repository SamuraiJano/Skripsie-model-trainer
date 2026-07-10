#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#include "ApiServer.hpp"

namespace fs = std::filesystem;

// Command-line configuration ------------------------------------------------
struct Args {
    int port = 8080;
    int camera = 0;
    int imgsz = 640;
    float conf = 0.25f;
    float iou = 0.50f;
    int frame_skip = 2;
    bool run_both_models = false;
    int cpu_threads = 4;
    int camera_width = 640;
    int camera_height = 480;
    double camera_fps = 15.0;
    fs::path rgb_model = "models/rgb_yolo_seg.onnx";
    fs::path hsv_model = "models/hsv_yolo_seg.onnx";
    fs::path output_dir = "output_frames";
    fs::path log_dir = "logs";
    cv::Rect roi;
};

// CLI parsing ---------------------------------------------------------------
void printUsage() {
    std::cout
        << "naartjie_backend options:\n"
        << "  --port <int>          API port, default 8080\n"
        << "  --camera <int>        OpenCV camera index, default 0\n"
        << "  --imgsz <int>         YOLO input size, default 640\n"
        << "  --conf <float>        Confidence threshold, default 0.25\n"
        << "  --iou <float>         NMS IoU threshold, default 0.50\n"
        << "  --frame-skip <int>    Run inference every Nth camera frame, default 2\n"
        << "  --run-both-models     Run RGB and HSV on every inference frame; default is active model only\n"
        << "  --threads <int>       ONNX Runtime intra-op CPU threads, default 4\n"
        << "  --camera-width <int>  Requested camera width, default 640\n"
        << "  --camera-height <int> Requested camera height, default 480\n"
        << "  --camera-fps <float>  Requested camera FPS, default 15\n"
        << "  --rgb-model <path>    RGB ONNX model path\n"
        << "  --hsv-model <path>    HSV ONNX model path\n"
        << "  --output-dir <path>   Annotated frame directory\n"
        << "  --log-dir <path>      CSV log directory\n"
        << "  --roi x,y,w,h         Optional crop ROI before resizing\n";
}

cv::Rect parseRoi(const std::string& value) {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    if (std::sscanf(value.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) != 4) {
        throw std::invalid_argument("--roi must be formatted as x,y,w,h");
    }
    return cv::Rect(x, y, w, h);
}

// OpenCV runtime tuning -----------------------------------------------------
void configureOpenCvRuntime(int cpu_threads) {
    cv::setUseOptimized(true);
    cv::setNumThreads(std::max(1, cpu_threads));
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
    }

    std::cout
        << "OpenCV " << CV_VERSION
        << " optimized=" << (cv::useOptimized() ? "on" : "off")
        << " threads=" << cv::getNumThreads()
        << " opencl=" << (cv::ocl::useOpenCL() ? "on" : "off")
        << "\n";
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            printUsage();
            std::exit(0);
        }
        if (key == "--run-both-models") {
            args.run_both_models = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--port") args.port = std::stoi(value);
        else if (key == "--camera") args.camera = std::stoi(value);
        else if (key == "--imgsz") args.imgsz = std::stoi(value);
        else if (key == "--conf") args.conf = std::stof(value);
        else if (key == "--iou") args.iou = std::stof(value);
        else if (key == "--frame-skip") args.frame_skip = std::stoi(value);
        else if (key == "--threads") args.cpu_threads = std::stoi(value);
        else if (key == "--camera-width") args.camera_width = std::stoi(value);
        else if (key == "--camera-height") args.camera_height = std::stoi(value);
        else if (key == "--camera-fps") args.camera_fps = std::stod(value);
        else if (key == "--rgb-model") args.rgb_model = value;
        else if (key == "--hsv-model") args.hsv_model = value;
        else if (key == "--output-dir") args.output_dir = value;
        else if (key == "--log-dir") args.log_dir = value;
        else if (key == "--roi") args.roi = parseRoi(value);
        else throw std::invalid_argument("Unknown option: " + key);
    }
    return args;
}

// Program entry -------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        configureOpenCvRuntime(args.cpu_threads);
        ApiServer server(
            args.port,
            args.camera,
            args.output_dir,
            args.log_dir,
            args.rgb_model,
            args.hsv_model,
            args.imgsz,
            args.conf,
            args.iou,
            args.roi,
            args.frame_skip,
            args.run_both_models,
            args.cpu_threads,
            args.camera_width,
            args.camera_height,
            args.camera_fps
        );
        std::cout << "Naartjie defect background service listening on http://localhost:" << args.port << "\n";
        server.run();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
