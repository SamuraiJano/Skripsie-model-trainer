#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CameraManager.hpp"
#include "Preprocessor.hpp"
#include "ResultLogger.hpp"
#include "YoloSegONNX.hpp"

struct LatestState {
    bool running = false;
    bool camera_running = false;
    bool inference_running = false;
    ModelType active_model = ModelType::Rgb;
    int camera_index = 0;
    bool has_camera_frame = false;
    uint64_t frame_index = 0;
    std::string updated_at;
};

struct ModelFeedState {
    bool loaded = false;
    bool has_result = false;
    std::string message;
    TimingInfo timing;
    std::vector<Detection> detections;
    bool major_defect_detected = false;
    uint64_t frame_index = 0;
    std::string updated_at;
};

struct DashboardState : LatestState {
    ModelFeedState rgb;
    ModelFeedState hsv;
};

class ApiServer {
public:
    ApiServer(
        int port,
        int camera_index,
        std::filesystem::path output_dir,
        std::filesystem::path log_dir,
        std::filesystem::path rgb_model,
        std::filesystem::path hsv_model,
        int input_size,
        float conf_threshold,
        float nms_threshold,
        cv::Rect roi,
        int frame_skip,
        bool run_both_models,
        int cpu_threads,
        int camera_width,
        int camera_height,
        double camera_fps
    );
    ~ApiServer();

    void run();

private:
    bool startCamera(std::string& message);
    void stopCamera();
    bool startInference(std::string& message);
    void stopInference();
    bool setCameraIndex(int camera_index, std::string& message);
    void workerLoop();
    void setActiveModel(ModelType model);
    ModelType activeModel() const;
    DashboardState latestState() const;

    static bool hasMajorDefect(const std::vector<Detection>& detections);

    int port_;
    std::atomic<int> camera_index_;
    std::filesystem::path output_dir_;
    std::filesystem::path latest_camera_path_;
    std::filesystem::path latest_rgb_frame_path_;
    std::filesystem::path latest_hsv_frame_path_;
    int frame_skip_;
    bool run_both_models_;
    int camera_width_;
    int camera_height_;
    double camera_fps_;

    CameraManager camera_;
    Preprocessor preprocessor_;
    YoloSegONNX detector_;
    ResultLogger logger_;

    std::atomic<bool> camera_running_{false};
    std::atomic<bool> inference_running_{false};
    std::atomic<int> active_model_{0};
    std::thread worker_;

    mutable std::mutex state_mutex_;
    DashboardState latest_;
};
