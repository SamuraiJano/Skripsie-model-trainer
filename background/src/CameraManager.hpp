#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include <opencv2/opencv.hpp>

class CameraManager {
public:
    ~CameraManager();

    bool open(int camera_index, int width = 0, int height = 0, double fps = 0.0);
    bool read(cv::Mat& frame);
    bool read(cv::Mat& frame, uint64_t& frame_id);
    void release();
    bool isOpen() const;

private:
    void captureLoop();

    mutable std::mutex capture_mutex_;
    mutable std::mutex frame_mutex_;
    std::atomic<bool> running_{false};
    std::thread capture_thread_;
    cv::VideoCapture capture_;
    cv::Mat latest_frame_;
    uint64_t latest_frame_id_ = 0;
};
