#include "CameraManager.hpp"

#include <chrono>

// Lifecycle -----------------------------------------------------------------
CameraManager::~CameraManager() {
    release();
}

bool CameraManager::open(int camera_index, int width, int height, double fps) {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (capture_.isOpened()) {
        return true;
    }
    if (!capture_.open(camera_index)) {
        return false;
    }
    capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (width > 0) {
        capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    }
    if (height > 0) {
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    if (fps > 0.0) {
        capture_.set(cv::CAP_PROP_FPS, fps);
    }

    {
        std::lock_guard<std::mutex> frame_lock(frame_mutex_);
        latest_frame_.release();
        latest_frame_id_ = 0;
    }

    running_.store(true);
    capture_thread_ = std::thread(&CameraManager::captureLoop, this);
    return true;
}

// Latest-frame access -------------------------------------------------------
bool CameraManager::read(cv::Mat& frame) {
    uint64_t ignored = 0;
    return read(frame, ignored);
}

bool CameraManager::read(cv::Mat& frame, uint64_t& frame_id) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (latest_frame_.empty()) {
        return false;
    }
    frame = latest_frame_.clone();
    frame_id = latest_frame_id_;
    return true;
}

void CameraManager::release() {
    running_.store(false);
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (capture_.isOpened()) {
        capture_.release();
    }
    std::lock_guard<std::mutex> frame_lock(frame_mutex_);
    latest_frame_.release();
    latest_frame_id_ = 0;
}

// Capture worker ------------------------------------------------------------
bool CameraManager::isOpen() const {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    return capture_.isOpened();
}

void CameraManager::captureLoop() {
    while (running_.load()) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(capture_mutex_);
            if (!capture_.isOpened() || !capture_.read(frame) || frame.empty()) {
                frame.release();
            }
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = std::move(frame);
            ++latest_frame_id_;
        }
    }
}
