#pragma once

#include <filesystem>
#include <mutex>
#include <string>

#include "Preprocessor.hpp"
#include "YoloSegONNX.hpp"

class ResultLogger {
public:
    explicit ResultLogger(std::filesystem::path log_dir);

    void log(
        const std::string& timestamp,
        ModelType model,
        const TimingInfo& timing,
        size_t detection_count,
        bool major_defect_detected
    );

    static std::string timestampNow();

private:
    std::mutex mutex_;
    std::filesystem::path csv_path_;
};
