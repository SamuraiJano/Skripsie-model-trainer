#include "ResultLogger.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

ResultLogger::ResultLogger(fs::path log_dir) {
    fs::create_directories(log_dir);
    csv_path_ = log_dir / "metrics.csv";
    if (!fs::exists(csv_path_) || fs::file_size(csv_path_) == 0) {
        std::ofstream out(csv_path_, std::ios::out);
        out << "timestamp,model,fps,preprocess_ms,hsv_ms,inference_ms,total_ms,"
               "detection_count,major_defect_detected\n";
    }
}

std::string ResultLogger::timestampNow() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

void ResultLogger::log(
    const std::string& timestamp,
    ModelType model,
    const TimingInfo& timing,
    size_t detection_count,
    bool major_defect_detected
) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(csv_path_, std::ios::app);
    out << timestamp << ','
        << modelTypeToString(model) << ','
        << timing.fps << ','
        << timing.preprocess_ms << ','
        << timing.hsv_ms << ','
        << timing.inference_ms << ','
        << timing.total_ms << ','
        << detection_count << ','
        << (major_defect_detected ? 1 : 0) << '\n';
}
