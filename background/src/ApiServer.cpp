#include "ApiServer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/ocl.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;

namespace {
constexpr int kCameraProbeCount = 8;

// Timing and state-to-JSON helpers -----------------------------------------
double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void addCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

json timingToJson(const TimingInfo& timing) {
    return {
        {"fps", timing.fps},
        {"preprocess_ms", timing.preprocess_ms},
        {"hsv_ms", timing.hsv_ms},
        {"inference_ms", timing.inference_ms},
        {"total_ms", timing.total_ms},
    };
}

json detectionToJson(const Detection& det) {
    return {
        {"class_id", det.class_id},
        {"class_name", det.class_name},
        {"confidence", det.confidence},
        {"box", {
            {"x1", det.box.x},
            {"y1", det.box.y},
            {"x2", det.box.x + det.box.width},
            {"y2", det.box.y + det.box.height},
        }},
    };
}

json detectionsToJson(const std::vector<Detection>& source) {
    json detections = json::array();
    for (const auto& det : source) {
        detections.push_back(detectionToJson(det));
    }
    return detections;
}

json modelFeedToJson(const ModelFeedState& feed) {
    return {
        {"loaded", feed.loaded},
        {"has_result", feed.has_result},
        {"message", feed.message},
        {"frame_index", feed.frame_index},
        {"updated_at", feed.updated_at},
        {"major_defect_detected", feed.major_defect_detected},
        {"timing", timingToJson(feed.timing)},
        {"detections", detectionsToJson(feed.detections)},
    };
}

const ModelFeedState& feedFor(const DashboardState& state, ModelType model) {
    return model == ModelType::Hsv ? state.hsv : state.rgb;
}

ModelFeedState& feedFor(DashboardState& state, ModelType model) {
    return model == ModelType::Hsv ? state.hsv : state.rgb;
}

void clearFeed(ModelFeedState& feed, bool loaded, const std::string& message) {
    feed.loaded = loaded;
    feed.has_result = false;
    feed.message = message;
    feed.timing = TimingInfo{};
    feed.detections.clear();
    feed.major_defect_detected = false;
}

void pauseFeed(ModelFeedState& feed, bool loaded, const std::string& message, uint64_t frame_index, const std::string& timestamp) {
    clearFeed(feed, loaded, message);
    feed.frame_index = frame_index;
    feed.updated_at = timestamp;
}

void storeFeedResult(
    ModelFeedState& feed,
    const std::string& message,
    TimingInfo timing,
    std::vector<Detection>&& detections,
    bool major_defect_detected,
    uint64_t frame_index,
    const std::string& timestamp
) {
    feed.loaded = true;
    feed.has_result = true;
    feed.message = message;
    feed.timing = timing;
    feed.detections = std::move(detections);
    feed.major_defect_detected = major_defect_detected;
    feed.frame_index = frame_index;
    feed.updated_at = timestamp;
}

json stateToJson(const DashboardState& state) {
    const auto& active = feedFor(state, state.active_model);
    return {
        {"running", state.running},
        {"camera_running", state.camera_running},
        {"inference_running", state.inference_running},
        {"active_model", modelTypeToString(state.active_model)},
        {"camera_index", state.camera_index},
        {"has_camera_frame", state.has_camera_frame},
        {"has_result", active.has_result},
        {"frame_index", state.frame_index},
        {"updated_at", state.updated_at},
        {"major_defect_detected", state.rgb.major_defect_detected || state.hsv.major_defect_detected},
        {"timing", timingToJson(active.timing)},
        {"detections", detectionsToJson(active.detections)},
        {"models", {
            {"rgb", modelFeedToJson(state.rgb)},
            {"hsv", modelFeedToJson(state.hsv)},
        }},
    };
}

// HTTP response helpers -----------------------------------------------------
json cameraListToJson(int selected_camera, bool camera_running) {
    json cameras = json::array();
    for (int index = 0; index < kCameraProbeCount; ++index) {
        bool available = index == selected_camera;
        if (!camera_running) {
            cv::VideoCapture probe;
            available = probe.open(index);
            probe.release();
        }
        if (available) {
            cameras.push_back({
                {"index", index},
                {"label", "Camera " + std::to_string(index)},
                {"selected", index == selected_camera},
            });
        }
    }
    if (cameras.empty()) {
        cameras.push_back({
            {"index", selected_camera},
            {"label", "Camera " + std::to_string(selected_camera)},
            {"selected", true},
        });
    }
    return {
        {"selected_camera", selected_camera},
        {"camera_running", camera_running},
        {"cameras", cameras},
    };
}

void jsonResponse(httplib::Response& res, const json& body, int status = 200) {
    res.status = status;
    addCors(res);
    res.set_content(body.dump(2), "application/json");
}

void imageResponse(httplib::Response& res, const fs::path& image_path, const std::string& missing_message) {
    addCors(res);
    if (!fs::exists(image_path)) {
        res.status = 404;
        res.set_content(missing_message, "text/plain");
        return;
    }

    std::ifstream in(image_path, std::ios::binary | std::ios::ate);
    if (!in) {
        res.status = 404;
        res.set_content(missing_message, "text/plain");
        return;
    }
    const std::streamoff size = static_cast<std::streamoff>(in.tellg());
    std::string bytes(static_cast<size_t>(std::max<std::streamoff>(0, size)), '\0');
    in.seekg(0);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    res.set_content(bytes, "image/jpeg");
}
}

// Construction and lifecycle -----------------------------------------------
ApiServer::ApiServer(
    int port,
    int camera_index,
    fs::path output_dir,
    fs::path log_dir,
    fs::path rgb_model,
    fs::path hsv_model,
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
) : port_(port),
    camera_index_(camera_index),
    output_dir_(std::move(output_dir)),
    latest_camera_path_(output_dir_ / "latest-camera.jpg"),
    latest_rgb_frame_path_(output_dir_ / "latest-rgb.jpg"),
    latest_hsv_frame_path_(output_dir_ / "latest-hsv.jpg"),
    frame_skip_(std::max(1, frame_skip)),
    run_both_models_(run_both_models),
    camera_width_(camera_width),
    camera_height_(camera_height),
    camera_fps_(camera_fps),
    preprocessor_(input_size, roi),
    detector_(std::move(rgb_model), std::move(hsv_model), input_size, conf_threshold, nms_threshold, cpu_threads),
    logger_(std::move(log_dir)) {
    fs::create_directories(output_dir_);
    latest_.rgb.loaded = detector_.hasModel(ModelType::Rgb);
    latest_.rgb.message = detector_.modelMessage(ModelType::Rgb);
    latest_.hsv.loaded = detector_.hasModel(ModelType::Hsv);
    latest_.hsv.message = detector_.modelMessage(ModelType::Hsv);
    latest_.camera_index = camera_index;
}

ApiServer::~ApiServer() {
    stopCamera();
}

void ApiServer::run() {
    httplib::Server server;

    // Routes are kept intentionally simple: JSON endpoints expose state and
    // commands, while JPEG endpoints serve the latest camera/model frames.
    server.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        addCors(res);
        res.status = 204;
    });

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        jsonResponse(res, {
            {"ok", true},
            {"opencv", {
                {"version", CV_VERSION},
                {"optimized", cv::useOptimized()},
                {"threads", cv::getNumThreads()},
                {"opencl", cv::ocl::useOpenCL()},
            }},
        });
    });

    server.Post("/start", [this](const httplib::Request&, httplib::Response& res) {
        std::string message;
        bool ok = startCamera(message);
        if (ok || message == "camera already running") {
            ok = startInference(message);
        }
        jsonResponse(res, {{"ok", ok}, {"message", message}, {"status", stateToJson(latestState())}}, ok ? 200 : 409);
    });

    server.Post("/stop", [this](const httplib::Request&, httplib::Response& res) {
        stopCamera();
        jsonResponse(res, {{"ok", true}, {"status", stateToJson(latestState())}});
    });

    server.Get("/cameras", [this](const httplib::Request&, httplib::Response& res) {
        jsonResponse(res, cameraListToJson(camera_index_.load(), camera_running_.load()));
    });

    server.Post("/set-camera", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = json::parse(req.body.empty() ? "{}" : req.body);
            std::string message;
            const bool ok = setCameraIndex(body.value("camera", camera_index_.load()), message);
            jsonResponse(res, {{"ok", ok}, {"message", message}, {"status", stateToJson(latestState())}}, ok ? 200 : 409);
        } catch (const std::exception& exc) {
            jsonResponse(res, {{"ok", false}, {"error", exc.what()}}, 400);
        }
    });

    server.Post("/start-camera", [this](const httplib::Request&, httplib::Response& res) {
        std::string message;
        const bool ok = startCamera(message);
        jsonResponse(res, {{"ok", ok}, {"message", message}, {"status", stateToJson(latestState())}}, ok ? 200 : 409);
    });

    server.Post("/stop-camera", [this](const httplib::Request&, httplib::Response& res) {
        stopCamera();
        jsonResponse(res, {{"ok", true}, {"status", stateToJson(latestState())}});
    });

    server.Post("/start-inference", [this](const httplib::Request&, httplib::Response& res) {
        std::string message;
        const bool ok = startInference(message);
        jsonResponse(res, {{"ok", ok}, {"message", message}, {"status", stateToJson(latestState())}}, ok ? 200 : 409);
    });

    server.Post("/stop-inference", [this](const httplib::Request&, httplib::Response& res) {
        stopInference();
        jsonResponse(res, {{"ok", true}, {"status", stateToJson(latestState())}});
    });

    server.Post("/set-model", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = json::parse(req.body.empty() ? "{}" : req.body);
            const auto model = modelTypeFromString(body.value("model", "rgb"));
            setActiveModel(model);
            jsonResponse(res, {{"ok", true}, {"active_model", modelTypeToString(model)}});
        } catch (const std::exception& exc) {
            jsonResponse(res, {{"ok", false}, {"error", exc.what()}}, 400);
        }
    });

    server.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
        jsonResponse(res, stateToJson(latestState()));
    });

    server.Get("/latest-result", [this](const httplib::Request&, httplib::Response& res) {
        jsonResponse(res, stateToJson(latestState()));
    });

    server.Get("/camera-frame.jpg", [this](const httplib::Request&, httplib::Response& res) {
        imageResponse(res, latest_camera_path_, "camera frame not found");
    });

    server.Get("/latest-frame.jpg", [this](const httplib::Request&, httplib::Response& res) {
        imageResponse(res, latest_camera_path_, "camera frame not found");
    });

    server.Get("/rgb-frame.jpg", [this](const httplib::Request&, httplib::Response& res) {
        imageResponse(res, latest_rgb_frame_path_, "RGB model frame not found");
    });

    server.Get("/hsv-frame.jpg", [this](const httplib::Request&, httplib::Response& res) {
        imageResponse(res, latest_hsv_frame_path_, "HSV model frame not found");
    });

    if (!server.listen("0.0.0.0", port_)) {
        throw std::runtime_error("Failed to listen on port " + std::to_string(port_));
    }
}

// Camera and inference controls --------------------------------------------
bool ApiServer::startCamera(std::string& message) {
    bool expected = false;
    if (!camera_running_.compare_exchange_strong(expected, true)) {
        message = "camera already running";
        return false;
    }

    const int selected_camera = camera_index_.load();
    if (!camera_.open(selected_camera, camera_width_, camera_height_, camera_fps_)) {
        camera_running_.store(false);
        message = "could not open camera index " + std::to_string(selected_camera);
        return false;
    }

    if (worker_.joinable()) {
        worker_.join();
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest_.running = true;
        latest_.camera_running = true;
        latest_.camera_index = selected_camera;
        latest_.active_model = activeModel();
        latest_.rgb.loaded = detector_.hasModel(ModelType::Rgb);
        latest_.rgb.message = detector_.modelMessage(ModelType::Rgb);
        latest_.hsv.loaded = detector_.hasModel(ModelType::Hsv);
        latest_.hsv.message = detector_.modelMessage(ModelType::Hsv);
    }
    worker_ = std::thread(&ApiServer::workerLoop, this);
    message = "camera started";
    return true;
}

void ApiServer::stopCamera() {
    inference_running_.store(false);
    camera_running_.store(false);
    if (worker_.joinable()) {
        worker_.join();
    }
    camera_.release();
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_.running = false;
    latest_.camera_running = false;
    latest_.inference_running = false;
}

bool ApiServer::startInference(std::string& message) {
    if (!camera_running_.load()) {
        message = "start the camera before starting inference";
        return false;
    }
    inference_running_.store(true);
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_.inference_running = true;
    latest_.running = true;
    message = "inference started";
    return true;
}

void ApiServer::stopInference() {
    inference_running_.store(false);
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_.inference_running = false;
}

bool ApiServer::setCameraIndex(int camera_index, std::string& message) {
    if (camera_index < 0) {
        message = "camera index must be zero or greater";
        return false;
    }
    if (camera_running_.load()) {
        message = "stop the camera before changing camera";
        return false;
    }
    camera_index_.store(camera_index);
    std::error_code ec;
    fs::remove(latest_camera_path_, ec);
    fs::remove(latest_rgb_frame_path_, ec);
    fs::remove(latest_hsv_frame_path_, ec);
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_.camera_index = camera_index;
    latest_.has_camera_frame = false;
    latest_.frame_index = 0;
    latest_.updated_at.clear();
    message = "camera selected";
    return true;
}

void ApiServer::setActiveModel(ModelType model) {
    active_model_.store(static_cast<int>(model));
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_.active_model = model;
}

ModelType ApiServer::activeModel() const {
    return active_model_.load() == static_cast<int>(ModelType::Hsv) ? ModelType::Hsv : ModelType::Rgb;
}

DashboardState ApiServer::latestState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    DashboardState state = latest_;
    state.camera_running = camera_running_.load();
    state.inference_running = inference_running_.load();
    state.running = state.camera_running;
    state.active_model = activeModel();
    state.camera_index = camera_index_.load();
    return state;
}

// Frame processing ----------------------------------------------------------
bool ApiServer::hasMajorDefect(const std::vector<Detection>& detections) {
    return std::any_of(detections.begin(), detections.end(), [](const Detection& det) {
        return det.class_id == 2;
    });
}

void ApiServer::workerLoop() {
    uint64_t last_camera_frame_id = 0;
    while (camera_running_.load()) {
        cv::Mat frame;
        uint64_t camera_frame_id = 0;
        if (!camera_.read(frame, camera_frame_id) || camera_frame_id == last_camera_frame_id) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        last_camera_frame_id = camera_frame_id;

        const std::string timestamp = ResultLogger::timestampNow();
        cv::imwrite(latest_camera_path_.string(), frame);

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            latest_.running = true;
            latest_.camera_running = true;
            latest_.inference_running = inference_running_.load();
            latest_.camera_index = camera_index_.load();
            latest_.active_model = activeModel();
            latest_.has_camera_frame = true;
            latest_.frame_index = camera_frame_id;
            latest_.updated_at = timestamp;
        }

        if (!inference_running_.load()) {
            continue;
        }

        if (frame_skip_ > 1 && (camera_frame_id % static_cast<uint64_t>(frame_skip_)) != 0) {
            continue;
        }

        std::vector<ModelType> models;
        if (run_both_models_) {
            models = {ModelType::Rgb, ModelType::Hsv};
        } else {
            const ModelType selected = activeModel();
            models = {selected};
            const ModelType paused = selected == ModelType::Rgb ? ModelType::Hsv : ModelType::Rgb;
            const bool paused_loaded = detector_.hasModel(paused);
            const std::string paused_message = paused_loaded
                ? "Paused in Pi 4 single-model mode; select this model or start with --run-both-models."
                : detector_.modelMessage(paused);

            std::lock_guard<std::mutex> lock(state_mutex_);
            pauseFeed(feedFor(latest_, paused), paused_loaded, paused_message, camera_frame_id, timestamp);
        }

        for (const ModelType model : models) {
            ModelFeedState unloaded;
            unloaded.loaded = detector_.hasModel(model);
            unloaded.message = detector_.modelMessage(model);

            if (!unloaded.loaded) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                clearFeed(feedFor(latest_, model), false, unloaded.message);
                continue;
            }

            try {
                const auto total_start = Clock::now();
                auto preprocessed = preprocessor_.process(frame, model);
                auto inference = detector_.run(model, preprocessed);
                const double total_ms = elapsedMs(total_start, Clock::now());

                TimingInfo timing;
                timing.preprocess_ms = preprocessed.preprocess_ms;
                timing.hsv_ms = preprocessed.hsv_ms;
                timing.inference_ms = inference.inference_ms;
                timing.total_ms = total_ms;
                timing.fps = total_ms > 0.0 ? 1000.0 / total_ms : 0.0;

                const bool major = hasMajorDefect(inference.detections);
                const auto& frame_path = model == ModelType::Hsv ? latest_hsv_frame_path_ : latest_rgb_frame_path_;
                cv::imwrite(frame_path.string(), inference.annotated);
                logger_.log(timestamp, model, timing, inference.detections.size(), major);

                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    storeFeedResult(
                        feedFor(latest_, model),
                        detector_.modelMessage(model),
                        timing,
                        std::move(inference.detections),
                        major,
                        camera_frame_id,
                        timestamp
                    );
                }
            } catch (const std::exception& exc) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                clearFeed(feedFor(latest_, model), true, std::string("Inference error: ") + exc.what());
            }
        }
    }
}
