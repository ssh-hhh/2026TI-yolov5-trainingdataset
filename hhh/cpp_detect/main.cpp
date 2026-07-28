#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// ===================== Config =====================
const std::string MODEL_PATH = "D:/Edge/Elcetronics competition/yolov5/runs/train/steel_ball_v1/weights/best.onnx";
const int    IMG_SZ      = 320;
const float  CONF_THRES  = 0.55f;
const float  IOU_THRES   = 0.45f;
const int    SKIP_FRAMES = 1;    // 1=no skip (320 is fast enough)

struct Detection { float x1, y1, x2, y2, conf; };

cv::Mat preprocess(const cv::Mat& frame, float& ratio, cv::Mat& padded) {
    int h = frame.rows, w = frame.cols;
    ratio = (float)IMG_SZ / std::max(w, h);

    int nw = (int)(w * ratio), nh = (int)(h * ratio);
    int pw = IMG_SZ - nw, ph = IMG_SZ - nh;

    if (padded.empty())
        padded.create(IMG_SZ, IMG_SZ, CV_8UC3);

    cv::Mat roi = padded(cv::Rect(0, 0, nw, nh));
    if (nw == w && nh == h)        frame.copyTo(roi);                                        // 1:1 -> memcpy
    else if (ratio < 1.f)          cv::resize(frame, roi, roi.size(), 0, 0, cv::INTER_AREA);   // downscale
    else                           cv::resize(frame, roi, roi.size(), 0, 0, cv::INTER_LINEAR); // upscale

    if (ph) padded.rowRange(nh, IMG_SZ).setTo(cv::Scalar(114, 114, 114));
    if (pw) padded.colRange(nw, IMG_SZ).setTo(cv::Scalar(114, 114, 114));

    return cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(),
                                  cv::Scalar(), true, false);
}

void decode(cv::Mat& out, float ratio, int fw, int fh, std::vector<Detection>& dets) {
    float* d = (float*)out.data;
    int rows = out.size[1], cols = out.size[2];
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < rows; ++i) {
        float conf = d[i * cols + 4];
        if (conf < CONF_THRES) continue;
        float cx = d[i * cols + 0], cy = d[i * cols + 1];
        float bw = d[i * cols + 2], bh = d[i * cols + 3];

        float x1 = (cx - bw / 2.f) / ratio;
        float y1 = (cy - bh / 2.f) / ratio;
        float x2 = (cx + bw / 2.f) / ratio;
        float y2 = (cy + bh / 2.f) / ratio;

        x1 = std::max(0.f, std::min(x1, (float)fw));
        y1 = std::max(0.f, std::min(y1, (float)fh));
        x2 = std::max(0.f, std::min(x2, (float)fw));
        y2 = std::max(0.f, std::min(y2, (float)fh));
        if (x2 <= x1 || y2 <= y1) continue;

        boxes.emplace_back((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
        scores.push_back(conf);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRES, IOU_THRES, idx);
    for (int i : idx) {
        dets.push_back({(float)boxes[i].x, (float)boxes[i].y,
                        (float)(boxes[i].x + boxes[i].width),
                        (float)(boxes[i].y + boxes[i].height), scores[i]});
    }
}

int main() {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_PATH);
    std::cout << "ONNX loaded | CPU mode" << std::endl;

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) { std::cerr << "Camera fail" << std::endl; return -1; }

    int fc = 0, frameCnt = 0;
    double fps = 0, lastMs = 0;
    auto ft = std::chrono::steady_clock::now();
    cv::Mat frame;
    cv::Mat padded;   // reusable 640x640 letterbox buffer (allocated lazily)
    std::vector<Detection> lastDets;

    while (cap.read(frame)) {
        if (frameCnt % SKIP_FRAMES == 0) {
            auto t0 = std::chrono::steady_clock::now();
            float ratio;
            cv::Mat blob = preprocess(frame, ratio, padded);
            net.setInput(blob);
            cv::Mat out = net.forward();

            lastDets.clear();
            decode(out, ratio, frame.cols, frame.rows, lastDets);

            lastMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
        }
        ++frameCnt;

        for (auto& d : lastDets) {
            cv::rectangle(frame, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2),
                          cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("ball %.2f", d.conf),
                        cv::Point(d.x1 + 2, d.y1 - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        if (++fc % 10 == 0) {
            auto now = std::chrono::steady_clock::now();
            fps = 10.0 / std::chrono::duration<double>(now - ft).count();
            ft = now;
        }

        cv::putText(frame, cv::format("FPS:%.1f | %dms | Det:%zu | skip:%d",
                     fps, (int)lastMs, lastDets.size(), SKIP_FRAMES),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);
        cv::imshow("SteelBall C++", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
