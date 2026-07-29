#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <libserial/SerialPort.h>
#include <cmath>

using namespace std;
using namespace cv;
using namespace LibSerial;


class SerialPortManager {
public:

    /*串口初始化*/
    SerialPortManager(const std::string& port, LibSerial::BaudRate baud_rate)
        : port_(port), baud_rate_(baud_rate)//port:串口路径
    {                                       //baud_rate：波特率
        openPort();
    }

    /*打开串口*/
    void openPort() {
        try {
            std::cout << "正在打开串口: " << port_ << std::endl;
            serial_port_.Open(port_);
            configurePort();
            std::cout << "串口打开成功" << std::endl;
        } catch (const LibSerial::OpenFailed&) {
            std::cerr << "无法打开串口: " << port_ << std::endl;
            throw;
        }
    }

    /*初始化配置*/
    void configurePort() {
        serial_port_.SetBaudRate(baud_rate_);
        serial_port_.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);
        serial_port_.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE);
        serial_port_.SetParity(LibSerial::Parity::PARITY_NONE);
        serial_port_.SetStopBits(LibSerial::StopBits::STOP_BITS_1);
        serial_port_.FlushIOBuffers();
    }

    
    void sendCoordinates(int16_t x, int16_t y, uint8_t header = 0xA5) {
    if (!serial_port_.IsOpen()) {
        std::cerr << "串口未打开，无法发送坐标" << std::endl;
        return;
    }
    
    try {
        std::vector<uint8_t> frame;
        frame.push_back(header);
        frame.push_back((x >> 8) & 0xFF); // x高字节
        frame.push_back(x & 0xFF);        // x低字节
        frame.push_back((y >> 8) & 0xFF); // y高字节
        frame.push_back(y & 0xFF);        // y低字节
        
        serial_port_.Write(frame);
        serial_port_.DrainWriteBuffer();
        
        std::cout << "已发送坐标: x=" << x << ", y=" << y << " -> ";
        for (uint8_t byte : frame) {
            std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0') 
                      << static_cast<int>(byte) << " ";
        }
        std::cout << std::dec << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "发送坐标时出错: " << e.what() << std::endl;
    }
}

    void closePort() {
        if (serial_port_.IsOpen()) {
            serial_port_.Close();
            std::cout << "串口已关闭" << std::endl;
        }
    }

    ~SerialPortManager() {
        closePort();
    }

private:
    LibSerial::SerialPort serial_port_;
    std::string port_;
    LibSerial::BaudRate baud_rate_;
};

// 常量定义
const double MIN_AREA_RATIO = 0.01;
const double MAX_AREA_RATIO = 0.5;
const double MIN_ASPECT_RATIO = 1.2;
const double MAX_ASPECT_RATIO = 1.6;
const Size KERNEL_SIZE(5, 5);
const int CANNY_THRESH1 = 30;
const int CANNY_THRESH2 = 60;

// 对矩形顶点进行排序（左上、右上、右下、左下）
vector<Point> sortRectanglePoints(const vector<Point>& points) {
    if (points.size() != 4) return points;
    
    vector<Point> sorted(4);
    vector<int> sum, diff;

    for (const auto& p : points) {
        sum.push_back(p.x + p.y);
        diff.push_back(p.x - p.y);
    }

    // 左上：最小和
    sorted[0] = points[min_element(sum.begin(), sum.end()) - sum.begin()];
    // 右下：最大和
    sorted[2] = points[max_element(sum.begin(), sum.end()) - sum.begin()];
    // 右上：最小差
    sorted[1] = points[min_element(diff.begin(), diff.end()) - diff.begin()];
    // 左下：最大差
    sorted[3] = points[max_element(diff.begin(), diff.end()) - diff.begin()];

    return sorted;
}

// 计算轮廓的长宽比
double calculateAspectRatio(const vector<Point>& points) {
    RotatedRect rect = minAreaRect(points);
    float width = rect.size.width;
    float height = rect.size.height;
    
    if (width < height) swap(width, height);
    return (height > 0) ? width / height : 0;
}

// 边缘检测与矩形处理
Point detectRedObject(Mat& frame) {

    Point center_in_src(-1, -1);

    // 预处理
    Mat gray, blurred, edges;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    Mat morph_kernel = getStructuringElement(MORPH_RECT, KERNEL_SIZE);
    GaussianBlur(gray, blurred, KERNEL_SIZE, 0);
    Canny(blurred, edges, CANNY_THRESH1, CANNY_THRESH2);
    
    // 轮廓检测
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    
    // 面积过滤
    double img_area = frame.rows * frame.cols;
    double min_area = img_area * MIN_AREA_RATIO;
    double max_area = img_area * MAX_AREA_RATIO;

    // 矩形检测变量
    vector<Point> max_rect_points;
    double max_area_val = 0;
    double max_aspect_ratio = 0;

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < min_area || area > max_area) continue;
        
        vector<Point> approx;
        double epsilon = 0.02 * arcLength(cnt, true);
        approxPolyDP(cnt, approx, epsilon, true);
        
        // 检查是否为凸四边形
        if (approx.size() != 4 || !isContourConvex(approx)) continue;
        
        // 计算并检查长宽比
        double aspect_ratio = calculateAspectRatio(approx);
        
        
        if (aspect_ratio < MIN_ASPECT_RATIO || aspect_ratio > MAX_ASPECT_RATIO) continue;
        
        // 更新最大矩形
        if (area > max_area_val) {
            max_area_val = area;
            max_rect_points = approx;
            max_aspect_ratio = aspect_ratio;
        }
    }
    
    // 处理最大矩形
    if (!max_rect_points.empty()) {
        vector<Point> sorted_points = sortRectanglePoints(max_rect_points);
        
        // 计算目标矩形尺寸
        double width1 = norm(sorted_points[1] - sorted_points[0]);
        double width2 = norm(sorted_points[2] - sorted_points[3]);
        double height1 = norm(sorted_points[3] - sorted_points[0]);
        double height2 = norm(sorted_points[2] - sorted_points[1]);
        double max_width = max(width1, width2);
        double max_height = max(height1, height2);
        
        // 准备透视变换
        vector<Point2f> src_pts, dst_pts;
        for (const auto& p : sorted_points) {
            src_pts.emplace_back(p);
        }
        
        dst_pts = {
            Point2f(0, 0),
            Point2f(static_cast<float>(max_width), 0),
            Point2f(static_cast<float>(max_width), static_cast<float>(max_height)),
            Point2f(0, static_cast<float>(max_height))
        };
        
        // 计算中心点
        Mat M = getPerspectiveTransform(src_pts, dst_pts);
        Point2f center_in_dst(max_width/2, max_height/2);
        vector<Point2f> src_center;
        perspectiveTransform(vector<Point2f>{center_in_dst}, src_center, M.inv());
        center_in_src = Point(static_cast<int>(src_center[0].x), 
                          static_cast<int>(src_center[0].y));

        // 绘制结果
        drawContours(frame, vector<vector<Point>>{sorted_points}, -1, Scalar(0, 255, 0), 2);
        for (const auto& p : sorted_points) {
            circle(frame, p, 5, Scalar(255, 0, 0), -1);
        }
        circle(frame, center_in_src, 8, Scalar(0, 0, 255), -1);
        putText(frame, "Center: " + to_string(center_in_src.x) + "," + to_string(center_in_src.y), 
                center_in_src + Point(10, -10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        
    }
    return center_in_src;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);
    int frame_width = 320;
    int frame_height = 240;
    const int CENTER_X =frame_width / 2;
    const int CENTER_Y =frame_height / 2;

    
    // 打开串口
    try {
        SerialPortManager serial_manager("/dev/ttyUSB0", LibSerial::BaudRate::BAUD_115200);
        
        Mat frame;
        Point last_valid_point(0, 0);

        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            Point center = detectRedObject(frame);

            Point detectRedObject(Mat& frame);
            if (center.x >= 0 && center.y >= 0) {
                    // 发送坐标（转换为16位整数）
                    int16_t x = static_cast<int16_t>(center.x - CENTER_X);
                    int16_t y = static_cast<int16_t>(CENTER_Y - center.y);

                    serial_manager.sendCoordinates(x, y);
                    last_valid_point = center;
                } else {
                    int16_t x = static_cast<int16_t>(last_valid_point.x - CENTER_X);
                int16_t y = static_cast<int16_t>(CENTER_Y - last_valid_point.y);
                serial_manager.sendCoordinates(x, y);
                }
            
            putText(frame, "Rectangle Detection", Point(10, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            
            imshow("Rectangle Detection", frame);
            
            if (waitKey(30) == 27) break;
        }
    
    } catch (const exception& e) {
        cerr << "串口错误: " << e.what() << endl;
        return -1;
    }
    destroyAllWindows();
    return 0;
}