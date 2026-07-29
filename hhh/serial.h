#include <libserial/SerialPort.h>
#include <iostream>
#include <vector>
#include <cstdint>

class SerialPortManager {
public:
    SerialPortManager(const std::string& port, LibSerial::BaudRate baud_rate)
        : port_(port), baud_rate_(baud_rate) {
        openPort();
    }

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
            frame.push_back((x >> 8) & 0xFF);   // x 高字节
            frame.push_back(x & 0xFF);          // x 低字节
            frame.push_back((y >> 8) & 0xFF);   // y 高字节
            frame.push_back(y & 0xFF);          // y 低字节

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
