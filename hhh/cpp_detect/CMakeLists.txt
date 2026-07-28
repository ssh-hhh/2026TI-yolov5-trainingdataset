cmake_minimum_required(VERSION 3.16)
project(steel_ball_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options("/utf-8")

# OpenCV
find_package(OpenCV REQUIRED)

add_executable(detect_camera main.cpp)
target_link_libraries(detect_camera ${OpenCV_LIBS})
