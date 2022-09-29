cmake_minimum_required(VERSION 3.23)

set(APP_NAME yolo_opencv)
project(${APP_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${APP_NAME} main.cpp)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenCV
include(UseOpenCV)
useOpenCV(${APP_NAME})
