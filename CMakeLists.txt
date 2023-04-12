cmake_minimum_required(VERSION 3.23)

set(APP_NAME yolo_opencv)
project(${APP_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${APP_NAME} src/main.cpp src/YOLODetector.h src/YOLOSegmentor.h)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(${APP_NAME} PUBLIC "-framework Carbon")
target_link_libraries(${APP_NAME} PUBLIC ${OpenCV_LIBS})

target_link_libraries(${APP_NAME} PUBLIC
        ${tesseract_lib}
        ${leptonica_lib}
        ${TESSERACT_LIBRARIES}
        ${LEPTONICA_LIBRARIES}
        )

target_include_directories(${APP_NAME} PUBLIC
        ${tesseract_include}
        ${leptonica_include})

# OnnxRuntime
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/$ENV{ONNXRUNTIME_ROOT}")
else()
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-osx-arm64-1.12.1")
endif()

find_package(ONNXRUNTIME REQUIRED)
target_link_libraries(${APP_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
target_include_directories(${APP_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
