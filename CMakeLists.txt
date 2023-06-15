cmake_minimum_required(VERSION 3.21)

set(APP_NAME yolo_opencv)
project(${APP_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${APP_NAME} src/main.cpp src/YOLODetector.h src/YOLOSegmentor.h)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenCV
set(OpenCV_DIR "C:/tools/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)
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

# download https://github.com/microsoft/onnxruntime/releases/tag/v1.12.1 and extract to externals/onnxruntime-win-x64-1.12.1
# !! NOTE THAT you need to copy onnxruntime.dll to cmake-build-debug folder !! https://github.com/microsoft/onnxruntime/issues/11230#issuecomment-1101616411
# OnnxRuntime
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-win-x64-1.12.1")

find_package(ONNXRUNTIME REQUIRED)
target_link_libraries(${APP_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
target_include_directories(${APP_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
