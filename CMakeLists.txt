cmake_minimum_required(VERSION 3.23)

set(APP_NAME yolo_opencv)
project(${APP_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${APP_NAME} main.cpp)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenCV
function(useOpenCV TARGET_NAME)
    find_package(OpenCV REQUIRED)
    target_link_libraries(${TARGET_NAME} PUBLIC "-framework Carbon")
    target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})

    target_link_libraries(${TARGET_NAME} PUBLIC
            ${tesseract_lib}
            ${leptonica_lib}
            ${TESSERACT_LIBRARIES}
            ${LEPTONICA_LIBRARIES}
            )

    target_include_directories(${TARGET_NAME} PUBLIC
            ${tesseract_include}
            ${leptonica_include})
endfunction()

useOpenCV(${APP_NAME})

# OnnxRuntime
# set(ONNXRUNTIME_ROOT "/opt/homebrew/Cellar/onnxruntime/1.12.1")
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-osx-arm64-1.12.1")
find_package(ONNXRUNTIME REQUIRED)
target_link_libraries(${APP_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
target_include_directories(${APP_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
