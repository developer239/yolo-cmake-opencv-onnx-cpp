cmake_minimum_required(VERSION 3.21) # use the lower version requirement

set(APP_NAME yolo_opencv)
project(${APP_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${APP_NAME} src/main.cpp src/YOLODetector.h src/YOLOSegmentor.h)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenCV
if(APPLE)
    find_package(OpenCV REQUIRED)
    target_link_libraries(${APP_NAME} PUBLIC "-framework Carbon")
    target_link_libraries(${APP_NAME} PUBLIC ${OpenCV_LIBS})
elseif(WIN32)
    set(OpenCV_DIR "C:/tools/opencv/build/x64/vc15/lib")
    find_package(OpenCV REQUIRED)
    target_link_libraries(${APP_NAME} PUBLIC ${OpenCV_LIBS})
endif()

# Link Tesseract and Leptonica
target_link_libraries(${APP_NAME} PUBLIC
        ${tesseract_lib}
        ${leptonica_lib}
        ${TESSERACT_LIBRARIES}
        ${LEPTONICA_LIBRARIES}
        )

target_include_directories(${APP_NAME} PUBLIC
        ${tesseract_include}
        ${leptonica_include}
        )

# OnnxRuntime
if(APPLE)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/$ENV{ONNXRUNTIME_ROOT}")
    else()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-osx-arm64-1.12.1")
    endif()
elseif(WIN32)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-win-x64-1.12.1")
endif()

find_package(ONNXRUNTIME REQUIRED)
target_link_libraries(${APP_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
target_include_directories(${APP_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})

# Copy onnxruntime.dll on Windows
if(WIN32)
    add_custom_command(TARGET ${APP_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime-win-x64-1.12.1/lib/onnxruntime.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll"
            )
endif()
