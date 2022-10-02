# YOLO CMake OpenCV ONNX CPP

Implementation is basically cleaned up version of [hpc203/yolov7-opencv-onnxrun-cpp-py](github.com/hpc203/yolov7-opencv-onnxrun-cpp-py) repo.

I mostly just cleaned up the code and then this repository is more of a documentation for my own purposes to show how to run these libraries with CMake.

## How to run

1) Install dependencies:
```bash
$ brew install cmake
$ brew install onnxruntime
$ brew install opencv
```

2) Download ONNX runtime files:
- Either pick correct version from releases [here](github.com/microsoft/onnxruntime/releases)
- Or download arm64-1.12.1.tgz version [here](https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-osx-arm64-1.12.1.tgz)
- If you are not sure what version to download use the official Optimize Inferencing picker [here](https://onnxruntime.ai)

3) Put ONNX runtime files to `extenrals/onnxruntime-osx-arm64-1.12.1` (**note:** if you use different version rename `ONNXRUNTIME_ROOT` in CMakeList.txt)

4) `.onnx` model and `.jpg` sample files were small so i already commited them. You can replace them in `main.cpp` and use your own weights.

## Preview

![preview](preview.png)
![preview-2](preview-2.png)
![preview-3](preview-3.png)
