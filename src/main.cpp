#include <iostream>

#include "YOLODetector.h"
#include "YOLOSegmentor.h"

void InstanceSegmentation(std::string& imagePath) {
  // Initialize the detector
  SegNetConfig DetectorConfig = {
      0.3,
      0.3,
      "../models/best-s-640-seg.onnx",
      "../models/class.names"};
  YOLOSegmentor net(DetectorConfig);

  // Initialize the image
  cv::Mat sourceImage = cv::imread(imagePath);

  // Run detection
  auto start = std::chrono::steady_clock::now();

  net.Detect(sourceImage);

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;

  // Show the result
  static const std::string windowName = "YOLOv5 Instance Segmentation";
  namedWindow(windowName, cv::WINDOW_NORMAL);
  imshow(windowName, sourceImage);

  // End
  cv::waitKey(0);
  cv::destroyAllWindows();
}

void ObjectDetection(std::string& imagePath) {
  // Initialize the detector
  NetConfig DetectorConfig = {
      0.3,
      0.5,
      "../models/best-n-640.onnx",
      "../models/class.names"};
  YOLODetector net(DetectorConfig);

  // Initialize the image
  cv::Mat sourceImage = cv::imread(imagePath);

  // Run detection
  auto start = std::chrono::steady_clock::now();

  net.detect(sourceImage);

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;

  // Show the result
  static const std::string windowName = "YOLOv5 Object Detection";
  namedWindow(windowName, cv::WINDOW_NORMAL);
  imshow(windowName, sourceImage);

  // End
  cv::waitKey(0);
  cv::destroyAllWindows();
}

int main() {
  std::string imagePath = "../samples/game-1.jpg";
//  std::string imagePath = "../samples/game-2.jpg";
//  std::string imagePath = "../samples/game-3.jpg";

  InstanceSegmentation(imagePath);
  ObjectDetection(imagePath);

  return 0;
}
