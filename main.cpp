#include "YOLODetector.h"

int main() {
  // Initialize the detector
  NetConfig DetectorConfig = {
      0.3,
      0.5,
      "../models/best-n-640.onnx",
      "../coco.names"};
  YOLODetector net(DetectorConfig);

  // Initialize the image
  cv::Mat sourceImage = cv::imread("../samples/game-2.jpg");

  // Run detection
  auto start = std::chrono::steady_clock::now();

  net.detect(sourceImage);

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;

  // Show the result
  static const std::string windowName = "YOLO CMake OpenCV ONNX CPP";
  namedWindow(windowName, cv::WINDOW_NORMAL);
  imshow(windowName, sourceImage);

  // End
  cv::waitKey(0);
  cv::destroyAllWindows();
}
