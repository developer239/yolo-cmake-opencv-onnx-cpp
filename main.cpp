#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main() {
  auto matrix = cv::Mat(500, 500, CV_8UC3, cv::Scalar(0, 0, 255));

  cv::imshow("matrix", matrix);
  cv::waitKey(0);

  return 0;
}
