#pragma once

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "onnxruntime_cxx_api.h"

// Enable cuda part 1
// #include <cuda_provider_factory.h>bl

struct NetConfig {
  float confidenceThreshold;
  float nonMaximumSuppressionThreshold;
  std::string pathToModel;
  std::string pathToClasses;
};

struct BoxInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
};

class YOLODetector {
 public:
  explicit YOLODetector(const NetConfig& config) {
    this->confidenceThreshold = config.confidenceThreshold;
    this->nonMaximumSuppressionThreshold =
        config.nonMaximumSuppressionThreshold;

    // Enable cuda part 2
    // OrtStatus* status =
    // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    ortSession =
        Ort::Session(ortEnv, config.pathToModel.c_str(), sessionOptions);

    LoadTypeInfo();
    LoadClasses(config.pathToClasses);
  }

  void detect(cv::Mat& frame) {
    cv::Mat resizedImage;
    resize(frame, resizedImage, cv::Size(this->inputWidth, this->inputHeight));
    this->Normalize(resizedImage);

    auto allocatorInfo =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::array<int64_t, 4> inputShape{
        1,
        3,
        this->inputHeight,
        this->inputWidth};
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(
        allocatorInfo,
        inputImage.data(),
        inputImage.size(),
        inputShape.data(),
        inputShape.size()
    );

    std::vector<Ort::Value> ortOutputs = ortSession.Run(
        Ort::RunOptions{nullptr},
        &inputNames[0],
        &input_tensor_,
        1,
        outputNames.data(),
        outputNames.size()
    );

    std::vector<BoxInfo> generatedBoxes;
    float ratioHeight = (float)frame.rows / this->inputHeight;
    float ratioWidth = (float)frame.cols / this->inputWidth;

    const float* pData = ortOutputs[0].GetTensorMutableData<float>();
    for (int n = 0; n < this->numberOfProposals; n++) {
      float boxScore = pData[4];

      if (boxScore > this->confidenceThreshold) {
        int maxInd = 0;
        float maxClassScore = 0;

        for (int k = 0; k < numberOfClasses; k++) {
          if (pData[k + 5] > maxClassScore) {
            maxClassScore = pData[k + 5];
            maxInd = k;
          }
        }

        maxClassScore *= boxScore;
        if (maxClassScore > this->confidenceThreshold) {
          float cx = pData[0] * ratioWidth;
          float cy = pData[1] * ratioHeight;
          float w = pData[2] * ratioWidth;
          float h = pData[3] * ratioHeight;

          float xmin = cx - 0.5 * w;
          float ymin = cy - 0.5 * h;
          float xmax = cx + 0.5 * w;
          float ymax = cy + 0.5 * h;

          generatedBoxes.push_back(
              BoxInfo{xmin, ymin, xmax, ymax, maxClassScore, maxInd}
          );
        }
      }

      pData += outputNodeDim;
    }

    NonMaximumSuppression(generatedBoxes);

    // return or draw boxes
    DrawBoxes(frame, generatedBoxes);
  }

 private:
  int inputWidth;
  int inputHeight;
  int outputNodeDim;
  int numberOfProposals;

  std::vector<std::string> classNames;
  int numberOfClasses;

  float confidenceThreshold;
  float nonMaximumSuppressionThreshold;

  std::vector<float> inputImage;

  Ort::Env ortEnv = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "YOLODetector");
  Ort::SessionOptions sessionOptions = Ort::SessionOptions();
  Ort::Session ortSession = Ort::Session(nullptr);

  std::vector<char*> inputNames;
  std::vector<char*> outputNames;

  std::vector<std::vector<int64_t>> inputNodeDims;
  std::vector<std::vector<int64_t>> outputNodeDims;

  void LoadTypeInfo() {
    size_t numInputNodes = ortSession.GetInputCount();
    size_t numOutputNodes = ortSession.GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    for (int i = 0; i < numInputNodes; i++) {
      Ort::TypeInfo inputTypeInfo = ortSession.GetInputTypeInfo(i);
      auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
      auto inputDims = inputTensorInfo.GetShape();

      inputNodeDims.push_back(inputDims);
      inputNames.push_back(ortSession.GetInputName(i, allocator));
    }

    for (int i = 0; i < numOutputNodes; i++) {
      Ort::TypeInfo outputTypeInfo = ortSession.GetOutputTypeInfo(i);
      auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
      auto outputDims = outputTensorInfo.GetShape();

      outputNames.push_back(ortSession.GetOutputName(i, allocator));
      outputNodeDims.push_back(outputDims);
    }

    this->inputHeight = inputNodeDims[0][2];
    this->inputWidth = inputNodeDims[0][3];
    this->outputNodeDim = outputNodeDims[0][2];
    this->numberOfProposals = outputNodeDims[0][1];
  }

  void LoadClasses(const std::string& pathToClasses) {
    std::ifstream ifs(pathToClasses.c_str());
    std::string line;

    while (getline(ifs, line)) {
      this->classNames.push_back(line);
    }

    this->numberOfClasses = classNames.size();
  }

  void DrawBoxes(cv::Mat& frame, std::vector<BoxInfo> generatedBoxes) {
    for (auto& generatedBox : generatedBoxes) {
      int xMin = int(generatedBox.x1);
      int yMin = int(generatedBox.y1);
      rectangle(
          frame,
          cv::Point(xMin, yMin),
          cv::Point(int(generatedBox.x2), int(generatedBox.y2)),
          cv::Scalar(0, 0, 255),
          2
      );

      std::string label = cv::format("%.2f", generatedBox.score);
      label = this->classNames[generatedBox.label] + ":" + label;
      putText(
          frame,
          label,
          cv::Point(xMin, yMin - 5),
          cv::FONT_HERSHEY_SIMPLEX,
          0.75,
          cv::Scalar(0, 255, 0),
          1
      );
    }
  }

  void Normalize(cv::Mat img) {
    int rows = img.rows;
    int cols = img.cols;

    this->inputImage.resize(rows * cols * img.channels());
    for (int channels = 0; channels < 3; channels++) {
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
          float pixel = img.ptr<uchar>(row)[col * 3 + 2 - channels];
          this->inputImage[channels * rows * cols + row * cols + col] =
              pixel / 255.0;
        }
      }
    }
  }

  // Removes overlapping boxes with lower scores
  void NonMaximumSuppression(std::vector<BoxInfo>& inputBoxes) const {
    sort(inputBoxes.begin(), inputBoxes.end(), [](BoxInfo a, BoxInfo b) {
      return a.score > b.score;
    });

    std::vector<float> vArea(inputBoxes.size());
    for (int i = 0; i < int(inputBoxes.size()); ++i) {
      vArea[i] = (inputBoxes.at(i).x2 - inputBoxes.at(i).x1 + 1) *
                 (inputBoxes.at(i).y2 - inputBoxes.at(i).y1 + 1);
    }

    std::vector<bool> isSuppressed(inputBoxes.size(), false);
    for (int i = 0; i < int(inputBoxes.size()); ++i) {
      if (isSuppressed[i]) {
        continue;
      }

      for (int j = i + 1; j < int(inputBoxes.size()); ++j) {
        if (isSuppressed[j]) {
          continue;
        }

        float xx1 = (cv::max)(inputBoxes[i].x1, inputBoxes[j].x1);
        float yy1 = (cv::max)(inputBoxes[i].y1, inputBoxes[j].y1);
        float xx2 = (cv::min)(inputBoxes[i].x2, inputBoxes[j].x2);
        float yy2 = (cv::min)(inputBoxes[i].y2, inputBoxes[j].y2);

        float w = (cv::max)(float(0), xx2 - xx1 + 1);
        float h = (cv::max)(float(0), yy2 - yy1 + 1);
        float inter = w * h;
        float ovr = inter / (vArea[i] + vArea[j] - inter);

        if (ovr >= this->nonMaximumSuppressionThreshold) {
          isSuppressed[j] = true;
        }
      }
    }

    int idx_t = 0;
    inputBoxes.erase(
        remove_if(
            inputBoxes.begin(),
            inputBoxes.end(),
            [&idx_t, &isSuppressed](const BoxInfo& f) {
              return isSuppressed[idx_t++];
            }
        ),
        inputBoxes.end()
    );
  }
};
