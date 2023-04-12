#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "onnxruntime_cxx_api.h"

struct SegNetConfig {
  float confidenceThreshold;
  float nonMaximumSuppressionThreshold;
  std::string pathToModel;
  std::string pathToClasses;
  int segChannels = 32;
  int segWidth = 160;
  int segHeight = 160;
  int netWidth = 640;
  int netHeight = 640;
};

struct Segment {
  int id;
  float confidence;
  cv::Rect bbox;
  cv::Mat mask;
};

struct MaskParams {
  cv::Size srcSize;
  cv::Vec4d transformParams;
};

class YOLOSegmentor {
 public:
  YOLOSegmentor(const SegNetConfig& config)
      : memoryInfo(Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtDeviceAllocator,
            OrtMemType::OrtMemTypeCPUOutput
        )) {
    confidenceThreshold = config.confidenceThreshold;
    nonMaximumSuppressionThreshold = config.nonMaximumSuppressionThreshold;
    segChannels = config.segChannels;
    segWidth = config.segWidth;
    segHeight = config.segHeight;
    netWidth = config.netWidth;
    netHeight = config.netHeight;

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    session = Ort::Session(env, config.pathToModel.c_str(), sessionOptions);

    LoadTypeInfo();
    LoadClasses(config.pathToClasses);
  };

 public:
  bool LoadTypeInfo() {
    try {
      Ort::AllocatorWithDefaultOptions allocator;

      inputName = std::move(session.GetInputNameAllocated(0, allocator));
      inputNodeNames.push_back(inputName.get());

      Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
      auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

      inputTensorShape = inputTensorInfo.GetShape();

      // Set the batch size if the input shape is dynamic
      int IS_UNSET = -1;
      if (inputTensorShape[0] == IS_UNSET) {
        inputTensorShape[0] = 1;
      }

      // Set the input width and height if they are dynamic
      if (inputTensorShape[2] == IS_UNSET || inputTensorShape[3] == IS_UNSET) {
        inputTensorShape[2] = netHeight;
        inputTensorShape[3] = netWidth;
      }

      const size_t numOutputNodes = session.GetOutputCount();
      if (numOutputNodes != 2) {
        std::cout << "The model should have exactly two output nodes. "
                     "Please check the model."
                  << std::endl;
        return false;
      }

      outputName0 = std::move(session.GetOutputNameAllocated(0, allocator));
      outputName1 = std::move(session.GetOutputNameAllocated(1, allocator));

      Ort::TypeInfo typeInfoOutput0(nullptr);
      Ort::TypeInfo typeInfoOutput1(nullptr);

      const bool isOutput0First =
          strcmp(outputName0.get(), outputName1.get()) < 0;
      if (isOutput0First) {
        typeInfoOutput0 = session.GetOutputTypeInfo(0);
        typeInfoOutput1 = session.GetOutputTypeInfo(1);
        outputNodeNames.push_back(outputName0.get());
        outputNodeNames.push_back(outputName1.get());
      } else {
        typeInfoOutput0 = session.GetOutputTypeInfo(1);
        typeInfoOutput1 = session.GetOutputTypeInfo(0);
        outputNodeNames.push_back(outputName1.get());
        outputNodeNames.push_back(outputName0.get());
      }

      outputTensorShape =
          typeInfoOutput0.GetTensorTypeAndShapeInfo().GetShape();
      return true;
    } catch (const Ort::Exception& e) {
      std::cout << "Ort::Exception: " << e.what() << std::endl;
      return false;
    } catch (const std::exception& e) {
      std::cout << "std::exception: " << e.what() << std::endl;
      return false;
    }
  }

  void Detect(cv::Mat& inputImage) {
    std::vector<cv::Mat> inputImages = {inputImage};
    std::vector<std::vector<Segment>> batched_outputs;

    if (BatchDetect(inputImages, batched_outputs)) {
      inputImage = DrawPredictions(inputImage, batched_outputs[0], classNames);
    }
  }

  bool BatchDetect(
      const std::vector<cv::Mat>& srcImgs,
      std::vector<std::vector<Segment>>& output
  ) {
    output.clear();
    std::vector<cv::Vec4d> params;
    std::vector<cv::Mat> inputImages;
    const cv::Size inputSize(netWidth, netHeight);

    PreprocessImages(srcImgs, inputImages, params);

    cv::Mat blob = cv::dnn::blobFromImages(
        inputImages,
        1 / 255.0,
        inputSize,
        cv::Scalar(0, 0, 0),
        true,
        false
    );

    const size_t inputTensorLength = std::accumulate(
        inputTensorShape.begin(),
        inputTensorShape.end(),
        1,
        std::multiplies<int64_t>()
    );
    std::vector<Ort::Value> inputTensors, outputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo,
        (float*)blob.data,
        inputTensorLength,
        inputTensorShape.data(),
        inputTensorShape.size()
    ));

    outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNodeNames.data(),
        inputTensors.data(),
        inputNodeNames.size(),
        outputNodeNames.data(),
        outputNodeNames.size()
    );

    float* pdata = outputTensors[0].GetTensorMutableData<float>();
    outputTensorShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    outputMaskTensorShape =
        outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int> maskProtosShape = {
        1,
        static_cast<int>(outputMaskTensorShape[1]),
        static_cast<int>(outputMaskTensorShape[2]),
        static_cast<int>(outputMaskTensorShape[3])};
    const int maskProtosLength = std::accumulate(
        maskProtosShape.begin(),
        maskProtosShape.end(),
        1,
        std::multiplies<int64_t>()
    );

    const int netWidth = static_cast<int>(numberOfClasses) + 5 + segChannels;
    const int out0Width = static_cast<int>(outputTensorShape[2]);
    assert(netWidth == out0Width);

    const int netHeight = static_cast<int>(outputTensorShape[1]);
    for (int i = 0; i < srcImgs.size(); ++i) {
      std::vector<int> classIds;
      std::vector<float> confidences;
      std::vector<cv::Rect> boxes;
      std::vector<std::vector<float>> pickedProposals;

      for (int r = 0; r < netHeight; r++) {
        float boxScore = pdata[4];
        if (boxScore >= confidenceThreshold) {
          cv::Mat
              scores(1, static_cast<int>(numberOfClasses), CV_32FC1, pdata + 5);
          cv::Point classIdPoint;
          double maxClassScore;
          minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
          maxClassScore = static_cast<float>(maxClassScore);
          if (maxClassScore >= confidenceThreshold) {
            std::vector<float> tempProto(
                pdata + 5 + static_cast<int>(numberOfClasses),
                pdata + netWidth
            );
            pickedProposals.push_back(tempProto);
            const float x = (pdata[0] - params[i][2]) / params[i][0];
            const float y = (pdata[1] - params[i][3]) / params[i][1];
            const float w = pdata[2] / params[i][0];
            const float h = pdata[3] / params[i][1];
            const int left = std::max(static_cast<int>(x - 0.5 * w + 0.5), 0);
            const int top = std::max(static_cast<int>(y - 0.5 * h + 0.5), 0);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(maxClassScore * boxScore);
            boxes.push_back(cv::Rect(
                left,
                top,
                static_cast<int>(w + 0.5),
                static_cast<int>(h + 0.5)
            ));
          }
        }
        pdata += netWidth;
      }
      std::vector<int> nmsResult;
      cv::dnn::NMSBoxes(
          boxes,
          confidences,
          confidenceThreshold,
          nonMaximumSuppressionThreshold,
          nmsResult
      );

      std::vector<std::vector<float>> tempMaskProposals;
      cv::Rect holeImgRect(0, 0, srcImgs[i].cols, srcImgs[i].rows);
      std::vector<Segment> tempOutput;

      for (int j = 0; j < nmsResult.size(); ++j) {
        const int idx = nmsResult[j];
        Segment result;
        result.id = classIds[idx];
        result.confidence = confidences[idx];
        result.bbox = boxes[idx] & holeImgRect;
        tempMaskProposals.push_back(pickedProposals[idx]);
        tempOutput.push_back(result);
      }

      MaskParams maskParams;
      maskParams.transformParams = params[i];
      maskParams.srcSize = srcImgs[i].size();

      cv::Mat maskProtos = cv::Mat(
          maskProtosShape,
          CV_32F,
          outputTensors[1].GetTensorMutableData<float>() + i * maskProtosLength
      );

      for (int j = 0; j < tempMaskProposals.size(); ++j) {
        CreatePolygonMask(
            cv::Mat(tempMaskProposals[j]).t(),
            maskProtos,
            tempOutput[j],
            maskParams
        );
      }

      output.push_back(tempOutput);
    }
    return !output.empty();
  }

 private:
  std::vector<std::string> classNames;
  int numberOfClasses;

  float confidenceThreshold;
  float nonMaximumSuppressionThreshold;
  int segChannels;
  int segWidth;   // netWidth/mask_ratio
  int segHeight;  // netHeight/mask_ratio
  int netWidth;
  int netHeight;

  Ort::Env env =
      Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov5-Seg");
  Ort::SessionOptions sessionOptions = Ort::SessionOptions();
  Ort::Session session = Ort::Session(nullptr);
  Ort::MemoryInfo memoryInfo;


  std::shared_ptr<char> inputName, outputName0, outputName1;
  // NOTE: maybe use smart pointer (although Session::Run expects char*
  std::vector<char*> inputNodeNames;
  std::vector<char*> outputNodeNames;

  std::vector<int64_t> inputTensorShape;
  std::vector<int64_t> outputTensorShape;
  std::vector<int64_t> outputMaskTensorShape;

  int PreprocessImage(
      const cv::Mat& srcImg, cv::Mat& outImg, cv::Vec4d& params
  ) {
    const cv::Size inputSize(netWidth, netHeight);

    if (srcImg.size() != inputSize) {
      ResizeAndPadImage(
          srcImg,
          outImg,
          params,
          inputSize,
          false,
          false,
          true,
          32
      );
    } else {
      outImg = srcImg;
      params = {1, 1, 0, 0};
    }

    return 0;
  }

  int PreprocessImages(
      const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs,
      std::vector<cv::Vec4d>& params
  ) {
    const cv::Size inputSize(netWidth, netHeight);
    outSrcImgs.clear();
    params.clear();

    for (const auto& srcImg : srcImgs) {
      cv::Mat outImg;
      cv::Vec4d transformParam;
      PreprocessImage(srcImg, outImg, transformParam);
      outSrcImgs.push_back(outImg);
      params.push_back(transformParam);
    }

    int batchSize = 1;
    const int numLack = batchSize - srcImgs.size();
    if (numLack > 0) {
      for (int i = 0; i < numLack; ++i) {
        cv::Mat outImg = cv::Mat::zeros(inputSize, CV_8UC3);
        cv::Vec4d transformParam = {1, 1, 0, 0};
        outSrcImgs.push_back(outImg);
        params.push_back(transformParam);
      }
    }

    return 0;
  }

  void LoadClasses(const std::string& pathToClasses) {
    std::ifstream ifs(pathToClasses.c_str());
    std::string line;

    while (getline(ifs, line)) {
      this->classNames.push_back(line);
    }

    this->numberOfClasses = classNames.size();
  }

  void ResizeAndPadImage(
      const cv::Mat& sourceImage, cv::Mat& destinationImage,
      cv::Vec4d& paddingParams, const cv::Size& targetSize = cv::Size(640, 640),
      bool maintainAspectRatio = false, bool fillBackground = false,
      bool allowScaleUp = true, int alignment = 32,
      const cv::Scalar& backgroundColor = cv::Scalar(114, 114, 114)
  ) {
    cv::Size originalSize = sourceImage.size();
    float scalingRatio = std::min(
        (float)targetSize.height / (float)originalSize.height,
        (float)targetSize.width / (float)originalSize.width
    );
    if (!allowScaleUp) {
      scalingRatio = std::min(scalingRatio, 1.0f);
    }

    float ratio[2]{scalingRatio, scalingRatio};
    int newUnpaddedSize[2] = {
        (int)std::round((float)originalSize.width * scalingRatio),
        (int)std::round((float)originalSize.height * scalingRatio)};

    auto deltaWidth = (float)(targetSize.width - newUnpaddedSize[0]);
    auto deltaHeight = (float)(targetSize.height - newUnpaddedSize[1]);

    if (maintainAspectRatio) {
      deltaWidth = (float)((int)deltaWidth % alignment);
      deltaHeight = (float)((int)deltaHeight % alignment);
    } else if (fillBackground) {
      deltaWidth = 0.0f;
      deltaHeight = 0.0f;
      newUnpaddedSize[0] = targetSize.width;
      newUnpaddedSize[1] = targetSize.height;
      ratio[0] = (float)targetSize.width / (float)originalSize.width;
      ratio[1] = (float)targetSize.height / (float)originalSize.height;
    }

    deltaWidth /= 2.0f;
    deltaHeight /= 2.0f;

    bool needsResizing = originalSize.width != newUnpaddedSize[0] &&
                         originalSize.height != newUnpaddedSize[1];
    if (needsResizing) {
      cv::resize(
          sourceImage,
          destinationImage,
          cv::Size(newUnpaddedSize[0], newUnpaddedSize[1])
      );
    } else {
      destinationImage = sourceImage.clone();
    }

    int top = int(std::round(deltaHeight - 0.1f));
    int bottom = int(std::round(deltaHeight + 0.1f));
    int left = int(std::round(deltaWidth - 0.1f));
    int right = int(std::round(deltaWidth + 0.1f));
    paddingParams[0] = ratio[0];
    paddingParams[1] = ratio[1];
    paddingParams[2] = left;
    paddingParams[3] = top;
    cv::copyMakeBorder(
        destinationImage,
        destinationImage,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        backgroundColor
    );
  }

  void CreatePolygonMask(
      const cv::Mat& proposals, const cv::Mat& prototypes, Segment& output,
      const MaskParams& maskParams
  ) {
    cv::Rect bbox = output.bbox;

    // Crop from prototypes
    int cropX = floor(
        (bbox.x * maskParams.transformParams[0] + maskParams.transformParams[2]
        ) /
        netWidth * segWidth
    );
    int cropY = floor(
        (bbox.y * maskParams.transformParams[1] + maskParams.transformParams[3]
        ) /
        netHeight * segHeight
    );
    int cropW = ceil(
                    ((bbox.x + bbox.width) * maskParams.transformParams[0] +
                     maskParams.transformParams[2]) /
                    netWidth * segWidth
                ) -
                cropX;
    int cropH = ceil(
                    ((bbox.y + bbox.height) * maskParams.transformParams[1] +
                     maskParams.transformParams[3]) /
                    netHeight * segHeight
                ) -
                cropY;

    // Fix invalid ranges
    cropW = std::max(cropW, 1);
    cropH = std::max(cropH, 1);
    if (cropX + cropW > segWidth) {
      if (segWidth - cropX > 0) {
        cropW = segWidth - cropX;
      } else {
        cropX -= 1;
      }
    }
    if (cropY + cropH > segHeight) {
      if (segHeight - cropY > 0) {
        cropH = segHeight - cropY;
      } else {
        cropY -= 1;
      }
    }

    // Compute crop ranges for crop
    std::vector<cv::Range> cropRanges;
    cropRanges.emplace_back(0, 1);
    cropRanges.push_back(cv::Range::all());
    cropRanges.emplace_back(cropY, cropH + cropY);
    cropRanges.emplace_back(cropX, cropW + cropX);

    // Crop prototypes
    cv::Mat croppedPrototypes = prototypes(cropRanges).clone();
    cv::Mat prototypesMatrix =
        croppedPrototypes.reshape(0, {segChannels, cropW * cropH});

    // Compute mask features
    cv::Mat matmul_res = (proposals * prototypesMatrix).t();
    cv::Mat features_matrix = matmul_res.reshape(1, {cropH, cropW});

    // Sigmoid activation
    cv::Mat dest, mask;
    cv::exp(-features_matrix, dest);
    dest = 1.0 / (1.0 + dest);

    // Resize mask to bbox size
    int left = floor(
        (netWidth / segWidth * cropX - maskParams.transformParams[2]) /
        maskParams.transformParams[0]
    );
    int top = floor(
        (netHeight / segHeight * cropY - maskParams.transformParams[3]) /
        maskParams.transformParams[1]
    );
    int width =
        ceil(netWidth / segWidth * cropW / maskParams.transformParams[0]);
    int height =
        ceil(netHeight / segHeight * cropH / maskParams.transformParams[1]);
    cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);

    // Threshold mask
    mask = mask(bbox - cv::Point(left, top)) > confidenceThreshold;

    output.mask = mask;
  }

  cv::Mat DrawPredictions(
      cv::Mat& image, const std::vector<Segment>& results,
      const std::vector<std::string>& classNames,
      bool shouldDrawBoundingBox = false
  ) {
    cv::Mat imageWithMask = image.clone();
    for (const auto& result : results) {
      // Get the bounding box coordinates
      const int x = result.bbox.x;
      int y = result.bbox.y;

      // Draw the bounding box
      if(shouldDrawBoundingBox) {
        rectangle(imageWithMask, result.bbox, cv::Scalar(0, 0, 255), 2, 8);
      }

      // Add the mask to the image
      const cv::Mat mask = result.mask;
      imageWithMask(result.bbox).setTo(cv::Scalar(0, 0, 255), mask);

      // Add the label to the image
      const std::string label =
          classNames[result.id] + ":" + std::to_string(result.confidence);
      const cv::Size label_size =
          getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
      y = std::max(y, label_size.height);
      const cv::Point label_origin(x, y);
      putText(
          imageWithMask,
          label,
          label_origin,
          cv::FONT_HERSHEY_SIMPLEX,
          0.75,
          cv::Scalar(0, 255, 0),
          1
      );
    }

    addWeighted(imageWithMask, 0.5, image, 0.5, 0, imageWithMask);

    return imageWithMask;
  }
};
