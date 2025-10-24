// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pipeline.h" // NOLINT
#include <algorithm>  // NOLINT
#include <iostream>   // NOLINT
#include <sstream>    // NOLINT

cv::Mat GetRotateCropImage(cv::Mat srcimage,
                           std::vector<std::vector<int>> box) {
  cv::Mat image;
  srcimage.copyTo(image);
  std::vector<std::vector<int>> points = box;

  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = static_cast<int>(*std::min_element(x_collect, x_collect + 4));
  int right = static_cast<int>(*std::max_element(x_collect, x_collect + 4));
  int top = static_cast<int>(*std::min_element(y_collect, y_collect + 4));
  int bottom = static_cast<int>(*std::max_element(y_collect, y_collect + 4));

  cv::Mat img_crop;
  image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  int img_crop_width =
      static_cast<int>(sqrt(pow(points[0][0] - points[1][0], 2) +
                            pow(points[0][1] - points[1][1], 2)));
  int img_crop_height =
      static_cast<int>(sqrt(pow(points[0][0] - points[3][0], 2) +
                            pow(points[0][1] - points[3][1], 2)));

  cv::Point2f pts_std[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  cv::Point2f pointsf[4];
  pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
  pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
  pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
  pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

  cv::Mat dst_img;
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  const float ratio = 1.5;
  if (static_cast<float>(dst_img.rows) >=
      static_cast<float>(dst_img.cols) * ratio) {
    cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
    cv::flip(srcCopy, srcCopy, 0);
    return srcCopy;
  } else {
    return dst_img;
  }
}

std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str()); // NOLINT

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str()); // NOLINT

  char *p = std::strtok(strs, d);
  while (p) {
    std::string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::map<std::string, double> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, double> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = stod(res[1]);
  }
  return dict;
}

cv::Mat Visualization(cv::Mat srcimg,
                      std::vector<std::vector<std::vector<int>>> boxes,
                      std::string output_image_path) {
  cv::Point rook_points[boxes.size()][4];
  for (int n = 0; n < boxes.size(); n++) {
    for (int m = 0; m < boxes[0].size(); m++) {
      rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m][0]),
                                    static_cast<int>(boxes[n][m][1]));
    }
  }
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite(output_image_path, img_vis);
  std::cout << "The detection visualized image saved in "
            << output_image_path.c_str() << std::endl;
  return img_vis;
}

Pipeline::Pipeline(const std::string &detModelDir,
                   const std::string &clsModelDir,
                   const std::string &recModelDir,
                   const std::string &cPUPowerMode, const int cPUThreadNum,
                   const std::string &config_path,
                   const std::string &dict_path) {
  clsPredictor_.reset(
      new ClsPredictor(clsModelDir, cPUThreadNum, cPUPowerMode));
  detPredictor_.reset(
      new DetPredictor(detModelDir, cPUThreadNum, cPUPowerMode));
  recPredictor_.reset(
      new RecPredictor(recModelDir, cPUThreadNum, cPUPowerMode));
  Config_ = LoadConfigTxt(config_path);
  charactor_dict_ = ReadDict(dict_path);
  charactor_dict_.insert(charactor_dict_.begin(), "#"); // NOLINT
  charactor_dict_.push_back(" ");
}

bool Pipeline::Process(std::string img_path, std::string output_img_path) {
  cv::Mat rgbaImage = cv::imread(img_path, cv::IMREAD_COLOR);
  int use_direction_classify =
      static_cast<int>(Config_["use_direction_classify"]);
  cv::Mat srcimg;
  rgbaImage.copyTo(srcimg);

  auto start = std::chrono::system_clock::now();
  // det predict
  auto boxes =
      detPredictor_->Predict(srcimg, Config_, nullptr, nullptr, nullptr);

  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat img;
  rgbaImage.copyTo(img);
  cv::Mat crop_img;

  std::vector<std::string> rec_text;
  std::vector<float> rec_text_score;
  for (int i = boxes.size() - 1; i >= 0; i--) {
    crop_img = GetRotateCropImage(img, boxes[i]);
    if (use_direction_classify >= 1) {
      crop_img =
          clsPredictor_->Predict(crop_img, nullptr, nullptr, nullptr, 0.9);
    }
    auto res = recPredictor_->Predict(crop_img, nullptr, nullptr, nullptr,
                                      charactor_dict_);
    rec_text.push_back(res.first);
    rec_text_score.push_back(res.second);
  }
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  // 可视化输出（可选）
  if (!output_img_path.empty()) {
    auto img_vis = Visualization(rgbaImage, boxes, output_img_path);
  }
  
  // print recognized text
  for (int i = 0; i < rec_text.size(); i++) {
    std::cout << i << "\t" << rec_text[i] << "\t" << rec_text_score[i]
              << std::endl;
  }
  std::cout << "花费了"
            << double(duration.count()) *
                   std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << "秒" << std::endl;
  return true;
}

std::string Pipeline::ProcessWithJson(std::string img_path, std::string output_img_path) {
  cv::Mat rgbaImage = cv::imread(img_path, cv::IMREAD_COLOR);
  int use_direction_classify = static_cast<int>(Config_["use_direction_classify"]);
  cv::Mat srcimg;
  rgbaImage.copyTo(srcimg);

  auto start = std::chrono::system_clock::now();
  
  // det predict
  auto boxes = detPredictor_->Predict(srcimg, Config_, nullptr, nullptr, nullptr);

  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat img;
  rgbaImage.copyTo(img);
  cv::Mat crop_img;

  std::vector<OCRResult> results;
  for (int i = boxes.size() - 1; i >= 0; i--) {
    crop_img = GetRotateCropImage(img, boxes[i]);
    if (use_direction_classify >= 1) {
      crop_img = clsPredictor_->Predict(crop_img, nullptr, nullptr, nullptr, 0.9);
    }
    auto res = recPredictor_->Predict(crop_img, nullptr, nullptr, nullptr, charactor_dict_);
    
    OCRResult ocr_result;
    ocr_result.box = {boxes[i]};
    ocr_result.text = res.first;
    ocr_result.score = res.second;
    results.push_back(ocr_result);
  }
  
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double elapsed_time = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

  // 构建JSON结果
  nlohmann::json result_json;
  result_json["elapsed_time"] = elapsed_time;
  
  nlohmann::json results_array = nlohmann::json::array();
  for (int i = 0; i < results.size(); i++) {
    nlohmann::json item;
    item["index"] = i;
    item["text"] = results[i].text;
    item["score"] = results[i].score;
    
    // 添加坐标信息
    nlohmann::json box_array = nlohmann::json::array();
    for (const auto& point : results[i].box[0]) {
      nlohmann::json point_json;
      point_json["x"] = point[0];
      point_json["y"] = point[1];
      box_array.push_back(point_json);
    }
    item["box"] = box_array;
    
    results_array.push_back(item);
  }
  result_json["results"] = results_array;

  // 可视化输出（可选）
  if (!output_img_path.empty()) {
    auto img_vis = Visualization(rgbaImage, boxes, output_img_path);
    result_json["output_image"] = output_img_path;
  }

  return result_json.dump(2); // 返回格式化的JSON字符串
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "[ERROR] usage: "
              << " ./ocr_db_crnn_demo det_model_file cls_model_file "
                 "rec_model_file image_path"
                 " charactor_dict config [output_image_path]\n";
    std::cerr << "Example: ./ocr_db_crnn_demo det_model cls_model rec_model image.jpg dict.txt config.txt [output.jpg]\n";
    exit(1);
  }
  
  std::string det_model_file = argv[1];
  std::string rec_model_file = argv[2];
  std::string cls_model_file = argv[3];
  std::string img_path = argv[4];
  std::string dict_path = argv[5];
  std::string config_path = argv[6];
  std::string output_img_path = "";
  
  // 输出图片路径为可选参数
  if (argc >= 8) {
    output_img_path = argv[7];
  }
  
  std::string cPUPowerMode = "";
  int cPUThreadNum = 1;
  Pipeline *pipe =
      new Pipeline(det_model_file, cls_model_file, rec_model_file, cPUPowerMode,
                   cPUThreadNum, config_path, dict_path);
  
  // 使用JSON输出模式
  std::string json_result = pipe->ProcessWithJson(img_path, output_img_path);
  std::cout << json_result << std::endl;
  
  return 0;
}
