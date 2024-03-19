#include "postprocess_op.h"
#include "preprocess_op.h"
#include "Convert.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <dirent.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#ifdef __cplusplus
	#define EXTERN_C extern "C" 
#else
	#define EXTERN_C
#endif

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

struct OCRPredictResult {
    //std::vector<std::vector<int>> box;
    char* text;
    float score;
};
struct OCRdetResult {
    std::vector<std::vector<int>> box;
};

struct Det_ModelParam{
    char modelPath[256]; //.engine file path
    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    bool use_dilation_ = false;
    bool use_polygon_score_ = false; // if use_polygon_score_ is true, it will be slow
    // input image
    int max_side_len_ = 640;
    //std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    //std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};   
};

struct Rec_ModelParam{
    char modelPath[256]; //.engine file path
    char label_path_[256] = "../myModels/dict_txt/output.txt";
    int rec_batch_num_= 1;
    int rec_img_h_ = 48;
    int rec_img_w_ = 320;
    
};


EXTERN_C IExecutionContext* infer_init_ocr(const char* model_path);

//EXTERN_C std::vector<OCRPredictResult> run(cv::Mat &inputImg, const Det_ModelParam& det_param, const Rec_ModelParam& rec_param, IExecutionContext* det_context, IExecutionContext* rec_context);

//EXTERN_C int run(cv::Mat &inputImg, const Det_ModelParam& det_param, const Rec_ModelParam& rec_param, IExecutionContext* det_context, IExecutionContext* rec_context, OCRPredictResult* res, int& outSize);
EXTERN_C OCRPredictResult* run(cv::Mat &inputImg, const Det_ModelParam& det_param, const Rec_ModelParam& rec_param, IExecutionContext* det_context, IExecutionContext* rec_context, int& outSize);

void det(cv::Mat inputImg, std::vector<OCRPredictResult> &ocr_results,std::vector<OCRdetResult> &det_results, const Det_ModelParam& det_param, IExecutionContext* det_context);

void rec(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results, const Rec_ModelParam& rec_param, IExecutionContext* rec_context);

void Model_Infer_det(cv::Mat& Input_Image,  std::vector<vector<vector<int>>> &boxes, const Det_ModelParam& det_param, IExecutionContext* det_context);

void Model_Infer_rec(std::vector<cv::Mat> img_list, std::vector<char*> &rec_texts,std::vector<float> &rec_text_scores, const Rec_ModelParam& rec_param, IExecutionContext* rec_context);

void print_result(const std::vector<OCRPredictResult>& ocr_result);

void infer_release(IExecutionContext *context);