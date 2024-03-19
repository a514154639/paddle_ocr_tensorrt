#include "ocr_out.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <thread>
#include "NvInfer.h"
#include <dlfcn.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <time.h>


int main(int argc, char* argv[]) {

    void *handle = dlopen("./libinfer_OCR.so", RTLD_LAZY);

    if(!handle)
    {
        printf("open lib error\n");
        std::cout<<dlerror()<<std::endl;
        return -1;
    }
    
    typedef OCRPredictResult* (*_infer)(cv::Mat &image, const Det_ModelParam& det_param, const Rec_ModelParam& rec_param, nvinfer1::IExecutionContext* det_context, nvinfer1::IExecutionContext* rec_context, int& outSize);
    //typedef int (*_infer)(cv::Mat &image, const Det_ModelParam& det_param, const Rec_ModelParam& rec_param, nvinfer1::IExecutionContext* det_context, nvinfer1::IExecutionContext* rec_context,OCRPredictResult* res, int& outSize);
    typedef nvinfer1::IExecutionContext* (*_inimodel)(const char* model_path);

    _infer infer = (_infer) dlsym(handle, "run");
    _inimodel init = (_inimodel) dlsym(handle, "infer_init_ocr");

    Det_ModelParam det_model;
    strcpy(det_model.modelPath, "../myEngines/det_fp16.engine");
    
    Rec_ModelParam rec_model;
    strcpy(rec_model.modelPath, "../myEngines/ch_rec_v3_fp16.engine");

    nvinfer1::IExecutionContext* det_con = init(det_model.modelPath);
    nvinfer1::IExecutionContext* rec_con = init(rec_model.modelPath);

    cv::Mat img = cv::imread(argv[1]);

    if (img.empty()) {
        std::cout << "read image failed" << std::endl;
        return -1;
    }
    // int row = img.rows;
    // int clomn = img.cols;
    // int size = img.total() * img.elemSize();
    // unsigned char* bytes = new unsigned char[size];
    // std::memcpy(bytes, img.data, size * sizeof(unsigned char));
    //std::vector<OCRPredictResult> ocr_results = infer(img, det_model, rec_model, det_con, rec_con);
    
    int outSize;
    OCRPredictResult* ocr_results  = new OCRPredictResult[100];
    //OCRPredictResult* ocr_results = infer(bytes, row, clomn, det_model, rec_model, det_con, rec_con, outSize);
    //int res = infer(img, det_model, rec_model, det_con, rec_con, ocr_results, outSize);
    ocr_results = infer(img, det_model, rec_model, det_con, rec_con, outSize);

    // 创建一个 std::vector<OCRPredictResult> 并将 OCRPredictResult 对象添加到其中
    std::vector<OCRPredictResult> ocr_results_vector(ocr_results, ocr_results + outSize);

    // 使用 ocr_results_vector
    print_result(ocr_results_vector);

    // 使用完后，别忘记释放内存
    delete[] ocr_results;

    //print_result(ocr_results);  
    //infer_release(det_con);
    return 0;
}