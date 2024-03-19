# include "det.h"

namespace OCR {

void TextDetect::Model_Infer(cv::Mat& img, vector<vector<vector<int>>>& boxes, vector<double> &times){

    ////////////////////// preprocess ////////////////////////
    float ratio_h{}; // = resize_h / h
    float ratio_w{}; // = resize_w / w

    cv::Mat srcimg;
    cv::Mat resize_img;
    img.copyTo(srcimg);
    
    auto preprocess_start = std::chrono::steady_clock::now();

    //  图像的宽和高处理为32的倍数后输出，最长不会超出最大预设值
    this->resize_op_.Run(img, resize_img, this->max_side_len_, ratio_h, ratio_w);
    //
    this->normalize_op_.Run(&resize_img, this->mean_, this->scale_, true);

    auto preprocess_end = std::chrono::steady_clock::now();

    //////////////////////// inference //////////////////////////
    void* buffers[2];
    // 为buffer[0]指针（输入）定义空间大小
    float *inBlob = new float[1 * 3 * resize_img.rows * resize_img.cols];
    this->permute_op_.Run(&resize_img, inBlob);

    int inputIndex = 0;

    CHECK(cudaMalloc(&buffers[inputIndex], 1 * 3 * resize_img.rows * resize_img.cols * sizeof(float)));
    
    auto inference_start = std::chrono::steady_clock::now();
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // 将数据放到gpu上
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inBlob, 1 * 3 * resize_img.rows * resize_img.cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    //#### 将输入图像的大小写入context中 #######
    context->setOptimizationProfileAsync(0, stream); // 让convert.h创建engine的动态输入配置生效
    auto in_dims = context->getBindingDimensions(inputIndex); //获取带有可变维度的输入维度信息
    
    in_dims.d[0]=1;
    in_dims.d[2]=resize_img.rows;
    in_dims.d[3]=resize_img.cols;
    
    context->setBindingDimensions(inputIndex, in_dims); // 根据输入图像大小更新输入维度

    // 为buffer[1]指针（输出）定义空间大小
    int outputIndex = 1;
    auto out_dims = context->getBindingDimensions(outputIndex);
    int output_size=1;
    for(int j=0; j<out_dims.nbDims; j++) 
        output_size *= out_dims.d[j];

    float *outBlob = new float[output_size];
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // 做推理
    context->enqueueV2(buffers, stream, nullptr);
    // 从gpu取数据到cpu上
    CHECK(cudaMemcpyAsync(outBlob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    auto inference_end = std::chrono::steady_clock::now();

    ///////////////////// postprocess //////////////////////
    auto postprocess_start = std::chrono::steady_clock::now();
    vector<int> output_shape;
    for(int j=0; j<out_dims.nbDims; j++) 
        output_shape.push_back(out_dims.d[j]);
    int n2 = output_shape[2];
    int n3 = output_shape[3];
    int n = n2 * n3; // output_h * output_w

    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');

    for (int i = 0; i < n; i++) {
        pred[i] = float(outBlob[i]);
        cbuf[i] = (unsigned char)((outBlob[i]) * 255);
    }

    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

    const double threshold = this->det_db_thresh_ * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    if (this->use_dilation_) {
        cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    boxes = post_processor_.BoxesFromBitmap(
            pred_map, bit_map, this->det_db_box_thresh_,
            this->det_db_unclip_ratio_, this->use_polygon_score_);

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg); // 将resize_img中得到的bbox 映射回srcing中的bbox

    auto postprocess_end = std::chrono::steady_clock::now();
    std::cout << "Detected boxes num: " << boxes.size() << endl;

    std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
    times[0] = double(preprocess_diff.count() * 1000);
    std::chrono::duration<float> inference_diff = inference_end - inference_start;
    times[1] = double(inference_diff.count() * 1000);
    std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
    times[2] = double(postprocess_diff.count() * 1000);

    delete [] inBlob;
    delete [] outBlob;
}

TextDetect::~TextDetect(){
  
}

} //namespace OCR
