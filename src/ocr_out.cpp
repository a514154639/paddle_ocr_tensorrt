#include "ocr_out.h"
#include <cstring> // 用于 std::strcpy 和 std::strlen
#include <time.h>

//std::unique_ptr<nvinfer1::ICudaEngine> engine = nullptr;
//IExecutionContext* context = nullptr;
//std::unique_ptr<nvinfer1::IExecutionContext> context = nullptr;
IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
//IExecutionContext* context = nullptr;
std::vector<std::string> label_list_;
Logger gLogger; //日志
std::vector<float> det_mean_ = {0.485f, 0.456f, 0.406f};
std::vector<float> det_scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f}; 
std::vector<float> rec_mean_ = {0.5f, 0.5f, 0.5f};
std::vector<float> rec_scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f}; 
std::vector<int> rec_image_shape_ = {3, 48, 320};
std::vector<std::vector<int>> box;

IExecutionContext* infer_init_ocr(const char* modelPath){
    //std::ifstream file(engineName, std::ios::binary);
    IExecutionContext* context = nullptr;
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << modelPath << " error!" << std::endl;
        return nullptr;
    }
    
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    std::cout << "read " << modelPath << " success!" << std::endl;

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    //context = engine->createExecutionContext();
    context = engine->createExecutionContext();
    assert(context != nullptr); 
    //std::cout << "create context success!" << std::endl;

    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2); //check if is a input and a output
    return context;
}


void det(cv::Mat inputImg, std::vector<OCRPredictResult> &ocr_results, std::vector<OCRdetResult> &det_results, const Det_ModelParam& det_param, IExecutionContext* context) {
    
    if (inputImg.channels() != 3) {
        OCR::Utility::get_3channels_img(inputImg);
    }
    std::vector<std::vector<std::vector<int>>> boxes;
    Model_Infer_det(inputImg, boxes, det_param, context);
    for (std::size_t i = 0; i < boxes.size(); i++) {
        OCRPredictResult res;
        OCRdetResult det_res;
        det_res.box = boxes[i];
        res.score = -1;
        ocr_results.push_back(res);
        det_results.push_back(det_res);
    }
}


// void rec_old(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results, const Rec_ModelParam& rec_param, IExecutionContext* context) {
    
//     std::vector<std::string> rec_texts(img_list.size(), "");
//     std::vector<float> rec_text_scores(img_list.size(), 0);
//     Model_Infer_rec(img_list, rec_texts, rec_text_scores, rec_param, context);
//     //cout << "rec_text size: " << rec_texts.size() << endl;
//     for (std::size_t i = 0; i < rec_texts.size(); i++) {
//         ocr_results[i].text = rec_texts[i];
//         ocr_results[i].score = rec_text_scores[i];
//     }
// }



void rec(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results, const Rec_ModelParam& rec_param, IExecutionContext* context) {
    
    std::vector<char*> rec_texts(img_list.size(), nullptr);  // 使用 char* array
    std::vector<float> rec_text_scores(img_list.size(), 0);

    // ... 假设 Model_Infer_rec 已经适应 char* 并且填充 rec_texts ...
    Model_Infer_rec(img_list, rec_texts, rec_text_scores, rec_param, context);

    for (std::size_t i = 0; i < rec_texts.size(); i++) {
        int len = std::strlen(rec_texts[i]);
        ocr_results[i].text = new char[len + 1];  // 为 text 分配内存，包括 \0
        std::strcpy(ocr_results[i].text, rec_texts[i]); // 拷贝字符串到新分配的内存
        ocr_results[i].score = rec_text_scores[i];
        
        // 如果不再需要 rec_texts[i], 应该在这里释放它的内存
        delete[] rec_texts[i]; // 只有在确定你拥有 rec_texts[i] 的内存时才这么做
    }
}


void Model_Infer_det(cv::Mat& img, vector<vector<vector<int>>>& boxes, const Det_ModelParam& det_param, IExecutionContext* context){
    OCR::ResizeImgType0 resize_op_;
    OCR::Normalize normalize_op_;
    OCR::Permute permute_op_ ;
    OCR::PostProcessor post_processor_ ;

    ////////////////////// preprocess ////////////////////////
    float ratio_h{}; // = resize_h / h
    float ratio_w{}; // = resize_w / w

    cv::Mat srcimg;
    cv::Mat resize_img;
    img.copyTo(srcimg);

    //  图像的宽和高处理为32的倍数后输出，最长不会超出最大预设值
    resize_op_.Run(img, resize_img, 640, ratio_h, ratio_w);
    //std::cout << "max_side_len_:" << det_param.max_side_len_ << std::endl;
    //
    normalize_op_.Run(&resize_img, det_mean_, det_scale_, true);

    //////////////////////// inference //////////////////////////
    void* buffers[2];
    // 为buffer[0]指针（输入）定义空间大小
    float *inBlob = new float[1 * 3 * resize_img.rows * resize_img.cols];
    permute_op_.Run(&resize_img, inBlob);
    //permute_op_(&resize_img, inBlob);

    int inputIndex = 0;
    CHECK(cudaMalloc(&buffers[inputIndex], 1 * 3 * resize_img.rows * resize_img.cols * sizeof(float)));
    
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // 将数据放到gpu上
    //std::cout << "data->gpu " << std::endl;
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inBlob, 1 * 3 * resize_img.rows * resize_img.cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    //std::cout << "image -> context " << std::endl;
    //#### 将输入图像的大小写入context中 #######
    context->setOptimizationProfileAsync(0, stream); // 让convert.h创建engine的动态输入配置生效
    //std::cout << "context binding " << std::endl;
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
    //std::cout << "infer " << std::endl;
    // 做推理
    context->enqueueV2(buffers, stream, nullptr);
    // 从gpu取数据到cpu上
    //std::cout << "gpu -> cpu " << std::endl;
    CHECK(cudaMemcpyAsync(outBlob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    ///////////////////// postprocess //////////////////////
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

    const double threshold = det_param.det_db_thresh_ * 255;
    //std::cout << "det_param.det_db_thresh_ "<< det_param.det_db_thresh_ << std::endl;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    if (det_param.use_dilation_) {
        cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    boxes = post_processor_.BoxesFromBitmap(
            pred_map, bit_map, det_param.det_db_box_thresh_,
            det_param.det_db_unclip_ratio_, det_param.use_polygon_score_);
    //std::cout << "det_param.det_db_unclip_ratio_ "<< det_param.det_db_unclip_ratio_ << std::endl;
    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg); // 将resize_img中得到的bbox 映射回srcing中的bbox

    //std::cout << "Detected boxes num: " << boxes.size() << endl;

    delete [] inBlob;
    delete [] outBlob;
    //std::cout << "det finish " << std::endl;
}


  
void Model_Infer_rec(vector<cv::Mat> img_list, std::vector<char*> &rec_texts, std::vector<float> &rec_text_scores, const Rec_ModelParam& rec_param, IExecutionContext* context){
    OCR::CrnnResizeImg resize_op_rec;
    OCR::PermuteBatch permute_op_rec;
    OCR::Normalize normalize_op_;
    label_list_ = OCR::Utility::ReadDict(rec_param.label_path_);
    //std::cout << "rec_param.label_path_ "<< rec_param.label_path_ << std::endl;
    label_list_.insert(label_list_.begin(), "#");
    label_list_.push_back(" ");
    int img_num = img_list.size();
    std::vector<float> width_list;
    for (int i = 0; i < img_num; i++){
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    
    std::vector<int> indices = OCR::Utility::argsort(width_list);//对宽高比由小到大进行排序，并获取indices

    int rec_batch_num = rec_param.rec_batch_num_;
    //std::cout << "rec_param.rec_batch_num_ "<< rec_param.rec_batch_num_ << std::endl;
    // if(img_num > 0){
    //     if (rec_param.rec_batch_num_ > get_engine_max_batch()){
    //         rec_batch_num = get_engine_max_batch();
    //         std::cerr<<"Your rec_batch_num is: " <<rec_param.rec_batch_num_ <<
    //         " greater than MAX_DIMS_[0], and is reset to: "<<get_engine_max_batch()<<" !"<<std::endl;
    //     }
    // }
    for(int beg_img_no = 0; beg_img_no < img_num; beg_img_no += rec_batch_num){
        /////////////////////////// preprocess ///////////////////////////////
        int end_img_no = min(img_num, beg_img_no + rec_batch_num);
        int batch_num = end_img_no - beg_img_no;
        int imgH = rec_image_shape_[1];
        int imgW = rec_image_shape_[2];
        float max_wh_ratio = imgW * 1.0 / imgH;

    //    I do not think this step bellow is necessary, because we want to get max_wh_ratio,
    //    indices is just the index from smallest to largest according to wh_ratio
        max_wh_ratio = max(max_wh_ratio, width_list[indices[end_img_no-1]]); // maybe it will be faster
//        for (int ino = beg_img_no; ino < end_img_no; ino++) {
//            int h = img_list[indices[ino]].rows;
//            int w = img_list[indices[ino]].cols;
//            float wh_ratio = w * 1.0 / h;
//            max_wh_ratio = max(max_wh_ratio, wh_ratio);
//        }

        // 将img按照从小到大的宽高比依次处理并放入norm_img_batch中
        // 处理方法为resize到高为rec_img_h，宽为rec_img_h*max_wh_ratio
        // 并做归一化。
        int batch_width = imgW;
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino ++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            resize_op_rec.Run(srcimg, resize_img, max_wh_ratio, rec_image_shape_);
            normalize_op_.Run(&resize_img, rec_mean_, rec_scale_, true);
            norm_img_batch.push_back(resize_img);
            batch_width = max(resize_img.cols, batch_width);
        }
      
        ////////////////////////// inference /////////////////////////
        void* buffers[2];

        // 为buffer[0]指针（输入）定义空间大小
        int data_size = batch_num * 3 * imgH * batch_width;
        float *inBlob = new float[data_size];
        permute_op_rec.Run(norm_img_batch, inBlob);
        //permuteBatch_op_(norm_img_batch, inBlob);

//        int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        int inputIndex = 0;
        CHECK(cudaMalloc(&buffers[inputIndex], data_size * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        //std::cout << "data -> gpu!" << std::endl;
        // 将数据放到gpu上
        CHECK(cudaMemcpyAsync(buffers[inputIndex], inBlob, data_size * sizeof(float), cudaMemcpyHostToDevice, stream));

        //#### 将输入图像的大小写入context中 #######
        //std::cout << "iamge -> contex!" << std::endl;
        context->setOptimizationProfileAsync(0, stream);
        auto in_dims = context->getBindingDimensions(inputIndex);
        in_dims.d[0]=batch_num;
        in_dims.d[1]=3;
        in_dims.d[2]=imgH;
        in_dims.d[3]=batch_width;
        
        context->setBindingDimensions(inputIndex, in_dims);

        // 为buffer[1]指针（输出）定义空间大小
        int outputIndex = 1;
        auto out_dims = context->getBindingDimensions(outputIndex);

        int output_size=1;
        for(int j=0; j<out_dims.nbDims; j++) 
            output_size *= out_dims.d[j];

        float *outBlob = new float[output_size];
        CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

        //std::cout << "rec infer" << std::endl;
        // 做推理
        context->enqueue(1, buffers, stream, nullptr);
        // 从gpu取数据到cpu上
        //std::cout << "gpu -> cpu" << std::endl;
        CHECK(cudaMemcpyAsync(outBlob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        //std::cout << "release stream" << std::endl;
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
      
        ////////////////////// postprocess ///////////////////////////
        //std::cout << "post process" << std::endl;
        vector<int> predict_shape;
        for(int j=0; j<out_dims.nbDims; j++) 
            predict_shape.push_back(out_dims.d[j]);
        for (int m = 0; m < predict_shape[0]; m++) { // m = batch_size
//            pair<vector<string>, double> temp_box_res;
            std::string str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape[1]; n++) { // n = 2*l + 1
                argmax_idx =
                    int(OCR::Utility::argmax(&outBlob[(m * predict_shape[1] + n) * predict_shape[2]],
                                        &outBlob[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                max_value =
                    float(*std::max_element(&outBlob[(m * predict_shape[1] + n) * predict_shape[2]],
                                            &outBlob[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res += label_list_[argmax_idx];
                    //str_res = str_res.append(label_list_[argmax_idx]);
                    //std::cout <<"argmax_idx: " << argmax_idx << " "<< "label_list_[argmax_idx]: " << label_list_[argmax_idx] << std::endl;                 
                }
                last_index = argmax_idx;
               
            }
            score /= count;
            if (isnan(score)){
                continue;
            }
            //std::cout << "rec text: " << str_res << std::endl;
            //rec_texts[indices[beg_img_no + m]] = str_res;
            //rec_text_scores[indices[beg_img_no + m]] = score;

            // Allocate memory for the C-style string
            char* c_string = new char[str_res.length() + 1];
            //cout << "str_res.length() + 1: " << str_res.length() + 1 << endl;
            std::strcpy(c_string, str_res.c_str());

            // Store the pointer in the vector (or you can directly use it as needed)
            rec_texts[indices[beg_img_no + m]] = c_string;

            // Remember, we will need to delete [] this memory when done
            rec_text_scores[indices[beg_img_no + m]] = score;
        }

        delete [] inBlob;
        delete [] outBlob;
    }
    //std::cout << "rec finish " << std::endl;
}



OCRPredictResult* run(cv::Mat &inputImg, const Det_ModelParam& det_param, const Rec_ModelParam& rec_param, IExecutionContext* det_context, IExecutionContext* rec_context, int& outSize) {
    std::vector<OCRPredictResult> ocr_result_vector;
    std::vector<OCRdetResult> det_result_vector;
    // ---------------------- det ----------------------
    //cout <<"det start"<< endl;
    //cv::imwrite("./out.jpg", inputImg);
    auto det_start = std::chrono::high_resolution_clock::now();
    det(inputImg, ocr_result_vector,det_result_vector, det_param, det_context);
    auto det_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = det_end - det_start;
    std::cout << "det time: " << elapsed.count() << " ms" << std::endl;

    // crop image
    std::vector<cv::Mat> img_list;
    for (std::size_t j = 0; j < det_result_vector.size(); j++) {
        cv::Mat crop_img;
        crop_img = OCR::Utility::GetRotateCropImage(inputImg, det_result_vector[j].box);
        img_list.push_back(crop_img);
    }
    //cout << "crop_img num:" << img_list.size() << endl;
    // ---------------------- rec ----------------------
    //cout <<"rec start"<< endl;
    auto rec_start = std::chrono::high_resolution_clock::now();
    rec(img_list, ocr_result_vector, rec_param, rec_context);
    auto rec_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_rec = rec_end - rec_start;
    std::cout << "rec time: " << elapsed_rec.count() << " ms" << std::endl;
   
    
    std::cout << "infer success!" << std::endl;
    // Convert the vector to a dynamic array
    outSize = ocr_result_vector.size();
     for (std::size_t i = 0; i < outSize; i++) {

        // rec
        if (ocr_result_vector[i].score != -1.0) {
            std::cout << " rec text: " << ocr_result_vector[i].text << " " <<std::endl
                        << " rec score: " << ocr_result_vector[i].score << " " << std::endl;
        }

    }
    
    OCRPredictResult* ocr_result_array = new OCRPredictResult[outSize];

    std::copy(ocr_result_vector.begin(), ocr_result_vector.end(), ocr_result_array);
    
    return ocr_result_array;
}


void print_result(const std::vector<OCRPredictResult>& ocr_result) {
        
    std::ofstream output_file("output.txt"); // 打开输出文件
    output_file << ocr_result.size() << std::endl;
    for (std::size_t i = 0; i < ocr_result.size(); i++) {
        output_file << i << "\t";

        // rec
        if (ocr_result[i].score != -1.0) {
            output_file << " rec text: " << ocr_result[i].text << " "
                        << " rec score: " << ocr_result[i].score << " ";
        }

        output_file << std::endl;
    }

    output_file.close(); // 关闭输出文件
}

void infer_release(IExecutionContext *context)
{
	// Release stream and buffers
    context->destroy();
    engine->destroy();
    runtime->destroy();
}