#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "calibrator.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
#define CONF_THRESH 0.5
#define IOU_THRESH 0.4

// retinaface engine file path (only deserialize from)
const char* ENGINE_FILE_DET = "retina_mnet.engine";
// lprnet engine file path (only deserialize from)
const char* ENGINE_FILE_REC = "lpr_20231102_all_types_fp16.engine";
// images for inference
const char* pattern_imgs = "E:/plate_recognition/CCPD2020/ccpd_green/val/*.jpg";

// retinaface
static const int INPUT_H_DET = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
static const int INPUT_W_DET = decodeplugin::INPUT_W;;
//static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
static const int OUTPUT_SIZE_DET = (INPUT_H_DET / 8 * INPUT_W_DET / 8 + INPUT_H_DET / 16 * INPUT_W_DET / 16 + INPUT_H_DET / 32 * INPUT_W_DET / 32) * 2 * 13 + 1; // ##########
const char* INPUT_BLOB_NAME_DET = "data";
const char* OUTPUT_BLOB_NAME_DET = "prob";

// lprnet
static const int INPUT_H_REC = 48; // 24 -> 48
static const int INPUT_W_REC = 96; // 94 -> 96
static const int OUTPUT_SIZE_REC = 18 * 79; // 18 * 68 -> 18 * 79
const char* INPUT_BLOB_NAME_REC = "input";  // "data"
const char* OUTPUT_BLOB_NAME_REC = "output"; // "prob"

const std::string alphabet[] = { "¾©", "»¦", "½ò", "Óå", "¼½", "½ú", "ÃÉ", "ÁÉ", "¼ª", "ºÚ",
        "ËÕ", "Õã", "Íî", "Ãö", "¸Ó", "Â³", "Ô¥", "¶õ", "Ïæ", "ÔÁ",
        "¹ð", "Çí", "´¨", "¹ó", "ÔÆ", "²Ø", "ÉÂ", "¸Ê", "Çà", "Äþ",
        "ÐÂ", "Ñ§", "¸Û", "°Ä", "¾¯", "Ê¹", "Áì", "Ó¦", "¼±", "¹Ò",
        "Ãñ", "º½", "ÁÙ",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z", "-"
};

static Logger gLogger;


class Model
{
public:
    Model() {
        std::cout << "init..." << std::endl;
    };
    ~Model() {
        std::cout << "destroy..." << std::endl;
    };
    
    // Read images from diretory and push them to queue
    void readFrame();

    // trt inference for retinaface
    void doInferenceDet(IExecutionContext& context, float* input, float* output, int batchSize);

    // trt inference for lprnet
    void doInferenceRec(IExecutionContext& context, float* input, float* output, int batchSize);

    // Get images from queue. Do preprocessing, inference and postprocessing
    void inference();

private:

    std::queue <cv::Mat>   mFrameQ;
    std::mutex             mFrameQ_mtx;
    std::atomic_bool mEndFlag = false;
};

void Model::readFrame() {
    std::vector<cv::String> fn;
    cv::glob(pattern_imgs, fn, false);
    size_t count = fn.size(); //number of jpg files in images folder
    std::cout << "found " << count << " images..." << std::endl;
    auto start = std::chrono::system_clock::now();
    for (size_t idx_file = 0; idx_file < count; idx_file++)
    {
        // preprocess input data
        std::cout << "file name: " << fn[idx_file] << std::endl;

        cv::Mat img = cv::imread(fn[idx_file]);

        mFrameQ_mtx.lock();
        mFrameQ.push(img);
        mFrameQ_mtx.unlock();
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "read " << count << " images in " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

    mEndFlag = true;

}

void Model::inference() {

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStreamDet{ nullptr };
    char* trtModelStreamRec{ nullptr };
    size_t size_det{ 0 };
    size_t size_rec{ 0 };

    std::cout << "deserializing from " << ENGINE_FILE_DET << " ..." << std::endl;
    std::ifstream file(ENGINE_FILE_DET, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size_det = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size_det];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size_det);
        file.close();
    }

    std::cout << "deserializing from " << ENGINE_FILE_REC << " ..." << std::endl;
    std::ifstream file_rec(ENGINE_FILE_REC, std::ios::binary);
    if (file_rec.good()) {
        file_rec.seekg(0, file_rec.end);
        size_rec = file_rec.tellg();
        file_rec.seekg(0, file_rec.beg);
        trtModelStreamRec = new char[size_rec];
        assert(trtModelStreamRec);
        file_rec.read(trtModelStreamRec, size_rec);
        file_rec.close();
    }

    static float data_det[BATCH_SIZE * 3 * INPUT_H_DET * INPUT_W_DET];
    static float data_rec[BATCH_SIZE * 3 * INPUT_H_REC * INPUT_W_REC];

    IRuntime* runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    ICudaEngine* engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size_det);
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine_det != nullptr);
    IExecutionContext* context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);


    IRuntime* runtime_rec = createInferRuntime(gLogger);
    assert(runtime_rec != nullptr);
    ICudaEngine* engine_rec = runtime_rec->deserializeCudaEngine(trtModelStreamRec, size_rec);
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine_rec != nullptr);
    IExecutionContext* context_rec = engine_rec->createExecutionContext();
    assert(context_rec != nullptr);

    // Run inference
    static float prob_det[BATCH_SIZE * OUTPUT_SIZE_DET];
    static float prob_rec[BATCH_SIZE * OUTPUT_SIZE_REC];

    size_t frame_count = 0;

    auto start_consume = std::chrono::system_clock::now();
    bool get_first_frame = false;
    while (!mEndFlag || !mFrameQ.empty())
    {
        if (!mFrameQ.empty())
        {   
            if (!get_first_frame) {
                get_first_frame = true;
                start_consume = std::chrono::system_clock::now();
            }
            auto start_total = std::chrono::system_clock::now();

            mFrameQ_mtx.lock();
            auto img = mFrameQ.front();
            frame_count += 1;
            mFrameQ.pop();
            mFrameQ_mtx.unlock();

            // preprocess input data
            cv::Mat pr_img = preprocess_img(img, INPUT_W_DET, INPUT_H_DET);

            // For multi-batch, I feed the same image multiple times.
            // If you want to process different images in a batch, you need adapt it.
            for (int b = 0; b < BATCH_SIZE; b++) {
                float* p_data = &data_det[b * 3 * INPUT_H_DET * INPUT_W_DET];
                for (int i = 0; i < INPUT_H_DET * INPUT_W_DET; i++) {
                    p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
                    p_data[i + INPUT_H_DET * INPUT_W_DET] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
                    p_data[i + 2 * INPUT_H_DET * INPUT_W_DET] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
                }
            }


            auto start = std::chrono::system_clock::now();
            doInferenceDet(*context_det, data_det, prob_det, BATCH_SIZE);
            auto end = std::chrono::system_clock::now();
            std::cout << "retinaface inference time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

            for (int b = 0; b < BATCH_SIZE; b++) {
                std::vector<decodeplugin::Detection> res;
                nms(res, &prob_det[b * OUTPUT_SIZE_DET], IOU_THRESH);
                std::cout << "number of detections -> " << prob_det[b * OUTPUT_SIZE_DET] << std::endl;
                std::cout << " -> " << prob_det[b * OUTPUT_SIZE_DET + 10] << std::endl;
                std::cout << "after nms -> " << res.size() << std::endl;
                cv::Mat tmp = img.clone();
                for (size_t j = 0; j < res.size(); j++) {
                    if (res[j].class_confidence < CONF_THRESH) continue;
                    //cv::Rect r = get_rect_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
                    auto xywh = get_rect_adapt_landmark(tmp, INPUT_W_DET, INPUT_H_DET, res[j].bbox, res[j].landmark);
                    auto x1 = xywh[0];
                    auto y1 = xywh[1];
                    auto w_bbox = xywh[2];
                    auto h_bbox = xywh[3];


                    if (x1 < 0 || (x1 + w_bbox) > INPUT_W_DET || y1 < 0 || (y1 + h_bbox) > INPUT_H_DET) continue; // ########################

                    cv::Point2f src_points[] = {
                    cv::Point2f(res[j].landmark[4] - x1,res[j].landmark[5] - y1),
                    cv::Point2f(res[j].landmark[6] - x1,res[j].landmark[7] - y1),
                    cv::Point2f(res[j].landmark[2] - x1,res[j].landmark[3] - y1),
                    cv::Point2f(res[j].landmark[0] - x1,res[j].landmark[1] - y1)
                    };

                    cv::Point2f dst_points[] = {
                        cv::Point2f(0, 0),
                        cv::Point2f(94, 0),
                        cv::Point2f(0, 24),
                        cv::Point2f(94, 24)
                    };

                    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);

                    cv::Mat processed;

                    cv::warpPerspective(img(cv::Range((int)y1, (int)(y1 + h_bbox)), cv::Range((int)x1, (int)(x1 + w_bbox))), processed, M, cv::Size(94, 24), cv::INTER_LINEAR);

                    cv::Mat pr_img_rec;
                    // preprocess 
                    cv::resize(processed, pr_img_rec, cv::Size(INPUT_W_REC, INPUT_H_REC), 0, 0, cv::INTER_LINEAR); // #####################################

                    int i = 0;
                    for (int row = 0; row < INPUT_H_REC; ++row) {
                        uchar* uc_pixel = pr_img_rec.data + row * pr_img_rec.step;
                        for (int col = 0; col < INPUT_W_REC; ++col) {
                            data_rec[i + 2 * INPUT_H_REC * INPUT_W_REC] = ((float)uc_pixel[2] - 127.5) * 0.0078125;
                            data_rec[i + INPUT_H_REC * INPUT_W_REC] = ((float)uc_pixel[1] - 127.5) * 0.0078125;
                            data_rec[i] = ((float)uc_pixel[0] - 127.5) * 0.0078125;
                            uc_pixel += 3;
                            ++i;
                        }
                    }

                    auto start = std::chrono::system_clock::now();
                    doInferenceRec(*context_rec, data_rec, prob_rec, BATCH_SIZE);
                    auto end = std::chrono::system_clock::now();
                    std::cout << "lprnet inference time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

                    std::vector<int> preds;

                    // decode
                    for (int i = 0; i < 18; i++) {
                        int maxj = 0;
                        for (int j = 0; j < 79; j++) { // ################ 68 -> 79
                            // if (prob[i + 18 * j] > prob[i + 18 * maxj]) maxj = j;
                            if (prob_rec[79 * i + j] > prob_rec[79 * i + maxj]) maxj = j; // ###########################
                        }
                        // printf("%d ", maxj);
                        preds.push_back(maxj);
                    }
                    int pre_c = preds[0];
                    std::vector<int> no_repeat_blank_label;

                    if (pre_c != 79 - 1) no_repeat_blank_label.push_back(pre_c); //##########################

                    for (auto c : preds) {
                        if (c == pre_c || c == 79 - 1) { // ################ 68 -> 79
                            if (c == 79 - 1) pre_c = c; // ################ 68 -> 79
                            continue;
                        }
                        no_repeat_blank_label.push_back(c);
                        pre_c = c;
                    }
                    std::string str;
                    for (auto v : no_repeat_blank_label) {
                        str += alphabet[v];
                    }

                    auto end_total = std::chrono::system_clock::now();

                    std::cout << "result:" << str << std::endl;
                    std::cout << "total time:" << std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count() << "us" << std::endl;
                    std::cout << std::endl;

                    //cv::imshow("aligned", processed);

                    cv::Rect r = cv::Rect(x1, y1, w_bbox, h_bbox);
                    cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
                    /*for (int k = 0; k < 10; k += 2) {
                        cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
                    }*/
                    for (int k = 0; k < 8; k += 2) {
                        cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 6), 255 * (k < 4)), 4);
                    }
                }
                // cv::imwrite(std::to_string(b) + "_result.jpg", tmp);
                //cv::imshow("result", tmp);
                //cv::waitKey(0);
            }

        }
        cv::waitKey(1);
        //std::this_thread::sleep_for(std::chrono::milliseconds(5));
        auto end_consume = std::chrono::system_clock::now();
        std::cout << "processed " << frame_count << " frames..." << std::endl;
    }

    auto end_consume = std::chrono::system_clock::now();
    std::cout << "use:" << std::chrono::duration_cast<std::chrono::microseconds>(end_consume - start_consume).count() << "us" << std::endl;

    // Destroy the engine
    context_det->destroy();
    engine_det->destroy();
    runtime_det->destroy();
    context_rec->destroy();
    engine_rec->destroy();
    runtime_rec->destroy();

}


void Model::doInferenceDet(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME_DET);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME_DET);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H_DET * INPUT_W_DET * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE_DET * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H_DET * INPUT_W_DET * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE_DET * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void Model::doInferenceRec(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME_REC);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME_REC);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H_REC * INPUT_W_REC * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE_REC * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H_REC * INPUT_W_REC * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE_REC * sizeof(float), cudaMemcpyDeviceToHost,
        stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    Model model;
    std::thread t_consume(&Model::inference, &model);
    std::thread t_read(&Model::readFrame, &model);
    t_consume.join();
    t_read.join();
    return 0;
}
