#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <sstream>

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

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

// engine file path (deserialize from)
const char* ENGINE_FILE = "lpr_20231102_all_types_fp16.engine";
// images for inference
const char* pattern_imgs = "E:/plate_recognition/CBLPRD-330k_v1/val_yellow/*.jpg";
static const int INPUT_H = 48; // 24 -> 48
static const int INPUT_W = 96; // 94 -> 96
static const int OUTPUT_SIZE = 18 * 79; // 18 * 68 -> 18 * 79
const char *INPUT_BLOB_NAME = "input";  // "data"
const char *OUTPUT_BLOB_NAME = "output"; // "prob"
static Logger gLogger;
using namespace nvinfer1;
//const std::string alphabet[] = {"¾©", "»¦", "½ò", "Óå", "¼½", "½ú", "ÃÉ", "ÁÉ", "¼ª", "ºÚ",
//                                "ËÕ", "Õã", "Íî", "Ãö", "¸Ó", "Â³", "Ô¥", "¶õ", "Ïæ", "ÔÁ",
//                                "¹ð", "Çí", "´¨", "¹ó", "ÔÆ", "²Ø", "ÉÂ", "¸Ê", "Çà", "Äþ",
//                                "ÐÂ",
//                                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
//                                "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
//                                "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
//                                "W", "X", "Y", "Z", "I", "O", "-"
//};

const std::string alphabet[] = {"¾©", "»¦", "½ò", "Óå", "¼½", "½ú", "ÃÉ", "ÁÉ", "¼ª", "ºÚ",
        "ËÕ", "Õã", "Íî", "Ãö", "¸Ó", "Â³", "Ô¥", "¶õ", "Ïæ", "ÔÁ",
        "¹ð", "Çí", "´¨", "¹ó", "ÔÆ", "²Ø", "ÉÂ", "¸Ê", "Çà", "Äþ",
        "ÐÂ", "Ñ§", "¸Û", "°Ä", "¾¯", "Ê¹", "Áì", "Ó¦", "¼±", "¹Ò",
        "Ãñ", "º½", "ÁÙ",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z", "-"
};


void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        // std::cerr << "./retina_mnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./LPRnet_multiline.exe -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    //if (argc == 2 && std::string(argv[1]) == "-s") {
    //    IHostMemory *modelStream{nullptr};
    //    APIToModel(BATCH_SIZE, &modelStream);
    //    assert(modelStream != nullptr);
    //    std::ofstream p("LPRnet.engine", std::ios::binary);
    //    if (!p) {
    //        std::cerr << "could not open plan output file" << std::endl;
    //        return -1;
    //    }
    //    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    //    modelStream->destroy();
    //    return 0;
    //} else if (argc == 2 && std::string(argv[1]) == "-d") {
    //    std::ifstream file("lpr_20231102_all_types_fp16.engine", std::ios::binary);
    //    if (file.good()) {
    //        file.seekg(0, file.end);
    //        size = file.tellg();
    //        file.seekg(0, file.beg);
    //        trtModelStream = new char[size];
    //        assert(trtModelStream);
    //        file.read(trtModelStream, size);
    //        file.close();
    //    }
    //} else {
    //    std::cerr << "arguments not right!" << std::endl;
    //    std::cerr << "./LPRnet -s  // serialize model to plan file" << std::endl;
    //    std::cerr << "./LPRnet -d ../samples  // deserialize plan file and run inference" << std::endl;
    //    return -1;
    //}

    // deserialize
    std::cout << "deserializing from " << ENGINE_FILE << std::endl;
    std::ifstream file(ENGINE_FILE, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    std::vector<cv::String> fn;
    cv::glob(pattern_imgs, fn, false);

    size_t count = fn.size(); //number of jpg files in images folder


    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);


    for (size_t idx_file = 0; idx_file < count; idx_file++) {
        cv::Mat img = cv::imread(fn[idx_file]);
        cv::Mat pr_img;
        // preprocess 
        //cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_CUBIC);
        cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR); // #####################################
        std::cout << "image: " << idx_file + 1 << " of " << count << std::endl; // 11
        cv::imshow("input", img);
        cv::waitKey(0);

        // For multi-batch, I feed the same image multiple times.
        // If you want to process different images in a batch, you need adapt it.
       //cv::Mat blob = cv::dnn::blobFromImage(pr_img, 0.0078125, pr_img.size(), cv::Scalar(127.5, 127.5, 127.5), true,
                                              //false);
        int i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[2] - 127.5) * 0.0078125;
                data[i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] - 127.5) * 0.0078125;
                data[i] = ((float)uc_pixel[0] - 127.5) * 0.0078125;
                uc_pixel += 3;
                ++i;
            }
        }


        /*   for (int i = 0; i < INPUT_H; i++) {
               for (int j = 0; j < INPUT_W; j++) {
               data[i * INPUT_W+j] = (float)(pr_img.at<cv::Vec3b>(i, j)[0] - 127.5) * 0.0078125;
               data[i * INPUT_W + j + INPUT_H * INPUT_W] = (float)(pr_img.at<cv::Vec3b>(i, j)[1] - 127.5) * 0.0078125;
               data[i * INPUT_W + j + 2 * INPUT_H * INPUT_W] = (float)(pr_img.at<cv::Vec3b>(i, j)[2] - 127.5) * 0.0078125;
               }
           }*/

           /* printf("%f ", (float)pr_img.at<cv::Vec3b>(0, 0)[0]);
            printf("%f ", (float)pr_img.at<cv::Vec3b>(0, 0)[1]);
            printf("%f ", (float)pr_img.at<cv::Vec3b>(0, 0)[2]);

            printf("%f ", data[0]);
            printf("%f ", data[0 + INPUT_H * INPUT_W]);
            printf("%f ", data[0 + 2 * INPUT_H * INPUT_W]);*/

        // Run inference
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
        std::vector<int> preds;
        //std::cout << std::endl;


        // decode
        for (int i = 0; i < 18; i++) {
            int maxj = 0;
            for (int j = 0; j < 79; j++) { // ################ 68 -> 79
                // if (prob[i + 18 * j] > prob[i + 18 * maxj]) maxj = j;
                if (prob[79 * i + j] > prob[79 * i + maxj]) maxj = j; // ###########################
            }
            // printf("%d ", maxj);
            preds.push_back(maxj);
        }
        int pre_c = preds[0];
        std::vector<int> no_repeat_blank_label;

        if (pre_c != 79 - 1) no_repeat_blank_label.push_back(pre_c); //##########################

        for (auto c: preds) {
            if (c == pre_c || c == 79 - 1) { // ################ 68 -> 79
                if (c == 79 - 1) pre_c = c; // ################ 68 -> 79
                continue;
            }
            no_repeat_blank_label.push_back(c);
            pre_c = c;
        }
        std::string str;
        for (auto v: no_repeat_blank_label) {
            str += alphabet[v];
        }
        std::cout<<"result:"<<str<<std::endl;
        std::cout << std::endl;
    }
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    return 0;
}
