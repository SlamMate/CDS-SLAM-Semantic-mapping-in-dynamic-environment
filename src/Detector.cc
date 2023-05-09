#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#include <thread>
#include <iomanip>
#include "Detector.h"

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

 #define DEVICE 0  // GPU id GPU设备的编号，需要在终端中查看
 #define NMS_THRESH 0.45  //非极大值抑制阈值
 #define BBOX_CONF_THRESH 0.3 //目标检测框概率阈值

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
// 输入图像的长和宽，要识别的类别，和blob的名称
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 80;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static Logger gLogger;
char* class_names[]= {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
const char* engine;

namespace ORB_SLAM3
{
//创建一个目标检测器类，从文件中初始化模型
Detector::Detector()
{
	//先初始化是否是新的图像为false
	mbNewImgFlag = false;
	//初始化请求功能是否完成为false
	mbFinishRequested = false;
	
	//引擎的路径
	engine = "engine/model_trt.engine";

}


//目标检测器的运行函数
void Detector::Run()
{

	while(1)
    {		
	//一微秒运行一次
        usleep(1);
        //如果不是新来的图像就不做处理
        if(!isNewImgArrived()){
			continue;
		}
	//如果是信赖的图像就清空people_boxes,并且开始目标检测
		// empty people_boxes vector
		//people_boxes.clear();
		dynamic_boxes.clear();
		boxes1.clear();
		for(int i = 0 ; i < 80 ; i++)
		label[i] = 0;
		// half_boxes.clear();
		// stay_boxes.clear();
		
		// perform YOLO detection
		Detect();
	//如果目标检测完成就结束Run功能
		if(isFinished())
        {
            break;
        }

	}

}

// 查询是否目标检测过程已经完成
bool Detector::isFinished()
{
    // 这是独占锁，这个进程生命周期结束后，其他的进程才能运行
    unique_lock<mutex> lock(mMutexFinish);
    
    // 返回一个标志变量，这个变量将在过程完成时被标记为true
    return mbFinishRequested;
}

// 过程已经完成时候将标记变量赋值为true  
void Detector::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested=true;
}

// 给mp跟踪器赋值
void Detector::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}


cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    //格式转换
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
{

    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)//如果大于置信度的话
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;//这是类别的id
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

//注意，这个能规定通道数
float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}


static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(strides, grid_strides);
        generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
        // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        // std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}
//每一种物体对应的颜色
const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

//要检测的类别名
static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

       /** fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height); **/

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        //sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    // 这个是结果的保存   
    //cv::imwrite("det_res.jpg", image);
    //cv::imshow("Detected Frame",image);
    //int cvWaitKey(int k);
    // 函数的功能是刷新图像，其中参数k单位是毫秒，表示刷新频率。
    //cv::waitKey(1);
    // fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
}


void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // 指向要传递到引擎的输入和输出设备缓冲区的指针。
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // 为了绑定缓冲区，我们需要知道输入和输出张量的名称。
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void Detector::Detect() {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    
    fig_area = mImg.rows * mImg.cols;
    // 初始化aprox_area		
    aprox_area = 0;
    //factor初始化？这是用来干啥的
    fig_factor = fig_area/fig_param;

    //argc是从终端读入的参数的数量
    //argv是从终端读入的具体的参数的数值
    	//第一个参数是引擎的路径
        const std::string engine_file_path {engine};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    //第三个参数是读入的图片的路径
    // const std::string input_image_path {inputv[3]};

    //std::vector<std::string> file_names;
    //if (read_files_in_dir(argv[2], file_names) < 0) {
        //std::cout << "read_files_in_dir failed." << std::endl;
        //return -1;
    //}

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    //置信度概率数组的大小是输出结果的size
    static float* prob = new float[output_size];

    //读入图片
    cv::Mat img = mImg;
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);
    // std::cout << "blob image" << std::endl;

    float* blob;
    blob = blobFromImage(pr_img);
    //取行数列数的较小值
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));

    // run inference 这应该是推理过程
    auto start = std::chrono::system_clock::now();
    doInference(*context, blob, prob, output_size, pr_img.size());
    auto end = std::chrono::system_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<Object> objects;
    //这相当于后处理，NMS抑制
    decode_outputs(prob, objects, scale, img_w, img_h);
    draw_objects(img, objects);
    //定义计数器，看看检测出多少人
//    int people_counter {0};
    int dynamic_counter {0};
//    int half_counter {0};
//    int stay_counter {0};
    //遍历整个classId数组
    for (size_t i = 0; i < objects.size(); ++i){
        label[i] = objects[i].label;
        // boxes[i] = objects[i].rect; 这个操作数组内存越界了
        boxes1.push_back(objects[i].rect);
	if(class_names[objects[i].label]=="person"){
		dynamic_counter++;		
		dynamic_boxes.push_back(objects[i].rect);

	}

	}
	//这段代码是用来调试，输出检测出来的人的总数
	//std::cout << "people: " << people_counter << std::endl;
	SetDetectionFlag();
    // 下面及时销毁指针，防止内存泄漏。
    // delete the pointer to the float
    delete blob;
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}
//判断图像是否是新来的，是新来的就改为false，返回true。线程独占锁。
bool Detector::isNewImgArrived()
{
    unique_lock<mutex> lock(mMutexGetNewImg);
    if(mbNewImgFlag)
    {
        mbNewImgFlag=false;
        return true;
    }
    else
    	return false;
}
//设置目标检测标志量（将新的这个）。线程独占锁。
void Detector::SetDetectionFlag()
{
    std::unique_lock <std::mutex> lock(mMutexNewImgDetection);
   
   //这个标志量需要看tracking
    mpTracker->mbNewDetImgFlag=true;
}
} //namespace ORB_SLAM3

