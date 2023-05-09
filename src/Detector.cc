/**
* This file is part of Crowd-SLAM.
* Copyright (C) 2020 João Carlos Virgolino Soares - Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: Marcelo Gattass and Marco Antonio Meggiolaro
* For more information see <https://github.com/virgolinosoares/Crowd-SLAM>.
* Please report suggestions and comments to virgolinosoares@gmail.com
* 
* YOLO detection and processing functions are based on the OpenCV project. They are subject to the license terms at http://opencv.org/license.html
*/
//cc是c++的源文件后缀，这是Yolo目标检测分割的代码，相当与yolo的一个小demo
#include <thread>
#include <iomanip>
#include "Detector.h"
#include <fstream>
#include <sstream>
#include <iostream>


namespace ORB_SLAM3
{

//创建一个目标检测器类，从文件中初始化模型
Detector::Detector()
{
	//先初始化是否是新的图像为false
	mbNewImgFlag = false;
	//初始化请求功能是否完成为false
	mbFinishRequested = false;
	
	std::cout << "\n--------------------------------------------"  << std::endl;
    
	//读入YOLO的设置文件
    cv::FileStorage YOLOsettings("cfg/YOLOpar.yaml", cv::FileStorage::READ);
    //如果YOLO设置文件没有打开就会报错，然后异常退出
    if(!YOLOsettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << "cfg/YOLOpar.yaml" << endl;
       exit(-1);
    }
	//如果正常打开了及就显示正在加载yolo文件
	std::cout << "loading YOLO network" << std::endl;
    	
    	//打开写有类的文件
	cout << "loading classes...";    	
	string classesFile = "cfg/CYTI.names";
    // 用文件输入流读入文件名为classesFile这个文件，从c_str()是字符流的一种
    ifstream ifs(classesFile.c_str());
    string line;
    //循环在文件输入流中读入一行字符串，加载类别名
    while (getline(ifs, line)) classes.push_back(line);
	std::cout << "done\n";

	// 从文件中读入YOLO的参数
	confThreshold = YOLOsettings["yolo.confThreshold"];
	nmsThreshold = YOLOsettings["yolo.nmsThreshold"];
	inpWidth = YOLOsettings["yolo.inpWidth"];
	inpHeight = YOLOsettings["yolo.inpHeight"];
    
    //加载权重文件
    std::cout << "loading weight files...";      
	string model_cfg = "cfg/CYTI.cfg";
    string model_weights = "weights/CYTI.weights";

    std::cout << "done\n";

    // 加载网络
	cout << "loading network...";
    net = cv::dnn::readNetFromDarknet(model_cfg, model_weights);
    
    //这是CPU加载我们要用GPU加载
     net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
     net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    // 英特尔GPU加速
    // net.setPreferableBackend(DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(DNN_TARGET_OPENCL);
        
    //英伟达GPU加速
    //https://blog.csdn.net/beingod0/article/details/102862945
    //net.setPreferableTarget(cv::dnn::dnn4_v20200609::DNN_BACKEND_CUDA);
    //net.setPreferableBackend(cv::dnn::dnn4_v20200609::DNN_BACKEND_CUDA);

	std::cout << "done\n";  
	std::cout << "--------------------------------------------"  << std::endl;


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
		people_boxes.clear();
		
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

// 整个目标检测过程实现的函数
void Detector::Detect(){

		// 将输入的图片尺寸赋予fig_area
		fig_area = mImg.rows * mImg.cols;
		// 初始化aprox_area
		aprox_area = 0;

		//factor初始化？这是用来干啥的
		fig_factor = fig_area/fig_param;

		cv::Mat blob;
		//从一帧图像初始化blob
		//blob就是 NCHW  多少个(number) 通道数(channel) 高度(height) 宽度(weight)
        //主要是用来对图片进行预处理
        //mImag是输入的图像
        //1,0指的是当我们将图片减去平均值之后，还可以对剩下的像素值进行一定的尺度缩放，1.0表示不变，0.5表示缩放半，1/255表示缩放255倍数        
        //Size：是我们神经网络在训练的时候要求输入的图片尺寸
        //Scalar():需要将图片整体减去的平均值,如果我们需要对RGB图片的三个通道分别减去不同的值，那么可以使用3组平均值，如果只使用一组，那么就默认对三个通道减去一样的值。
        //这里的平均值是为了消除光照对于神经网络的影响，对图片的R、G、B通道的像素求一个平均值，然后将每个像素值减去我们的平均值。
        //倒数第二个true指的是是否要进行RB两个通道的交换，OpenCV中认为我们的图片通道顺序是BGR，但是我平均值假设的顺序是RGB，所以如果需要交换R和G，那么就为true
        //倒数第一个参数false指的是是否要进行剪切，默认false
    	cv::dnn::blobFromImage(mImg, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);

	//blob = cv::dnn::blobFromImage(mImg, 1.0, cv::Size(32,32));
        
    	//将初始化的blob放在输入
    	net.setInput(blob);
	
    	//运行前向传播以从输出层获取输出，outs将输出结果存储
    	std::vector<cv::Mat> outs;
    	//getOutsNames是获取输出层的类别名
    	net.forward(outs, getOutsNames(net));

		std::vector<int> classId;
		std::vector<int> centersX;
		std::vector<int> centersY;
		std::vector<cv::Rect> boxes;	      

    	//将前向传播后的数据进行处理
    	postprocess(mImg, outs, classId, centersX, centersY, boxes);

	//将图像im在名称为ORB-SLAM2: Current Frame的窗口中显示，这是用来调试的代码。
	//必须和下面waitkey一起用，不然刷新得太快人眼捕捉不到。（因为一条指令的执行速率约为0.0000000001s）
	//cv::imshow("ORB-SLAM2: Current Frame",mImg);
	//int cvWaitKey(int k)函数的功能是刷新图像，其中参数k单位是毫秒，表示刷新频率。
        //cv::waitKey(1);

		//定义计数器，看看检测出多少人
		int people_counter {0};
		//遍历整个classId数组
		for (size_t i = 0; i < classId.size(); ++i){
			//如果目标检测出，该类别是人的话，人总数的计数器增加1，且将该目标检测框放进people_boxes中
			if(classes[classId[i]] == "person"){
				people_counter++;		
				people_boxes.push_back(boxes[i]);

				//总共的区域是aprox_area不停地加上目标检测框的尺寸，这是干嘛的？
				aprox_area = aprox_area + (boxes[i].width * boxes[i].height)/fig_area;				
			}
		}
		//这段代码是用来调试，输出检测出来的人的总数
		//std::cout << "people: " << people_counter << std::endl;
		SetDetectionFlag();
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

// 获取输出层检测出来的类别名字，即out_Names初始化
std::vector<cv::String> Detector::getOutsNames(const cv::dnn::Net& net)
{
    // 定义一个变量存放名字
    static std::vector<cv::String> out_names;
    // 如果没有获取过输出层名字就要进行获取
    if (out_names.empty())
    {
        //获取输出层
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        //获取输出层的类别名
        std::vector<cv::String> layersNames = net.getLayerNames();
        // 将out_names数组的大小设置为输出层数组的大小
        out_names.resize(outLayers.size());
        // 将输出层的类别名依次放去out_names数组中
		for (size_t i = 0; i < outLayers.size(); ++i){
			out_names[i] = layersNames[outLayers[i] - 1];
		}
    }
    // 将带有输出层类别名的变量返回
    return out_names;
}

// 非最大值抑制移除置信度较低的边界框
void Detector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<int>& classIds, std::vector<int> &centersX, std::vector<int> &centersY, std::vector<cv::Rect>& boxes)
{
    //定义置信度数组来储存每个边界框的置信度
    std::vector<float> confidences;
    //遍历outs中的每个边界框    
    /**
    
    找到最高置信度的值和位置
    （1）将所有框的得分排序，选中最高分及其对应的框
    
     **/
    //扫描网络输出的所有边界框，只保留置信度高的边界框。
    for (size_t i = 0; i < outs.size(); ++i)
    {
        //输出层每个边界框数据存入data
        float* data = (float*)outs[i].data;
        //遍历边界框每行，data也随着换行，data是将几行并称一行，所以加列数就相当与换行
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            //score等于第一个边界框的j行5～n列
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            
            //某一个检测框的坐标
            cv::Point classIdPoint;
            //某一个检测框的置信度
            double confidence;

            // minMaxLoc寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置. 
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            //如果其置信度大于阈值，则将其中心点坐标，框的长宽，左外边距（左边的边离检测框边界的距离），上外边距算出来
            if (confidence > confThreshold)
            {   //NMS需要预测框左上角和右下角的尺寸
                //类似yolov3这种最后预测的坐标是中心点和长宽，那么要转化一下，下面提供一个转化函数：
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols)-10;
                int height = (int)(data[3] * frame.rows)-150;
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                //将该点的x坐标记录下来
                classIds.push_back(classIdPoint.x);
                //将置信度记录下来
                confidences.push_back((float)confidence);
                //将该目标框记录下来
                boxes.push_back(cv::Rect(left, top, width, height));
				centersX.push_back(centerX);
				centersY.push_back(centerY);
            }
        }
    }
    /**
    
    （2）遍历其余的框，如果和当前最高分框的重叠面积(IOU)大于一定阈值，我们就将框删除。
    
    **/
    //执行非最大抑制，以消除置信度较低的冗余重叠框
    std::vector<int> indices;
    //input:  boxes: 原始检测框集合;
    //input:  confidences：原始检测框对应的置信度值集合
    //input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值,nms阈值即和最大面积重叠阈值
    //output:  indices  经过上面两个阈值过滤后剩下的检测框的index
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    //将剩下的框画出来
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
               box.x + box.width, box.y + box.height, frame);
    }
}

//绘制预测的边界框
void Detector::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //绘制一个显示边框的矩形
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //获取类名及其可信度的标签
    string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //在边界框顶部显示标签
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    //void rectangle(InputOutputArray img, Point pt1, Point pt2,const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0);
    //使用对角线的两点，绘制出一个矩形框框
    //参数1是输入的图像
    //参数2,3是对角线的两点
    //scalar是矩形框颜色
    //cv::filled表示线条的宽度，表示矩形框已填充
    //有两个参数是默认的，就是线条的类型，默认为8，shift默认为0
    rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}


} //namespace ORB_SLAM3
