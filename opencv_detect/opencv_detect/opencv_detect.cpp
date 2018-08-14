#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

void detectAndDisplay(Mat frame, int i)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);  //彩色图转换成灰度图
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces 检测人脸
	long t0 = cv::getTickCount();
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(40, 40));
	long t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();
	cout << "Detections takes " << secs*1000 << " ms " << endl;

	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);   //矩形框
		Mat faceROI = frame_gray(faces[i]);

	}
	imwrite("D:\\人脸识别\\opencv_detect\\result\\result" + to_string(i)+ ".jpg", frame);
	ofstream out("costtime.txt", ios::in | ios::app);
	out << secs << "ms" << endl;
	out.close();
	//system("pause");
	//imshow("Test", frame);
	//waitKey(0);
}

int main(int argc, const char** argv)
{
	// Load the cascades 导入级联（Haar分类器，实际是一个XML文件）
	if (!face_cascade.load("D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml")) { printf("--(!)Error loading face cascade\n"); return -1; };

	for (int i = 0; i <= 10; i++)
	{
		Mat frame;
		frame = imread("D:\\人脸识别\\SeetaFaceEngine-master\\SeetaFaceEngine-master\\FaceDetection\\data\\picture" + to_string(i) + ".jpg");

		detectAndDisplay(frame, i);
	}	
	return 0;
}