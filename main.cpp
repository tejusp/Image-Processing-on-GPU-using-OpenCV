#include <iostream>
#include <stdint.h>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <typeinfo>
#include <argp.h>
#include <sys/time.h>

using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/gpu/gpu.hpp>

using namespace cv;

struct Arguments {
	string project;
	string input;
	string output;
	int padding;
	int frames;
	string extension;
	int width;
	int height;
	int area_min;
	int area_max;
	int search_win_size;
	int blur_radius;
	int threshold_win_size;
	float threshold_ratio;
	string log;
	bool verbose;

	Arguments() :
			input("data/"), output("output.txt"), padding(7), frames(40), extension(
					".jpg"), width(24), height(10), area_min(200), area_max(
					400), search_win_size(100), blur_radius(3), threshold_win_size(
					25), threshold_ratio(0.9), log("wormSeg.log"), verbose(true) {
	}
} cla;

int timeToUpload = 0;

int findCentroidFromImage(cv::Mat, int*, int*, int*);

template<typename T> string NumberToString(T pNumber) {
	ostringstream oOStrStream;
	oOStrStream << pNumber;

	return oOStrStream.str();
}

string intToFileName(string fileNameFormat, int fileNumber) {
	string temp = NumberToString(fileNumber);

	return fileNameFormat.replace(fileNameFormat.size() - temp.size(),
			temp.size(), temp);
}
void func(const float*, float*, size_t, const size_t, int, int);

void callKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst) {
	float* p = (float*) src.data;
	float* p2 = (float*) dst.data;
	func(p, p2, src.step, dst.step, src.cols, src.rows);
}

int findCentroidFromImage(cv::Mat src, int *pX, int *pY, int *pArea) {
	cout << "size=" << sizeof(bool);
	//GPU Mat... Copy from CPU memory to GPU memory...
	cv::gpu::GpuMat gpu_src(src);
	cout << "\nsrc\n " << src << endl;

	cv::gpu::GpuMat matAfterBlur;
	//Filters on GPU...
	cv::gpu::blur(gpu_src, matAfterBlur,
			Size(cla.blur_radius, cla.blur_radius));
	cv::gpu::GpuMat matAfterThreshold;
	//Convert into Binary image on GPU...
	cv::gpu::threshold(matAfterBlur, matAfterThreshold,
			int(cla.threshold_ratio * 255), 255, THRESH_BINARY_INV);

	cv::gpu::GpuMat floatMatForKernel;
	matAfterThreshold.convertTo(floatMatForKernel, CV_32FC1);
	callKernel(floatMatForKernel, gpu_src);
	//Copy from GPU memory to CPU memory...
	cv::Mat cpu_src(gpu_src);
	cout << "\nafter thr=\n" << cpu_src << endl;

	vector<vector<Point> > contours;

	vector<Vec4i> hierarchy;

	//findContours on CPU...
	//Need to write this method on GPU... To improve overall performance...
	//Check Sobel Algorithm...
	cv::findContours(cpu_src, contours, hierarchy, CV_RETR_CCOMP,
			CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() > 0) {
		int largest_contour_index = 0;
		int largest_area = 0;

		for (int i = 0; i < contours.size(); i++) {
			double a = cv::contourArea(contours[i], false);

			if (a > largest_area) {
				largest_area = a;

				largest_contour_index = i;
			}
		}

		cv::Rect bRect = cv::boundingRect(contours[largest_contour_index]);

		*pX = bRect.x + (bRect.width / 2);
		*pY = bRect.y + (bRect.height / 2);
		*pArea = largest_area;
	} else {
		*pX = -1;
		*pY = -1;
		*pArea = -1;
	}

	return 0;
}

int wormSegmenter() {

	fstream outputFile;

	outputFile.open(cla.output.c_str(), ios::out);

	int x = -1, y = -1, area = -1;
	int adjustX = 0, adjustY = 0;

	for (int fileNumber = 0; fileNumber < cla.frames; fileNumber++) {
		string fileName = cla.input + intToFileName("0000000", fileNumber)
				+ cla.extension;
		cv::Mat src = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

		if (!src.data) {
			cout << endl << "Exited." << endl;
			exit(1);
		}

		if ((x == -1) && (y == -1)) {
			findCentroidFromImage(src, &x, &y, &area);
			src = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
			adjustX = x - (cla.search_win_size / 2);
			adjustY = y - (cla.search_win_size / 2);
		} else {
			src = src(
					cv::Rect(x - (cla.search_win_size / 2),
							y - (cla.search_win_size / 2), cla.search_win_size,
							cla.search_win_size));

			findCentroidFromImage(src, &x, &y, &area);

			if ((x > 0) && (y > 0)) {
				x += adjustX;
				y += adjustY;
				adjustX = x - (cla.search_win_size / 2);
				adjustY = y - (cla.search_win_size / 2);
			}
		}

		outputFile << fileNumber << ", " << x << ", " << y << ", " << area
				<< endl;
	}

	outputFile.close();

	return 0;
}

int main(int argc, char **argv) {
	wormSegmenter();

	cout << "time=" << timeToUpload << endl;

	return 0;
}

//int main() {
//	Mat input = imread("0000001.jpg", 0);
//	std::cout << "matrix input=\n " << input;
//	Mat float_input;
//	input.convertTo(float_input, CV_32FC1);
//	cv::gpu::GpuMat d_frame, d_output;
//	Size size = float_input.size();
//	d_frame.upload(float_input);
//	d_output.create(size, CV_32FC1);
//	callKernel(d_frame, d_output);
//	Mat output(d_output);
//	return 0;
//}
