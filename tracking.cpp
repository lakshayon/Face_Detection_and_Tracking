
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

bool addRemovePt = false;

int main(int argc, char** argv)
{
	VideoCapture cap(0);
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	vector < vector<uchar> > status;
	vector < vector<float> > err;
	vector < vector<Point2f> > tmp;
	vector<uchar>  tempStatus;
	vector<float>  tempErr;
	vector<Point2f>  tempTmp;
	

	Mat gray, prevGray, image, frame,framehsv,gray2;
	Mat mask;  // type of mask is CV_8U
	Mat roi;
	vector <Mat> channels;
	vector < vector<Point2f> > points;
	vector < vector<Point2f> > pointsPrev;
	vector<Point2f>  tempPoints;
	vector<Point2f> centroid;
	vector<Point2f> variance;
	vector<Point2f> prevvariance;
	vector<Point2f> prevdim;
	vector<Point2f> newdim;

	size_t i, j, k;

	const int MAX_COUNT = 500;
	bool needToInit = true, needToDetect = true;
	int countFrames = 0;
	int nFaces = 0;
	std::vector<cv::Rect> faces;
	bool bSuccess;
	
	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	face_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml");

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;
	cout << "Press Esc to exit.";
	namedWindow("LK Demo", WINDOW_AUTOSIZE);
//	namedWindow("mask", WINDOW_AUTOSIZE);

	cap >> frame;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	gray.copyTo(mask);		//memory is allocated to mask

	while(1)
	{
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if (needToDetect) {
			// Detect faces
			faces.clear();
			face_cascade.detectMultiScale(gray, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
			//		face_cascade.detectMultiScale(frame, faces, 1.1, 10, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, cvSize(10, 10), cvSize(300, 300));
			nFaces = faces.size();
			needToDetect = false;
			newdim.resize(nFaces);
			for (i = 0; i < nFaces; i++) {
				newdim[i].x = faces[i].width;
				newdim[i].y = faces[i].height;
			}
		}

		if (needToInit)
		{
			// automatic initialization
			points.clear();
			centroid.resize(nFaces);
			variance.resize(nFaces);

			for (i = 0; i < nFaces; i++) {
				mask = Scalar(0,0,0);
				mask(faces[i]) = Scalar(255);
//				imshow("mask", mask);
				//goodFeaturesToTrack(gray, points[i], MAX_COUNT, 0.01, 10, mask, 3, 0, 0.04);
				tempPoints.clear();
				goodFeaturesToTrack(gray, tempPoints, MAX_COUNT, 0.01, 10, mask, 3, 0, 0.04);
				points.push_back(tempPoints);
				cornerSubPix(gray, points[i], subPixWinSize, Size(-1, -1), termcrit);
				newdim[i].x = faces[i].width;
				newdim[i].y = faces[i].height;
				centroid[i].x = faces[i].x + faces[i].width / 2.0;
				centroid[i].y = faces[i].y + faces[i].height / 2.0;
			}
			///////////////////////
			for (i = 0; i < nFaces; i++) {
				variance[i].x = 0, variance[i].y = 0;
				for (j = 0, k = 0; j < points[i].size(); j++)
				{
					circle(image, points[i][j], 1, Scalar(255, 255, 255), -1, 8);
					k++;
				}

				/*				for (int p = 0; p < points[i].size(); p++) {
									centroid[i].x = centroid[i].x + points[i][p].x;
									centroid[i].y = centroid[i].y + points[i][p].y;

								}
								centroid[i].x = centroid[i].x / points[i].size(); //Mean cordinates of points[i]
								centroid[i].y = centroid[i].y / points[i].size(); // Mean cordinates of points[i]
				*/				for (int p = 0; p < points[i].size(); p++) {
					variance[i].x = variance[i].x + (points[i][p].x - centroid[i].x)*(points[i][p].x - centroid[i].x);
					variance[i].y = variance[i].y + (points[i][p].y - centroid[i].y)*(points[i][p].y - centroid[i].y);
				}
				variance[i].x = variance[i].x / points[i].size();
				variance[i].y = variance[i].y / points[i].size();
				//				newdim[i].x = (prevdim[i].x * variance[i].x) / prevvariance[i].x;
				//				newdim[i].y = (prevdim[i].y * variance[i].y) / prevvariance[i].y;
			}


			needToInit = false;
		}
		else
		{
			if (prevGray.empty()) {
				gray.copyTo(prevGray);
			}
			status.clear();
			err.clear();
			for (i = 0; i < nFaces; i++) {
				tempStatus.clear();
				tempErr.clear();
				calcOpticalFlowPyrLK(prevGray, gray, pointsPrev[i], points[i], tempStatus, tempErr, winSize,
					3, termcrit, 0, 0.001);
				status.push_back(tempStatus);
				err.push_back(tempErr);
			}

			centroid.resize(nFaces);
			variance.resize(nFaces);
			newdim.resize(nFaces);

			for (i = 0; i < nFaces; i++) {
				centroid[i].x = 0, centroid[i].y = 0;
				variance[i].x = 0, variance[i].y = 0;
				for (j = 0, k = 0; j < points[i].size(); j++)
				{
					if (!status[i][j])
						continue;
					points[i][k++] = points[i][j];
					circle(image, points[i][j], 1, Scalar(255, 255, 255), -1, 8);
				}
				points[i].resize(k);

				for (int p = 0; p < points[i].size(); p++) {
					centroid[i].x = centroid[i].x + points[i][p].x;
					centroid[i].y = centroid[i].y + points[i][p].y;

				}
				centroid[i].x = centroid[i].x / points[i].size(); //Mean cordinates of points[i]
				centroid[i].y = centroid[i].y / points[i].size(); // Mean cordinates of points[i]
				for (int p = 0; p < points[i].size(); p++) {
					variance[i].x = variance[i].x + (points[i][p].x- centroid[i].x)*(points[i][p].x - centroid[i].x);
					variance[i].y = variance[i].y + (points[i][p].y - centroid[i].y)*(points[i][p].y - centroid[i].y);
				}
				variance[i].x = variance[i].x / points[i].size();
				variance[i].y = variance[i].y / points[i].size();
				newdim[i].x = (prevdim[i].x * variance[i].x) / prevvariance[i].x;
				newdim[i].y = (prevdim[i].y * variance[i].y) / prevvariance[i].y;
				
				circle(image, centroid[i], 2, Scalar(0, 0, 255), -1, 8);

			}
		}

		for (i = 0; i < nFaces; i++) {
			rectangle(image, Rect{(int)(centroid[i].x - newdim[i].x/2),(int)(centroid[i].y - newdim[i].y / 2),(int)newdim[i].x,(int)newdim[i].y }, Scalar(255, 0, 0), 2, 8, 0);
		}
		imshow("LK Demo", image);

		char c = (char)waitKey(10);
		if (c == 27)
			break;

		pointsPrev.clear();
		pointsPrev = points;
		prevdim.clear();
		prevdim = newdim;
		prevvariance.clear();
		prevvariance = variance;
		gray.copyTo(prevGray);
		countFrames++;

		if (countFrames == 50) {
			pointsPrev.clear();
			prevdim.clear();
			prevvariance.clear();
			needToDetect = true;
			needToInit = true;
			countFrames = 0;
		}
	}

	return 0;
}