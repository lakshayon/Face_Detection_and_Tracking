/*

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;


int main(){

	VideoCapture cap(0); // open the video camera no. 0
	Mat frame, grayframe, framehsv;
	vector <Mat>channels;
	int c = -1, d = -1;
	bool bSuccess;
	vector<Rect> faces;

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

//	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	namedWindow("Detected Face", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"


	while (1)
	{

		cap >> frame; // read a new frame from video
		cvtColor(frame, grayframe, COLOR_BGR2GRAY);
		//split(framehsv, channels);
		//grayframe = channels[0];
		// Detect faces
		faces.clear();
//		detect_faces(grayframe, &faces, face_cascade);
		face_cascade.detectMultiScale(grayframe, faces, 1.1, 3, CV_HAAR_SCALE_IMAGE, Size(30, 30));
//		face_cascade.detectMultiScale(frame, faces, 1.1, 10, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, cvSize(10, 10), cvSize(300, 300));

		// Draw circles on the detected faces
		imshow("MyVideo", frame);

		for (int i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			//ellipse(grayframe, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
		}

		imshow("Detected Face", frame);
		c = waitKey(10);
		printf("displaying\n");
		if (c == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			//	cout << "esc key is pressed by user" << endl;
			printf("Esc is pressed.");
			break;
		}


	}

	face_cascade.~CascadeClassifier();
	faces.clear();
	return 0;
}

*/
