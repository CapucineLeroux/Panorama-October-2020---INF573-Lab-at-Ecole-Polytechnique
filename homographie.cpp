#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
        Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
        Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);
        //Mat I2 = imread("../IMG_0046r.JPG", IMREAD_GRAYSCALE);
        //Mat I1 = imread("../computer1.JPG", IMREAD_GRAYSCALE);
        //Mat I2 = imread("../computer2.JPG", IMREAD_GRAYSCALE);


	imshow("I1", I1);
	imshow("I2", I2);

	// the goal of this lab is to create a panorama creation pipeline from 2 images, using opencv functions
	// that is thus a higher level lab, where the goal is to check that you understood the lecture concepts
	// that you are able to find and use their implementation in the opencv library and to put
	// in place the full pipeline.
	// the documentation is available at https://docs.opencv.org/master, mostly in the feature2D and calib3d modules.


        // Q1. use an opencv feature extractor and descriptor to detect and compute features on both images

        //AKAZE to get feature detection (stored in m1,m2) and description (stored in descriptors1 and descriptors2)
        Ptr<AKAZE> D = AKAZE::create();
        vector<KeyPoint> m1, m2;
        Mat descriptors1, descriptors2;

        D->detectAndCompute( I1, noArray(), m1, descriptors1 );
        D->detectAndCompute( I2, noArray(), m2, descriptors2 );

        Mat I1kp;
        drawKeypoints(I1,m1,I1kp);
        imshow("KeyPoints I1",I1kp);

        Mat I2kp;
        drawKeypoints(I2,m2,I2kp);
        imshow("KeyPoints I2",I2kp);

	
	// Q2. use a descriptor matcher, to compute feature correspondences
	//BFMatcher M ...

	// drawMatches ...
	
        //I used norm hamming because we used akaze method
        BFMatcher M(NORM_HAMMING);
        vector<vector<DMatch>> matches;
        //catches the 2 closest neighbours of descriptors1 in descriptors2
        M.knnMatch(descriptors1,descriptors2,matches,2);

        //compares the 2 closest neighbours, if there is a big difference of distance, then it's a good match
        float threshold = 0.8f;
        vector<DMatch> matched;
        vector<KeyPoint> keypoints1, keypoints2;

        DMatch first_neighbour;
        float distance1;
        float distance2;
        int new_i;
        //only keeps good matches in matched and keypoints1/2
        for(size_t i = 0; i < matches.size(); i++) {

            first_neighbour = matches[i][0];
            distance1 = matches[i][0].distance;
            distance2 = matches[i][1].distance;

            if(distance1 < threshold*distance2) {
                new_i = static_cast<int>(keypoints1.size());
                keypoints1.push_back(m1[first_neighbour.queryIdx]);
                keypoints2.push_back(m2[first_neighbour.trainIdx]);
                matched.push_back(DMatch(new_i, new_i, 0));
            }
        }


        Mat I_matches;
        drawMatches(I1, keypoints1, I2, keypoints2, matched, I_matches);
        imshow("Matches Q2", I_matches);

	// Q3. Organize the matched feature pairs into vectors and estimate an homography using RANSAC
	// and a model reprojection threshold of 3 pixels
	// provide a mask input to draw the inlier matches
	// vector<Point2f> matches1, matches2;
	// Mat H = findHomography(...
	// drawMatches ...


        vector<Point2f> points1, points2;
        for(size_t i=0 ; i<matched.size() ; i++){
            points1.push_back( keypoints1[matched[i].queryIdx].pt );
            points2.push_back( keypoints2[matched[i].trainIdx].pt );
        }

        //calculates the homography and the corresponding mask to draw the inliers matches
        vector<char> mask;
        Mat H;
        H = findHomography(points1, points2, RANSAC, 3, mask);

        Mat I_matches_Q3;
        drawMatches( I1, keypoints1, I2, keypoints2, matched, I_matches_Q3, Scalar::all(-1), Scalar::all(-1), mask);
        imshow("Matches Q3", I_matches_Q3);

	// Q4. copy I1 to a new (bigger image) K using the identity homography
	// warp I2 to K using the computed homography
	// Mat K(2 * I1.cols, I1.rows, CV_8U);
	// warpPerspective( ...
	// show your panorama !

        Mat Id = Mat::eye(3,3,CV_32F);
        Mat K1(2*I1.rows, 3*I1.cols, CV_8U);
        Mat K2(2*I1.rows, 3*I1.cols, CV_8U);
        Mat K(2*I1.rows, 3*I1.cols, CV_8U);

        warpPerspective(I1, K1, Id, K1.size());
        warpPerspective(I2, K2, H.inv(), K2.size());

        K = max(K1,K2);
        imshow("Panorama",K);

	// Q5. does it work when the images are rotated so that they are not approximately aligned at first ?
	// make a panorama with the rotated version IMG_0046r.JPG of image IMG_0046.JPG

        //Answer : it works, you just have to decomment IMG_0046r.JPG and commment IMG_0046.JPG instead
	
	// Q6. make it work on 2 images of your own
	// submit this cpp file, a screenshot of the labs' panorama, your input files and a screenshot of your own panorama.

        //Answer : it works, you just have to decomment photo1.JPG and photo2.JPG and commment IMG_0045.JPG and IMG_0046.JPG instead

	// Q7. extra credits
	// program and submit a panorama with more than 2 images
	// you can use your own or images from https://github.com/holynski/cse576_sp20_hw5/tree/master/pano 
	
	waitKey(0);
	return 0;
}
