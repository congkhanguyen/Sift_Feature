// Sift_Feature.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
/*
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;

int main(int argc, const char* argv[])
{
    const cv::Mat input = cv::imread("C:\\Users\\congkha\\Desktop\\3.png", 0); //Load as grayscale

    SiftFeatureDetector *detector = new SiftFeatureDetector(0, 15);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(input, keypoints);

    // Add results to image and save.
    Mat output;
    drawKeypoints(input, keypoints, output, Scalar::all(-1) , 4);
    imwrite("C:\\Users\\congkha\\Desktop\\14.png", output);

    return 0;
}
*/

/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 
*/
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */

int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SIFT detector (30) ;

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;
  extractor.extended = true; // For 128 dimesion descriptors
  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= 0.5) // 0.45 - 0.46, 0.39
    { 
	  //Distance between q and centroid
	  float cqx = keypoints_1[matches[i].queryIdx].pt.x  - img_1.cols;
	  float cqy = keypoints_1[matches[i].queryIdx].pt.y  - img_1.rows;
	  float cq = sqrt (cqx * cqx + cqy*cqy);

	   //Distance between nearest and centroid
	  float cnx = keypoints_2[matches[i].trainIdx].pt.x  - img_2.cols;
	  float cny = keypoints_2[matches[i].trainIdx].pt.y  - img_2.rows;
	  float cn = sqrt ( cnx *  cnx +  cny* cny);

	   if (abs(cq - cn) < cn/30)
	   {
		 good_matches.push_back( matches[i]); 
	   }
	}
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  imshow( "Good Matches", img_matches );

  for( int i = 0; i < (int)good_matches.size(); i++ )
  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

  waitKey(0);

  return 0;
}

/**
 * @function readme
*/
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; } 