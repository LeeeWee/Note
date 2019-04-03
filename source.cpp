#include <opencv2\opencv.hpp>
#include <time.h>
using namespace cv;
using namespace std;

Mat getObjectRegion(string file_path)
{
	Mat image = imread(file_path);

	Rect roi = Rect(135, 645, 2355, 755);
	int pinNums = 2;

	//判断图片是否为空
	if (image.empty())
	{
		return image;
	}
	//裁剪出ROI区域
	Mat src = image(roi);
	Mat gray, binary, binary_inv;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	//求取自适应阈值
	double thresh0 = threshold(gray, binary, 0, 255, THRESH_OTSU);

	//黑色环的位置
	vector<Point2f> rect_img_contours_center;
	vector<double> rect_img_contours_radius;

	//通过黑色环求取的中间白点的位置
	vector<Point2f> results_center;
	vector<double> results_radius;

	//最外侧圆的位置
	vector<Point2f> rect_img_out_circle_center;
	vector<double> rect_img_out_circle_radius;
	//通过最外侧白圆求取的中间白点的位置
	vector<Point2f> results_center_out_circle;
	vector<double> results_radius_out_circle;

	//阈值遍历，求取目标点
	for (int t = 30; t < 150; t = t + 10)
	{
		if (t >= 255)
		{
			break;
		}
		threshold(gray, binary, t, 255, THRESH_BINARY);
		threshold(gray, binary_inv, t, 255, THRESH_BINARY_INV);

		//形态学闭运算
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		cv::Mat closed;
		cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, element);//高级形态学运算函数

																   //查找黑色环轮廓
		vector<vector<Point>>contours;
		vector<Vec4i>hierarchy;
		//未做形态学处理的图像
		vector<vector<Point>>contours_src;
		vector<Vec4i>hierarchy_src;

		//筛选出最外圈的大圆
		vector<vector<Point>>contours_out_circle;
		vector<Vec4i>hierarchy_out_circle;

		findContours(closed, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		findContours(binary, contours_src, hierarchy_src, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		findContours(binary_inv, contours_out_circle, hierarchy_out_circle, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		//筛选轮廓
		vector<vector<Point>>contours_area;
		vector<RotatedRect> rot_rects_ret;
		//筛选出最外圈的大圆
		vector<vector<Point>>contours_out_circle_ret;
		for (int j = 0; j < contours.size(); j++)
		{
			//面积筛选
			double area = contourArea(contours[j]);
			//圆度筛选
			double factor = (contourArea(contours[j]) * 4 * CV_PI) /
				(pow(arcLength(contours[j], true), 2));
			if (area < 1000 || area > 8000)
			{
				continue;
			}
			if (factor < 0.8)
			{
				continue;
			}
			contours_area.push_back(contours[j]);
		}
		for (int j = 0; j < contours_src.size(); j++)
		{
			//面积筛选
			double area = contourArea(contours_src[j]);

			//圆度筛选
			double factor = (contourArea(contours_src[j]) * 4 * CV_PI) /
				(pow(arcLength(contours_src[j], true), 2));
			if (area < 1000 || area > 8000)
			{
				continue;
			}
			if (factor < 0.7)
			{
				continue;
			}
			contours_area.push_back(contours_src[j]);
		}
		for (int j = 0; j < contours_out_circle.size(); j++)
		{
			//面积筛选
			double area = contourArea(contours_out_circle[j]);
			//筛选出最外圈的大圆
			Point2f center; float radius;
			minEnclosingCircle(contours_out_circle[j], center, radius);
			double factor1 = contourArea(contours_out_circle[j]) / (CV_PI * radius * radius);
			//圆度筛选
			double factor = (contourArea(contours_out_circle[j]) * 4 * CV_PI) /
				(pow(arcLength(contours_out_circle[j], true), 2));
			if (area >= 20000 && area <= 50000 && factor >= 0.7 && factor1 >= 0.85)
			{
				contours_out_circle_ret.push_back(contours_out_circle[j]);
			}
		}
		//根据上述筛选出的轮廓内是否有白色圆心筛选区域
		if ((contours_area.size() + contours_out_circle_ret.size()) < pinNums)
		{
			continue;
		}
		vector<vector<Point>>rect_img_contours_ret;

		for (int j = 0; j < contours_area.size(); j++)
		{
			double area0 = contourArea(contours_area[j]);
			Rect rect = boundingRect(contours_area[j]);
			Mat rect_img = gray(rect);
			Mat rect_img_ = src(rect);
			Mat rect_img_binary;
			threshold(rect_img, rect_img_binary, 0, 255, THRESH_OTSU);
			//形态学闭运算
			Mat rect_img_element = getStructuringElement(MORPH_RECT, Size(5, 5));
			cv::Mat rect_img_closed;
			cv::morphologyEx(rect_img_binary, rect_img_closed, cv::MORPH_OPEN, rect_img_element);//高级形态学运算函数

			vector<vector<Point>>rect_img_contours;
			vector<Vec4i>rect_img_hierarchy;
			//查找轮廓
			findContours(rect_img_closed, rect_img_contours, rect_img_hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			//根据上述筛选出的轮廓内是否有白色圆心筛选区域
			if (rect_img_contours.size() < 1)
			{
				continue;
			}
			for (int m = 0; m < rect_img_contours.size(); m++)
			{
				Mat mask = Mat(rect_img.rows, rect_img.cols, CV_8UC1, Scalar(0));
				drawContours(mask, rect_img_contours, m, Scalar(255, 255, 255), CV_FILLED);
				Scalar tmp_m = mean(rect_img, mask);
				double area = contourArea(rect_img_contours[m]);
				double factor = (contourArea(rect_img_contours[m]) * 4 * CV_PI) /
					(pow(arcLength(rect_img_contours[m], true), 2));
				if (factor < 0.65)
				{
					continue;
				}
				if (area < area0 / 30)
				{
					continue;
				}
				if (tmp_m[0] < 150)
				{
					continue;
				}

				Point2f center; float radius;
				minEnclosingCircle(contours_area[j], center, radius);
				Point2f center1; float radius1;
				minEnclosingCircle(rect_img_contours[m], center1, radius1);
				//最小外接矩形
				RotatedRect rotate_rect = minAreaRect(rect_img_contours[m]);
				//根据白色中心是否在中心点出进行筛选
				double dist = sqrt((rect_img.cols / 2 - center1.x)*(rect_img.cols / 2 - center1.x) + (rect_img.rows / 2 - center1.y)*(rect_img.rows / 2 - center1.y));
				double dist_delta = 4 + (max(rotate_rect.size.width, rotate_rect.size.height) - min(rotate_rect.size.width, rotate_rect.size.height)) / 2.0;
				if (dist > dist_delta)
				{
					continue;
				}

				double dist_cc = 999999;

				int index = 0;
				float radius_index = 0.0;
				Point2f center_index = Point2f(0.0, 0.0);
				//筛选重复区域
				for (int n = 0; n < rect_img_contours_center.size(); n++)
				{
					double dist_cc_ = sqrt((rect_img_contours_center[n].x - center.x)*(rect_img_contours_center[n].x - center.x) + (rect_img_contours_center[n].y - center.y)*(rect_img_contours_center[n].y - center.y));
					if (dist_cc_ < dist_cc)
					{
						dist_cc = dist_cc_;
						index = n;
					}
				}
				if (dist_cc < 22)
				{
					if (radius <= rect_img_contours_radius[index])
					{
						continue;
					}
					else
					{
						rect_img_contours_center[index] = center;
						rect_img_contours_radius[index] = radius;
					}
				}
				else
				{
					results_center.push_back(Point2f(rect.x + center1.x, rect.y + center1.y));
					results_radius.push_back(radius1);

					rect_img_contours_center.push_back(center);
					rect_img_contours_radius.push_back(radius);
				}
			}
		}
		//最外侧圆
		//中间白点的位置
		vector<Point2f> results_center;
		vector<double> results_radius;
		for (int j = 0; j < contours_out_circle_ret.size(); j++)
		{
			double area0 = contourArea(contours_out_circle_ret[j]);
			Rect rect = boundingRect(contours_out_circle_ret[j]);
			Mat rect_img = gray(rect);
			Mat rect_img_ = src(rect);
			Mat rect_img_binary;
			threshold(rect_img, rect_img_binary, 0, 255, THRESH_OTSU);
			//求取轮廓重心
			Point2f gravityCenter = Point2f(0, 0);
			double m00, m10, m01;
			Moments moment = cv::moments(rect_img_binary, 1);
			m00 = moment.m00;
			m10 = moment.m10;
			m01 = moment.m01;
			gravityCenter.x = (float)(m10 / m00);
			gravityCenter.y = (float)(m01 / m00);

			//形态学闭运算
			Mat rect_img_element = getStructuringElement(MORPH_RECT, Size(3, 3));
			cv::Mat rect_img_closed;
			cv::morphologyEx(rect_img_binary, rect_img_closed, cv::MORPH_OPEN, rect_img_element);//高级形态学运算函数

			vector<vector<Point>>rect_img_contours;
			vector<Vec4i>rect_img_hierarchy;
			//查找轮廓
			findContours(rect_img_closed, rect_img_contours, rect_img_hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			//根据上述筛选出的轮廓内是否有白色圆心筛选区域
			if (rect_img_contours.size() < 1)
			{
				continue;
			}
			//判断是否有白色圆点
			bool white_circle_exited = false;
			for (int m = 0; m < rect_img_contours.size(); m++)
			{
				Mat mask = Mat(rect_img.rows, rect_img.cols, CV_8UC1, Scalar(0));
				drawContours(mask, rect_img_contours, m, Scalar(255, 255, 255), CV_FILLED);
				Scalar tmp_m = mean(rect_img, mask);
				double area1 = contourArea(rect_img_contours[m]);
				double factor1 = (contourArea(rect_img_contours[m]) * 4 * CV_PI) /
					(pow(arcLength(rect_img_contours[m], true), 2));
				if (factor1 < 0.65)
				{
					continue;
				}
				if (area1 < area0 / 90 || area1 > area0 / 5)
				{
					continue;
				}
				if (tmp_m[0] < 150)
				{
					continue;
				}

				Point2f center; float radius;
				minEnclosingCircle(contours_out_circle_ret[j], center, radius);
				Point2f center1; float radius1;
				minEnclosingCircle(rect_img_contours[m], center1, radius1);

				double dist_cc = 999999;

				int index = 0;
				float radius_index = 0.0;
				Point2f center_index = Point2f(0.0, 0.0);
				//筛选重复区域
				for (int n = 0; n < results_center_out_circle.size(); n++)
				{
					double dist_cc_ = sqrt((results_center_out_circle[n].x - center1.x - rect.x)*
						(results_center_out_circle[n].x - center1.x - rect.x) +
						(results_center_out_circle[n].y - center1.y - rect.y)*
						(results_center_out_circle[n].y - center1.y - rect.y));
					if (dist_cc_ < dist_cc)
					{
						dist_cc = dist_cc_;
						index = n;
					}
				}
				if (dist_cc < 22)
				{
					if (radius <= rect_img_out_circle_radius[index])
					{
						continue;
					}
					else
					{
						rect_img_out_circle_center[index] = Point2f(rect.x + center.x, rect.y + center.y);
						rect_img_out_circle_radius[index] = radius;
					}
				}
				else
				{
					results_center_out_circle.push_back(Point2f(rect.x + center1.x, rect.y + center1.y));
					results_radius_out_circle.push_back(radius1);

					rect_img_out_circle_center.push_back(Point2f(rect.x + center.x, rect.y + center.y));
					rect_img_out_circle_radius.push_back(radius);
				}
				white_circle_exited = true;
			}
			if (!white_circle_exited)
			{
				double dist_cc = 999999;

				int index = 0;
				float radius_index = 0.0;
				Point2f center_index = Point2f(0.0, 0.0);
				//筛选重复区域
				for (int n = 0; n < results_center_out_circle.size(); n++)
				{
					double dist_cc_ = sqrt((results_center_out_circle[n].x - gravityCenter.x - rect.x)*
						(results_center_out_circle[n].x - gravityCenter.x - rect.x) +
						(results_center_out_circle[n].y - gravityCenter.y - rect.y)*
						(results_center_out_circle[n].y - gravityCenter.y - rect.y));
					if (dist_cc_ < dist_cc)
					{
						dist_cc = dist_cc_;
						index = n;
					}
				}
				if (dist_cc < 22)
				{

				}
				else
				{
					results_center_out_circle.push_back(Point2f(rect.x + gravityCenter.x, rect.y + gravityCenter.y));
					results_radius_out_circle.push_back(20);

					rect_img_out_circle_center.push_back(Point2f(rect.x + gravityCenter.x, rect.y + gravityCenter.y));
					rect_img_out_circle_radius.push_back(20);
				}
			}
		}
	}
	for (int ind = 0; ind < results_center_out_circle.size(); ind++)
	{
		bool equal = false;
		for (int ind0 = 0; ind0 < results_center.size(); ind0++)
		{
			double dist_cc = sqrt((results_center_out_circle[ind].x - results_center[ind0].x)*
				(results_center_out_circle[ind].x - results_center[ind0].x) +
				(results_center_out_circle[ind].y - results_center[ind0].y)*
				(results_center_out_circle[ind].y - results_center[ind0].y));
			if (dist_cc < 50)
			{
				equal = true;
			}
		}
		if (!equal)
		{
			results_center.push_back(results_center_out_circle[ind]);
			results_radius.push_back(results_radius_out_circle[ind]);
			rect_img_contours_center.push_back(results_center_out_circle[ind]);
			rect_img_contours_radius.push_back(2 * results_radius_out_circle[ind]);
		}
	}
	if (rect_img_contours_center.size() < pinNums)
	{
		return image;
	}
	for (int i = 0; i < rect_img_contours_center.size(); i++)
	{
		circle(image, Point(int(135 + results_center[i].x), int(645 + results_center[i].y)), int(results_radius[i]), Scalar(0, 255, 0), 2);
	}
	line(image, Point(int(135 + results_center[0].x), int(645 + results_center[0].y)), Point(int(135 + results_center[1].x), int(645 + results_center[1].y)), Scalar(0, 0, 255), 2);
	return image;
}

int main()
{
	clock_t startTime, endTime;
	startTime = clock();

	
	

	Mat image = getObjectRegion("C:\\Users\\wwx439753\\Desktop\\测试程序\\222.jpg");
	imwrite("D://1_1_1_1.bmp", image);
	endTime = clock();
	cout << "Totle Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	system("pause");
	return 0;
}

