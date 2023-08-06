#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <queue>
#include <ctime>
#include <cmath>

using namespace std;

// function prototyping
float[] luminance_flash_count(float[] color, float[] prev_color);
float[] sRGB_to_linearRGB(float[] sRGB);
float[] inverse_gamma_transform(float[] signal);
float[] linearRBG_to_Ls(float[] linearRGB);
float[] is_luminance_flash(float[] ls, float[] prev_ls);
float[] saturated_red_flash_count(float[] color, float[] prev_color);
float[] red_ratio(float[] sRGB);
float[] pure_red(float[] sRGB);

int main()
{
	const int BUFFER_SIZE = 16;
	const int frame_scaling_factor = 0.25;

	int frame_width;
	int frame_height;
	int quarter_area_threshold;

	// buffers 
	deque<float[]> frame_buffer;
	deque<time_t> time_buffer;
	deque<float[]> luminous_flash_buffer;
	deque<float[]> red_flash_buffer;

	// setting up the web cam
	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FPS, 100);
	
	if (!cap.isOpened()) {
		cout << "Change the camera port number" << endl;
		return -1;
	}

	float[] frame;

	cap.read(frame);
	cv::resize(frame, frame, cv::Size(), 0.25, 0.25);

	frame_width = frame.size().width;
	frame_height = frame.size().height;

	quarter_area_threshold = (frame_width * frame_height) / 4;

	float[] luminous_flashes = float[]::zeros(frame_height, frame_width, CV_32S);
	float[] red_flashes = float[]::zeros(frame_height, frame_width, CV_32S);

	time_t prev_time = time(0);
	
	while (1) {
		cap.read(frame);
		cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

		if (frame.empty()) {
			cout << "The frame is not captured properly" << endl;
			break;
		}

		time_t s = time(0);

		cv::resize(frame, frame, cv::Size(), frame_scaling_factor, frame_scaling_factor);

		frame_buffer.push_back(frame);
		time_buffer.push_back(time(0));

		if (frame_buffer.size() > 1) {
			float[] cur_frame = frame_buffer[frame_buffer.size() - 1];
			float[] prev_frame = frame_buffer[frame_buffer.size() - 2];

			float[] luminous = luminance_flash_count(cur_frame, prev_frame);
			float[] red = saturated_red_flash_count(cur_frame, prev_frame);

			luminous_flash_buffer.push_back(luminous);
			luminous_flashes = luminous_flashes + luminous;
			red_flash_buffer.push_back(red);
			red_flashes = red_flashes + red;
		}

		time_t interval = time_buffer[time_buffer.size() - 1] - time_buffer[0];

		if (frame_buffer.size() >= BUFFER_SIZE && interval >= 1) {
			float[] luminous_flash_freq = float[]::zeros(luminous_flashes.rows, luminous_flashes.cols, CV_32F);
			float[] red_flash_freq = float[]::zeros(red_flashes.rows, red_flashes.cols, CV_32F);

			luminous_flash_freq = (luminous_flashes / 2) / interval;
			red_flash_freq = (red_flashes / 2) / interval;

			int luminous_count = 0;
			int red_count = 0;

			for (int i = 0; i < luminous_flash_freq.rows; i++) {
				for (int j = 0; j < luminous_flash_freq.cols; j++) {
					if (luminous_flash_freq.at<float>(i, j) >= 3) {
						luminous_count++;
					}
				}
			}

			for (int i = 0; i < red_flash_freq.rows; i++) {
				for (int j = 0; j < red_flash_freq.cols; j++) {
					if (red_flash_freq.at<float>(i, j) >= 3) {
						red_count++;
					}
				}
			}

			if (luminous_count >= quarter_area_threshold || red_count >= quarter_area_threshold) {
				cout << "Flashing Detected!" << endl;
			}

			frame_buffer.pop_front();
			time_buffer.pop_front();

			luminous_flashes = luminous_flashes - luminous_flash_buffer.front();
			luminous_flash_buffer.pop_front();

			red_flashes = red_flashes - red_flash_buffer.front();
			red_flash_buffer.pop_front();

			cout << time(0) - s << endl;
		}
	}

	cap.release();

	return 0;
}

float[] luminance_flash_count(float[] color, float[] prev_color) {
	color = color / 255;
	prev_color = prev_color / 255;

	float[] linear_color = sRGB_to_linearRGB(color);
	float[] linear_prev_color = sRGB_to_linearRGB(prev_color);

	float[] Ls = linearRBG_to_Ls(linear_color);
	float[] prev_Ls = linearRBG_to_Ls(linear_prev_color);

	return is_luminance_flash(Ls, prev_Ls);
}

float[] sRGB_to_linearRGB(float[] sRGB) {
	float[] linear = inverse_gamma_transform(sRGB);
	return linear;
}

float[] inverse_gamma_transform(float[] signal) {
	for (int i = 0; i < signal.rows; i++) {
		for (int j = 0; j < signal.cols; j++) {
			for (int k = 0; k < 3; k++) {
				if (signal.at<cv::Vec3b>(i, j)[k] <= 0.03928) {
					signal.at<cv::Vec3b>(i, j)[k] = signal.at<cv::Vec3b>(i, j)[k] / 12.92;
				}
				else {
					signal.at<cv::Vec3b>(i, j)[k] = pow((signal.at<cv::Vec3b>(i, j)[k] + 0.055) / 1.055, 2.4);
				}
			}
		}
	}

	return signal;
}

float[] linearRBG_to_Ls(float[] linearRGB) {
	float[] ls = float[]::zeros(linearRGB.rows, linearRGB.cols, CV_32F);

	for (int i = 0; i < linearRGB.rows; i++) {
		for (int j = 0; j < linearRGB.cols; j++) {
			ls.at<float>(i, j) = 0.2126 * linearRGB.at<cv::Vec3b>(i, j)[0] + 0.7152 * linearRGB.at<cv::Vec3b>(i, j)[1] + 0.0722 * linearRGB.at<cv::Vec3b>(i, j)[2];
		}
	}

	return ls;
}

float[] is_luminance_flash(float[] ls, float[] prev_ls) {
	float[] brighter_ls = float[]::zeros(ls.rows, ls.cols, CV_32S);
	float[] darker_ls = float[]::zeros(ls.rows, ls.cols, CV_32S);

	float[] flashing_grid = float[]::zeros(ls.rows, ls.cols, CV_32S);

	for (int i = 0; i < ls.rows; i++) {
		for (int j = 0; j < ls.cols; j++) {
			if (ls.at<int>(i, j) > prev_ls.at<int>(i, j)) {
				brighter_ls.at<int>(i, j) = ls.at<int>(i, j);
				darker_ls.at<int>(i, j) = prev_ls.at<int>(i, j);
			}
			else {
				brighter_ls.at<int>(i, j) = prev_ls.at<int>(i, j);
				darker_ls.at<int>(i, j) = ls.at<int>(i, j);
			}
		}
	}

	for (int i = 0; i < ls.rows; i++) {
		for (int j = 0; j < ls.cols; j++) {
			if ((brighter_ls.at<int>(i, j) - darker_ls.at<int>(i, j)) >= 0.1 && darker_ls.at<int>(i, j) < 0.8) {
				flashing_grid.at<int>(i, j) = 1;
			}
			else {
				flashing_grid.at<int>(i, j) = 0;
			}
		}
	}

	return flashing_grid;
}

float[] saturated_red_flash_count(float[] color, float[] prev_color) {
	color = color / 255;
	prev_color = prev_color / 255;

	float[] linear_color = sRGB_to_linearRGB(color);
	float[] linear_prev_color = sRGB_to_linearRGB(prev_color);

	return is_saturated_red_flash(linear_color, linear_prev_color);
}

float[] is_saturated_red_flash(float[] linear_color, float[] prev_linear_color) {
	return float[]::zeros(1, 1, CV_32S); // change this
}

float[] red_ratio(float[] sRGB) {
	float[] result = float[]::zeros(sRGB.rows, sRGB.cols, CV_32F);

	for (int i = 0; i < sRGB.rows; i++) {
		for (int j = 0; j < sRGB.cols; j++) {
			result.at<float>(i, j) = sRGB.at<cv::Vec3b>(i, j)[0] / (sRGB.at<cv::Vec3b>(i, j)[0] + sRGB.at<cv::Vec3b>(i, j)[1] + sRGB.at<cv::Vec3b>(i, j)[2] + 1e-10);
		}
	}

	return result;
}

float[] pure_red(float[] sRGB) {
	float[] comp = float[]::zeros(sRGB.rows, sRGB.cols, CV_32S);

	for (int i = 0; i < comp.rows; i++) {
		for (int j = 0; j < comp.cols; j++) {
			if (sRGB.at<cv::Vec3b>(i, j)[0] - sRGB.at<cv::Vec3b>(i, j)[1] - sRGB.at<cv::Vec3b>(i, j)[2] > 0) {
				comp.at<int>(i, j) = 320 * (sRGB.at<cv::Vec3b>(i, j)[0] - sRGB.at<cv::Vec3b>(i, j)[1] - sRGB.at<cv::Vec3b>(i, j)[2]);
			}
			else {
				comp.at<int>(i, j) = 0;
			}
		}
	}

	return comp;
}