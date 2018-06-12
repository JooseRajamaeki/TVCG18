/*

MIT License

Copyright (c) 2017 Joose Rajamäki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


*/

#include "MiscUtils.hpp"

std::string get_time_string() {
	time_t now = time(0);
	tm localtm;
	localtime_s(&localtm, &now);
	std::string result(1000, ' ');

	asctime_s((char*)result.data(), 1000, &localtm);


	do {

		if (result.back() == '\n') {
			result.pop_back();
			break;
		}

		result.pop_back();

		if (result.size() == 0) {
			break;
		}

	} while (true);

	result = result.substr(4, result.size());

	std::string year_string = result.substr(result.size() - 4, result.size());
	result = result.substr(0, result.size() - 5);

	result = year_string + "_" + result;

	for (char& character : result) {
		if (character == ':') {
			character = '.';
		}
		if (character == ' ') {
			character = '_';
		}
	}

	size_t index = 0;
	while (true) {
		/* Locate the substring to replace. */
		index = result.find("__", index);
		if (index == std::string::npos) {
			break;
		}

		/* Make the replacement. */
		result.replace(index, 2, "_");
	}

	return result;
}

void repulse(float** sequence1, int dim, int data_points, int repulse_points, float shift_amount)
{

	std::vector<std::vector<float>> data;
	data.reserve(data_points*dim);


	std::vector<int> random_order(data_points, 0);
	for (int i = 0; i < data_points; i++) {
		random_order[i] = i;
	}

	for (int i = 0; i < data_points; i++) {
		int rand_idx = rand() % data_points;

		int tmp = random_order[i];
		random_order[i] = random_order[rand_idx];
		random_order[rand_idx] = tmp;
	}


	for (int ii = 0; ii < data_points; ii++) {

		int i = random_order[ii];

		std::vector<float> tmp(dim, 0.0f);

		Eigen::Map<Eigen::VectorXf> datum(tmp.data(), dim);

		Eigen::Map<Eigen::VectorXf> current(sequence1[i], dim);

		Eigen::VectorXf diff;

		if (data_points == repulse_points) {

			for (int j = 0; j < data_points; j++) {

				if (i == j) {
					continue;
				}

				Eigen::Map<Eigen::VectorXf> other(sequence1[j], dim);

				diff = current - other;
				float norm = diff.norm();
				if (norm > std::numeric_limits<float>::min()) {
					diff /= norm;
				}

				datum += diff;

			}

		}
		else {

			for (int k = 0; k < repulse_points; k++) {

				int j = rand() % data_points;

				if (i == j) {
					continue;
				}

				Eigen::Map<Eigen::VectorXf> other(sequence1[j], dim);

				diff = current - other;
				float norm = diff.norm();
				if (norm > std::numeric_limits<float>::min()) {
					diff /= norm;
				}

				datum += diff;

			}

		}

		float norm = datum.norm();
		if (norm > std::numeric_limits<float>::min()) {
			datum /= norm;
		}
		datum *= shift_amount;

		datum += current;

		data.push_back(tmp);

	}


	for (int i = 0; i < data_points; i++) {
		for (int j = 0; j < dim; j++) {
			sequence1[i][j] = data[i][j];
		}
	}
}


void swap_closest(float** sequence1, int dim, int data_points, int repulse_points)
{

	if (data_points <= 1 || repulse_points <= 1) {
		return;
	}

	std::vector<int> random_order(data_points, 0);
	for (int i = 0; i < data_points; i++) {
		random_order[i] = i;
	}

	for (int i = 0; i < data_points; i++) {
		int rand_idx = rand() % data_points;

		int tmp = random_order[i];
		random_order[i] = random_order[rand_idx];
		random_order[rand_idx] = tmp;
	}

	for (int ii = 0; ii < data_points; ii++) {

		int i = random_order[ii];

		Eigen::Map<Eigen::VectorXf> current(sequence1[i], dim);
		float* closest = nullptr;
		float min_dist = std::numeric_limits<float>::infinity();


		if (data_points == repulse_points) {

			for (int j = 0; j < data_points; j++) {

				if (i == j) {
					continue;
				}

				Eigen::Map<Eigen::VectorXf> other(sequence1[j], dim);

				float dist = (current - other).norm();

				if (dist < min_dist) {
					min_dist = dist;
					closest = sequence1[j];
				}

			}

		}
		else {

			for (int k = 0; k < repulse_points; k++) {

				int j = rand() % data_points;

				if (i == j) {
					continue;
				}

				Eigen::Map<Eigen::VectorXf> other(sequence1[j], dim);

				float dist = (current - other).norm();

				if (dist < min_dist) {
					min_dist = dist;
					closest = sequence1[j];
				}

			}

		}

		if (closest) {
			Eigen::VectorXf tmp;

			Eigen::Map<Eigen::VectorXf> other(closest, dim);
			tmp = other;
			other = current;
			current = tmp;
		}

	}


}

std::vector<std::pair<int, float>> indices_of_k_nearest(float ** sequence1, int dim, int data_points, int point_idx, int k)
{



	Eigen::Map<const Eigen::VectorXf> current(sequence1[point_idx], dim);

	std::vector<std::pair<int, float>> closest;

	for (int i = 0; i < data_points; ++i) {

		if (i == point_idx) {
			continue;
		}

		Eigen::Map<const Eigen::VectorXf> other(sequence1[i], dim);

		float dist = (other - current).norm();


		auto order_meas = [](const std::pair<int, float>& sample1, const std::pair<int, float>& sample2) {
			return sample1.second < sample2.second;
		};


		if (closest.size() < k) {

			closest.push_back(std::make_pair(i, dist));
			std::sort(closest.begin(), closest.end(), order_meas);

		}
		else {

			if (closest.back().second > dist) {

				closest.back().second = dist;
				closest.back().first = i;

				std::sort(closest.begin(), closest.end(), order_meas);

			}

		}

	}

	return closest;



}


std::vector<std::pair<int, float>> indices_of_k_nearest(float ** sequence1, const Eigen::VectorXf& current,int dim, int data_points, int k)
{

	std::vector<std::pair<int, float>> closest;

	for (int i = 0; i < data_points; ++i) {

		Eigen::Map<const Eigen::VectorXf> other(sequence1[i], dim);

		float dist = (other - current).norm();

		auto order_meas = [](const std::pair<int, float>& sample1, const std::pair<int, float>& sample2) {
			return sample1.second < sample2.second;
		};


		if (closest.size() < k) {

			closest.push_back(std::make_pair(i, dist));
			std::sort(closest.begin(), closest.end(), order_meas);

		}
		else {

			if (closest.back().second > dist) {

				closest.back().second = dist;
				closest.back().first = i;

				std::sort(closest.begin(), closest.end(), order_meas);

			}

		}

	}

	return closest;



}



void project_to_unit_sphere(float** sequence, int dim, int data_points)
{

	for (int i = 0; i < data_points; i++) {
		Eigen::Map<Eigen::VectorXf> current(sequence[i], dim);
		current.normalize();
	}


}

void softmax(const float* nums, float* result, int dim)
{
	float max = -std::numeric_limits<float>::infinity();
	float sum = 0.0f;
	int max_idx = -1;

	for (int i = 0; i < dim; ++i) {
		if (nums[i] > max) {
			max = nums[i];
			max_idx = i;
		}

		sum += nums[i];
	}


	//Here we have a one hot vector.
	if (sum == max) {

		for (int i = 0; i < dim; ++i) {
			result[i] = 0.0f;
		}
		result[max_idx] = 1.0f;

		return;
	}

	sum = 0.0f;

	for (int i = 0; i < dim; ++i) {
		result[i] = nums[i] - max;
		result[i] = std::exp(result[i]);
		sum += result[i];
	}

	for (int i = 0; i < dim; ++i) {
		 result[i] /= sum;
	}

}
