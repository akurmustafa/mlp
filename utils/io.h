
#ifndef UTILS_IO_H
#define UTILS_IO_H

#include <algorithm>
#include <cstddef>
#include <cassert>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "dpu_config.h"

namespace io 
{
// TEMPLATE FUNCTION DECLARATIONS
template<typename T>
inline void print_elements(const T& coll, const std::string& optional_str = "");

template<typename T>
void print_arr(const T& arr, std::size_t length);

template<typename T>
void print_arr(const T arr[][10], int row = 10, int col = 10);

template<typename T>
void print_vector(const std::vector<T>& vec);

template<typename T>
void print_vector(const std::vector<std::vector<T>>& vec);

template<typename T>
void print_deque(const std::deque<T>& deck);

template<typename T>
int save_data_as_bin(const std::vector<T>& data, const std::string& file_path);

template<typename T>
int save_data_as_bin(const std::vector<std::vector<T>>& data, const std::string& file_path);

template<typename T>
std::vector<std::vector<T>> read_bin_file(const std::string& file_path, int channel_num = 5150, int channel_start = 0, int channel_end = 5150, \
	int time_start_sec = 0, int time_end_sec = 3600, int fs = 2000);

template<typename T>
std::vector<T> read_file(const std::string& file_path, int channel_num = 5150);

template<typename T>
std::vector<std::vector<T>> read_norm_power(const std::string& file_path, int channel_num = 5150, int channel_start = 0, int channel_end = 5150, \
	int time_start_sec = 0, int time_end_sec = 3600, int fs = 5, int slice_start = 1, int slice_stride = 2, int slice_end = -1);

template<typename T>
std::vector<std::vector<T>> read_prob(const std::string& file_path, int class_no, int class_num = 14, int channel_num = 5150, int channel_start = 0, int channel_end = 5150, \
	int time_start_sec = 0, int time_end_sec = 3600, int fs = 5);

// NON TEMPLATE FUNCTION DECLARATIONS DEFINITIONS ARE IN THE io.cpp
std::vector<std::string> get_files(const std::string& directory, const std::string& extension);
std::string get_parent_folder(const std::string& file_path);
std::string get_file_name(const std::string& file_path);
void create_folder(const std::string& dir);
void partition_bin_file(const std::string& file_path, const std::string& extension = ".pow", int channel_num = 5150, int partition_num = 10);
// void save_higher_prob_input_spectrograms(float* mel_in, float* norm_power_db, float* prob_data, int channel_num, int class_num, 
//										int time_counter, HighProbLogParameters high_prob_log_parameters);


// TEMPLATE FUNCTION DEFINITIONS

// This function prints elemensts inside a container to the screen
template<typename T>
inline void print_elements(const T& coll, const std::string& optional_str) {
	std::cout << optional_str;
	for (const auto& elem : coll) {
		std::cout << elem << ", ";
	}
	std::cout << "\n";
}


// This function prints arrays to the screen, arguments address of 
// the first elements of the array and array length, format: value0, value1, etc.
template<typename T>
void print_arr(const T& arr, std::size_t length) {
	for (int i = 0; i < length; ++i) {
		std::cout << arr[i] << ", ";
	}
	std::cout << '\n';
}


/* This function prints 2D arrays to the screen, assumes that 2nd dimension of the array is 10, give the fist dimension and the second dimension of the array to print.
Format: value00, value01, etc
value10, value11, etc.*/
template<typename T>
void print_arr(const T arr[][10], int row, int col) {
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			std::cout << arr[i][j] << ", ";
		}
		std::cout << '\n';
	}
}

// This function prints vector values to the screen. in the format value0, value1, etc.
template<typename T>
void print_vector(const std::vector<T>& vec) {
	for (const auto& elem : vec) {
		std::cout << elem << ", ";
	}
	std::cout << '\n';
}

/* This function prints 2D vectors to the screen.
Format:		value00, value01, etc
value10, value11, etc.*/
template<typename T>
void print_vector(const std::vector<std::vector<T>>& vec) {
	for (const auto& elem : vec) {
		print_vector(elem);
	}
}

template<typename T>
void print_deque(const std::deque<T>& deck) {
	for (const auto& elem : deck) {
		std::cout << elem << ", ";
	}
	std::cout << '\n';
}


// This function saves 1d vector data as bin file to the given path
template<typename T>
int save_data_as_bin(const std::vector<T>& data, const std::string& file_path) {
	std::ofstream bin_file(file_path, std::ios::out | std::ios::binary | std::ios::beg);
	namespace fs = std::experimental::filesystem;
	if (fs::exists(file_path)) {
		bin_file.clear();
	}
	if (bin_file.is_open()) {
		// std::cout << "writing to file: " << file_path << "\n";
		bin_file.write((char*)(&data[0]), sizeof(T)*data.size());
		bin_file.close();
		return 0; // means success
	}
	else {
		std::cout << "unable to open file: " << file_path << "\n";
		return 1; // means failure
	}
}


/* This function saves 2d vector data as bin file to the given path, it writes vector
row by row to the bin format: 0th row, 1st row, ..,  nth row,
(this is different from MATLAB. MATLAB first writes columns)*/
template<typename T>
int save_data_as_bin(const std::vector<std::vector<T>>& data, const std::string& file_path) {
	std::ofstream bin_file(file_path, std::ios::out | std::ios::binary | std::ios::beg);
	namespace fs = std::filesystem;
	if (fs::exists(file_path)) {
		bin_file.clear();
	}
	if (bin_file.is_open()) {
		std::cout << "writing to file: " << file_path << "\n";
		for (const auto& cur_row : data) {
			bin_file.write((char*)(&cur_row[0]), sizeof(T)*cur_row.size());
		}
		bin_file.close();
		return 0;
	}
	else {
		std::cout << "unable to open file: " << file_path << "\n";
		return 1;
	}
}


/* This function reads data (midas raw data) from bin file in the given file_path,
channel_num is the channel_num in the recording,
channel_start is the starting channel we wish to rad data,
channel_end is the channel we wish to end reading,
time_start_sec is the time index in seconds we wish to read data,
time_end_sec is the time index we wish to end reading,
fs is the sampling interval of the recording.
For instance assume data as below
0------ - 5149
0------ - 5149
0------ - 5149
0------ - 5149
0------ - 5149
0------ - 5149
read_bin_file(path, 5150, 100, 1000, 0, 3, 1) reads the portion of the data as below
100------999
100------999
100------999
Meaning end points are exclusive*/
template<typename T>
std::vector<std::vector<T>> read_bin_file(const std::string& file_path, int channel_num, int channel_start, int channel_end, int time_start_sec, int time_end_sec, int fs) {
	assert((channel_start >= 0 && channel_end <= channel_num && channel_end>channel_start && time_end_sec > time_start_sec) && "Inputs are not valid!");
	std::vector<std::vector<T>> res;
	std::cout << res.size() << "\n";
	std::ifstream bin_file(file_path, std::ios::in | std::ios::binary | std::ios::beg);
	if (bin_file.is_open()) {
		int col_num = channel_end - channel_start;
		bin_file.seekg(0, std::ios::end);
		const size_t num_elements = bin_file.tellg() / sizeof(T);
		int64_t row_num = num_elements / channel_num;
		std::cout << row_num << ", " << col_num << "\n";
    int64_t time_start = time_start_sec * fs;
    int64_t time_end = time_end_sec * fs;
		if (time_start > row_num) {
			bin_file.close();
			std::cout << "File is not that long! Change time_start\n";
			return res;
		}
    int64_t min_val = row_num < time_end ? row_num : time_end;//min(row_num, time_end);
		res = std::vector<std::vector<T>>(min_val - time_start, std::vector<T>(col_num, 0));
		for (int64_t i = time_start; i < min_val; ++i) {
			bin_file.seekg((channel_num*(i)+channel_start)*sizeof(T), std::ios::beg);
			std::vector<T> data(col_num);
			bin_file.read(reinterpret_cast<char*>(&data[0]), col_num*sizeof(T));
			res[i-time_start] = data;
		}
		std::cout << "Entire file content is in memory\n";
		bin_file.close();
	}
	else {
		std::cout << "unable to open file: " << file_path << "\n";
	}
	return res;
}


/* This function is basically same with the above read_bin_file function but returning vector is 1-D.
0-------5149
0-------5149
0-------5149
0-------5149
0-------5149
0-------5149
read_file(path, 5150, 100, 1000, 0, 3, 1) reads the portion of the data as below
100------999, 100------999, 100------999   */
template<typename T>
std::vector<T> read_file(const std::string& file_path, int channel_num) {
	std::vector<T> res;
	std::ifstream bin_file(file_path, std::ios::in | std::ios::binary | std::ios::beg);
	int i{ 0 }; int j{ 0 };
	if (bin_file.is_open()) {
		bin_file.seekg(0, std::ios::end);
		const size_t num_elements = bin_file.tellg() / sizeof(T);
    int64_t row_num = num_elements / channel_num;
		std::cout << row_num << ", " << channel_num << "\n";
		bin_file.seekg(0, std::ios::beg);
		std::vector<T> data(num_elements);
		bin_file.read(reinterpret_cast<char*>(&data[0]), num_elements*sizeof(T));
		std::cout << "Entire file content is in memory\n";
		return data;
		bin_file.close();
	}
	else {
		std::cout << "unable to open file: " << file_path << "\n";
	}
	return res;
}

/*// This function reads data (midas  norm power [waterfall]) from bin file
in the given file_path,
channel_num is the channel_num in the recording,
channel_start is the starting channel we wish to rad data,
channel_end is the channel we wish to end reading,
time_start_sec is the time index in seconds we wish to read data,
time_end_sec is the time index we wish to end reading,
fs is the sampling interval of the recording,
slice_start is the starting index we wish to read data,
slice_stride is the number showing at which intervals we wish to read data(for norm power 2),
slice_end is the index we wish to end reading (-1 means last).
For instance assume data as below
0-> 0-------5149
1-> 0-------5149
2-> 0-------5149
3-> 0-------5149
4-> 0-------5149
5-> 0-------5149
read_norm_power(path, channel_num = 5150, channel_start = 100, channel_end = 1000,
time_start_sec = 0, time_start_end = 3, fs = 1, slice_start = 1,
slice_stride = 2, slice_end = -1) reads the portion of the data as below
1-> 100------999
3-> 100------999
5-> 100------999
Meaning end points are exclusive
*/
template<typename T>
std::vector<std::vector<T>> read_norm_power(const std::string& file_path, int channel_num, int channel_start, int channel_end, \
	int time_start_sec, int time_end_sec, int fs, int slice_start, int slice_stride, int slice_end) {
	assert((channel_start >= 0 && channel_end <= channel_num && channel_end>channel_start && time_end_sec > time_start_sec) && "Inputs are not valid!");
	int norm_power_mult{ 2 }; // 2 because raw_data produces 2 values while calculating norm_power, we want to get normalized result
	std::vector<std::vector<T>> res;
	std::vector<std::vector<T>> res_sliced;
	std::cout << res.size() << "\n";
	std::ifstream bin_file(file_path, std::ios::in | std::ios::binary | std::ios::beg);
	if (bin_file.is_open()) {
		int col_num = channel_end - channel_start;
		bin_file.seekg(0, std::ios::end);
		const std::size_t num_elements = bin_file.tellg() / sizeof(T);
		std::size_t row_num = num_elements / channel_num;
		std::cout << row_num << ", " << col_num << "\n";
		int time_start = time_start_sec * fs * norm_power_mult; 
		int time_end = time_end_sec * fs * norm_power_mult;
		if (time_start > row_num) {
			bin_file.close();
			std::cout << "File is not that long! Change time_start\n";
			return res;
		}
		std::size_t min_val = row_num < time_end ? row_num : time_end;	//std::min(row_num, time_end);
		if (slice_end >= 0) {
			min_val = slice_end * fs;
		}
		std::size_t res_row_num = std::ceil((min_val - time_start - slice_start) / static_cast<double>(slice_stride));
		std::cout << res_row_num << ", " << col_num << "\n";
		res = std::vector<std::vector<T>>(res_row_num, std::vector<T>(col_num, 0));
		for (int i = time_start + slice_start; i < min_val; i += slice_stride) {
			// std::cout << (i - time_start - slice_start) / slice_stride << ", ";
			bin_file.seekg((channel_num*(i)+channel_start)*sizeof(T), std::ios::beg);
			std::vector<T> data(col_num);
			bin_file.read(reinterpret_cast<char*>(&data[0]), col_num*sizeof(T));
			res[(i - time_start - slice_start) / slice_stride ] = data;
		}
		std::cout << "Entire file content is in memory\n";
		bin_file.close();
	}
	else {
		std::cout << "unable to open file: " << file_path << "\n";
	}
	return res;
}

/*// This function reads data (midas prob data[classification results])from bin file in the given file_path, class_no is the class of clasificaiton result we wish to see prob values,  class_num is the total number of classes in the output of the classifier (for our case 14), channel_num is the channel_num in the recording, channel_start is the starting channel we wish to rad data, channel_end is the channel we wish to end reading, time_start_sec is the time index in seconds we wish to read data, time_end_sec is the time index we wish to end reading, fs is the sampling interval of the recording.
For instance assume data as below
0, 1, 2, .., 13,0,1,2, .., 13, ..
read_prob(file_path, class_no = 1, class_numn = 14, channel_num = 5150, channel_ start = 0, channel_end = 1000, time_Start_sec = 0, time_end_sec = 1, fs = 5) obtains the

0...       999			5 = time_end_sec* fs;
0->	1, 1,1 ..,  1
1->	1, 1,1 ..,  1
2->	1, 1,1 ..,  1
3->	1, 1,1 ..,  1
4->	1, 1,1 ..,  1
5->	1, 1,1 ..,  1
*/

template<typename T>
std::vector<std::vector<T>> read_prob_old(const std::string& file_path, int class_no, int class_num, int channel_num, int channel_start, int channel_end, \
	int time_start_sec, int time_end_sec, int fs) {
	assert((channel_start >= 0 && channel_end <= channel_num && channel_end>channel_start && time_end_sec > time_start_sec) && "Inputs are not valid!");
	std::vector<std::vector<T>> res;
	std::cout << res.size() << "\n";
	std::ifstream bin_file(file_path, std::ios::in | std::ios::binary | std::ios::beg);
	if (bin_file.is_open()) {
		bin_file.seekg(0, std::ios::end);
		const size_t num_elements = bin_file.tellg() / sizeof(T);
		int num_elements_per_prob = num_elements / class_num;
		int col_num = channel_end - channel_start;
		int row_num = num_elements_per_prob / channel_num;
		std::cout << row_num << ", " << col_num << "\n";
		int time_start = time_start_sec * fs;
		int time_end = time_end_sec * fs;
		int prob_data_size = std::min(num_elements_per_prob, time_end*channel_num);
		std::vector<std::vector<T>> prob_data(prob_data_size, std::vector<T>(class_num, 0));
		for (int i = 0; i < prob_data_size; ++i) {
			bin_file.seekg((class_num)*i*sizeof(T), std::ios::beg);
			std::vector<T> cur_class_data(class_num);
			bin_file.read(reinterpret_cast<char*>(&cur_class_data[0]), class_num*sizeof(T));
			prob_data[i] = cur_class_data;
		}
		if (time_start > row_num) {
			bin_file.close();
			std::cout << "File is not that long! Change time_start\n";
			return res;
		}
		int min_val = prob_data_size / channel_num;
		res = std::vector<std::vector<T>>(min_val - time_start, std::vector<T>(col_num, 0));
		for (int i = time_start; i < min_val; ++i) {
			std::vector<T> cur_row(col_num, 0);
			for (int j = 0; j < col_num; ++j) {
				cur_row[j] = prob_data[i*channel_num + channel_start +j][class_no];
			}
			res[i - time_start] = cur_row;
		}
		std::cout << "Entire file content is in memory\n";
		bin_file.close();
	}
	else {
		std::cout << "unable to open file: " << file_path << "\n";
	}
	return res;
}

template<typename T>
std::vector<std::vector<T>> read_prob(const std::string& file_path, int class_no, int class_num, int channel_num, int channel_start, int channel_end, \
	int time_start_sec, int time_end_sec, int fs) {
	assert((channel_start >= 0 && channel_end <= channel_num && channel_end>channel_start && time_end_sec > time_start_sec) && "Inputs are not valid!");
	std::vector<std::vector<T>> res;
	std::cout << res.size() << "\n";
	std::ifstream bin_file(file_path, std::ios::in | std::ios::binary | std::ios::beg);
	if (bin_file.is_open()) {
		bin_file.seekg(0, std::ios::end);
		const size_t num_elements = bin_file.tellg() / sizeof(T);
		int num_elements_per_prob = num_elements / class_num;
		int col_num = channel_end - channel_start;
		int row_end = num_elements_per_prob / channel_num;
		int time_start = time_start_sec * fs;
		int time_end = time_end_sec * fs;
		row_end = std::min(time_end, row_end);
		if (time_start >= row_end) {
			std::cout << "Input is not that long. Change time_start !\n";
			return res;
		}
		int row_num = row_end - time_start;
		std::cout << row_num << ", " << col_num << "\n";
		int prob_data_size = row_end*channel_num;
		std::vector<T> prob_data(prob_data_size*class_num, 0);
		bin_file.seekg(0, std::ios::beg);
		bin_file.read(reinterpret_cast<char*>(&prob_data[0]), prob_data_size*class_num * sizeof(T));
		bin_file.close();
		res = std::vector<std::vector<T>>(row_num, std::vector<T>(col_num, 0));
		for (int i = time_start; i < row_end; ++i) {
			for (int j = channel_start; j < channel_end; ++j) {
				res[i - time_start][j - channel_start] = prob_data[class_num*(channel_num*i + j) + class_no];
			}
		}
		std::cout << "Entire file content is in memory\n";
	}
	else {
		std::cout << "unable to open file: " << file_path << "\n";
	}
	return res;
}

}  // namespace io

#endif  // end of UTILS_IO_H
