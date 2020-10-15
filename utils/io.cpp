
#include "utils/io.h"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <unordered_map>

namespace io 
{
/* This function returns the all files with extension under the directory
For instance get_files(“img/”, “.png”) return all the files with png extension
under img folder, however it doesn’t return files in the deeper layers.
For instance it doesn’t return .img/fold1/0.png
*/
std::vector<std::string> get_files(const std::string& directory, const std::string& extension) {
	namespace fs = std::filesystem;
	std::vector<std::string> res;
	if (!fs::exists(directory)) {
		std::cout << "Directory doesn't exist check the directory!\n";
		return res;
	}
	for (auto& entry : fs::directory_iterator(directory)) {
		// std::cout << entry << "\n";
		std::string s = entry.path().string();
		std::string dot = ".";
		auto index = std::find_end(s.begin(), s.end(), dot.begin(), dot.end());
		if (index != s.end()) {
			std::string remaining;
			for (; index != s.end(); ++index) {
				remaining.push_back(*index);
			}
			if (remaining == extension) {
				res.push_back(s);
			}
		}
	}
	return res;
}

/*This function returns the parent folder given a file path, for instance get_parent_folder(“./img/img1.png”) return “./img/”	*/
std::string get_parent_folder(const std::string& file_path) {
	std::string parent_folder;
	std::string fold_separate1 = "\\";
	std::string fold_separate2 = "/";
	auto it_start1 = std::find_end(file_path.begin(), file_path.end(), fold_separate1.begin(), fold_separate1.end());
	auto it_start2 = std::find_end(file_path.begin(), file_path.end(), fold_separate2.begin(), fold_separate2.end());
	auto it_end = file_path.end();
	auto it_start = std::max(std::min(it_start1, it_end), std::min(it_start2, it_end));
	if (it_start == it_end) {
		it_start = std::min(std::min(it_start1, it_end), std::min(it_start2, it_end));
	}
	if (it_start != file_path.end()) {
		for (auto idx = file_path.begin(); idx != it_start + 1; ++idx) {
			parent_folder.push_back(*idx);
		}
	}
	return parent_folder;
}

// This function returns the file_name given a file_path, for instance get_file_name(“./img/img1.png”) returns “img1”
std::string get_file_name(const std::string& file_path) {
	std::string file_name;
	std::string fold_separate1 = "\\";
	std::string fold_separate2 = "/";
	std::string dot = ".";
	auto it_start1 = std::find_end(file_path.begin(), file_path.end(), fold_separate1.begin(), fold_separate1.end());
	auto it_start2 = std::find_end(file_path.begin(), file_path.end(), fold_separate2.begin(), fold_separate2.end());
	auto it_end = std::find_end(file_path.begin(), file_path.end(), dot.begin(), dot.end());
	auto it_start = std::max(std::min(it_start1, it_end), std::min(it_start2, it_end));
	if (it_start == it_end) {
		it_start = std::min(std::min(it_start1, it_end), std::min(it_start2, it_end));
	}
	if (it_start != file_path.end() && it_end != file_path.end() && it_end > it_start) {
		for (auto idx = it_start + 1; idx != it_end; ++idx) {
			file_name.push_back(*idx);
		}
	}
	return file_name;
}

// This function creates the folder given in the argument, for instance create_folder(“./res/img/”) creates img folder 
// given “./res/” folder exists
void create_folder(const std::string& dir) {
	namespace fs = std::filesystem;
	// auto bool1 = fs::is_directory(dir);
	auto bool2 = fs::exists(dir);
	if (!bool2) {
		fs::create_directory(dir);
		std::cout << dir << " created\n";
	}
	else {
		std::cout << dir << " already exists\n";
	}
}

/*// This function partitions bin file given in the file_path, into partition_num parts
For instance assume data as below
0------ 5149
0-------5149
0------ 5149
0-------5149
0------ 5149
0-------5149
partition_bin_file(const std::string& file_path, const std::string& extension = ".pow", int channel_num = 5150, int partition_num = 10)
results in
0------ 514		515------1029		.. ..	  4636-----5149
0-------514		515------1029		.. ..	  4636-----5149
0------ 514		515------1029		.. ..	  4636-----5149
0-------514		515------1029		.. ..	  4636-----5149
0------ 514		515------1029		.. ..	  4636-----5149
0-------514		5515------1029		.. ..	  4636-----5149
saves these partitions to the bin file
*/
void partition_bin_file(const std::string& file_path, const std::string& extension, int channel_num, int partition_num) {
	int channel_width = channel_num / partition_num;
	auto parent_folder = io::get_parent_folder(file_path);
	auto partition_folder = parent_folder + "partitions/";
	io::create_folder(partition_folder);
	int channel_start{ 0 }, channel_end = channel_width;
	while (channel_end <= channel_num) {
		auto file_name = io::get_file_name(file_path);
		auto cur_part_file_name = file_name + "-" + std::to_string(channel_start) + "_" + std::to_string(channel_end) + extension;
		auto cur_part_file_path = partition_folder + cur_part_file_name;
		auto cur_part = io::read_norm_power<float>(file_path, channel_num, channel_start, channel_end, 0, 3600, 5, 1, 2, -1);
		int failure = io::save_data_as_bin(cur_part, cur_part_file_path);
		channel_start = channel_end;
		channel_end = channel_start + channel_width;
		if (failure)
			std::cout << cur_part_file_path << " couldn't written\n";
		else
			std::cout << cur_part_file_path << " is written\n";
	}
}

//void save_higher_prob_input_spectrograms(float* mel_in, float* norm_power_db, float* prob_data, int channel_num, int class_num, int time_counter, HighProbLogParameters high_prob_log_parameters_) {
//	namespace tp = tracker_params;
//	int channel_end = std::min(high_prob_log_parameters_.spectrogram_dump_channel_end, channel_num);
//	int channel_start = std::max(0, high_prob_log_parameters_.spectrogram_dump_channel_start);
//	for (int i = channel_start; i < channel_end; ++i) {
//		float cur_norm_power = norm_power_db[i];
//		for (int idx = 0; idx < high_prob_log_parameters_.active_categories_enabled_ptr->enabled.size(); ++idx) {
//			if (high_prob_log_parameters_.active_categories_enabled_ptr->enabled[idx]) {
//				std::string category_name = high_prob_log_parameters_.active_categories_enabled_ptr->categories[idx];
//				float cur_prob = prob_data[i*class_num + idx];
//				if (cur_prob > high_prob_log_parameters_.spectrogram_dump_high_prob_thresh && cur_norm_power > high_prob_log_parameters_.norm_power_anding_thresh) {
//					int cur_probi = 999 < std::round(cur_prob*1e3) ? 999 : std::round(cur_prob*1e3);
//					std::string spectrogram_input_path = high_prob_log_parameters_.spectrogram_folder + category_name + "/"
//						"time_" + std::to_string(time_counter) + "-channel_" + std::to_string(i) + \
//						"-prob_" + process::convert_string(cur_probi, 3) + ".bin";
//					std::vector<float> mel_data(mel_in + 1536 * i, mel_in + 1536 * (i + 1));
//					int failure = save_data_as_bin(mel_data, spectrogram_input_path);
//					if (failure) {
//						std::cout << "data couldn't be written!!!\n";
//					}
//				}
//			}
//		}
//	}
//}

}  // namespace io
