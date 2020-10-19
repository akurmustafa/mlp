
#include "src/matrices.h"
#include "src/mlp.h"

#include "utils/io.h"
#include "utils/rand_wrapper.h"
#include "utils/time_wrapper.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <random>
#include <vector>

int main()
{
  time_wrapper::Timer timer{};
  auto rng = std::default_random_engine{};

  std::string dataset_folder_path{ "./mnist/" };
  auto train_data_vec = io::read_file<std::uint8_t>(dataset_folder_path + "train-images.idx3-ubyte", 784, 16);
  auto train_data_labels_vec = io::read_file<std::uint8_t>(dataset_folder_path + "train-labels.idx1-ubyte", 1, 8);
  std::size_t train_data_num{ train_data_labels_vec.size() };
  auto test_data_vec = io::read_file<std::uint8_t>(dataset_folder_path + "t10k-images.idx3-ubyte", 784, 16);
  auto test_data_labels_vec = io::read_file<std::uint8_t>(dataset_folder_path + "t10k-labels.idx1-ubyte", 1, 8);
  std::size_t test_data_num{ test_data_labels_vec.size() };
  matrices::Matrix<std::uint8_t> train_data{ train_data_num, 784, train_data_vec };
  matrices::Matrix<std::uint8_t> test_data{ test_data_num, 784, test_data_vec };
  auto train_data_norm = nn::util::normalize_images<double>(train_data);
  matrices::Matrix<std::uint8_t> train_data_labels{ train_data_num, 1, train_data_labels_vec };
  auto test_data_norm = nn::util::normalize_images<double>(test_data);
  matrices::Matrix<std::uint8_t> test_data_labels{ test_data_num, 1, test_data_labels_vec };

  auto train_data_labels_onehot = nn::util::onehot_encode(train_data_labels);
  auto test_data_labels_onehot = nn::util::onehot_encode(test_data_labels);


  double lr{ 0.001 };
  std::string loss_cat{ "categorical_cross_entropy" };
  int category_num{ 10 };
  std::string output_activation{ "softmax" };
  /*std::string loss_cat{ "cross_entropy" };
  int category_num{ 1 };
  std::string output_activation{ "sigmoid" };*/
  int input_dim{ 784 };
  std::vector<int> layers{ 64, 32, category_num };
  std::vector<std::string> activations{ "relu", "relu", output_activation };
  nn::Model nn_model{ input_dim, layers, activations, loss_cat };
  nn::Loss<double, std::uint8_t> loss_obj{ loss_cat };
  int print_every{ 25 };
  int batch_size{ 16 };
  int gradient_check{ 0 };


  /*int train_data_num{ 10000 };
  int test_data_num{ 1000 };
  std::vector<double> train_data_vec(train_data_num * input_dim, double{ 0 });
  matrices::Matrix<double> train_data_norm{ train_data_num, input_dim, train_data_vec };

  std::vector<std::uint8_t> train_labels_vec(train_data_num, int{ 0 });
  matrices::Matrix<std::uint8_t> train_data_labels{ train_data_num, 1, train_labels_vec };

  std::vector<double> test_data_vec(test_data_num * input_dim, double{ 0 });
  matrices::Matrix<double> test_data_norm{ test_data_num, input_dim, test_data_vec };

  std::vector<std::uint8_t> test_labels_vec(test_data_num, int{ 0 });
  matrices::Matrix<std::uint8_t> test_data_labels{ test_data_num, 1, test_labels_vec };
  for (int i = 0; i < train_data_num; ++i) {
    double mean = rand_wrapper::randn(0.0, 1.0);
    auto cur_train_data = rand_wrapper::randn(input_dim, mean, 1);
    double total = std::accumulate(cur_train_data.begin(), cur_train_data.end(), 0.0);
    int cur_label_data = total > 0 ? 1 : 0;
    std::vector<std::uint8_t> cur_label(1, cur_label_data);
    train_data_norm.set_row(i, cur_train_data);
    train_data_labels.set_row(i, cur_label);
  }
  for (int i = 0; i < test_data_num; ++i) {
    double mean = rand_wrapper::randn(0.0, 1.0);
    auto cur_test_data = rand_wrapper::randn(input_dim, mean, 1);
    double total = std::accumulate(cur_test_data.begin(), cur_test_data.end(), 0.0);
    int cur_label_data = total > 0 ? 1 : 0;
    std::vector<std::uint8_t> cur_label(1, cur_label_data);
    test_data_norm.set_row(i, cur_test_data);
    test_data_labels.set_row(i, cur_label);
  }
  matrices::Matrix<std::uint8_t> train_data_labels_onehot;
  matrices::Matrix<std::uint8_t> test_data_labels_onehot;
  if (category_num == 1) {
    train_data_labels_onehot = train_data_labels;
    test_data_labels_onehot = test_data_labels;
  }
  else if(category_num == 2) {
    train_data_labels_onehot = nn::util::onehot_encode(train_data_labels);
    test_data_labels_onehot = nn::util::onehot_encode(test_data_labels);
  }
  else {
    assert(0 && "train label generation doesnt support more than 2 category");
  }*/


  for (int epoch = 0; epoch < 10; ++epoch) {
    int count{ 0 };
    for (int data_idx = batch_size; data_idx < train_data_num; data_idx+= batch_size) {
      int row_start_idx = data_idx - batch_size;
      auto cur_input = train_data_norm.get_row_btw(row_start_idx, data_idx);
      auto probs = nn_model.forward(cur_input);
      auto cur_label = train_data_labels_onehot.get_row_btw(row_start_idx, data_idx);
      nn_model.backward(cur_label, gradient_check);
      nn_model.update(lr);

      if (count % print_every == 0) {
        // auto cur_accuracy = nn::util::get_accuracy(probs, cur_label);
        std::vector<int> indices(test_data_num, int{ 0 });
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        int sub_sample_data_num = 100 < test_data_num ? 100 : test_data_num;
        matrices::Matrix<double> test_data_norm_sampled{ sub_sample_data_num, input_dim };
        matrices::Matrix<std::uint8_t> test_data_norm_labels_sampled{ sub_sample_data_num, category_num };
        for (std::size_t i = 0; i < sub_sample_data_num; ++i) {
          test_data_norm_sampled.set_row(i, test_data_norm.get_row(indices[i]));
          test_data_norm_labels_sampled.set_row(i, test_data_labels_onehot.get_row(indices[i]));
        }
        auto all_probs_sampled = nn_model.forward(test_data_norm_sampled);
        auto cur_accuracy = nn::util::get_accuracy(all_probs_sampled, test_data_norm_labels_sampled);
        std::cout << "accuracy: " << std::to_string(cur_accuracy) << "\n";
        auto cur_loss = loss_obj(all_probs_sampled, test_data_norm_labels_sampled);
        auto cur_loss_mean = nn::util::get_mean(cur_loss);
        std::string msg_str = "iter #" + std::to_string(epoch) + ", mean loss: ";
        matrices::util::print_elements(cur_loss_mean.data, " ", msg_str);
        std::cout << "\n";
      }
      ++count;
    }
  }

  std::cout << timer;

	return 0;
}
