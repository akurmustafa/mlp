
#include "src/matrices.h"
#include "src/mlp.h"

#include "utils/io.h"
#include "utils/rand_wrapper.h"
#include "utils/time_wrapper.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <random>
#include <utility>
#include <vector>

int main()
{
  time_wrapper::Timer timer{};

  std::vector<double> in_data{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  matrices::Matrix<double> in{ 1, 6, in_data };
  nn::Dropout<double> dropout{ 0.5 };
  auto out = dropout.forward(in);
  // auto rng = std::default_random_engine{};

  using data_type = double;
  double lr{ 0.001 };
  std::string loss_cat{ "categorical_cross_entropy" };
  int category_num{ 10 };
  std::string output_activation{ "softmax" };
  /*std::string loss_cat{ "cross_entropy" };
  int category_num{ 1 };
  std::string output_activation{ "sigmoid" };*/
  int input_dim{ 784 };
  std::vector<int> layers{ 64, 64, category_num };
  std::vector<std::string> activations{ "relu", "relu", output_activation };
  std::vector<data_type> dropout_probs{ 0.2, 0.2, 0.0 };
  nn::Model<data_type> nn_model{ input_dim, layers, activations, dropout_probs, loss_cat };
  nn::Loss<data_type, std::uint8_t> loss_obj{ loss_cat };
  int print_every{ 100 };
  int batch_size{ 8 };
  int n_epoch{ 10 };
  int gradient_check{ 0 };

  std::string dataset_folder_path{ "./mnist/" };
  auto train_data_vec = io::read_file<std::uint8_t>(dataset_folder_path + "train-images.idx3-ubyte", 784, 16);
  auto train_data_labels_vec = io::read_file<std::uint8_t>(dataset_folder_path + "train-labels.idx1-ubyte", 1, 8);
  std::size_t train_data_num{ train_data_labels_vec.size() };
  auto test_data_vec = io::read_file<std::uint8_t>(dataset_folder_path + "t10k-images.idx3-ubyte", 784, 16);
  auto test_data_labels_vec = io::read_file<std::uint8_t>(dataset_folder_path + "t10k-labels.idx1-ubyte", 1, 8);
  std::size_t test_data_num{ test_data_labels_vec.size() };
  matrices::Matrix<std::uint8_t> train_data{ train_data_num, 784, std::move(train_data_vec) };
  matrices::Matrix<std::uint8_t> test_data{ test_data_num, 784, std::move(test_data_vec) };

  auto train_data_norm = nn::util::normalize_images<data_type>(train_data);
  matrices::Matrix<std::uint8_t> train_data_labels{ train_data_num, 1, std::move(train_data_labels_vec) };
  auto test_data_norm = nn::util::normalize_images<data_type>(test_data);
  matrices::Matrix<std::uint8_t> test_data_labels{ test_data_num, 1, std::move(test_data_labels_vec) };

  auto train_data_labels_onehot = nn::util::onehot_encode(train_data_labels);
  auto test_data_labels_onehot = nn::util::onehot_encode(test_data_labels);

  double train_split_ratio{ 0.9 };
  nn::util::shuffle_in_place(train_data_norm, train_data_labels_onehot);
  
  int train_split = std::round(train_split_ratio * train_data_norm.get_row_num());
  auto training_labels = train_data_labels_onehot.get_row_btw(0, train_split);
  auto training_data = train_data_norm.get_row_btw(0, train_split);
  auto validation_labels = train_data_labels_onehot.get_row_btw(train_split, train_data_num);
  auto validation_data = train_data_norm.get_row_btw(train_split, train_data_num);
  std::size_t training_data_num{ training_data.get_row_num() };
  std::size_t val_data_num{ validation_data.get_row_num() };


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

  std::vector<double> training_losses{};
  std::vector<double> validation_losses{};
  for (int epoch = 0; epoch < n_epoch; ++epoch) {
    
    /*auto res_pair = nn::util::shuffle(training_data, training_labels);
    auto training_data = res_pair.first;
    auto training_labels = res_pair.second;*/

    nn::util::shuffle_in_place(training_data, training_labels);
    int count{ 0 };
    for (int data_idx = batch_size; data_idx < training_data_num; data_idx+= batch_size) {
      int row_start_idx = data_idx - batch_size;
      auto cur_batch = training_data.get_row_btw(row_start_idx, data_idx);
      auto cur_batch_labels = training_labels.get_row_btw(row_start_idx, data_idx);
      auto cur_batch_probs = nn_model.forward(cur_batch);
      nn_model.backward(cur_batch_labels, gradient_check);
      nn_model.update(lr);

      if (count % print_every == 0) {
        auto cur_loss_training = loss_obj(cur_batch_probs, cur_batch_labels);
        auto cur_loss_training_mean = nn::util::get_mean(cur_loss_training);
        std::string training_msg_str = "iter #" + std::to_string(epoch) + ", mean training loss: ";
        matrices::util::print_elements(cur_loss_training_mean.data, " ", training_msg_str);
        training_losses.push_back(cur_loss_training_mean.data[0]);

        std::vector<int> indices(val_data_num, int{ 0 });
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rand_wrapper::rng);
        // int sub_sample_data_num = 100 < val_data_num ? 100 : val_data_num;
        int sub_sample_data_num = std::min(100u, val_data_num);
        matrices::Matrix<data_type> val_data_sampled{ sub_sample_data_num, input_dim };
        matrices::Matrix<std::uint8_t> val_data_labels_sampled{ sub_sample_data_num, category_num };
        for (std::size_t i = 0; i < sub_sample_data_num; ++i) {
          val_data_sampled.set_row(i, validation_data.get_row(indices[i]));
          val_data_labels_sampled.set_row(i, validation_labels.get_row(indices[i]));
        }
        auto val_probs_sampled = nn_model.forward(val_data_sampled);
        auto cur_val_accuracy = nn::util::get_accuracy(val_probs_sampled, val_data_labels_sampled);
        std::cout << "validation accuracy: " << std::to_string(cur_val_accuracy) << "\n";
        auto cur_val_loss = loss_obj(val_probs_sampled, val_data_labels_sampled);
        auto cur_val_loss_mean = nn::util::get_mean(cur_val_loss);
        std::string msg_str = "iter #" + std::to_string(epoch) + ", mean val loss: ";
        matrices::util::print_elements(cur_val_loss_mean.data, " ", msg_str);
        validation_losses.push_back(cur_val_loss_mean.data[0]);
        std::cout << "\n";
      }
      ++count;
    }
  }

  auto test_data_probs = nn_model.forward(test_data_norm);
  auto test_data_accuracy = nn::util::get_accuracy(test_data_probs, test_data_labels_onehot);
  std::cout << "end of training, test accuracy: " << std::to_string(test_data_accuracy) << "\n";
  auto test_data_loss = loss_obj(test_data_probs, test_data_labels_onehot);
  auto test_data_loss_mean = nn::util::get_mean(test_data_loss);
  std::string msg_str = "enf of training mean loss: ";
  matrices::util::print_elements(test_data_loss_mean.data, " ", msg_str);

  std::cout << timer;

	return 0;
}
