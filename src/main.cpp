
#include "src/matrices.h"
#include "src/mlp.h"
#include "utils/rand_wrapper.h"
#include "utils/time_wrapper.h"

#include <iostream>
#include <string>
#include <vector>

int main()
{
  time_wrapper::Timer timer{};

  double lr{ 0.01 };
  std::string loss_cat{ "categorical_cross_entropy" };
  int category_num{ 2 };
  std::string output_activation{ "softmax" };
  std::vector<int> layers{ 10, 5, category_num };
  std::vector<std::string> activations{ "relu", "tanh", output_activation };
  int input_dim{ 100 };
  nn::Model nn_model{ input_dim, layers, activations, loss_cat };
  nn::Loss<double> loss_obj{ loss_cat };
  int print_every{ 25 };
  int batch_size{ 16 };
  int gradient_check{ 1 };
  int train_data_num{ 1000 };
  std::vector<double> train_data_vec(train_data_num * input_dim, double{ 0 });
  std::vector<int> train_labels_vec(train_data_num, int{ 0 });
  matrices::Matrix<double> train_data{ train_data_num, input_dim, train_data_vec };
  matrices::Matrix<int> train_labels{ train_data_num, 1, train_labels_vec };
  for (int i = 0; i < train_data_num; ++i) {
    double mean = rand_wrapper::randn(0.0, 1.0);
    auto cur_train_data = rand_wrapper::randn(input_dim, mean, 1);
    double total = std::accumulate(cur_train_data.begin(), cur_train_data.end(), 0.0);
    int cur_label_data = total > 0 ? 1 : 0;
    std::vector<int> cur_label(1, cur_label_data);
    train_data.set_row(i, cur_train_data);
    train_labels.set_row(i, cur_label);
  }
  matrices::Matrix<int> train_labels_onehot;
  if (category_num == 1) {
    train_labels_onehot = train_labels;
  }
  else if(category_num == 2) {
    train_labels_onehot = nn::util::onehot_encode(train_labels);
  }
  else {
    assert(0 && "tran label generation doesnt support more than 2 category");
  }
  for (int epoch = 0; epoch < 10; ++epoch) {
    int count{ 0 };
    for (int data_idx = batch_size; data_idx < train_data_num; data_idx+= batch_size) {
      int row_start_idx = data_idx - batch_size;
      auto cur_input = train_data.get_row_btw(row_start_idx, data_idx);
      auto probs = nn_model.forward(cur_input);
      auto cur_label = train_labels_onehot.get_row_btw(row_start_idx, data_idx);
      nn_model.backward(cur_label, gradient_check);
      nn_model.update(lr);

      if (count % print_every == 0) {
        auto cur_loss = loss_obj(probs, cur_label);
        auto cur_loss_mean = nn::util::get_mean(cur_loss);
        std::string msg_str = "iter #" + std::to_string(epoch) + ", mean loss: ";
        matrices::util::print_elements(cur_loss_mean.data, " ", msg_str);
        // auto cur_accuracy = nn::util::get_accuracy(probs, cur_label);
        auto all_probs = nn_model.forward(train_data);
        auto cur_accuracy = nn::util::get_accuracy(all_probs, train_labels_onehot);
        std::cout << "accuracy: " << std::to_string(cur_accuracy) << "\n";
      }
      ++count;
    }
  }

  std::cout << timer;

	return 0;
}
