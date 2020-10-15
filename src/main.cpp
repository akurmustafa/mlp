
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

  /*std::vector<double> in_data{ 1.0 };
  matrices::Matrix<double> in{ 1, 1, in_data };
  nn::Linear linear{ 1, 2 };
  auto out = linear.forward(in);
  matrices::util::print_elements(linear.m_weights.data);
  matrices::util::print_elements(linear.m_bias.data);
  matrices::util::print_elements(in.data);
  matrices::util::print_elements(out.data);*/

  /*double lr{ 0.01 };
  std::vector<int> layers{ 5, 1 };
  std::vector<std::string> activations{"sigmoid", "sigmoid"};
  int input_dim = 10;
  int batch_size = 1;
  nn::Model nn_model{ input_dim, layers, activations };
  std::vector<double> in_data2 = rand_wrapper::randn(batch_size *input_dim, 0, 1);
  matrices::Matrix<double> input_chunk{ batch_size, input_dim, in_data2};
  std::vector<int> label_data(batch_size * 1, int{ 0 });
  matrices::Matrix<int> labels{ batch_size, 1, label_data };*/
  
  
  /*auto preds = nn_model.forward(input_chunk, labels, 0, 0);
  auto cur_loss = nn_model.m_loss_fn(preds, labels);
  matrices::util::print_elements(cur_loss.data, " ", "cur loss: ");

  std::string opt_str = "output row num: " + std::to_string(preds.get_row_num());
  opt_str += " col num: " + std::to_string(preds.get_col_num());
  opt_str += "\ndata: ";
  matrices::util::print_elements(preds.data, " ", opt_str);

  auto res2 = nn_model.backward(1, 0);
  std::string opt_str2 = "output row num: " + std::to_string(res2.get_row_num());
  opt_str2 += " col num: " + std::to_string(res2.get_col_num());
  opt_str2 += "\ndata: ";
  matrices::util::print_elements(res2.data, " ", opt_str2);
  
  nn_model.update(lr);*/

  double lr{ 0.01 };
  std::vector<int> layers{ 10, 10, 1 };
  std::vector<std::string> activations{ "relu", "relu", "sigmoid" };
  int input_dim = 100;
  nn::Model nn_model{ input_dim, layers, activations };
  int print_every{ 25 };
  int batch_size = 16;
  int gradient_check{ 0 };
  int train_data_num = 1000;
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

  for (int epoch = 0; epoch < 10; ++epoch) {
    int count{ 0 };
    for (int data_idx = batch_size; data_idx < train_data_num; data_idx+= batch_size) {
      int row_start_idx = data_idx - batch_size;
      auto cur_input = train_data.get_row_btw(row_start_idx, data_idx);
      auto cur_label = train_labels.get_row_btw(row_start_idx, data_idx);
      auto probs = nn_model.forward(cur_input, cur_label);
      nn_model.backward(gradient_check);
      nn_model.update(lr);

      if (count % print_every == 0) {
        auto cur_loss = nn_model.m_loss_fn(probs, cur_label);
        auto cur_loss_mean = nn::util::get_mean(cur_loss);
        std::string msg_str = "iter #" + std::to_string(epoch) + ", mean loss: ";
        matrices::util::print_elements(cur_loss_mean.data, " ", msg_str);
        // auto cur_accuracy = nn::util::get_accuracy(probs, cur_label);
        auto all_probs = nn_model.forward(train_data, train_labels);
        auto cur_accuracy = nn::util::get_accuracy(all_probs, train_labels);
        std::cout << "accuracy: " << std::to_string(cur_accuracy) << "\n";
      }
      ++count;
    }
  }

  std::cout << timer;

	return 0;
}
