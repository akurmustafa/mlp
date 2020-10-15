#ifndef SRC_MLP_H
#define SRC_MLP_H

#include "src/matrices.h"
#include "utils/rand_wrapper.h"

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace nn {
	namespace util {

		template<typename T>
		double get_accuracy(matrices::Matrix<T> const& probs, matrices::Matrix<int> labels) {
			assert((probs.get_col_num() == 1 && labels.get_col_num() == 1) && "col nums should be same");
			assert(probs.get_row_num() == labels.get_row_num() && "data nums should be same");
			int tp{ 0 };
			for (std::size_t i = 0; i != probs.get_row_num(); ++i) {
				int cur_pred = probs.data[i] > 0.5 ? 1 : 0;
				if (cur_pred == labels.data[i]) {
					++tp;
				}
			}
			return 1.0 * tp / probs.get_row_num();
		}

		template<typename T>
		matrices::Matrix<T> get_mean(matrices::Matrix<T> const& in) {
			matrices::Matrix<T> res{ 1, in.get_col_num(), std::vector<T>(in.get_col_num(), T{0}) };
			for (std::size_t i = 0; i < in.get_col_num(); ++i) {
				auto cur_col = in.get_col(i);
				T sum_val{ 0 };
				sum_val = std::accumulate(cur_col.begin(), cur_col.end(), sum_val);
				T mean_val = sum_val / in.get_row_num();
				res.data[i] = mean_val;
			}
			return res;
		}

		template<typename T>
		T relu(T in) {
			T out;
			out = in > 0 ? in : 0;
			return out;
		}

		template<typename T>
		std::vector<T> relu(std::vector<T> const& in) {
			std::vector<T> out(in.size(), T{0});
			auto it_out = std::begin(out);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out) {
				*it_out = relu<T>(*it_in);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> relu(matrices::Matrix<T> const& in) {
			auto out_data = relu<T>(in.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T>
		T d_relu(T in) {
			T out;
			out = in > 0 ? 1 : 0;
			return out;
		}

		template<typename T>
		std::vector<T> d_relu(std::vector<T> const& in) {
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out) {
				*it_out = d_relu<T>(*it_in);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> d_relu(matrices::Matrix<T> const& in) {
			auto out_data = d_relu<T>(in.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T>
		T sigmoid(T in) {
			T out = 1.0 / (1.0 + std::exp(-in));
			return out;
		}

		template<typename T>
		std::vector<T> sigmoid(std::vector<T> const& in) {
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out) {
				*it_out = sigmoid<T>(*it_in);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> sigmoid(matrices::Matrix<T> const& in) {
			auto out_data = sigmoid<T>(in.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T>
		T d_sigmoid(T in) {
			auto tmp = sigmoid(in);
			T out = tmp * (1.0 - tmp);
			return out;
		}

		template<typename T>
		std::vector<T> d_sigmoid(std::vector<T> const& in) {
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out) {
				*it_out = d_sigmoid<T>(*it_in);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> d_sigmoid(matrices::Matrix<T> const& in) {
			auto out_data = d_sigmoid<T>(in.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T>
		T cross_entropy(T in, int label) {
			// static_assert(std::enable_if_t<std::declval(T == U), std::true_type> && "Arguments are not comparable\n");
			T loss{ 0 };
			if (label) {
				loss = -std::log(in);
			}
			else {
				loss = -std::log(1-in);
			}
			return loss;
		}

		template<typename T>
		std::vector<T> cross_entropy(std::vector<T> const& in, std::vector<int> const& labels) {
			assert(labels.size() == in.size() && "Dimensions do not match");
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			auto it_label = std::begin(labels);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out, ++it_label) {
				*it_out = cross_entropy<T>(*it_in, *it_label);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> cross_entropy(matrices::Matrix<T> const& in, matrices::Matrix<int> const& labels) {
			assert(in.get_row_num() == labels.get_row_num() && "Dimensions do not match");
			assert(in.get_col_num() == labels.get_col_num() && "Dimensions do not match");
			auto out_data = cross_entropy<T>(in.data, labels.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T>
		T d_cross_entropy(T in, int label) {
			// static_assert(std::enable_if_t<std::declval(T == U), std::true_type> && "Arguments are not comparable\n");
			T d_loss{ 0 };
			d_loss = (in - label) / ((1.0 - in) * in);
			return d_loss;
		}

		template<typename T>
		std::vector<T> d_cross_entropy(std::vector<T> const& in, std::vector<int> const& labels) {
			assert(labels.size() == in.size() && "Dimensions do not match");
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			auto it_label = std::begin(labels);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out, ++it_label) {
				*it_out = d_cross_entropy<T>(*it_in, *it_label);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> d_cross_entropy(matrices::Matrix<T> const& in, matrices::Matrix<int> const& labels) {
			assert(in.get_row_num() == labels.get_row_num() && "Dimensions do not match");
			assert(in.get_col_num() == labels.get_col_num() && "Dimensions do not match");
			auto out_data = d_cross_entropy<T>(in.data, labels.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

	} // namespace util

	struct Loss {
		;
	};

	struct Linear {
		matrices::Matrix<double> m_weights;
		matrices::Matrix<double> m_bias;
		std::string m_activation{};
		std::string m_init_method{};
		mutable matrices::Matrix<double> z_cache;
		mutable matrices::Matrix<double> activation_cache;
		matrices::Matrix<double> d_bias;
		matrices::Matrix<double> d_weights;

		Linear() {}
		Linear(int n_row, int n_col, std::string const& activation="relu", std::string const& init_method="gaussian") {
			m_init_method = init_method;
			int elem_num = n_row * n_col;
			std::vector<double> weight_data(elem_num, double{ 0 });
			std::vector<double> bias_data(n_col, double{ 0 });
			if (m_init_method.compare("gaussian") == 0) {
				double mean{ 0.0 };
				double var = 1.0 / n_row;
				weight_data = rand_wrapper::randn(elem_num, mean, var);
				bias_data = rand_wrapper::randn(n_col, mean, var);
			}
			m_weights = matrices::Matrix<double>{ n_row, n_col, weight_data };
			m_bias = matrices::Matrix<double>{ 1, n_col, bias_data };
			m_activation = activation;
		}

		matrices::Matrix<double> forward(matrices::Matrix<double> const& in, int print_on=0) const {
			z_cache = matrices::mult(in, m_weights) + m_bias;
			if (m_activation.compare("relu") == 0) {
				activation_cache = util::relu<double>(z_cache);
			}
			else if (m_activation.compare("sigmoid") == 0) {
				activation_cache = util::sigmoid<double>(z_cache);
			}
			else {
				assert(0 && "Not a valid activation name");
			}
			if (print_on) {
				std::string opt_str = "input row num: " + std::to_string(in.get_row_num());
				opt_str += " col num: " + std::to_string(in.get_col_num());
				opt_str += "\ndata: ";
				matrices::util::print_elements(in.data, " ", opt_str);

				std::string opt_str2 = "weights row num: " + std::to_string(m_weights.get_row_num());
				opt_str2 += " col num: " + std::to_string(m_weights.get_col_num());
				opt_str2 += "\ndata: ";
				matrices::util::print_elements(m_weights.data, " ", opt_str2);

				std::string opt_str3 = "bias row num: " + std::to_string(m_bias.get_row_num());
				opt_str3 += " col num: " + std::to_string(m_bias.get_col_num());
				opt_str3 += "\ndata: ";
				matrices::util::print_elements(m_bias.data, " ", opt_str3);

				std::string opt_str4 = "activation row num: " + std::to_string(activation_cache.get_row_num());
				opt_str4 += " col num: " + std::to_string(activation_cache.get_col_num());
				opt_str4 += "\ndata: ";
				matrices::util::print_elements(activation_cache.data, " ", opt_str4);
			}

			return activation_cache;
		}

		matrices::Matrix<double> backward(matrices::Matrix<double> const& in, matrices::Matrix<double>prev_activation, 
			matrices::Matrix<double> const& next_weigth, 
			int print_on=0) {
			matrices::Matrix<double> out;
			matrices::Matrix<double> d_activation;
			if (m_activation.compare("relu") == 0) {
				d_activation = util::d_relu<double>(z_cache);
			}
			else if (m_activation.compare("sigmoid") == 0) {
				d_activation = util::d_sigmoid<double>(z_cache);
			}
			else {
				assert(0 && "Not a valid activation name");
			}
			auto d_bias_batch = matrices::mult(in, next_weigth.transpose()) * d_activation;
			d_weights = matrices::mult(prev_activation.transpose(), d_bias_batch);
			int batch_size = d_bias_batch.get_row_num();
			d_bias = util::get_mean(d_bias_batch);
			d_weights = d_weights / batch_size;

			if (print_on) {
				std::string opt_str = "d_activation row num: " + std::to_string(d_activation.get_row_num());
				opt_str += " col num: " + std::to_string(d_activation.get_col_num());
				opt_str += "\ndata: ";
				matrices::util::print_elements(d_activation.data, " ", opt_str);

				std::string opt_str1 = "input row num: " + std::to_string(in.get_row_num());
				opt_str += " col num: " + std::to_string(in.get_col_num());
				opt_str += "\ndata: ";
				matrices::util::print_elements(in.data, " ", opt_str1);

				std::string opt_str2 = "d_bias row num: " + std::to_string(d_bias.get_row_num());
				opt_str2 += " col num: " + std::to_string(d_bias.get_col_num());
				opt_str2 += "\ndata: ";
				matrices::util::print_elements(d_bias.data, " ", opt_str2);

				std::string opt_str3 = "d_weights row num: " + std::to_string(d_weights.get_row_num());
				opt_str3 += " col num: " + std::to_string(d_weights.get_col_num());
				opt_str3 += "\ndata: ";
				matrices::util::print_elements(d_weights.data, " ", opt_str3);

			}

			return d_bias_batch;
		}
	};

	struct Model {
		std::vector<int> m_layers{};
		std::vector<std::string> m_activations{};
		std::vector<Linear> m_steps{};
		matrices::Matrix<double> m_input{};
		matrices::Matrix<int> m_labels{};
		matrices::Matrix<double> m_preds{};
		matrices::Matrix<double>(*m_loss_fn)(matrices::Matrix<double> const&, matrices::Matrix<int> const&);
		matrices::Matrix<double>(*m_d_loss_fn)(matrices::Matrix<double> const&, matrices::Matrix<int> const&);
		matrices::Matrix<double> m_loss{};
		std::string m_loss_cat{};
		Model(int input_dim, std::vector<int> layers, std::vector<std::string> activations, std::string loss_cat="cross_entropy") {
			assert(layers.size() == activations.size() && "dimensions doesnt match");
			m_layers.resize(layers.size()+1);
			m_layers[0] = input_dim;
			std::copy(layers.begin(), layers.end(), m_layers.begin() + 1);
			m_activations = activations;
			m_steps.resize(m_activations.size());
			for (std::size_t i = 0; i < m_activations.size(); ++i) {
				m_steps[i] = Linear(m_layers[i], m_layers[i + 1], m_activations[i]);
			}
			m_loss_cat = loss_cat;
			if (m_loss_cat.compare("cross_entropy") == 0) {
				m_loss_fn = &util::cross_entropy<double>;
				m_d_loss_fn = &util::d_cross_entropy<double>;
			}
			else {
				assert(0 && "No valid loss function");
			}
		}
		matrices::Matrix<double> forward(matrices::Matrix<double> in, matrices::Matrix<int> labels, 
			int start_step = 0, int print_on = 0) {
			assert((start_step >= 0 && start_step < m_steps.size()) && "start step is not valid");
			if (start_step == 0) {
				m_input = in;
				m_labels = labels;
			}
			for (std::size_t i = start_step; i!=m_steps.size(); ++i) {
				in = m_steps[i].forward(in, print_on);
			}
			m_preds = in;
			auto loss = m_loss_fn(in, labels);
			m_loss = loss;
			return m_preds;
		};

		matrices::Matrix<double> backward( int gradient_check = 0, int print_on = 0) {
			std::vector<double> next_weigth_data(m_preds.get_col_num(), double{ 1 });
			matrices::Matrix<double> next_weigth{ 1, m_preds.get_col_num(), next_weigth_data };
			matrices::Matrix<double> d_err = m_d_loss_fn(m_preds, m_labels);
			for (int i = m_steps.size()-1; i != -1; --i) {
				matrices::Matrix<double> prev_activation = i == 0 ? m_input : m_steps[i - 1].activation_cache;
				d_err = m_steps[i].backward(d_err, prev_activation, next_weigth, print_on);
				next_weigth = m_steps[i].m_weights;
				if (gradient_check) {
					double epsilon = { 0.001 };
					for (std::size_t j = 0; j < m_steps[i].m_weights.data.size(); ++j) {
						auto temp = m_steps[i].m_weights.data[j];
						
						m_steps[i].m_weights.data[j] = temp + epsilon;
						auto m_preds1 = forward(prev_activation, m_labels, i, 0);
						auto loss1 = m_loss_fn(m_preds1, m_labels);

						m_steps[i].m_weights.data[j] = temp - epsilon;
						auto m_preds2 = forward(prev_activation, m_labels, i, 0);
						auto loss2 = m_loss_fn(m_preds2, m_labels);

					  // TODO: calc numeric gradient
						auto mean_loss1 = util::get_mean(loss1);
						auto mean_loss2 = util::get_mean(loss2);
						auto cur_grad = (mean_loss1.data[0] - mean_loss2.data[0]) / (2 * epsilon);

						m_steps[i].m_weights.data[j] = temp;
						auto diff = std::abs(cur_grad - m_steps[i].d_weights.data[j]);
						std::cout << "diff btw gradients " << std::to_string(diff) << "\n";
						assert(diff < 0.001 && "numerical gradient doesnt match");
					}
					for (std::size_t j = 0; j < m_steps[i].m_bias.data.size(); ++j) {
						auto temp = m_steps[i].m_bias.data[j];

						m_steps[i].m_bias.data[j] = temp + epsilon;
						auto m_preds1 = forward(prev_activation, m_labels, i, 0);
						auto loss1 = m_loss_fn(m_preds1, m_labels);

						m_steps[i].m_bias.data[j] = temp - epsilon;
						auto m_preds2 = forward(prev_activation, m_labels, i, 0);
						auto loss2 = m_loss_fn(m_preds2, m_labels);

						// TODO: calc numeric gradient
						auto mean_loss1 = util::get_mean(loss1);
						auto mean_loss2 = util::get_mean(loss2);
						auto cur_grad = (mean_loss1.data[0] - mean_loss2.data[0]) / (2 * epsilon);

						m_steps[i].m_bias.data[j] = temp;
						auto diff = std::abs(cur_grad - m_steps[i].d_bias.data[j]);
						std::cout << "diff btw gradients " << std::to_string(diff) << "\n";
						assert(diff < 1e-7 && "numerical gradient doesnt match");
					}
				}
			}
			return d_err;
		};

		void update(double learning_rate = 0.01) {
			for (std::size_t i = 0; i < m_steps.size(); ++i) {
				m_steps[i].m_weights -= m_steps[i].d_weights * learning_rate;
				m_steps[i].m_bias -= m_steps[i].d_bias * learning_rate;
			}
		}

	};
} // namespace nn

#endif // end of SRC_MLP_H

