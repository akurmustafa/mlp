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
		
		template<typename T, typename D>
		void shuffle_in_place(matrices::Matrix<T> & in, matrices::Matrix<D> & labels) {
			static_assert(std::is_integral_v<D>, "rhs type must be integral");
			static_assert(std::is_floating_point_v<T>, "lhs must have floating point type");
			assert(in.get_row_num() == labels.get_row_num() && "row nums should be same");
			matrices::Matrix<T> res{ in.get_row_num(), in.get_col_num() };
			matrices::Matrix<D> res_label{ labels.get_row_num(), labels.get_col_num() };
			std::vector<int> indices(in.get_row_num(), int{ 0 });
			std::iota(indices.begin(), indices.end(), 0);
			std::shuffle(indices.begin(), indices.end(), rand_wrapper::rng);
			for (std::size_t i = 0; i < res.get_row_num(); ++i) {
				res.set_row(i, in.get_row(indices[i]));
				res_label.set_row(i, labels.get_row(indices[i]));
			}
			in = res;
			labels = res_label;
		}

		template<typename T, typename D>
		std::pair<matrices::Matrix<T>, matrices::Matrix<D>> shuffle(matrices::Matrix<T> const& in, matrices::Matrix<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs type must be integral");
			static_assert(std::is_floating_point_v<T>, "lhs must have floating point type");
			assert(in.get_row_num() == labels.get_row_num() && "row nums should be same");
			matrices::Matrix<T> res{ in.get_row_num(), in.get_col_num() };
			matrices::Matrix<D> res_label{ labels.get_row_num(), labels.get_col_num()};
			std::vector<int> indices(in.get_row_num(), int{ 0 });
			std::iota(indices.begin(), indices.end(), 0);
			std::shuffle(indices.begin(), indices.end(), rand_wrapper::rng);
			for (std::size_t i = 0; i < res.get_row_num(); ++i) {
				res.set_row(i, in.get_row(indices[i]));
				res_label.set_row(i, labels.get_row(indices[i]));
			}
			return std::pair{ res, res_label };
		}

		template<typename T, typename D>
		matrices::Matrix<T> normalize_images(matrices::Matrix<D> const& in) {
			static_assert(std::is_integral_v<D>, "type of rhs is not integral");
			matrices::Matrix<T>res{ in.get_row_num(), in.get_col_num() };
			for (std::size_t i = 0; i < in.data.size(); ++i) {
				res.data[i] = in.data[i] / 255.0;
			}
			return res;
		}

		template<typename T>
		matrices::Matrix<T> onehot_encode(matrices::Matrix<T> const& labels) {
			static_assert(std::is_integral_v<T>, "Type is not integral");
			assert(labels.get_col_num() == 1 && "labels should be unique for onehot encoding");
			auto max_elem_it = std::max_element(labels.data.cbegin(), labels.data.cend());
			int diff_label_num = *max_elem_it + 1;
			matrices::Matrix<T> res{ labels.get_row_num(), diff_label_num };
			for (std::size_t i = 0; i < res.get_row_num(); ++i) {
				std::vector<T> cur_row(diff_label_num, int{ 0 });
				cur_row[labels.data[i]] = 1;
				res.set_row(i, cur_row);
			}
			return res;
		}

		namespace detail {
			// expects binary label
			template<typename T, typename D>
			double get_accuracy_impl(matrices::Matrix<T> const& probs, matrices::Matrix<D> labels, std::true_type) {
				int tp{ 0 };
				for (std::size_t i = 0; i != probs.get_row_num(); ++i) {
					D cur_pred = probs.data[i] > 0.5 ? 1 : 0;
					if (cur_pred == labels.data[i]) {
						++tp;
					}
				}
				return 1.0 * tp / probs.get_row_num();
			}

			// expects onehot encoded vector as label
			template<typename T, typename D>
			double get_accuracy_impl(matrices::Matrix<T> const& probs, matrices::Matrix<D> labels, std::false_type) {
				int tp{ 0 };
				for (std::size_t i = 0; i != probs.get_row_num(); ++i) {
					std::size_t offset = i * probs.get_col_num();
					std::size_t offset_end = (i + 1) * probs.get_col_num();
					auto cur_pred = std::distance(probs.data.cbegin()+ offset,
						std::max_element(probs.data.cbegin()+offset, probs.data.cbegin()+ offset_end));
					if (labels.data[i * labels.get_col_num() + cur_pred] == 1) {
						++tp;
					}
				}
				return 1.0 * tp / probs.get_row_num();
			}
		} // namespace detail

		template<typename T, typename D>
		double get_accuracy(matrices::Matrix<T> const& probs, matrices::Matrix<D> labels) {
			static_assert(std::is_integral_v<D>, "rhs type is not integral");
			assert((probs.get_col_num() && labels.get_col_num()) && "col nums should be same");
			assert(probs.get_row_num() == labels.get_row_num() && "data nums should be same");
			std::true_type t;
			std::false_type f;
			if (probs.get_col_num() == 1)
				return detail::template get_accuracy_impl<T>(probs, labels, t);
			else
				return detail::template get_accuracy_impl<T>(probs, labels, f);
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


		////////////// activations ///////////////

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
		T tanh(T in) {
			T tmp1 = std::exp(in);
			T tmp2 = std::exp(-in);
			T out = (tmp1 - tmp2) / (tmp1 + tmp2);
			return out;
		}

		template<typename T>
		std::vector<T> tanh(std::vector<T> const& in) {
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out) {
				*it_out = tanh<T>(*it_in);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> tanh(matrices::Matrix<T> const& in) {
			auto out_data = tanh<T>(in.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T>
		T d_tanh(T in) {
			T tmp1 = tanh(in);
			T out = (1.0 - tmp1 * tmp1);
			return out;
		}

		template<typename T>
		std::vector<T> d_tanh(std::vector<T> const& in) {
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out) {
				*it_out = d_tanh<T>(*it_in);
			}
			return out;
		}

		template <typename T>
		matrices::Matrix<T> d_tanh(matrices::Matrix<T> const& in) {
			auto out_data = d_tanh<T>(in.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T>
		std::vector<T> softmax(std::vector<T> const& in) {
			std::vector<T> res(in.size(), T{ 0 });
			res = in;
			auto max_elem_it = std::max_element(res.begin(), res.end());
			auto max_elem = *max_elem_it;
			// subtract max num from each one for numerical stability
			std::transform(res.begin(), res.end(),
				res.begin(),
				[max_elem](auto lhs) {return lhs - max_elem; });
			// take exponent of each elem
			std::transform(res.begin(), res.end(),
				res.begin(),
				[](auto elem) {return std::exp(elem); });
			auto total = std::accumulate(res.begin(), res.end(), T{ 0 });
			std::transform(res.begin(), res.end(),
				res.begin(),
				[total](auto elem) {return elem / total; });

			return res;
		}

		template<typename T>
		matrices::Matrix<T> softmax(matrices::Matrix<T> const& in) {
			matrices::Matrix<T> res{ in.get_row_num(), in.get_col_num(), std::vector<T>(in.get_row_num() * in.get_col_num(), T{0}) };
			for(std::size_t row_idx =0;row_idx<in.get_row_num();++row_idx){
				auto cur_row = in.get_row(row_idx);
				auto soft_max_res = softmax<T>(cur_row);
				res.set_row(row_idx, soft_max_res);
			}
			return res;
		}

		template<typename T>
		matrices::Matrix<T> d_softmax(std::vector<T> const& in) {
			matrices::Matrix<T> res{ in.size(), in.size(), std::vector<T>(in.size() * in.size(), T{0}) };
			for (std::size_t i = 0; i < res.get_row_num(); ++i) {
				std::vector<T> cur_row(res.get_col_num(), T{ 0 });
				for (std::size_t j = 0; j < res.get_col_num(); ++j) {
					if (i == j) {
						cur_row[j] = in[j] * (1.0 - in[j]);
					}
					else {
						cur_row[j] = -in[i] * in[j];
					}
				}
				res.set_row(i, cur_row);
			}
			return res;
		}

		template<typename T>
		matrices::Matrix<T> d_softmax(matrices::Matrix<T> const& in) {
			matrices::Matrix<T> res{ in.get_row_num() * in.get_col_num(), in.get_col_num() };
			for (std::size_t i = 0; i < in.get_row_num(); ++i) {
				auto cur_row = in.get_row(i);
				auto cur_d_softmax = d_softmax<T>(cur_row);
				for (std::size_t j = i; j < i + cur_d_softmax.get_row_num(); ++j) {
					res.set_row(j, cur_row);
				}
			}
			return res;
		}
		////////////// activations ///////////////

		////////////// loss functions ////////////
		template<typename T, typename D>
		T cross_entropy(T in, D label) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
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

		template<typename T, typename D>
		std::vector<T> cross_entropy(std::vector<T> const& in, std::vector<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			assert(labels.size() == in.size() && "Dimensions do not match");
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			auto it_label = std::begin(labels);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out, ++it_label) {
				*it_out = cross_entropy<T>(*it_in, *it_label);
			}
			return out;
		}

		template <typename T, typename D>
		matrices::Matrix<T> cross_entropy(matrices::Matrix<T> const& in, matrices::Matrix<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			assert(in.get_row_num() == labels.get_row_num() && "Dimensions do not match");
			assert(in.get_col_num() == labels.get_col_num() && "Dimensions do not match");
			auto out_data = cross_entropy<T>(in.data, labels.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T, typename D>
		T d_cross_entropy(T in, D label) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			// static_assert(std::enable_if_t<std::declval(T == U), std::true_type> && "Arguments are not comparable\n");
			T d_loss{ 0 };
			d_loss = (in - label) / ((1.0 - in) * in);
			return d_loss;
		}

		template<typename T, typename D>
		std::vector<T> d_cross_entropy(std::vector<T> const& in, std::vector<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			assert(labels.size() == in.size() && "Dimensions do not match");
			std::vector<T> out(in.size(), T{ 0 });
			auto it_out = std::begin(out);
			auto it_label = std::begin(labels);
			for (auto it_in = std::cbegin(in); it_in != std::cend(in); ++it_in, ++it_out, ++it_label) {
				*it_out = d_cross_entropy<T>(*it_in, *it_label);
			}
			return out;
		}

		template <typename T, typename D>
		matrices::Matrix<T> d_cross_entropy(matrices::Matrix<T> const& in, matrices::Matrix<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			assert(in.get_row_num() == labels.get_row_num() && "Dimensions do not match");
			assert(in.get_col_num() == labels.get_col_num() && "Dimensions do not match");
			auto out_data = d_cross_entropy<T>(in.data, labels.data);
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num(), out_data };
			return out;
		}

		template<typename T, typename D>
		std::vector<T> categorical_cross_entropy(std::vector<T> const& in, std::vector<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			assert(labels.size() == in.size() && "Dimensions do not match");
			std::vector<T> out(in.size(), T{ 0 });
			std::transform(in.cbegin(), in.cend(),
				labels.cbegin(),
				out.begin(),
				[](auto lhs, auto rhs) {return rhs * std::log(lhs); });
			T loss = -1.0*std::accumulate(out.begin(), out.end(), T{ 0 });
			return std::vector<T>{loss};
		}

		template <typename T, typename D>
		matrices::Matrix<T> categorical_cross_entropy(matrices::Matrix<T> const& in, matrices::Matrix<D> const& labels) {
			assert(in.get_row_num() == labels.get_row_num() && "Dimensions do not match");
			assert(in.get_col_num() == labels.get_col_num() && "Dimensions do not match");
			matrices::Matrix<T> out{ in.get_row_num(), 1 };
			for (std::size_t i = 0; i < in.get_row_num(); ++i) {
				auto cur_row_probs = in.get_row(i);
				auto cur_row_labels = labels.get_row(i);
				auto cur_loss = categorical_cross_entropy<T>(cur_row_probs, cur_row_labels);
				out.set_row(i, cur_loss);
			}
			return out;
		}

		template<typename T, typename D>
		std::vector<T> d_categorical_cross_entropy_with_softmax(std::vector<T> const& in, std::vector<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			assert(labels.size() == in.size() && "Dimensions do not match");
			std::vector<T> d_in(in.size(), T{ 0 });
			std::transform(in.cbegin(), in.cend(),
				labels.cbegin(),
				d_in.begin(),
				[](auto lhs, auto rhs) {return lhs - rhs; });
			return d_in;
		}

		template <typename T, typename D>
		matrices::Matrix<T> d_categorical_cross_entropy_with_softmax(matrices::Matrix<T> const& in, matrices::Matrix<D> const& labels) {
			static_assert(std::is_integral_v<D>, "rhs must be integral type");
			assert(in.get_row_num() == labels.get_row_num() && "Dimensions do not match");
			assert(in.get_col_num() == labels.get_col_num() && "Dimensions do not match");
			matrices::Matrix<T> out{ in.get_row_num(), in.get_col_num() };
			for (std::size_t i = 0; i < in.get_row_num(); ++i) {
				auto cur_row_probs = in.get_row(i);
				auto cur_row_labels = labels.get_row(i);
				auto cur_loss = d_categorical_cross_entropy_with_softmax<T>(cur_row_probs, cur_row_labels);
				out.set_row(i, cur_loss);
			}
			return out;
		}
		////////////// loss functions ////////////

	} // namespace util

	struct Layer {
		Layer() {};

	};

	template<typename T>
	struct Dropout {
		static_assert(std::is_floating_point_v<T>, "type of dropout must be floating point type");
		double m_dropout{};
		double m_ratio{};
		matrices::Matrix<T> m_activation_cache{};
		matrices::Matrix<double> m_dropout_cache;
		Dropout(double prob = 0.5) {
			m_dropout = prob;
			m_ratio = 1.0 / (1.0 - m_dropout);
		}

		matrices::Matrix<T> forward(matrices::Matrix<T> const& in, bool inference=false, bool init=true) {
			matrices::Matrix<T> out{};
			if (inference) {
				out = in;
			}
			else {
				if (init) {
					m_dropout_cache = matrices::Matrix<T>{ in.get_row_num(), in.get_col_num() };
					for (std::size_t i = 0; i < m_dropout_cache.data.size(); ++i) {
						if (rand_wrapper::rand(0.0, 1.0) > m_dropout) {
							m_dropout_cache.data[i] = 1;
						}
					}
				}
				out = in * m_dropout_cache * m_ratio;
				m_activation_cache = out;
			}
			return out;
		}

		matrices::Matrix<T> backward(matrices::Matrix<T> const& in) {
			return in * m_dropout_cache * m_ratio;
		}
	};

	template<typename T, typename D>
	struct Loss {
		static_assert(std::is_floating_point_v<T> && "lhs must be floating point");
		static_assert(std::is_integral_v<D>, "rhs must be integral type");
		std::string m_loss_cat;
		matrices::Matrix<T>(*m_loss_fn)(matrices::Matrix<T> const&, matrices::Matrix<D> const&);
		matrices::Matrix<T>(*m_d_loss_fn)(matrices::Matrix<T> const&, matrices::Matrix<D> const&);
		Loss(std::string const& loss_cat = "cross_entropy") {
			m_loss_cat = loss_cat;
			if (m_loss_cat.compare("cross_entropy") == 0) {
				m_loss_fn = &util::cross_entropy<T>;
				m_d_loss_fn = &util::d_cross_entropy<T>;
			}
			else if (m_loss_cat.compare("categorical_cross_entropy") == 0) {
				m_loss_fn = &util::categorical_cross_entropy<T>;
				m_d_loss_fn = &util::d_categorical_cross_entropy_with_softmax<T>;
			}
			else {
				assert(0 && "Loss function name is not recognized");
			}
		}
		matrices::Matrix<T> operator()(matrices::Matrix<T> const& probs, matrices::Matrix<D> const& labels) {
			return m_loss_fn(probs, labels);
		}
		matrices::Matrix<T> backward(matrices::Matrix<T> const& probs, matrices::Matrix<D> const& labels) {
			return m_d_loss_fn(probs, labels);
		}
	};

	struct Optimizer {
		std::string m_optimizer_cat;
		Optimizer(std::string const& optimizer_cat = "sgd") {
			m_optimizer_cat = optimizer_cat;
			if (m_optimizer_cat.compare("sgd")) {
				;
			}
			else {
				assert(0, "optimization name is not recognized");
			}
		}
	};

	struct Linear {
		matrices::Matrix<double> m_weights;
		matrices::Matrix<double> m_bias;
		std::string m_activation{};
		std::string m_init_method{};
		mutable matrices::Matrix<double> m_z_cache;
		mutable matrices::Matrix<double> m_activation_cache;
		matrices::Matrix<double> m_d_bias;
		matrices::Matrix<double> m_d_weights;

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
			else if (m_init_method.compare("uniform") == 0) {
				double high = std::sqrt(1.0 / n_row);
				double low = -1.0 * high;
				weight_data = rand_wrapper::rand(elem_num, low, high);
				bias_data = rand_wrapper::rand(n_col, low, high);
			}
			else {
				assert(0 && "init method is not defined");
			}
			m_weights = matrices::Matrix<double>{ n_row, n_col, weight_data };
			m_bias = matrices::Matrix<double>{ 1, n_col, bias_data };
			m_activation = activation;
		}

		matrices::Matrix<double> get_d_activation() const {
			matrices::Matrix<double> d_activation;
			if (m_activation.compare("relu") == 0) {
				d_activation = util::d_relu<double>(m_z_cache);
			}
			else if (m_activation.compare("sigmoid") == 0) {
				d_activation = util::d_sigmoid<double>(m_z_cache);
			}
			else if (m_activation.compare("tanh") == 0) {
				d_activation = util::d_tanh<double>(m_z_cache);
			}
			else if (m_activation.compare("softmax") == 0) {
				d_activation = util::d_softmax<double>(m_z_cache);
			}
			else {
				assert(0 && "Not a valid activation name");
			}
			return d_activation;
		}

		template<typename T>
		matrices::Matrix<T> forward(matrices::Matrix<T> const& in, int print_on=0) const {
			static_assert(std::is_floating_point_v<T>, "Type of input must be floating point type");
			m_z_cache = matrices::mult(in, m_weights) + m_bias;
			if (m_activation.compare("relu") == 0) {
				m_activation_cache = util::relu<T>(m_z_cache);
			}
			else if (m_activation.compare("sigmoid") == 0) {
				m_activation_cache = util::sigmoid<T>(m_z_cache);
			}
			else if (m_activation.compare("tanh") == 0) {
				m_activation_cache = util::tanh<T>(m_z_cache);
			}
			else if (m_activation.compare("softmax") == 0) {
				m_activation_cache = util::softmax<T>(m_z_cache);
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

				std::string opt_str4 = "activation row num: " + std::to_string(m_activation_cache.get_row_num());
				opt_str4 += " col num: " + std::to_string(m_activation_cache.get_col_num());
				opt_str4 += "\ndata: ";
				matrices::util::print_elements(m_activation_cache.data, " ", opt_str4);
			}

			return m_activation_cache;
		}

		matrices::Matrix<double> backward(matrices::Matrix<double> const& in, matrices::Matrix<double>prev_activation, 
			matrices::Matrix<double> const& next_weigth, 
			int print_on=0) {
			matrices::Matrix<double> out;
			matrices::Matrix<double> d_activation;
			if (m_activation.compare("relu") == 0) {
				d_activation = util::d_relu<double>(m_z_cache);
			}
			else if (m_activation.compare("sigmoid") == 0) {
				d_activation = util::d_sigmoid<double>(m_z_cache);
			}
			else if (m_activation.compare("tanh") == 0) {
				d_activation = util::d_tanh<double>(m_z_cache);
			}
			else if (m_activation.compare("softmax") == 0) {
				d_activation = util::d_softmax<double>(m_z_cache);
			}
			else {
				assert(0 && "Not a valid activation name");
			}
			// auto d_bias_batch = matrices::mult(in, matrices::mult(next_weigth, d_activation));
			auto d_bias_batch = matrices::mult(in, next_weigth.transpose()) * d_activation;
			m_d_weights = matrices::mult(prev_activation.transpose(), d_bias_batch);
			int batch_size = d_bias_batch.get_row_num();
			m_d_bias = util::get_mean(d_bias_batch);
			m_d_weights = m_d_weights / batch_size;

			if (print_on) {
				std::string opt_str = "d_activation row num: " + std::to_string(d_activation.get_row_num());
				opt_str += " col num: " + std::to_string(d_activation.get_col_num());
				opt_str += "\ndata: ";
				matrices::util::print_elements(d_activation.data, " ", opt_str);

				std::string opt_str1 = "input row num: " + std::to_string(in.get_row_num());
				opt_str += " col num: " + std::to_string(in.get_col_num());
				opt_str += "\ndata: ";
				matrices::util::print_elements(in.data, " ", opt_str1);

				std::string opt_str2 = "d_bias row num: " + std::to_string(m_d_bias.get_row_num());
				opt_str2 += " col num: " + std::to_string(m_d_bias.get_col_num());
				opt_str2 += "\ndata: ";
				matrices::util::print_elements(m_d_bias.data, " ", opt_str2);

				std::string opt_str3 = "d_weights row num: " + std::to_string(m_d_weights.get_row_num());
				opt_str3 += " col num: " + std::to_string(m_d_weights.get_col_num());
				opt_str3 += "\ndata: ";
				matrices::util::print_elements(m_d_weights.data, " ", opt_str3);

			}

			return d_bias_batch;
		}

		matrices::Matrix<double> backward2(matrices::Matrix<double> const& d_bias_batch, matrices::Matrix<double>prev_activation,
			int print_on = 0) {
			matrices::Matrix<double> out;
			// auto d_bias_batch = matrices::mult(in, next_weigth.transpose()) * d_activation;
			m_d_weights = matrices::mult(prev_activation.transpose(), d_bias_batch);
			int batch_size = d_bias_batch.get_row_num();
			m_d_bias = util::get_mean(d_bias_batch);
			m_d_weights = m_d_weights / batch_size;
			auto prev_d_err = matrices::mult(d_bias_batch, m_weights.transpose());

			if (print_on) {
				auto d_activation = get_d_activation();
				std::string opt_str = "d_activation row num: " + std::to_string(d_activation.get_row_num());
				opt_str += " col num: " + std::to_string(d_activation.get_col_num());
				opt_str += "\ndata: ";
				matrices::util::print_elements(d_activation.data, " ", opt_str);

				std::string opt_str2 = "d_bias row num: " + std::to_string(m_d_bias.get_row_num());
				opt_str2 += " col num: " + std::to_string(m_d_bias.get_col_num());
				opt_str2 += "\ndata: ";
				matrices::util::print_elements(m_d_bias.data, " ", opt_str2);

				std::string opt_str3 = "d_weights row num: " + std::to_string(m_d_weights.get_row_num());
				opt_str3 += " col num: " + std::to_string(m_d_weights.get_col_num());
				opt_str3 += "\ndata: ";
				matrices::util::print_elements(m_d_weights.data, " ", opt_str3);

			}

			return prev_d_err;
		}

	};

	template<typename D>
	struct Model {
		std::vector<int> m_layers{};
		std::vector<std::string> m_activations{};
		std::vector<Linear> m_steps{};
		std::vector<Dropout<D>> m_dropouts{};
		matrices::Matrix<D> m_input{};
		matrices::Matrix<D> m_probs{};
		Loss<D, std::uint8_t> loss_obj{};
		matrices::Matrix<D> m_loss{};
		std::string m_loss_cat{};
		Model(int input_dim, std::vector<int> layers, std::vector<std::string> activations, 
			std::vector<D>dropout_probs, std::string loss_cat="cross_entropy") {
			assert((layers.size() == activations.size()) && (layers.size() == dropout_probs.size()) && "dimensions doesnt match");
			m_layers.resize(layers.size()+1);
			m_layers[0] = input_dim;
			std::copy(layers.begin(), layers.end(), m_layers.begin() + 1);
			m_activations = activations;
			m_steps.resize(m_activations.size());
			m_dropouts.resize(m_activations.size());
			for (std::size_t i = 0; i < m_activations.size(); ++i) {
				m_steps[i] = Linear(m_layers[i], m_layers[i + 1], m_activations[i], "uniform");
				m_dropouts[i] = Dropout<D>{ dropout_probs[i] };
			}
			m_loss_cat = loss_cat;
			loss_obj = Loss<D, std::uint8_t>{ m_loss_cat };
		}

		template<typename T>
		matrices::Matrix<T> forward(matrices::Matrix<T> in,
			int start_step = 0, int inference = 0, int init_dropout = 1, int print_on = 0) {
			assert((start_step >= 0 && start_step < m_steps.size()) && "start step is not valid");
			if (start_step == 0) {
				m_input = in;
			}
			for (std::size_t i = start_step; i!=m_steps.size(); ++i) {
				in = m_steps[i].forward(in, print_on);
				in = m_dropouts[i].forward(in, inference, init_dropout);
			}
			m_probs = in;
			// m_loss = loss_obj(m_probs, labels);
			return m_probs;
		};

		template<typename T>
		matrices::Matrix<D> backward( matrices::Matrix<T> labels, int gradient_check = 0, int print_on = 0) {
			static_assert(std::is_integral_v<T>, "label type is not integral");
			matrices::Matrix<D> d_bias_batch;
			matrices::Matrix<D> d_err;
			if (m_loss_cat.compare("cross_entropy") == 0) {
				d_err = loss_obj.backward(m_probs, labels); 
				d_bias_batch = d_err * m_steps.back().get_d_activation();
			}
			else if (m_loss_cat.compare("categorical_cross_entropy") == 0) {
				assert(m_steps.back().m_activation.compare("softmax") == 0 && "Derivative implemented only for softmax+categorical_cross_entropy");
				d_bias_batch = loss_obj.backward(m_probs, labels);
			}
			else {
				assert(0 && "Loss function is not known");
			}
			std::vector<D> next_weigth_data(m_probs.get_col_num(), D{ 1 });
			matrices::Matrix<D> next_weigth{ 1, m_probs.get_col_num(), next_weigth_data };
			for (int i = m_steps.size()-1; i != -1; --i) {
				// matrices::Matrix<double> prev_activation = i == 0 ? m_input : m_steps[i - 1].m_activation_cache;
				// d_err = m_steps[i].backward(d_err, prev_activation, next_weigth, print_on);
				matrices::Matrix<D> prev_activation = i == 0 ? m_input : m_dropouts[i - 1].m_activation_cache;
				d_err = m_steps[i].backward2(d_bias_batch, prev_activation, print_on);
				if (i > 0) {
					d_err = m_dropouts[i-1].backward(d_err);
					d_bias_batch = d_err * m_steps[i - 1].get_d_activation();
				}
				next_weigth = m_steps[i].m_weights;
				if (gradient_check) {
					double err_thresh{ 1e-6 };
					double epsilon = { 0.001 };
					int inference{ 0 };
					int init_dropout{ 0 };
					int print_on{ 0 };
					for (std::size_t j = 0; j < m_steps[i].m_weights.data.size(); ++j) {
						if (i == 0 && j == 19) {
							std::cout << "for debugging\n";
						}
						auto temp = m_steps[i].m_weights.data[j];
						
						m_steps[i].m_weights.data[j] = temp + epsilon;
						auto m_preds1 = forward(prev_activation, i, inference, init_dropout, print_on);
						auto loss1 = loss_obj(m_preds1, labels);

						m_steps[i].m_weights.data[j] = temp - epsilon;
						auto m_preds2 = forward(prev_activation, i, inference, init_dropout, print_on);
						auto loss2 = loss_obj(m_preds2, labels);

					  // TODO: calc numeric gradient
						auto mean_loss1 = util::get_mean(loss1);
						auto mean_loss2 = util::get_mean(loss2);
						auto cur_grad = (mean_loss1.data[0] - mean_loss2.data[0]) / (2 * epsilon);

						m_steps[i].m_weights.data[j] = temp;
						auto diff = std::abs(cur_grad - m_steps[i].m_d_weights.data[j]);
						std::cout << "diff btw gradients " << std::to_string(diff) << ", layer:" << std::to_string(i);
						std::cout << ", idx:" << std::to_string(j) << "/" << std::to_string(m_steps[i].m_weights.data.size()) << "\n";
						assert(diff < err_thresh && "numerical gradient doesnt match");
					}
					for (std::size_t j = 0; j < m_steps[i].m_bias.data.size(); ++j) {
						auto temp = m_steps[i].m_bias.data[j];

						m_steps[i].m_bias.data[j] = temp + epsilon;
						auto m_preds1 = forward(prev_activation, i, inference, init_dropout, print_on);
						auto loss1 = loss_obj(m_preds1, labels);

						m_steps[i].m_bias.data[j] = temp - epsilon;
						auto m_preds2 = forward(prev_activation, i, inference, init_dropout, print_on);
						auto loss2 = loss_obj(m_preds2, labels);

						// TODO: calc numeric gradient
						auto mean_loss1 = util::get_mean(loss1);
						auto mean_loss2 = util::get_mean(loss2);
						auto cur_grad = (mean_loss1.data[0] - mean_loss2.data[0]) / (2 * epsilon);

						m_steps[i].m_bias.data[j] = temp;
						auto diff = std::abs(cur_grad - m_steps[i].m_d_bias.data[j]);
						std::cout << "diff btw gradients " << std::to_string(diff) << ", layer:" << std::to_string(i);
						std::cout << ", idx:" << std::to_string(j) << "/" << std::to_string(m_steps[i].m_bias.data.size()) << "\n";
						assert(diff < err_thresh && "numerical gradient doesnt match");
					}
				}
			}
			return d_bias_batch;
		};

		void update(double learning_rate = 0.01) {
			for (std::size_t i = 0; i < m_steps.size(); ++i) {
				m_steps[i].m_weights -= m_steps[i].m_d_weights * learning_rate;
				m_steps[i].m_bias -= m_steps[i].m_d_bias * learning_rate;
			}
		}

	};
} // namespace nn

#endif // end of SRC_MLP_H

