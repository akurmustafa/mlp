#ifndef SRC_MATRICES_H
#define SRC_MATRICES_H


#include <cassert>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace matrices {
	namespace util {
		template<typename T>
		inline void print_elements(const T& coll, const std::string& seperator_str = " ", const std::string& opt_str = "") {
			std::cout << opt_str;
			std::for_each(coll.begin(), coll.end(), [seperator_str](auto elem) {std::cout << elem << seperator_str; });
			std::cout << "\n";
		}

		template<typename It>
		inline void print_elements(It start_it, It end_it, const std::string& seperator_str = " ", const std::string& opt_str = "", std::int64_t n = 1) {
			std::cout << opt_str;
			/*auto dist = std::floor(std::distance(start_it, end_it) / step) * step;
			auto end_it_modif = start_it + dist;
			for (auto it = start_it; it != end_it_modif; it += step) {
				std::cout << *(it) << seperator_str;
			}
			if (end_it_modif != end_it) {
				std::cout << *(end_it_modif) << seperator_str;
			}*/
			auto dist = std::distance(start_it, end_it);
			for (std::int64_t i = dist / n; i--; std::advance(start_it, n)) {
				std::cout << *(start_it) << seperator_str;
			}
			if (start_it != end_it) {
				std::cout << *(start_it) << seperator_str;
			}
			std::cout << "\n";
		}

		template<class in_it, class out_it>
		inline out_it copy_every_n(in_it b, in_it e, out_it r, std::int64_t n) {
			for (std::int64_t i = std::distance(b, e) / n; i--; std::advance(b, n)) {
				*r++ = *b;
			}
			if (b != e) {
				*r++ = *b;
			}
			return r;
		}
	}  // namespace util
	template <typename T>
	class Matrix {
	private:
		std::size_t n_row;
		std::size_t n_col;
		int transposed = 0;
	public:
		std::vector<T> data;
		Matrix() : n_row(0), n_col(0), transposed(0), data(0, 0) {}  // default ctor
		// contructor
		Matrix(std::int64_t m, std::int64_t n) : n_row(m), n_col(n), transposed(0), data(std::vector<T>(m* n, T{ 0 })) {}

		// constructor with vector data
		template<typename T1>
		Matrix(std::int64_t m, std::int64_t n, std::vector<T1> const& vec, int transpose = 0) : n_row(m), n_col(n) {
			// using X = decltype(std::declval<T>() +std::declval<T1>());
			static_assert(std::is_assignable<std::vector<T>, std::vector<T1> const>::value, "Matrix type and Vector type are not assignable");
			static_assert(std::is_assignable<T&, T1>::value, "Matrix type and Vector type are not assignable");
			assert(m * n == vec.size() && "Dimension doesn't match");
			// std::cout << "vec constructor of Matrix\n";
			data = vec;
			transposed = transpose;
		}

		// copy constructor
		Matrix(const Matrix<T>& other) {
			// std::cout << "copy constructor of Matrix\n";
			n_row = other.n_row;
			n_col = other.n_col;
			data.resize(n_row * n_col);
			transposed = other.transposed;
			std::copy(other.data.begin(), other.data.end(), data.begin());
		}

		// copy assignment
		Matrix& operator=(const Matrix<T>& other){
			// std::cout << "copy assignment of Matrix\n";
			n_row = other.n_row;
			n_col =other.n_col;
			data.resize(n_row * n_col);
			transposed = other.transposed;
			std::copy(other.data.begin(), other.data.end(), data.begin());
			return *this;
		}

		// move constructor
		Matrix(Matrix<T>&& other) noexcept {
			// std::cout << "move constructor of Matrix\n";
			std::swap(n_row, other.n_row);
			std::swap(n_col, other.n_col);
			std::swap(transposed, other.transposed);
			data = std::move(other.data);
		}

		// move assignment
		template<typename T>
		Matrix& operator=(Matrix<T>&& other) {
			// std::cout << "move assignment of Matrix\n";
			std::swap(n_row, other.n_row);
			std::swap(n_col, other.n_col);
			std::swap(transposed, other.transposed);
			data = std::move(other.data);
			return *this;
		}
		std::size_t get_row_num() const { return n_row; }
		std::size_t get_col_num() const { return n_col; }

		std::vector<T> get_data() const { return data; }

		std::vector<T> get_row(std::int64_t row_idx) const {
			assert(row_idx >= 0 && "row idx cant be negative");
			assert(row_idx < n_row && "row idx is not valid\n");
			std::vector<T> cur_row(n_col, 0);
			if (transposed) {
				util::copy_every_n(data.begin() + row_idx, data.end(), cur_row.begin(), n_row);
				/*for (int i = 0; i < n_col; ++i) {
					cur_row[i] = data[i * n_row + row_idx];
				}*/
			}
			else {
				std::copy(data.begin() + row_idx * n_col, data.begin() + (row_idx + 1) * n_col,
					cur_row.begin());
				/*for (int i = 0; i < n_col; ++i) {
					cur_row[i] = data[row_idx * n_col + i];
				}*/
			}
			return cur_row;
		}

		Matrix<T> get_row_btw(std::int64_t row_start_idx, std::int64_t row_end_idx) {
			assert(row_start_idx >= 0 && "cannot be negative idx");
			assert(row_end_idx <= n_row && "row is not in the range");
			std::int64_t new_row_num = row_end_idx - row_start_idx;
			assert(new_row_num > 0 && "cannot request nonpositive row number");
			std::vector<T> res_mat_data(new_row_num * n_col, T{ 0 });
			std::copy(data.begin() + row_start_idx * n_col, data.begin() + (row_end_idx) * n_col,
				res_mat_data.begin());
			Matrix<T> res_mat{ new_row_num, n_col, res_mat_data };
			return res_mat;
		}

		void set_row(std::int64_t row_idx, std::vector<T> cur_row) {
			assert((row_idx >= 0) && "row idx cant be negative");
			assert((row_idx < n_row) && "row idx is not valid\n");
			assert((n_col == cur_row.size()) && "Dimensions are not equal");
			std::copy(cur_row.begin(), cur_row.end(),
				data.begin() + row_idx * n_col);
		}

		std::vector<T> get_col(std::int64_t col_idx) const {
			assert(col_idx >= 0 && "col idx cant be negative");
			assert(col_idx < n_col && "col idx is not valid\n");
			std::vector<T> cur_col(n_row, 0);
			if (transposed) {
				std::copy(data.begin() + col_idx * n_row, data.begin() + (col_idx + 1) * n_row,
					cur_col.begin());
				/*for (int i = 0; i < n_row; ++i) {
					cur_col[i] = data[col_idx * n_row + i];
				}*/
			}
			else {
				util::copy_every_n(data.begin() + col_idx, data.end(), cur_col.begin(), n_col);
				/*for (int i = 0; i < n_row; ++i) {
					cur_col[i] = data[i * n_col + col_idx];
				}*/
			}
			return cur_col;
		}

		bool is_transposed() const { return transposed; }
		//void transpose() {
		//	// std::cout << "transpose taken\n";
		//	transposed = transposed == 0 ? 1 : 0; // toggle
		//	std::swap(n_row, n_col);
		//}
		Matrix<T> transpose() const {
			std::vector<T> new_data(data.size(), T{ 0 });
			for (std::size_t i = 0; i < n_row; ++i) {
				for (std::size_t j = 0; j < n_col; ++j) {
					new_data[j*n_row+i] = data[i * n_col + j];
				}
			}
			Matrix<T> out{ n_col, n_row, new_data };
			return out;
		}

		void print() {
			if (transposed) {
				for (std::size_t i = 0; i < n_row; ++i) {
					util::print_elements(data.begin() + i, data.end(),
						" ", "row" + std::to_string(i) + ": ", n_row);
					std::cout << "\n";
				}
			}
			else {
				for (std::size_t i = 0; i < n_row; ++i) {
					util::print_elements(data.begin() + i * n_col, data.begin() + (i + 1) * n_col, 
						" ", "row" + std::to_string(i) + ": ", 1);
					std::cout << "\n";
				}
			}
		}

		T operator[](std::int64_t idx) const {
			return data[idx];
		}

		template<typename T1>
		friend void print(const Matrix<T1>&, const std::string);
	};

	template<typename T>
	Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		assert(lhs.get_col_num() == rhs.get_col_num() && "dimensions do not match!");
		assert((lhs.get_row_num() == rhs.get_row_num() || rhs.get_row_num() == 1) && "cannot broadcast!");
		int row_num = lhs.get_row_num();
		int col_num = lhs.get_col_num();
		std::vector<T> data(row_num * col_num, T{ 0 });
		for (std::size_t i = 0; i < lhs.data.size(); i += rhs.data.size()) {
			std::transform(lhs.data.begin()+i, lhs.data.begin()+i+rhs.data.size(),
				rhs.data.begin(),
				data.begin()+i,
				[](auto lhs, auto rhs) {return lhs + rhs; });
		}
		/*std::transform(lhs_data.begin(), lhs_data.end(),
			rhs_data.begin(),
			data.begin(),
			[](auto lhs, auto rhs) {return lhs + rhs; });*/

		/*for (int i = 0; i < row_num; ++i) {
			for (int j = 0; j < col_num; ++j) {
				data[i * col_num + j] = lhs_data[i * col_num + j] + rhs_data[i * rhs_col_num + (j % rhs_col_num)];
			}
		}*/
		Matrix<T> res(row_num, col_num, data);
		return res;
	}

	template<typename T>
	Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		assert(lhs.get_col_num() == rhs.get_col_num() && "dimensions do not match!");
		assert((lhs.get_row_num() == rhs.get_row_num() || rhs.get_row_num() == 1) && "dimensions do not match!");
		int row_num = lhs.get_row_num();
		int col_num = lhs.get_col_num();
		std::vector<T> data(row_num * col_num, T{ 0 });
		for (std::size_t i = 0; i < lhs.data.size(); i += rhs.data.size()) {
			std::transform(lhs.data.begin() + i, lhs.data.begin() + i + rhs.data.size(),
				rhs.data.begin(),
				data.begin() + i,
				[](auto lhs, auto rhs) {return lhs - rhs; });
		}
		Matrix<T> res(row_num, col_num, data);
		return res;
	}
	
	template<typename T>
	Matrix<T>& operator-=(Matrix<T>& lhs, const Matrix<T>& rhs) {
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T>
	Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		assert(lhs.get_col_num() == rhs.get_col_num() && "dimensions do not match!");
		assert(lhs.get_row_num() == rhs.get_row_num() && "dimensions do not match");
		assert((lhs.get_row_num() == rhs.get_row_num() || rhs.get_row_num() == 1) && "cannot broadcast!");
		auto row_num = lhs.get_row_num();
		auto col_num = lhs.get_col_num();
		std::vector<T> data(row_num * col_num, 0);
		for (std::size_t i = 0; i < lhs.data.size(); i += rhs.data.size()) {
			std::transform(lhs.data.begin() + i, lhs.data.begin() + i + rhs.data.size(),
				rhs.data.begin(),
				data.begin() + i,
				[](auto lhs, auto rhs) {return lhs * rhs; });
		}
		Matrix<T> res(row_num, col_num, data);
		return res;
	}

	template<typename T, typename D>
	Matrix<T> operator*(const Matrix<T>& lhs, D constant) {
		int row_num = lhs.get_row_num();
		int col_num = lhs.get_col_num();
		std::vector<T> data(row_num * col_num, 0);
		std::transform(lhs.data.begin(), lhs.data.end(),
			data.begin(),
			[constant](auto lhs) {return lhs * constant; });
		Matrix<T> res(row_num, col_num, data);
		return res;
	}

	template<typename D, typename T>
	Matrix<T> operator*(D constant, const Matrix<T>& lhs) {
		return lhs*constant;
	}

	template<typename T, typename D>
	Matrix<T> operator/(const Matrix<T>& lhs, D constant) {
		int row_num = lhs.get_row_num();
		int col_num = lhs.get_col_num();
		std::vector<T> data(row_num * col_num, 0);
		std::transform(lhs.data.begin(), lhs.data.end(),
			data.begin(),
			[constant](auto lhs) {return lhs / constant; });
		Matrix<T> res(row_num, col_num, data);
		return res;
	}

	template<typename T>
	void print(const Matrix<T>& mat, const std::string opt_str = "") {
		int n_row = mat.n_row;
		int n_col = mat.n_col;
		std::cout << opt_str;
		if (mat.transposed) {
			for (std::size_t i = 0; i < n_row; ++i) {
				for (std::size_t j = 0; j < n_col; ++j) {
					std::cout << mat.data[j * n_row + i] << ", ";
				}
				std::cout << "\n";
			}
		}
		else {
			for (std::size_t i = 0; i < n_row; ++i) {
				for (std::size_t j = 0; j < n_col; ++j) {
					std::cout << mat.data[i * n_col + j] << ", ";
				}
				std::cout << "\n";
			}
		}
	}

	template <typename T>
	T dot_product(const std::vector<T>& lhs, const std::vector<T>& rhs) {
		assert(lhs.size() == rhs.size() && "dimensions don't match");
		
		T res =std::inner_product(lhs.begin(), lhs.end(),
			rhs.begin(), T{ 0 });
		return res;
	}

	template<typename T>
	Matrix<T> mult(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		assert(lhs.get_col_num() == rhs.get_row_num());
		auto row_num = lhs.get_row_num();
		auto col_num = rhs.get_col_num();
		std::vector<T> res_data(row_num * col_num, T{ 0 });
		for (std::size_t i = 0; i < row_num; ++i) {
			for (std::size_t j = 0; j < col_num; ++j) {
				std::vector<T> cur_row = lhs.get_row(i);
				std::vector<T> cur_col = rhs.get_col(j);
				T total = dot_product<T>(cur_row, cur_col);
				res_data[i * col_num + j] = total;
			}
		}
		Matrix<T> res(row_num, col_num, std::move(res_data));
		return res;
	}

}  // namespace matrices

#endif // end of SRC_MATRICES_H
