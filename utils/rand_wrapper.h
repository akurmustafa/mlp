
#ifndef UTILS_RAND_WRAPPER_H
#define UTILS_RAND_WRAPPER_H

#include <random>
#include <type_traits>
#include <vector>

namespace rand_wrapper 
{
// all declarations in the namespace rand_wrapper
static std::random_device rd;
// std::mt19937 gen(rd());
static std::mt19937 rng(0); // seed for reproducability
int rand_int(int min_val, int max_val); // [) max_val exclusive
double randn(double mean = 0, double variance = 1);
std::vector<double> randn(int n, double mean = 0, double variance = 1);
double rand(double low = 0.0, double high = 0.0);
int poisson(int lambda);

template<typename T>
T rand(T low, T high) {
	static_assert(std::is_floating_point_v<T>, "T must be floating point type");
	std::uniform_real_distribution<T> uni(low, high);
	return uni(rng);
}

template<typename D, typename T>
std::vector<T> rand(D elem_num, T low, T high) {
	static_assert(std::is_integral_v<D>, "D must be integral type");
	static_assert(std::is_floating_point_v<T>, "T must be floating point type");
	assert(elem_num >= 0 && "elem_num can'be negative");
	std::vector<T> res(elem_num, T{ 0 });
	for (std::size_t i = 0; i < res.size(); ++i) {
		res[i] = rand<T>(low, high);
	}
	return res;
}

template <typename T>
void shuffle(std::vector<T>& vec) { // shuffles in place
	for (int i = 0; i < vec.size(); ++i) {
		int idx1 = rand_int(0, vec.size());
		int idx2 = rand_int(0, vec.size());
		// swap
		auto temp = vec[idx1];
		vec[idx1] = vec[idx2];
		vec[idx2] = temp;
	}
}

}  // namespace rand_wrapper

#endif  // end of UTILS_RAND_WRAPPER_H
