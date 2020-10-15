
#ifndef UTILS_RAND_WRAPPER_H
#define UTILS_RAND_WRAPPER_H

#include <random>
#include <vector>

namespace rand_wrapper 
{
// all declarations in the namespace rand_wrapper

int rand_int(int min_val, int max_val); // [) max_val exclusive
double randn(double mean = 0, double variance = 1);
std::vector<double> randn(int n, double mean = 0, double variance = 1);
double rand(double low = 0.0, double high = 0.0);
int poisson(int lambda);

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
