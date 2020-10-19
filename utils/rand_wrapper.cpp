
#include "utils/rand_wrapper.h"

#include <cassert>
#include <random>

namespace rand_wrapper 
{
// std::random_device rd;
// std::mt19937 gen(rd());
// std::mt19937 gen(0); // seed for reproducability
int rand_int(int min_val, int max_val) {
	// generates random integere between the interval
	// min_val and max_val (including min_val, excluding max_val)
	std::uniform_int_distribution<int> uni(min_val, max_val -1);
	return uni(rng);
}

double randn(double mean, double variance) {
	// generates gaussian number with mean, and variance int the arguments
	std::normal_distribution<double> normal(mean, variance);
	return normal(rng);
}

std::vector<double> randn(int n, double mean, double variance) {
	// generates gaussian number with mean, and variance int the arguments
	assert(n >= 0 && "n cant be negative");
	std::vector<double> out(n, double{ 0 });
	std::normal_distribution<double> normal(mean, variance);
	for (int i = 0; i < n; ++i) {
		out[i] = normal(rng);
	}
	return out;
}

double rand(double low, double high) {
	// generates uniform double number between
	// low and high
	std::uniform_real_distribution<double> uni(low, high);
	return uni(rng);
}

int poisson(int lambda) {
	assert(lambda >= 0 && "in poisson distribution lambda cannot be negative\n");
	if (lambda == 0) {
		return 0;
	}
	std::poisson_distribution<int> distribution(lambda);
	return distribution(rng);
}

}  // namespace rand_wrapper
