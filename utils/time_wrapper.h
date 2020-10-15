
#ifndef UTILS_TIME_WRAPPER_H
#define UTILS_TIME_WRAPPER_H

#include <chrono>
#include <cstddef>
#include <ostream>

namespace time_wrapper
{ // all declarations in the namespace time_wrapper
enum class duration_type{ nano, micro, mili, sec};
namespace stdc = std::chrono;

class Timer {
private:
	stdc::time_point<stdc::system_clock> t_start;
public:
	Timer();
	std::int64_t elapsed(duration_type dt = duration_type::mili, bool reset=false);
	void reset();
	friend std::ostream& operator<<(std::ostream& os, Timer& timer);
};

}  // namespace time_wrapper

#endif  // end of UTILS_TIME_WRAPPER_H
