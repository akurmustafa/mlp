
#include "utils/time_wrapper.h"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

namespace time_wrapper 
{
namespace stdc = std::chrono;
Timer::Timer() {
	t_start = std::chrono::system_clock::now();
}

std::int64_t Timer::elapsed(duration_type dt, bool reset) {
	auto t_cur = stdc::system_clock::now();
	std::int64_t elapsed{ 0 };
	switch (dt) {
	case duration_type::nano:
		elapsed = stdc::duration_cast<stdc::nanoseconds>(t_cur - t_start).count();
		break;
	case duration_type::micro:
		elapsed = stdc::duration_cast<stdc::microseconds>(t_cur - t_start).count();
		break;
	case duration_type::mili:
		elapsed = stdc::duration_cast<stdc::milliseconds>(t_cur - t_start).count();
		break;
	case duration_type::sec:
		elapsed = stdc::duration_cast<stdc::seconds>(t_cur - t_start).count();
		break;
	default:
		elapsed = -1; // indication of a problem, shouldn't happen
	}
	if (reset) {
		t_start = stdc::system_clock::now();
	}
	return elapsed;
}


void Timer::reset() {
	t_start = stdc::system_clock::now();
}


std::ostream& operator<<(std::ostream& os, Timer& timer) {
	auto time_elapsed = timer.elapsed(duration_type::nano, false);
	std::int64_t mili_nano{ 1000000 };	// 1e6
	std::int64_t sec_nano{ mili_nano*1000 };	// 1e9
	std::int64_t minute_nano{ sec_nano * 60 };
	std::int64_t hour_nano{ sec_nano * 60 * 60 };
	std::int64_t hh = (time_elapsed/(hour_nano));
	time_elapsed = time_elapsed % hour_nano;
	std::int64_t mm = (time_elapsed / minute_nano);
	time_elapsed = time_elapsed % minute_nano;
	std::int64_t ss = (time_elapsed / sec_nano);
	time_elapsed = time_elapsed % sec_nano;
	std::int64_t msec = (time_elapsed / mili_nano);
	time_elapsed = time_elapsed % mili_nano;
	os << std::setfill('0') << std::setw(2) << hh << " hours::" << std::setw(2) << mm;
	os << " mins::" << std::setw(2) << ss << " secs::" << std::setw(3) << msec << " msecs\n";
	return os;
}

}  // namespace time_wrapper
