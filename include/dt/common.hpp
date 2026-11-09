#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dt {

constexpr int kMaxClasses = 16;
constexpr int kMaxBins = 256;

// Histogram counter type. Counts are integers, so every reduction is exact and
// independent of the order in which partial histograms are combined. This is
// what makes the parallel trainers produce bit-identical trees.
using hist_t = std::uint32_t;

inline int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

class Timer {
 public:
  Timer() : start_(Clock::now()) {}
  void reset() { start_ = Clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }
  // Returns the elapsed time and restarts the timer.
  double lap() {
    const double s = seconds();
    reset();
    return s;
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

// Minimal "--key value" / "--flag" command-line parser.
class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + key);
      key = key.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        values_[key] = argv[++i];
      } else {
        values_[key] = "";
      }
    }
  }

  bool has(const std::string& key) const { return values_.count(key) > 0; }

  std::string str(const std::string& key, const std::string& def = "") const {
    auto it = values_.find(key);
    return it == values_.end() ? def : it->second;
  }

  std::string required(const std::string& key) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.empty()) {
      throw std::runtime_error("missing required option --" + key);
    }
    return it->second;
  }

  long long integer(const std::string& key, long long def) const {
    auto it = values_.find(key);
    return it == values_.end() ? def : std::stoll(it->second);
  }

  double real(const std::string& key, double def) const {
    auto it = values_.find(key);
    return it == values_.end() ? def : std::stod(it->second);
  }

 private:
  std::map<std::string, std::string> values_;
};

}  // namespace dt
