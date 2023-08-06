#ifndef INCREMENTALSOLVER_COMMON_TYPES_H
#define INCREMENTALSOLVER_COMMON_TYPES_H
#include <cstdint>
#include <limits>

namespace incremental_solver {

using Integer = int64_t;
using Double = double;

constexpr Integer INTEGER_MIN = std::numeric_limits<Integer>::min();
constexpr Integer INTEGER_MAX = std::numeric_limits<Integer>::max();
constexpr Double DOUBLE_MIN = std::numeric_limits<Double>::lowest();
constexpr Double DOUBLE_MAX = std::numeric_limits<Double>::max();

enum class ValueType {
  INTEGER,
  DOUBLE,
};

} // namespace incremental_solver

#endif /* INCREMENTALSOLVER_COMMON_TYPES_H */
