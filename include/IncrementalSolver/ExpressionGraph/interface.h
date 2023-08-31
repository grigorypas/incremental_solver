#ifndef INCREMENTALSOLVER_EXPRESSIONGRAPH_INTERFACE_H
#define INCREMENTALSOLVER_EXPRESSIONGRAPH_INTERFACE_H
#include <cassert>
#include <memory>
#include <variant>

#include "IncrementalSolver/Common/Types.h"

namespace incremental_solver {
namespace expression_graph {
class Node;
class ExpressionGraphBuilder;

class ValueExpression {
  int64_t id_;
  ValueType valueType_;

protected:
  Node *node_;

public:
  ValueExpression(ValueType valueType, Node *node, int64_t id)
      : id_(id), valueType_(valueType), node_(node) {}
  ValueExpression(const ValueExpression &other) = default;
  ValueExpression(ValueExpression &&other) = default;
  ValueExpression &operator=(const ValueExpression &other) = default;
  ValueExpression &operator=(ValueExpression &&other) = default;
  ValueType getValueType() const { return valueType_; }
  int64_t getId() const { return id_; }
  bool getBoolValue() const { return bool(getIntValue()); };
  Integer getIntValue() const;
  Double getDoubleValue() const;
};

class DecisionVariable : public ValueExpression {
  std::variant<Integer, Double> lb_;
  std::variant<Integer, Double> ub_;
  std::shared_ptr<ExpressionGraphBuilder> graph_;

public:
  DecisionVariable(ValueType valueType, Node *node, int64_t id,
                   std::variant<Integer, Double> lb,
                   std::variant<Integer, Double> ub,
                   std::shared_ptr<ExpressionGraphBuilder> graph)
      : ValueExpression(valueType, node, id), lb_(std::move(lb)),
        ub_(std::move(ub)), graph_(std::move(graph)) {
    assert((lb_.index() == 0) == (valueType == ValueType::INTEGER));
    assert((ub_.index() == 0) == (valueType == ValueType::INTEGER));
  }
  DecisionVariable(const DecisionVariable &other) = default;
  DecisionVariable(DecisionVariable &&other) = default;
  DecisionVariable &operator=(const DecisionVariable &other) = default;
  DecisionVariable &operator=(DecisionVariable &&other) = default;
  void setBoolValue(bool value) {
    assert(getValueType() == ValueType::INTEGER);
    setIntValue(value ? 1 : 0);
  }
  void setIntValue(Integer value);
  void setDoubleValue(Double value);

private:
  template <typename T> void checkBounds(T value) {
    assert(value >= std::get<T>(lb_) && "Lower bound is violated");
    assert(value <= std::get<T>(ub_) && "Upper bound is violated");
  }
};

} // namespace expression_graph
} // namespace incremental_solver

#endif /* INCREMENTALSOLVER_EXPRESSIONGRAPH_INTERFACE_H */
