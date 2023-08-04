#ifndef INCREMENTALSOLVER_EXPRESSIONGRAPH_INTERFACE_H
#define INCREMENTALSOLVER_EXPRESSIONGRAPH_INTERFACE_H
#include <cassert>
#include <memory>
#include <variant>

namespace incremental_solver {
namespace expression_graph {
using Integer = int64_t;
using Double = double;
class Node;
class ExpressionGraphBuilder;

enum class ValueType {
  INTEGER,
  DOUBLE,
};

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
    assert(lb_.index() == 0);
    assert((ub_.index() == 0) == (valueType == ValueType::INTEGER));
  }
  void setBoolValue(bool value) {
    assert(getValueType() == ValueType::INTEGER);
    setIntValue(value ? 1 : 0);
  }
  void setIntValue(Integer value);
  void setDoubleValue(Double value);

private:
  template <typename T> void checkBounds(T value) {
    assert(value >= std::get<Integer>(lb_) && "Lower bound is violated");
    assert(value <= std::get<Integer>(ub_) && "Upper bound is violated");
  }
};

} // namespace expression_graph
} // namespace incremental_solver

#endif /* INCREMENTALSOLVER_EXPRESSIONGRAPH_INTERFACE_H */
