#include "IncrementalSolver/ExpressionGraph/ExpressionGraph.h"
#include "IncrementalSolver/ExpressionGraph/Expressions.h"
#include "IncrementalSolver/ExpressionGraph/interface.h"

using namespace incremental_solver::expression_graph;
using namespace incremental_solver;

Integer ValueExpression::getIntValue() const {
  assert((valueType_ == ValueType::INTEGER) &&
         "Expression is of DOUBLE type. Please use getDoubleValue() instead.");
  return node_->getNewValue<Integer>();
}
Double ValueExpression::getDoubleValue() const {
  assert((valueType_ != ValueType::DOUBLE) &&
         "Expression is not of DOUBLE type. Please use getIntValue() instead.");
  return node_->getNewValue<Double>();
}

void DecisionVariable::setIntValue(Integer value) {
  assert(getValueType() == ValueType::INTEGER);
  checkBounds(value);
  graph_->setValue<Integer>(node_, value);
}

void DecisionVariable::setDoubleValue(Double value) {
  assert(getValueType() == ValueType::DOUBLE);
  checkBounds(value);
  graph_->setValue<Double>(node_, value);
}

ExpressionGraphBuilder::~ExpressionGraphBuilder() {
  for (auto node : nodes_) {
    if (node != nullptr) {
      delete node;
    }
  }
  for (auto edge : edges_) {
    if (edge != nullptr) {
      delete edge;
    }
  }
}

void ExpressionGraphBuilder::propagate() {
  while (!queue_.empty()) {
    Node *node = queue_.front();
    queue_.pop();
    auto edges = node->getOutEdges();
    for (auto edge : edges) {
      edge->propagate(queue_);
    }
    node->resetOldValue();
  }
}

void AnyOp::propagate(NodeQueue &queue) {
  Integer varDifference = fromNode_->getDifference<Integer>();
  assert(varDifference == 0 || varDifference == -1 || varDifference == 1);
  if (varDifference == 0) {
    return;
  }
  toNode_->storageValue() += varDifference;
  if (toNode_->storageValue() > 0) {
    toNode_->setNewValue<Integer>(1);
  } else {
    toNode_->setNewValue<Integer>(0);
  }
  if (toNode_->getDifference<Integer>() != 0) {
    queue.push(toNode_);
  }
}