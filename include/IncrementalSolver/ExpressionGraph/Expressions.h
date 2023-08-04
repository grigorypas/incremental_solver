#ifndef INCREMENTALSOLVER_EXPRESSIONGRAPH_EXPRESSIONS_H
#define INCREMENTALSOLVER_EXPRESSIONGRAPH_EXPRESSIONS_H
#include "IncrementalSolver/ExpressionGraph/ExpressionGraph.h"

namespace incremental_solver {
namespace expression_graph {

template <typename T> class StorageNode : public Node {
  T storageVal_;

public:
  template <typename U>
  StorageNode(U currentVal, T storageVal)
      : Node(currentVal), storageVal_(storageVal) {}
  T &storageValue() { return storageVal_; }
};

class AnyOp final : public SimpleEdge<Node, StorageNode<Integer>> {
public:
  AnyOp(Node *fromNode, StorageNode<Integer> *toNode)
      : SimpleEdge(fromNode, toNode) {}
  void propagate(NodeQueue &queue) override;
};

template <typename T> class AddOp final : public SimpleEdge<Node, Node> {
public:
  AddOp(Node *fromNode, Node *toNode) : SimpleEdge(fromNode, toNode) {}
  void propagate(NodeQueue &queue) override {
    T varDifference = fromNode_->getDifference<T>();
    if (varDifference == 0) {
      return;
    }
    toNode_->increaseNewValueBy(varDifference);
    queue.push(toNode_);
  }
};

template <typename T> class NegAddOp final : public SimpleEdge<Node, Node> {
public:
  NegAddOp(Node *fromNode, Node *toNode) : SimpleEdge(fromNode, toNode) {}
  void propagate(NodeQueue &queue) override {
    T varDifference = fromNode_->getDifference<T>();
    if (varDifference == 0) {
      return;
    }
    toNode_->increaseNewValueBy(-varDifference);
    queue.push(toNode_);
  }
};

template <typename T> class MulAddOp final : public SimpleEdge<Node, Node> {
  T coefficient_;

public:
  MulAddOp(Node *fromNode, Node *toNode, T coefficient)
      : SimpleEdge(fromNode, toNode), coefficient_(coefficient) {}
  void propagate(NodeQueue &queue) override {
    T varDifference = fromNode_->getDifference<T>();
    if (varDifference == 0) {
      return;
    }
    toNode_->increaseNewValueBy(varDifference * coefficient_);
    queue.push(toNode_);
  }
};

} // namespace expression_graph
} // namespace incremental_solver

#endif /* INCREMENTALSOLVER_EXPRESSIONGRAPH_EXPRESSIONS_H */
