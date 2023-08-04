#ifndef INCREMENTALSOLVER_EXPRESSIONGRAPH_EXPRESSIONGRAPH_H
#define INCREMENTALSOLVER_EXPRESSIONGRAPH_EXPRESSIONGRAPH_H
#include "IncrementalSolver/ExpressionGraph/interface.h"

#include "llvm/ADT/ArrayRef.h"

#include <queue>
#include <vector>

namespace incremental_solver {
namespace expression_graph {
class Node;
using NodeQueue = std::queue<Node *>;

class Edge;
class Node {
  union VariantValue {
    Integer intValue;
    Double doubleValue;
  };
  VariantValue newValue_;
  VariantValue oldValue_;
  std::vector<Edge *> outEdges_;

public:
  template <typename T> Node(T currentValue) {
    llvm::report_fatal_error("unsupported value type");
  };
  void addEdge(Edge *edge) { outEdges_.emplace_back(edge); }
  llvm::ArrayRef<Edge *> getOutEdges() const {
    if (outEdges_.empty()) {
      return llvm::ArrayRef<Edge *>();
    }
    return llvm::ArrayRef<Edge *>(&outEdges_[0], outEdges_.size());
  }
  template <typename T> T getNewValue() const {
    llvm::report_fatal_error("unsupported value type");
  }
  template <typename T> T getOldValue() const {
    llvm::report_fatal_error("unsupported value type");
  }
  template <typename T> T getDifference() const {
    return getNewValue<T>() - getOldValue<T>();
  }
  template <typename T> void setNewValue(T newValue) {
    llvm::report_fatal_error("unsupported value type");
  }
  template <typename T> void increaseNewValueBy(T newValueDelta) {
    setNewValue<T>(getNewValue<T>() + newValueDelta);
  }
  void resetOldValue() { oldValue_ = newValue_; }
};

template <> inline Node::Node(Integer currentValue) {
  newValue_.intValue = currentValue;
  oldValue_ = newValue_;
}

template <> inline Node::Node(Double currentValue) {
  newValue_.doubleValue = currentValue;
  oldValue_ = newValue_;
}

template <> inline Integer Node::getNewValue<Integer>() const {
  return newValue_.intValue;
}

template <> inline double Node::getNewValue<Double>() const {
  return newValue_.doubleValue;
}

template <> inline Integer Node::getOldValue<Integer>() const {
  return oldValue_.intValue;
}

template <> inline double Node::getOldValue<Double>() const {
  return oldValue_.doubleValue;
}

template <> inline void Node::setNewValue<Integer>(Integer newValue) {
  newValue_.intValue = newValue;
}

template <> inline void Node::setNewValue<Double>(Double newValue) {
  newValue_.doubleValue = newValue;
}

class Edge {
public:
  virtual void propagate(NodeQueue &queue) = 0;
  virtual ~Edge(){};
};

template <class T1, class T2> class SimpleEdge : public Edge {
protected:
  T1 *fromNode_;
  T2 *toNode_;

public:
  SimpleEdge(T1 *fromNode, T2 *toNode) : fromNode_(fromNode), toNode_(toNode) {}
};

class ExpressionGraphBuilder {
  std::vector<Node *> nodes_;
  std::vector<Edge *> edges_;
  NodeQueue queue_;

public:
  ExpressionGraphBuilder() = default;
  template <class T, class... Args> T *createNode(Args &&...args) {
    T *node = new T(std::forward<Args>(args)...);
    nodes_.emplace_back(node);
    return node;
  }

  template <class T, class N1, class N2, class... Args>
  T *createSimpleEdge(N1 *fromNode, N2 *toNode, Args &&...args) {
    T *edge = new T(fromNode, toNode, std::forward<Args>(args)...);
    edges_.emplace_back(edge);
    fromNode->addEdge(edge);
    return edge;
  }

  template <class T, class... Args> T *createEdge(Args &&...args) {
    T *edge = new T(std::forward<Args>(args)...);
    edges_.emplace_back(edge);
    return edge;
  }

  template <typename T> void setValue(Node *node, T newValue) {
    node->setNewValue<T>(newValue);
    queue_.push(node);
    propagate();
  }

  ~ExpressionGraphBuilder();

private:
  void propagate();
};

} // namespace expression_graph
} // namespace incremental_solver

#endif /* INCREMENTALSOLVER_EXPRESSIONGRAPH_EXPRESSIONGRAPH_H */
