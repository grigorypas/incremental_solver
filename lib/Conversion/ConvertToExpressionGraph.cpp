#include "IncrementalSolver/Conversion/ConvertToExpressionGraph.h"
#include "IncrementalSolver/Dialect/ModelFormulationOps.h"
#include "IncrementalSolver/ExpressionGraph/ExpressionGraph.h"
#include "IncrementalSolver/ExpressionGraph/Expressions.h"
#include "IncrementalSolver/ExpressionGraph/interface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <unordered_map>
#include <variant>
#include <vector>

using namespace incremental_solver;
using namespace incremental_solver::expression_graph;

static bool isConst(mlir::Value value) {
  auto op = value.getDefiningOp();
  return mlir::isa<mlir::arith::ConstantIntOp>(op) ||
         mlir::isa<mlir::arith::ConstantFloatOp>(op);
}

template <typename T> bool isOperationConst(T &op);

template <> bool isOperationConst(mlir::arith::MulIOp &op) {
  return isConst(op.getLhs()) && isConst(op.getRhs());
}

class ConverterToExpressionGraph {
  const mlir::ModuleOp &module_;
  std::shared_ptr<ExpressionGraphBuilder> graphBuilder_;
  std::vector<DecisionVariable> decisionVariablesList_;
  std::vector<ValueExpression> trackedValueExpressionsList_;
  std::unordered_map<mlir::Operation *, Node *> opToNodeMap_;
  std::unordered_map<mlir::Operation *, std::variant<int64_t, double>>
      currentValues_;

public:
  ConverterToExpressionGraph(const mlir::ModuleOp &module)
      : module_(module),
        graphBuilder_(std::make_shared<ExpressionGraphBuilder>()) {}

  void buildGraph();
  void swapDecisionVariablesList(std::vector<DecisionVariable> &varList) {
    varList.swap(decisionVariablesList_);
  }
  void swapTrackedValueExpressionsList(
      std::vector<ValueExpression> &trackedExprList) {
    trackedExprList.swap(trackedValueExpressionsList_);
  }

private:
  template <typename T, class O> T getCurrentVal(O &op) {
    return std::get<T>(currentValues_.at(op.getOperation()));
  }
  template <typename T> T getCurrentVal(mlir::Value value) {
    return std::get<T>(currentValues_.at(value.getDefiningOp()));
  }
  template <typename T, class O> Node *createSimpleNode(O &op) {
    T currentVal = getCurrentVal<T>(op);
    auto node = graphBuilder_->createNode<Node>(currentVal);
    opToNodeMap_.emplace(op.getOperation(), node);
    return node;
  }
  void extractValAndFactor(mlir::arith::MulIOp &op, mlir::Value &value,
                           Integer &factor);
#define OPERATION(T)                                                           \
  void actOn(T &op);                                                           \
  void calculateCurrentValue(T &op);
#include "Operations.def"
#undef OPERATION
};

void ConverterToExpressionGraph::extractValAndFactor(mlir::arith::MulIOp &op,
                                                     mlir::Value &value,
                                                     Integer &factor) {
  bool lhsConst = isConst(op.getLhs());
  bool rhsConst = isConst(op.getRhs());
  if (lhsConst) {
    factor = getCurrentVal<Integer>(op.getLhs());
    value = op.getRhs();
  } else if (rhsConst) {
    factor = getCurrentVal<Integer>(op.getRhs());
    value = op.getLhs();
  } else {
    llvm::report_fatal_error(
        "Multiplication of two variables is not supported");
  }
}

void incremental_solver::convertToExpressionGraph(
    const mlir::ModuleOp &module,
    std::vector<expression_graph::DecisionVariable> &decisionVariablesList,
    std::vector<expression_graph::ValueExpression>
        &trackedValueExpressionsList) {
  ConverterToExpressionGraph converter(module);
  converter.buildGraph();
  converter.swapDecisionVariablesList(decisionVariablesList);
  converter.swapTrackedValueExpressionsList(trackedValueExpressionsList);
}

void ConverterToExpressionGraph::buildGraph() {
  const auto &opList = module_->getRegion(0).getOps();
  for (mlir::Operation &op : opList) {
    mlir::TypeSwitch<mlir::Operation *>(&op)
#define OPERATION(T) .Case<T>([this](T &sOp) { actOn(sOp); })
#include "Operations.def"
#undef OPERATION
        .Default([](mlir::Operation *op) {
          llvm_unreachable("Not implemented conversion from operation.");
        });
  }
}

void ConverterToExpressionGraph::calculateCurrentValue(
    model::IntegerDecisionVarDecl &op) {
  currentValues_.emplace(op.getOperation(), op.getValue());
}

void ConverterToExpressionGraph::calculateCurrentValue(
    mlir::arith::ConstantIntOp &op) {
  currentValues_.emplace(op.getOperation(), op.value());
}

void ConverterToExpressionGraph::calculateCurrentValue(
    mlir::arith::MulIOp &op) {
  Integer curValue =
      std::get<Integer>(currentValues_.at(op.getLhs().getDefiningOp())) *
      std::get<Integer>(currentValues_.at(op.getRhs().getDefiningOp()));
  currentValues_.emplace(op.getOperation(), curValue);
}

void ConverterToExpressionGraph::calculateCurrentValue(
    model::IntegerSumOp &op) {
  Integer curValue = 0;
  for (auto var : op.getInputVector()) {
    curValue += std::get<Integer>(currentValues_.at(var.getDefiningOp()));
  }
  currentValues_.emplace(op.getOperation(), curValue);
}

void ConverterToExpressionGraph::actOn(model::IntegerDecisionVarDecl &op) {
  calculateCurrentValue(op);
  auto node = createSimpleNode<Integer>(op);
  decisionVariablesList_.emplace_back(ValueType::INTEGER, node, op.getId(),
                                      op.getLowerBound(), op.getUpperBound(),
                                      graphBuilder_);
}

void ConverterToExpressionGraph::actOn(mlir::arith::MulIOp &op) {
  calculateCurrentValue(op);
  auto result = op.getResult();
  int countUsers = 0;
  bool userIsSum = false;
  for (auto userOp : result.getUsers()) {
    countUsers++;
    if (mlir::isa<model::IntegerSumOp>(userOp)) {
      userIsSum = true;
    }
  }
  if (countUsers != 1 || !userIsSum) {
    auto node = createSimpleNode<Integer>(op);
    if (!isOperationConst(op)) {
      mlir::Value val;
      Integer factor;
      extractValAndFactor(op, val, factor);
      graphBuilder_->createSimpleEdge<MulAddOp<Integer>>(
          opToNodeMap_.at(val.getDefiningOp()), node, factor);
    }
  }
}

void ConverterToExpressionGraph::actOn(mlir::arith::ConstantIntOp &op) {
  calculateCurrentValue(op);
  // We will only create node for constant if it feeds into tracked value
  auto result = op.getResult();
  bool isTracked = false;
  for (auto userOp : result.getUsers()) {
    if (mlir::isa<model::IntegerTrackedVarDecl>(userOp)) {
      isTracked = true;
      break;
    }
  }
  if (isTracked) {
    createSimpleNode<Integer>(op);
  }
}

void ConverterToExpressionGraph::actOn(model::IntegerSumOp &op) {
  calculateCurrentValue(op);
  auto node = createSimpleNode<Integer>(op);
  for (auto inputVar : op.getInputVector()) {
    mlir::Operation *defOperation = inputVar.getDefiningOp();
    if (mlir::isa<mlir::arith::ConstantIntOp>(defOperation)) {
      // This input will not change, so we can ignore the
      // dependence.
      continue;
    }
    auto opToNodeIt = opToNodeMap_.find(defOperation);
    if (mlir::isa<mlir::arith::MulIOp>(defOperation) &&
        opToNodeIt == opToNodeMap_.end()) {
      auto mulOp = mlir::dyn_cast<mlir::arith::MulIOp>(defOperation);
      bool lhsConst = isConst(mulOp.getLhs());
      bool rhsConst = isConst(mulOp.getRhs());
      if (lhsConst && rhsConst) {
        continue;
      }
      int64_t factor;
      mlir::Value var;
      if (lhsConst) {
        factor = getCurrentVal<Integer>(mulOp.getLhs());
        var = mulOp.getRhs();
      } else if (rhsConst) {
        factor = getCurrentVal<Integer>(mulOp.getRhs());
        var = mulOp.getLhs();
      } else {
        llvm::report_fatal_error(
            "Multiplication of two variables is not supported");
      }
      graphBuilder_->createSimpleEdge<MulAddOp<Integer>>(
          opToNodeMap_.at(var.getDefiningOp()), node, factor);
      continue;
    }
    assert(opToNodeIt != opToNodeMap_.end());
    graphBuilder_->createSimpleEdge<AddOp<Integer>>(opToNodeIt->second, node);
  }
}

void ConverterToExpressionGraph::actOn(model::IntegerTrackedVarDecl &op) {
  int64_t id = op.getId();
  auto node = opToNodeMap_.at(op.getValue().getDefiningOp());
  trackedValueExpressionsList_.emplace_back(ValueType::INTEGER, node, id);
}