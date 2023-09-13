#ifndef INCREMENTALSOLVER_BINDINGS_MODELBUILDERWRAPPER_H
#define INCREMENTALSOLVER_BINDINGS_MODELBUILDERWRAPPER_H
#include "IncrementalSolver/ExpressionGraph/interface.h"
#include "IncrementalSolver/ModelBuilder/ModelFormulationBuilder.h"

#include <vector>

namespace incremental_solver {
class ModelBuilderWrapper {
  std::vector<mlir::Value> values_;
  ModelFormulationBuilder builder_;
  std::vector<expression_graph::DecisionVariable> decisionVariables_;
  std::vector<expression_graph::ValueExpression> trackedExpressions_;

public:
  ModelBuilderWrapper() {}
  void compile();
  void printMlir();
  std::vector<expression_graph::DecisionVariable> getDecisionVariables() const {
    return decisionVariables_;
  }

  std::vector<expression_graph::ValueExpression> getTrackedExpressions() const {
    return trackedExpressions_;
  }

  size_t addIntegerVarDecl(Integer initialValue, int64_t id,
                           std::optional<Integer> lowerBound = std::nullopt,
                           std::optional<Integer> upperBound = std::nullopt) {
    values_.emplace_back(
        builder_.addIntegerVarDecl(initialValue, id, lowerBound, upperBound));
    return curInd();
  }

  size_t addDoubleVarDecl(Double initialValue, int64_t id,
                          std::optional<Double> lowerBound = std::nullopt,
                          std::optional<Double> upperBound = std::nullopt) {
    values_.emplace_back(
        builder_.addDoubleVarDecl(initialValue, id, lowerBound, upperBound));
    return curInd();
  }

  void markAsTracked(int64_t id) { builder_.markAsTracked(id); }

  size_t emitIntegerConstAssignment(Integer value,
                                    std::optional<int64_t> id = std::nullopt) {
    values_.emplace_back(builder_.emitIntegerConstAssignment(value, id));
    return curInd();
  }

  size_t emitDoubleConstAssignment(Double value,
                                   std::optional<int64_t> id = std::nullopt) {
    values_.emplace_back(builder_.emitDoubleConstAssignment(value, id));
    return curInd();
  }
  size_t emitAnyFunc(std::vector<size_t> inputValues,
                     std::optional<int64_t> id = std::nullopt) {
    values_.emplace_back(
        builder_.emitAnyFunc(convertToIndeces(inputValues), id));
    return curInd();
  }

  size_t emitIntegerSum(std::vector<size_t> inputValues,
                        std::optional<int64_t> id = std::nullopt) {
    values_.emplace_back(
        builder_.emitIntegerSum(convertToIndeces(inputValues), id));
    return curInd();
  }

  size_t emitDoubleSum(std::vector<size_t> inputValues,
                       std::optional<int64_t> id = std::nullopt) {
    values_.emplace_back(
        builder_.emitDoubleSum(convertToIndeces(inputValues), id));
    return curInd();
  }
  size_t emitIntegerMultiply(size_t var, int64_t factor,
                             std::optional<int64_t> id = std::nullopt) {
    values_.emplace_back(
        builder_.emitIntegerMultiply(values_[var], factor, id));
    return curInd();
  }

  size_t emitDoubleMultiply(size_t var, double factor,
                            std::optional<int64_t> id = std::nullopt) {
    values_.emplace_back(builder_.emitDoubleMultiply(values_[var], factor, id));
    return curInd();
  }

private:
  size_t curInd() const { return values_.size() - 1; }
  std::vector<mlir::Value>
  convertToIndeces(const std::vector<size_t> &inputVec) const {
    std::vector<mlir::Value> valueVector;
    valueVector.reserve(inputVec.size());
    for (size_t ind : inputVec) {
      valueVector.emplace_back(values_[ind]);
    }
    return valueVector;
  }
};

} // namespace incremental_solver

#endif /* INCREMENTALSOLVER_BINDINGS_MODELBUILDERWRAPPER_H */
