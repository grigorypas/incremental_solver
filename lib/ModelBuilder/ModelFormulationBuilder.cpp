#include "IncrementalSolver/ModelBuilder/ModelFormulationBuilder.h"

using namespace incremental_solver;

mlir::Value
ModelFormulationBuilder::registerVariable(mlir::Value val,
                                          std::optional<int64_t> id) {
  if (id) {
    int64_t idVal = *id;
    if (variables_.find(idVal) != variables_.end()) {
      llvm::report_fatal_error(
          llvm::formatv("Variable with id {0} already registered", idVal));
    }
    variables_.emplace(idVal, val);
  }
  return val;
}

mlir::Value ModelFormulationBuilder::getValueFromId(int64_t id) {
  auto it = variables_.find(id);
  if (it == variables_.end()) {
    llvm::report_fatal_error(
        llvm::formatv("Variable with id {0} has not beed declared", id));
  }
  return it->second;
}

mlir::Value
ModelFormulationBuilder::addIntegerVarDecl(Integer initialValue, int64_t id,
                                           std::optional<Integer> lowerBound,
                                           std::optional<Integer> upperBound) {
  return registerVariable(
      builder_.create<incremental_solver::model::IntegerDecisionVarDecl>(
          builder_.getUnknownLoc(), mlirIntegerType_, initialValue, id,
          getIntLowerBound(lowerBound), getIntUpperBound(upperBound)),
      id);
}

mlir::Value
ModelFormulationBuilder::addDoubleVarDecl(Double initialValue, int64_t id,
                                          std::optional<Double> lowerBound,
                                          std::optional<Double> upperBound) {
  return registerVariable(
      builder_.create<incremental_solver::model::DoubleDecisionVarDecl>(
          builder_.getUnknownLoc(), mlirDoubleType_,
          builder_.getF64FloatAttr(initialValue),
          builder_.getI64IntegerAttr(id),
          builder_.getF64FloatAttr(getDoubleLowerBound(lowerBound)),
          builder_.getF64FloatAttr(getDoubleUpperBound(upperBound))),
      id);
}

mlir::Value
ModelFormulationBuilder::emitCastToDouble(mlir::Value val,
                                          std::optional<int64_t> id) {
  assert(val.getType().isInteger(64) && "Input value must be integer");
  return registerVariable(
      builder_.create<incremental_solver::model::CastToDoubleOp>(
          builder_.getUnknownLoc(), val),
      id);
}

void ModelFormulationBuilder::makeIntegerVarTracked(int64_t id) {
  auto value = getValueFromId(id);
  builder_.create<incremental_solver::model::IntegerTrackedVarDecl>(
      builder_.getUnknownLoc(), value, id);
}
void ModelFormulationBuilder::makeDoubleVarTracked(int64_t id) {
  auto value = getValueFromId(id);
  builder_.create<incremental_solver::model::DoubleTrackedVarDecl>(
      builder_.getUnknownLoc(), value, id);
}

void ModelFormulationBuilder::markAsTracked(int64_t id) {
  auto value = getValueFromId(id);
  if (value.getType().isInteger(64)) {
    makeIntegerVarTracked(id);
  } else {
    makeDoubleVarTracked(id);
  }
}

mlir::Value
ModelFormulationBuilder::emitIntegerConstAssignment(Integer value,
                                                    std::optional<int64_t> id) {
  return registerVariable(
      builder_.create<mlir::arith::ConstantIntOp>(builder_.getUnknownLoc(),
                                                  value, mlirIntegerType_),
      id);
}

mlir::Value
ModelFormulationBuilder::emitDoubleConstAssignment(Double value,
                                                   std::optional<int64_t> id) {
  return registerVariable(
      builder_.create<mlir::arith::ConstantFloatOp>(
          builder_.getUnknownLoc(), llvm::APFloat(value), mlirDoubleType_),
      id);
}
mlir::Value
ModelFormulationBuilder::emitAnyFunc(std::vector<mlir::Value> inputValues,
                                     std::optional<int64_t> id) {
  return registerVariable(builder_.create<incremental_solver::model::AnyOp>(
                              builder_.getUnknownLoc(), builder_.getI64Type(),
                              std::move(inputValues)),
                          id);
}

mlir::Value
ModelFormulationBuilder::emitIntegerSum(std::vector<mlir::Value> inputValues,
                                        std::optional<int64_t> id) {
  return registerVariable(
      builder_.create<incremental_solver::model::IntegerSumOp>(
          builder_.getUnknownLoc(), mlirIntegerType_, std::move(inputValues)),
      id);
}

mlir::Value
ModelFormulationBuilder::emitDoubleSum(std::vector<mlir::Value> inputValues,
                                       std::optional<int64_t> id) {
  return registerVariable(
      builder_.create<incremental_solver::model::DoubleSumOp>(
          builder_.getUnknownLoc(), mlirDoubleType_, std::move(inputValues)),
      id);
}

mlir::Value
ModelFormulationBuilder::emitIntegerMultiply(mlir::Value var, int64_t factor,
                                             std::optional<int64_t> id) {
  auto vFactor = emitIntegerConstAssignment(factor);
  auto op = builder_.create<mlir::arith::MulIOp>(builder_.getUnknownLoc(), var,
                                                 vFactor);
  return registerVariable(op, id);
}

mlir::Value
ModelFormulationBuilder::emitDoubleMultiply(mlir::Value var, double factor,
                                            std::optional<int64_t> id) {
  auto vFactor = emitDoubleConstAssignment(factor);
  if (var.getType().isInteger(64)) {
    var = emitCastToDouble(var);
  }
  return registerVariable(builder_.create<mlir::arith::MulFOp>(
                              builder_.getUnknownLoc(), var, vFactor),
                          id);
}