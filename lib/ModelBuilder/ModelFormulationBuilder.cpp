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
  auto v = registerVariable(
      builder_.create<incremental_solver::model::IntegerDecisionVarDecl>(
          builder_.getUnknownLoc(), mlirIntegerType_, initialValue, id,
          getIntLowerBound(lowerBound), getIntUpperBound(upperBound)),
      id);
  return setBounds(v, lowerBound, upperBound);
}

mlir::Value
ModelFormulationBuilder::addDoubleVarDecl(Double initialValue, int64_t id,
                                          std::optional<Double> lowerBound,
                                          std::optional<Double> upperBound) {
  auto v = registerVariable(
      builder_.create<incremental_solver::model::DoubleDecisionVarDecl>(
          builder_.getUnknownLoc(), mlirDoubleType_,
          builder_.getF64FloatAttr(initialValue),
          builder_.getI64IntegerAttr(id),
          builder_.getF64FloatAttr(getDoubleLowerBound(lowerBound)),
          builder_.getF64FloatAttr(getDoubleUpperBound(upperBound))),
      id);
  return setBounds(v, lowerBound, upperBound);
}

mlir::Value
ModelFormulationBuilder::emitCastToDouble(mlir::Value val,
                                          std::optional<int64_t> id) {
  assert(val.getType().isInteger(64) && "Input value must be integer");
  auto v = registerVariable(
      builder_.create<incremental_solver::model::CastToDoubleOp>(
          builder_.getUnknownLoc(), val),
      id);
  std::optional<Integer> valLb = getLowerBound<Integer>(val);
  std::optional<Integer> valUb = getUpperBound<Integer>(val);
  std::optional<Double> lb, ub;
  if (valLb) {
    lb = valLb.value();
  }
  if (valUb) {
    ub = valUb.value();
  }
  return setBounds(v, lb, ub);
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
  auto v =
      registerVariable(builder_.create<mlir::arith::ConstantIntOp>(
                           builder_.getUnknownLoc(), value, mlirIntegerType_),
                       id);
  return setBounds<Integer>(v, value, value);
}

mlir::Value
ModelFormulationBuilder::emitDoubleConstAssignment(Double value,
                                                   std::optional<int64_t> id) {
  auto v = registerVariable(
      builder_.create<mlir::arith::ConstantFloatOp>(
          builder_.getUnknownLoc(), llvm::APFloat(value), mlirDoubleType_),
      id);
  return setBounds<Double>(v, value, value);
}
mlir::Value
ModelFormulationBuilder::emitAnyFunc(std::vector<mlir::Value> inputValues,
                                     std::optional<int64_t> id) {
  auto v = registerVariable(builder_.create<incremental_solver::model::AnyOp>(
                                builder_.getUnknownLoc(), builder_.getI64Type(),
                                std::move(inputValues), 0, 1),
                            id);
  return setBounds<Integer>(v, 0, 1);
}

mlir::Value
ModelFormulationBuilder::emitIntegerSum(std::vector<mlir::Value> inputValues,
                                        std::optional<int64_t> id) {
  std::optional<Integer> lb{0}, ub{0};
  for (auto &v : inputValues) {
    auto curLb = getLowerBound<Integer>(v);
    if (!curLb) {
      lb = std::nullopt;
      break;
    }
    lb = lb.value() + curLb.value();
  }

  for (auto &v : inputValues) {
    auto curUb = getUpperBound<Integer>(v);
    if (!curUb) {
      ub = std::nullopt;
      break;
    }
    ub = ub.value() + curUb.value();
  }

  auto v = registerVariable(
      builder_.create<incremental_solver::model::IntegerSumOp>(
          builder_.getUnknownLoc(), mlirIntegerType_, std::move(inputValues),
          getIntLowerBound(lb), getIntUpperBound(ub)),
      id);
  return setBounds(v, lb, ub);
}

mlir::Value
ModelFormulationBuilder::emitDoubleSum(std::vector<mlir::Value> inputValues,
                                       std::optional<int64_t> id) {
  for (size_t ind = 0; ind < inputValues.size(); ++ind) {
    if (inputValues[ind].getType().isInteger(64)) {
      inputValues[ind] = emitCastToDouble(inputValues[ind]);
    }
  }

  std::optional<Double> lb{0}, ub{0};
  for (auto &v : inputValues) {
    auto curLb = getLowerBound<Double>(v);
    if (!curLb) {
      lb = std::nullopt;
      break;
    }
    lb = lb.value() + curLb.value();
  }

  for (auto &v : inputValues) {
    auto curUb = getUpperBound<Double>(v);
    if (!curUb) {
      ub = std::nullopt;
      break;
    }
    ub = ub.value() + curUb.value();
  }
  auto v = registerVariable(
      builder_.create<incremental_solver::model::DoubleSumOp>(
          builder_.getUnknownLoc(), mlirDoubleType_, std::move(inputValues),
          llvm::APFloat(getDoubleLowerBound(lb)),
          llvm::APFloat(getDoubleUpperBound(ub))),
      id);
  return setBounds(v, lb, ub);
}

mlir::Value
ModelFormulationBuilder::emitIntegerMultiply(mlir::Value var, int64_t factor,
                                             std::optional<int64_t> id) {
  std::optional<Integer> lb = getLowerBound<Integer>(var);
  if (lb) {
    lb = lb.value() * factor;
  }
  std::optional<Integer> ub = getLowerBound<Integer>(var);
  if (ub) {
    ub = ub.value() * factor;
  }
  auto vFactor = emitIntegerConstAssignment(factor);
  auto op = builder_.create<mlir::arith::MulIOp>(builder_.getUnknownLoc(), var,
                                                 vFactor);
  return setBounds(registerVariable(op, id), lb, ub);
}

mlir::Value
ModelFormulationBuilder::emitDoubleMultiply(mlir::Value var, double factor,
                                            std::optional<int64_t> id) {
  std::optional<Double> lb = getLowerBound<Double>(var);
  if (lb) {
    lb = lb.value() * factor;
  }
  std::optional<Double> ub = getLowerBound<Double>(var);
  if (ub) {
    ub = ub.value() * factor;
  }
  auto vFactor = emitDoubleConstAssignment(factor);
  if (var.getType().isInteger(64)) {
    var = emitCastToDouble(var);
  }
  auto v = registerVariable(builder_.create<mlir::arith::MulFOp>(
                                builder_.getUnknownLoc(), var, vFactor),
                            id);
  return setBounds(v, lb, ub);
}