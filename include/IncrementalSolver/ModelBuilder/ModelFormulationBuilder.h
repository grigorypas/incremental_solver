#ifndef incremental_solver_BUILDER_MODELFORMULATIONBUILDER_H
#define incremental_solver_BUILDER_MODELFORMULATIONBUILDER_H
#include "IncrementalSolver/Common/Types.h"
#include "IncrementalSolver/Dialect/ModelFormulationOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/FormatVariadic.h"

#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace incremental_solver {

class ModelFormulationBuilder {
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  std::unordered_map<int, mlir::Value> variables_;
  mlir::IntegerType mlirIntegerType_;
  mlir::FloatType mlirDoubleType_;

public:
  ModelFormulationBuilder(mlir::MLIRContext &context) : builder_(&context) {
    module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToEnd(module_.getBody());
    mlirIntegerType_ = builder_.getI64Type();
    mlirDoubleType_ = builder_.getF64Type();
  }
  mlir::OwningOpRef<mlir::ModuleOp> getModule() { return module_; }
  mlir::Value getValueFromId(int64_t id);

  mlir::Value
  addIntegerVarDecl(Integer initialValue, int64_t id,
                    std::optional<Integer> lowerBound = std::nullopt,
                    std::optional<Integer> upperBound = std::nullopt);

  mlir::Value addDoubleVarDecl(Double initialValue, int64_t id,
                               std::optional<Double> lowerBound = std::nullopt,
                               std::optional<Double> upperBound = std::nullopt);

  void makeIntegerVarTracked(int64_t id);
  void makeDoubleVarTracked(int64_t id);

  mlir::Value
  emitIntegerConstAssignment(Integer value,
                             std::optional<int64_t> id = std::nullopt);

  mlir::Value
  emitDoubleConstAssignment(Double value,
                            std::optional<int64_t> id = std::nullopt);
  mlir::Value emitAnyFunc(std::vector<mlir::Value> inputValues,
                          std::optional<int64_t> id = std::nullopt);

  mlir::Value emitIntegerSum(std::vector<mlir::Value> inputValues,
                             std::optional<int64_t> id = std::nullopt);

  mlir::Value emitDoubleSum(std::vector<mlir::Value> inputValues,
                            std::optional<int64_t> id = std::nullopt);
  mlir::Value emitIntegerMultiply(mlir::Value var, int64_t factor,
                                  std::optional<int64_t> id = std::nullopt);

  mlir::Value emitDoubleMultiply(mlir::Value var, double factor,
                                 std::optional<int64_t> id = std::nullopt);

private:
  Integer getIntLowerBound(std::optional<Integer> lb) {
    return lb ? lb.value() : INTEGER_MIN;
  }
  Integer getIntUpperBound(std::optional<Integer> ub) {
    return ub ? ub.value() : INTEGER_MAX;
  }
  Double getDoubleLowerBound(std::optional<Double> lb) {
    return lb ? lb.value() : DOUBLE_MIN;
  }
  Double getDoubleUpperBound(std::optional<Double> ub) {
    return ub ? ub.value() : DOUBLE_MAX;
  }
  mlir::Value registerVariable(mlir::Value val,
                               std::optional<int64_t> id = std::nullopt);
};

} // namespace incremental_solver
#endif /* incremental_solver_BUILDER_MODELFORMULATIONBUILDER_H */
