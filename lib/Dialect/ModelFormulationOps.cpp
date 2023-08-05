#include "IncrementalSolver/Dialect/ModelFormulationOps.h"
#include "IncrementalSolver/Dialect/ModelFormulationDialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace plco;

static bool isTrueConst(mlir::Value val) {
  if (!val.getType().isInteger(1)) {
    return false;
  }
  auto op = val.getDefiningOp();

  if (auto const_op = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
    return bool(const_op.getValue());
  }
  return false;
}

struct EliminateConstantsFromAnyOp
    : public mlir::OpRewritePattern<model::AnyOp> {
  EliminateConstantsFromAnyOp(mlir::MLIRContext *context)
      : OpRewritePattern<model::AnyOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(model::AnyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    for (auto operand : op.getInputVector()) {
      if (isTrueConst(operand)) {
        rewriter.replaceOp(op, {operand});
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void model::AnyOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                               mlir::MLIRContext *context) {
  results.add<EliminateConstantsFromAnyOp>(context);
}

#include "IncrementalSolver/Dialect/ModelFormulationOpsEnums.cpp.inc"