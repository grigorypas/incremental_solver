#ifndef IncrementalSolver_MODEL_DIALECT
#define IncrementalSolver_MODEL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// PLCO dialect definition.
//===----------------------------------------------------------------------===//

def IS_ModelFormulationDialect : Dialect {
    let name = "ModelFormulation";
    let summary = "A PLCO model formulation dialect.";
    let description = [{
        This dialect is desing to express model.
    }];
    let cppNamespace = "incremental_solver::model";

    //let useDefaultTypePrinterParser = 1;
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::math::MathDialect"
  ];
}

// The predicate indicates the type of the comparison to perform
def IS_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "Comparison operator",
    [
      I64EnumAttrCase<"EQ", 0>,
      I64EnumAttrCase<"NEQ", 1>,
      I64EnumAttrCase<"LEQ", 2>,
      I64EnumAttrCase<"GEQ", 3>,
    ]> {
   let cppNamespace = "incremental_solver::model";
}


#endif // IncrementalSolver_MODEL_DIALECT