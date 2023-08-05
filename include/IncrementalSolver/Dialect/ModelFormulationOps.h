#ifndef MLIR_IR_MODELFORMULATIONOPS_H
#define MLIR_IR_MODELFORMULATIONOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "IncrementalSolver/Dialect/ModelFormulationOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "IncrementalSolver/Dialect/ModelFormulationOps.h.inc"

#endif /* MLIR_IR_MODELFORMULATIONOPS_H */
