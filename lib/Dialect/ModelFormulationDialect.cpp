#include "IncrementalSolver/Dialect/ModelFormulationDialect.h"
#include "IncrementalSolver/Dialect/ModelFormulationOps.h"

using namespace plco::model;

#include "IncrementalSolver/Dialect/ModelFormulationOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Model Formulation dialect.
//===----------------------------------------------------------------------===//

void ISModelFormulationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IncrementalSolver/Dialect/ModelFormulationOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "IncrementalSolver/Dialect/ModelFormulationOps.cpp.inc"