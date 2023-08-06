#ifndef INCREMENTALSOLVER_CONVERSION_CONVERTTOEXPRESSIONGRAPH_H
#define INCREMENTALSOLVER_CONVERSION_CONVERTTOEXPRESSIONGRAPH_H
#include "IncrementalSolver/ExpressionGraph/interface.h"

#include <vector>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace incremental_solver {

void convertToExpressionGraph(
    const mlir::ModuleOp &module,
    std::vector<expression_graph::DecisionVariable> &decisionVariablesList,
    std::vector<expression_graph::ValueExpression>
        &trackedValueExpressionsList);

} // namespace incremental_solver

#endif /* INCREMENTALSOLVER_CONVERSION_CONVERTTOEXPRESSIONGRAPH_H */
