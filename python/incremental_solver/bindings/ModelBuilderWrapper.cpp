#include "ModelBuilderWrapper.h"
#include "IncrementalSolver/Conversion/ConvertToExpressionGraph.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace incremental_solver;

void ModelBuilderWrapper::compile() {
  const auto &mlirModule = builder_.getModule();
  mlir::PassManager pm(mlirModule->getName());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(mlirModule)))
    llvm::report_fatal_error("Failed to apply optimization.");
  convertToExpressionGraph(builder_.getModule(), decisionVariables_,
                           trackedExpressions_);
}

void incremental_solver::ModelBuilderWrapper::printMlir() {
  builder_.getModule()->dump();
}

void init_model_builder(py::module &m) {
  py::class_<ModelBuilderWrapper>(m, "ModelBuilder")
      .def(py::init())
      .def("compile", &ModelBuilderWrapper::compile)
      .def("get_decision_variables", &ModelBuilderWrapper::getDecisionVariables)
      .def("get_tracked_expressions",
           &ModelBuilderWrapper::getTrackedExpressions)
      .def("add_integer_var_decl", &ModelBuilderWrapper::addIntegerVarDecl)
      .def("add_double_var_decl", &ModelBuilderWrapper::addDoubleVarDecl)
      .def("mark_as_tracked", &ModelBuilderWrapper::markAsTracked)
      .def("emit_integer_const_assignment",
           &ModelBuilderWrapper::emitIntegerConstAssignment)
      .def("emit_double_const_assignment",
           &ModelBuilderWrapper::emitDoubleConstAssignment)
      .def("emit_any", &ModelBuilderWrapper::emitAnyFunc)
      .def("emit_integer_sum", &ModelBuilderWrapper::emitIntegerSum)
      .def("emit_double_sum", &ModelBuilderWrapper::emitDoubleSum)
      .def("emit_integer_multiply", &ModelBuilderWrapper::emitIntegerMultiply)
      .def("emit_double_multiply", &ModelBuilderWrapper::emitDoubleMultiply)
      .def("print_mlir", &ModelBuilderWrapper::printMlir);
}