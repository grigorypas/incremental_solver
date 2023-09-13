#include "IncrementalSolver/Conversion/ConvertToExpressionGraph.h"
#include "IncrementalSolver/ExpressionGraph/interface.h"
#include "IncrementalSolver/ModelBuilder/ModelFormulationBuilder.h"

#include <gtest/gtest.h>
#include <unordered_map>

using namespace incremental_solver::expression_graph;
using namespace incremental_solver;

template <typename T>
std::unordered_map<int64_t, T> convertToMap(const std::vector<T> input) {
  std::unordered_map<int64_t, T> result;
  for (const auto &var : input) {
    result.emplace(var.getId(), var);
  }
  return result;
}

TEST(Conversion, TrackEqualInput) {
  ModelFormulationBuilder modelBuilder;
  modelBuilder.addIntegerVarDecl(10, 1);
  modelBuilder.makeIntegerVarTracked(1);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(modelBuilder.getModule(), vars, expressions);
  ASSERT_EQ(vars.size(), 1);
  ASSERT_EQ(expressions.size(), 1);
  auto &var = vars[0];
  auto &expr = expressions[0];
  ASSERT_EQ(expr.getIntValue(), 10);
  var.setIntValue(15);
  ASSERT_EQ(expr.getIntValue(), 15);
}

TEST(Conversion, SumInt) {
  ModelFormulationBuilder modelBuilder;
  auto v1 = modelBuilder.addIntegerVarDecl(10, 1);
  auto v2 = modelBuilder.addIntegerVarDecl(15, 2);
  modelBuilder.emitIntegerSum({v1, v2}, 3);
  modelBuilder.makeIntegerVarTracked(3);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(modelBuilder.getModule(), vars, expressions);
  auto varMap = convertToMap(vars);
  ASSERT_EQ(expressions.size(), 1);
  auto &expr = expressions[0];
  ASSERT_EQ(expr.getIntValue(), 25);
}

TEST(Conversion, SumWithConst) {
  ModelFormulationBuilder modelBuilder;
  auto v1 = modelBuilder.addIntegerVarDecl(10, 0);
  auto v2 = modelBuilder.addIntegerVarDecl(5, 1);
  auto v2_mult = modelBuilder.emitIntegerMultiply(v2, 3);
  auto c = modelBuilder.emitIntegerConstAssignment(7);
  modelBuilder.emitIntegerSum({v1, v2_mult, c}, 2);
  modelBuilder.markAsTracked(2);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(modelBuilder.getModule(), vars, expressions);
  auto &expr = expressions[0];
  ASSERT_EQ(expr.getIntValue(), 32);
}

TEST(Conversion, DoubleVar) {
  ModelFormulationBuilder modelBuilder;
  modelBuilder.addDoubleVarDecl(12.5, 0);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(modelBuilder.getModule(), vars, expressions);
  ASSERT_EQ(vars.size(), 1);
  auto &var = vars[0];
  ASSERT_EQ(var.getDoubleValue(), 12.5);
  var.setDoubleValue(20.3);
  ASSERT_EQ(var.getDoubleValue(), 20.3);
}

TEST(Converstion, DoubleMultiply) {
  ModelFormulationBuilder modelBuilder;
  auto v1 = modelBuilder.addDoubleVarDecl(10.5, 0);
  modelBuilder.emitDoubleMultiply(v1, 2.0, 1);
  modelBuilder.markAsTracked(1);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(modelBuilder.getModule(), vars, expressions);
  auto &expr = expressions[0];
  ASSERT_NEAR(expr.getDoubleValue(), 21.0, 0.001);
}

TEST(Converstion, IntVarMultiplyByDouble) {
  ModelFormulationBuilder modelBuilder;
  auto v1 = modelBuilder.addIntegerVarDecl(10, 0);
  modelBuilder.emitDoubleMultiply(v1, 0.5, 1);
  modelBuilder.markAsTracked(1);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(modelBuilder.getModule(), vars, expressions);
  auto &expr = expressions[0];
  ASSERT_NEAR(expr.getDoubleValue(), 5.0, 0.001);
}

TEST(Conversion, DoubleSum) {
  ModelFormulationBuilder modelBuilder;
  auto v1 = modelBuilder.addIntegerVarDecl(5, 0);
  auto v2 = modelBuilder.addIntegerVarDecl(7, 1);
  modelBuilder.emitDoubleSum({modelBuilder.emitDoubleMultiply(v1, 0.5),
                              modelBuilder.emitDoubleMultiply(v2, 0.5)},
                             2);
  modelBuilder.markAsTracked(2);
  std::vector<DecisionVariable> vars;
  std::vector<ValueExpression> expressions;
  convertToExpressionGraph(modelBuilder.getModule(), vars, expressions);
  auto &expr = expressions[0];
  ASSERT_NEAR(expr.getDoubleValue(), 6.0, 0.01);
  vars[0].setIntValue(6);
  ASSERT_NEAR(expr.getDoubleValue(), 6.5, 0.01);
}