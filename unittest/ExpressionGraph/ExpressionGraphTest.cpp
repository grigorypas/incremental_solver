#include "IncrementalSolver/ExpressionGraph/ExpressionGraph.h"
#include "IncrementalSolver/ExpressionGraph/Expressions.h"

#include <gtest/gtest.h>
using namespace incremental_solver::expression_graph;
using namespace incremental_solver;

TEST(ExpressionGraph, LinearExpressionOp) {
  ExpressionGraphBuilder graphBuilder;
  auto intVar1 = graphBuilder.createNode<Node>((Integer)5);
  auto intVar2 = graphBuilder.createNode<Node>((Integer)10);
  graphBuilder.createSimpleEdge<MulAddOp<Integer>>(intVar1, intVar2, 4);
  EXPECT_EQ(intVar2->getNewValue<Integer>(), 10);
  graphBuilder.setValue(intVar1, (Integer)6);
  EXPECT_EQ(intVar2->getNewValue<Integer>(), 14);
  EXPECT_EQ(intVar1->getOldValue<Integer>(), 6);
  EXPECT_EQ(intVar2->getOldValue<Integer>(), 14);
}

TEST(ExpressionGraph, AnyOp) {
  ExpressionGraphBuilder graphBuilder;
  auto intVar1 = graphBuilder.createNode<Node>((Integer)0);
  auto intVar2 = graphBuilder.createNode<Node>((Integer)1);
  auto intVar3 = graphBuilder.createNode<Node>((Integer)0);
  auto intVar4 = graphBuilder.createNode<Node>((Integer)1);
  auto result = graphBuilder.createNode<StorageNode<Integer>>(
      /*currentValue=*/(Integer)1, /*storageValue=*/(Integer)2);
  graphBuilder.createSimpleEdge<AnyOp>(intVar1, result);
  graphBuilder.createSimpleEdge<AnyOp>(intVar2, result);
  graphBuilder.createSimpleEdge<AnyOp>(intVar3, result);
  graphBuilder.createSimpleEdge<AnyOp>(intVar4, result);
  EXPECT_EQ(result->getNewValue<Integer>(), 1);
  graphBuilder.setValue(intVar1, (Integer)1);
  EXPECT_EQ(result->getNewValue<Integer>(), 1);
  graphBuilder.setValue(intVar4, (Integer)0);
  EXPECT_EQ(result->getNewValue<Integer>(), 1);
  graphBuilder.setValue(intVar1, (Integer)0);
  graphBuilder.setValue(intVar2, (Integer)0);
  EXPECT_EQ(result->getNewValue<Integer>(), 0);
}