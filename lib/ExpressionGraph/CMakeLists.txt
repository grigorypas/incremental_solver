add_library(ExpressionGraph
  ExpressionGraph.cpp
  )

target_link_libraries(ExpressionGraph
LLVMSupport
)