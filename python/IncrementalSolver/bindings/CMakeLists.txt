pybind11_add_module(ISBindings 
    incremental_solver.cpp
    variables_interface.cpp
    ModelBuilderWrapper.cpp 
)

target_link_libraries(ISBindings
PRIVATE
ExpressionGraph
)