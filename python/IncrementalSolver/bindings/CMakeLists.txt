pybind11_add_module(isbindings 
    incremental_solver.cpp
    variables_interface.cpp
    ModelBuilderWrapper.cpp 
)

target_link_libraries(isbindings
PRIVATE
ExpressionGraph
modelConversion
ModelFormulationBuilder
)