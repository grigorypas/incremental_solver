pybind11_add_module(isbindings 
    incremental_solver.cpp
    variables_interface.cpp
    ModelBuilderWrapper.cpp 
)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(isbindings
PRIVATE
${conversion_libs}
ExpressionGraph
modelConversion
ModelFormulationBuilder
)

install(TARGETS isbindings DESTINATION .)