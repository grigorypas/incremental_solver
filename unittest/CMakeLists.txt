include (${CMAKE_CURRENT_SOURCE_DIR}/googletest.cmake)

include(GoogleTest)
enable_testing()

function(add_gtest_ut)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS LIBS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${__NAME}
          COMMAND ${__NAME})
  add_executable(
          ${__NAME}
          ${__SRCS})
  target_link_libraries(
          ${__NAME}
          PRIVATE
          GTest::gtest_main
          gmock
          ${__LIBS})

  gtest_discover_tests(${__NAME})
endfunction()

add_subdirectory(ExpressionGraph)
add_subdirectory(Conversion)

