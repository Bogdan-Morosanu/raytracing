include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  raytest
  triangletest.cu
  spheretest.cu
  ../src/check_error.cu
)

set_target_properties(raytest PROPERTIES CUDA_ARCHITECTURES "75")

target_include_directories(
  raytest
  PUBLIC
  ../include
  ../deps/eigen/
)

target_link_libraries(
  raytest
  GTest::gtest_main
)

# against eigen warnings
target_compile_options(raytest PUBLIC --expt-relaxed-constexpr)

include(GoogleTest)
gtest_discover_tests(raytest)
