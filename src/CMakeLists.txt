set(SOURCES main.cu check_error.cu)

add_executable(raytracing ${SOURCES})

set_target_properties(raytracing PROPERTIES CUDA_ARCHITECTURES "75")

find_package(CUDAToolkit)

target_include_directories(raytracing PUBLIC
  ${PROJECT_SOURCE_DIR}/deps/eigen
  ${PROJECT_SOURCE_DIR}/deps/stb
  ${PROJECT_SOURCE_DIR}/include
)

# against eigen warnings
target_compile_options(raytracing PUBLIC --expt-relaxed-constexpr)

target_link_libraries(raytracing PUBLIC CUDA::curand)


