set(SOURCES main.cu)

add_executable(raytracing ${SOURCES})

target_include_directories(raytracing PUBLIC
  ${PROJECT_SOURCE_DIR}/deps/eigen
  ${PROJECT_SOURCE_DIR}/deps/stb
)

# against eigen warnings
target_compile_options(raytracing PUBLIC --expt-relaxed-constexpr)
