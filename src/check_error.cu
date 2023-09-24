#include <cstdio>
#include <cstdlib>

namespace rt {
  namespace internal {

    void check_cuda_error(cudaError_t result, const char *const val,
			  const char *const file, int line)
    {
      if (result) {
	std::fprintf(stderr, "CUDA error: %d (%s) at %s:%d\n", (int)result, val, file, line);
	cudaDeviceReset();
	std::exit(EXIT_FAILURE);
      }
    }
  }
}
