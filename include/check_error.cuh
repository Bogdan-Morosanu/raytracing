#pragma once

#define checkCudaErrors(val) rt::internal::check_cuda_error((val), #val, __FILE__, __LINE__)

namespace rt {
  namespace internal {
        void check_cuda_error(cudaError_t result, const char *const val,
			      const char *const file, int line);
  }
}
