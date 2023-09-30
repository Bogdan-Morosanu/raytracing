#pragma once

#include <cstdlib>

#include <curand_kernel.h>

#include "check_error.cuh"
#include "work_partition.cuh"

namespace rt {

  namespace internal {
    __global__ void curand_init_state(std::size_t num_states, curandState *device_rand_states)
    {
      int i = threadIdx.x + blockIdx.x * blockDim.x;
      if (i > num_states) {
	return;
      }

      curand_init(1984, i, 0, &device_rand_states[i]);
    }
  }
  
  class CurandState {
  public:
    explicit
    CurandState(std::size_t num_states)
      : num_states_(num_states)
    {
      checkCudaErrors(cudaMalloc((void**)&device_rand_state_,
				 num_states * sizeof(curandState)));

      std::size_t threads{128};
      auto work_partition = make_work_partition(num_states, threads);
      std::size_t blocks{work_partition.elements_per_thread};
      std::printf("rand init job running on (%lu, %lu)\n", blocks, threads);
      internal::curand_init_state<<<blocks, threads>>>(num_states_, device_rand_state_);
      checkCudaErrors(cudaDeviceSynchronize());
    }


    __device__ curandState *state(std::size_t i)
    {
      assert(i < num_states_);
      return &device_rand_state_[i];
    }

    ~CurandState() {
      checkCudaErrors(cudaFree(device_rand_state_));
    }
  private:
    std::size_t num_states_;
    curandState *device_rand_state_;
  };
}
