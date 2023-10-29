#pragma once

#include <cstdlib>

#include <curand_kernel.h>

#include "cuda_utils/check_error.cuh"
#include "cuda_utils/work_partition.cuh"

#include <Eigen/Dense>

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

  class CurandStateView;
  
  class CurandState {
    friend class CurandStateView;

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

  class CurandStateView {
  public:
    CurandStateView(CurandState &states)
      : num_states_(states.num_states_)
      , device_rand_state_(states.device_rand_state_)
    { }

    __device__ curandState *state(std::size_t i)
    {
      assert(i < num_states_);
      return &device_rand_state_[i];
    }

  private:
    std::size_t num_states_;
    curandState *device_rand_state_;
  };

  __host__ inline CurandStateView make_view(CurandState &states)
  {
    return CurandStateView{states};
  }

  /// vector coords are in [0.0, 1.0] range
  __device__ inline Eigen::Vector3f random_vector(curandState *state)
  {
    return Eigen::Vector3f(curand_uniform(state),
			   curand_uniform(state),
			   curand_uniform(state));
  }

  /// vector contained within unit sphere centered at (0, 0, 0)
  __device__ inline Eigen::Vector3f random_in_unit_sphere(curandState *state)
  {
    const Eigen::Vector3f original_center{0.5f, 0.5f, 0.5f};

    auto in_sphere = [&](const Eigen::Vector3f &v)
		     {
		       Eigen::Vector3f v_centered = v - original_center;
		       return v_centered.dot(v_centered) <= 1.0f;
		     };

    auto v = random_vector(state);
    while (!in_sphere(v)) {
      v = random_vector(state);
    }

    return v;
  }
}
