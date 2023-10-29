#pragma once

#include "container/alloc.cuh"
#include "core/ray.cuh"
#include "cuda_utils/curand_state.cuh"

namespace rt {

  class MaterialColorFunction {
  public:

    explicit
    __device__ MaterialColorFunction(Eigen::Vector3f multiplier)
      : multiplier_(multiplier)
    { }

    __device__ Eigen::Vector3f operator()(Eigen::Vector3f color) const
    {
      return Eigen::Vector3f{
          color.x() * multiplier_.x(),
	  color.y() * multiplier_.y(),
	  color.z() * multiplier_.z(),
      };
    }

    __device__ void compose_with(const MaterialColorFunction &other)
    {
      multiplier_.x() *= other.multiplier_.x();
      multiplier_.y() *= other.multiplier_.y();
      multiplier_.z() *= other.multiplier_.z();
    }
    
  private:
    Eigen::Vector3f multiplier_;
  };

  struct BounceResult {
    Ray ray;
    MaterialColorFunction color_function;
  };
  
  class DiffuseMaterial {
  public:

    explicit
    __host__ __device__ DiffuseMaterial(Eigen::Vector3f color)
      : color_(color)
    { }

    // bounces ray off of material and advances random sequence
    // in rand_state
    __device__ BounceResult bounce(const Ray &incoming,
				   Eigen::Vector3f intersect,
				   Eigen::Vector3f normal,
				   curandState &rand_state) const
    {
      // todo: implement bounce ray
      auto bounce_delta = random_in_unit_sphere(&rand_state);
      Eigen::Vector3f bounce_dir = normal + bounce_delta;
      Eigen::Vector3f ray_origin = intersect;
      return BounceResult{Ray{ray_origin, bounce_dir}, MaterialColorFunction{color_}};
    }

  private:
    Eigen::Vector3f color_;    
  };

  inline __host__ __device__ DiffuseMaterial diffuse_red()
  {
    return DiffuseMaterial{Eigen::Vector3f{1.0f/16.0f, 0.0f, 0.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_green()
  {
    return DiffuseMaterial{Eigen::Vector3f{0.0f, 1.0f/16.0f, 0.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_blue()
  {
    return DiffuseMaterial{Eigen::Vector3f{0.0f, 0.0f, 1.0f/16.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_yellow()
  {
    return DiffuseMaterial{Eigen::Vector3f{1.0f/16.0f, 1.0f/16.0f, 0.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_purple()
  {
    return DiffuseMaterial{Eigen::Vector3f{1.0f/16.0f, 0.0f, 1.0f/16.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_white()
  {
    return DiffuseMaterial{Eigen::Vector3f{1.0f/16.0f, 1.0f/16.0f, 1.0f/16.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_black()
  {
    return DiffuseMaterial{Eigen::Vector3f{0.001f, 0.001f, 0.001f}};
  }
}
