#pragma once

#include "container/alloc.cuh"
#include "core/ray.cuh"

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
    
    __device__ BounceResult bounce(const Ray &incoming,
				   Eigen::Vector3f intersect,
				   Eigen::Vector3f normal) const
    {
      // todo: implement bounce ray
      return BounceResult{Ray{intersect, normal}, MaterialColorFunction{color_}};
    }

  private:
    Eigen::Vector3f color_;    
  };

  inline __host__ __device__ DiffuseMaterial diffuse_red()
  {
    return DiffuseMaterial{Eigen::Vector3f{1.0f, 0.0f, 0.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_green()
  {
    return DiffuseMaterial{Eigen::Vector3f{0.0f, 1.0f, 0.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_blue()
  {
    return DiffuseMaterial{Eigen::Vector3f{0.0f, 0.0f, 1.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_white()
  {
    return DiffuseMaterial{Eigen::Vector3f{1.0f, 1.0f, 1.0f}};
  }

  inline __host__ __device__ DiffuseMaterial diffuse_black()
  {
    return DiffuseMaterial{Eigen::Vector3f{0.01f, 0.01f, 0.01f}};
  }
}
