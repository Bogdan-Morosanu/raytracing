#pragma once

#include "alloc.cuh"
#include "ray.cuh"

namespace rt {

  class MaterialColorFunction {
  public:

    explicit
    __device__ MaterialColorFunction(Eigen::Vector3f multiplier)
      : multiplier_(multriplier)
    { }

    __device__ Eigen::Vector3f operator()(Eigen::Vector3f color)
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

    __device__ BounceResult bounce(const Ray &incoming,
				   Eigen::Vector3f intersect,
				   Eigen::Vector3f normal)
    {

    }

  private:
    Eigen::Vector3f color_;
    Eigen::Vector3f attenuation_;
  };
}
