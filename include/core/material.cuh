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
      auto bounce_delta = random_in_unit_sphere(&rand_state);
      Eigen::Vector3f bounce_dir = normal + bounce_delta;
      Eigen::Vector3f ray_origin = intersect;
      return BounceResult{Ray{ray_origin, bounce_dir}, MaterialColorFunction{color_}};
    }

    __device__ Eigen::Vector3f light_bounce(const Ray &incoming_light,
					    Eigen::Vector3f light_color,
					    Eigen::Vector3f intersect,
					    Eigen::Vector3f normal,
					    const Ray &) const
    {
      auto scale = -incoming_light.direction().dot(normal);
      if (scale > 0.0f) {
	return scale * light_color;
      } else {
	return Eigen::Vector3f{0.0f, 0.0f, 0.0f};
      }
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

  class MetallicMaterial {
  public:
    
    explicit
    __host__ __device__ MetallicMaterial(Eigen::Vector3f color)
      : color_(color)
    { }

    // bounces ray off of material and does not advance random sequence
    // in rand_state
    __device__ BounceResult bounce(const Ray &incoming,
				   Eigen::Vector3f intersect,
				   Eigen::Vector3f normal,
				   curandState &) const
    {
      return bounce_impl(incoming, intersect, normal);
    }

    __device__ Eigen::Vector3f light_bounce(const Ray &incoming_light,
					    Eigen::Vector3f light_color,
					    Eigen::Vector3f intersect,
					    Eigen::Vector3f normal,
					    const Ray &viewing_direction) const
    {
      auto light_reflect = bounce_impl(incoming_light, intersect, normal);
      if (light_reflect.ray.direction().dot(viewing_direction.direction()) < -0.99f) {
	return light_color;

      } else {
	return Eigen::Vector3f{0.001f, 0.001f, 0.001f};
      }
    }

  private:

    __device__ BounceResult bounce_impl(const Ray &incoming,
					Eigen::Vector3f intersect,
					Eigen::Vector3f normal) const
    {
      Eigen::Vector3f bounce_dir = incoming.direction() - 2.0f * incoming.direction().dot(normal) * normal;
      Eigen::Vector3f ray_origin = intersect;
      return BounceResult{Ray{ray_origin, bounce_dir}, MaterialColorFunction{color_}};
    }

    Eigen::Vector3f color_;    
  };

  inline __host__ __device__ MetallicMaterial metallic_red()
  {
    return MetallicMaterial{Eigen::Vector3f{1.0f, 0.0f, 0.0f}};
  }

  inline __host__ __device__ MetallicMaterial metallic_green()
  {
    return MetallicMaterial{Eigen::Vector3f{0.0f, 1.0f, 0.0f}};
  }

  inline __host__ __device__ MetallicMaterial metallic_blue()
  {
    return MetallicMaterial{Eigen::Vector3f{0.0f, 0.0f, 1.0f}};
  }

  inline __host__ __device__ MetallicMaterial metallic_yellow()
  {
    return MetallicMaterial{Eigen::Vector3f{1.0f, 1.0f, 0.0f}};
  }

  inline __host__ __device__ MetallicMaterial metallic_purple()
  {
    return MetallicMaterial{Eigen::Vector3f{1.0f, 0.0f, 1.0f}};
  }

  inline __host__ __device__ MetallicMaterial metallic_white()
  {
    return MetallicMaterial{Eigen::Vector3f{1.0f, 1.0f, 1.0f}};
  }

  inline __host__ __device__ MetallicMaterial metallic_black()
  {
    return MetallicMaterial{Eigen::Vector3f{0.001f, 0.001f, 0.001f}};
  }

}
