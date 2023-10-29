#pragma once


#include "core/material.cuh"
#include "core/plane.cuh"
#include "core/sphere.cuh"
#include "core/triangle.cuh"
#include "container/variant.cuh"
#include "cuda_utils/curand_state.cuh"

namespace rt {

  using GeometryVariant = Variant<Sphere, Plane, Triangle>;

  using MaterialVariant = Variant<DiffuseMaterial, MetallicMaterial>;
  
  class SceneObject {
  public:

    __host__ __device__ SceneObject() = default;
    
    __host__ __device__ SceneObject(GeometryVariant geometry,
				    MaterialVariant material)
      : geometry_(std::move(geometry))
      , material_(std::move(material))
    { }
    
    __device__ bool can_hit(const Ray &r) const
    {
      auto can_hit_visitor = [&r](auto &geometry) { return geometry.can_hit(r); };
      return geometry_.visit(can_hit_visitor);
    }

    __device__ Optional<HitResult> hit(const Ray &r,
				       const Interval &t_interval) const
    {
      auto hit_visitor = [&r, &t_interval](auto &geometry)
      			 {
      			   return geometry.hit(r, t_interval);
      			 };
      return geometry_.visit(hit_visitor);
    }

    __device__ BounceResult bounce(const Ray &incoming,
				   Eigen::Vector3f intersect,
				   Eigen::Vector3f normal,
				   curandState &state) const
    {
      auto bounce_visitor = [&incoming, &intersect, &normal, &state](auto &material)
			    {
			      return material.bounce(incoming,
						     intersect,
						     normal,
						     state);
			    };

      return material_.visit(bounce_visitor);
    }

    __device__ Eigen::Vector3f light_bounce(const Ray &incoming_light,
					    Eigen::Vector3f light_color,
					    Eigen::Vector3f intersect,
					    Eigen::Vector3f normal,
					    const Ray &viewing_direction) const
    {
      auto light_bounce_visitor = [&incoming_light, &light_color, &intersect,
				   &normal, &viewing_direction](auto &material)
				  {
				    return material.light_bounce(incoming_light,
								 light_color,
								 intersect,
								 normal,
								 viewing_direction);
				  };

      return material_.visit(light_bounce_visitor);
    }

  private:
    GeometryVariant geometry_;
    MaterialVariant material_;
  };
}
