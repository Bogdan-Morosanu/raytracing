#pragma once

#include "core/material.cuh"
#include "core/plane.cuh"
#include "core/sphere.cuh"
#include "core/triangle.cuh"
#include "container/variant.cuh"

namespace rt {

  using GeometryVariant = Variant<Sphere, Plane, Triangle>;

  using MaterialVariant = Variant<DiffuseMaterial>;
  
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
				   Eigen::Vector3f normal) const
    {
      auto bounce_visitor = [&incoming, &intersect, &normal](auto &material)
			    {
			      return material.bounce(incoming,
						     intersect,
						     normal);
			    };

      return material_.visit(bounce_visitor);
    }

  private:
    Variant<Sphere, Plane, Triangle> geometry_;
    Variant<DiffuseMaterial> material_;
  };
}
