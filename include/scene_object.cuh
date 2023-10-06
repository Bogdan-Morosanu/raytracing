#pragma once

#include "sphere.cuh"
#include "plane.cuh"
#include "triangle.cuh"
#include "variant.cuh"

namespace rt {

  class SceneObject {
  public:

    SceneObject() = default;
    
    SceneObject(Sphere sphere)
      : object_(std::move(sphere))
    { }

    SceneObject(Plane plane)
      : object_(std::move(plane))
    { }

    SceneObject(Triangle triangle)
      : object_(std::move(triangle))
    { }
    
    __device__ bool can_hit(const Ray &r) const
    {
      auto can_hit_visitor = [&r](auto &o) { return o.can_hit(r); };
      return object_.visit(can_hit_visitor);
    }

    __device__ Optional<HitResult> hit(const Ray &r,
				       const Interval &t_interval) const
    {
      auto hit_visitor = [&r, &t_interval](auto &object) -> Optional<HitResult>
      			 {
      			   return object.hit(r, t_interval);
      			 };
      return object_.visit(hit_visitor);
    }
    
  private:
    Variant<Sphere, Plane, Triangle> object_;
  };
}
