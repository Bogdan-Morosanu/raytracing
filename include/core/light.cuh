#pragma once

#include <cmath>

#include <Eigen/Dense>

#include "core/ray.cuh"
#include "core/hit_result.cuh"
#include "core/scene_object.cuh"
#include "core/sphere.cuh"


namespace rt {

  /// encodes the ray from a light to a given object
  /// within the ray, the object point is parametrised at t = 0
  /// and the closest point on the light surface is parametrised
  /// by the hit_result structure.
  /// In other words, an object o' obstructs the light source if
  /// o'.hit(point_to_light_ray.ray) returns a hit result with
  /// a t param that is in the interval:
  /// (0, point_to_light_ray.hit_result.t)
  struct PointToLightRay {
    Ray ray;
    HitResult hit_result;
    Eigen::Vector3f color;

    __device__ bool obstructed_by(const SceneObject &object) const
    {
      constexpr auto eps = 1e-3f;
      auto other_hit = object.hit(ray, Interval{eps, hit_result.t});
      return other_hit.is_valid();
    }
  };

  class PointLight {
  public:
    __host__ __device__ PointLight(Eigen::Vector3f pos, Eigen::Vector3f color)
      : pos_(pos)
      , color_(color)
    { }

    __host__ __device__ Eigen::Vector3f &pos() { return pos_; }

    __host__ __device__ const Eigen::Vector3f &pos() const { return pos_; }

    __device__ PointToLightRay ray_from(Eigen::Vector3f object_point) const
    {
      auto ray = Ray{object_point, pos_ - object_point};
      auto hit_result = HitResult{1.0f, pos_, (object_point - pos_).normalized()};

      return PointToLightRay{ray, hit_result, color_};
    }
    
  private:
    Eigen::Vector3f pos_;
    Eigen::Vector3f color_;
  };

  class DirectionalLight {
  public:
    DirectionalLight() = default;
    
    __host__ __device__ DirectionalLight(Eigen::Vector3f dir, Eigen::Vector3f color)
      : dir_(dir.normalized())
      , color_(color)
    { }

    __device__ Eigen::Vector3f &dir() { return dir_; }

    __device__ const Eigen::Vector3f &dir() const { return dir_; }

    __device__ PointToLightRay ray_from(Eigen::Vector3f object_point) const
    {
      auto ray = Ray{object_point, -dir_};
      constexpr auto inf = float(INFINITY);
      auto inf_point = Eigen::Vector3f{inf, inf, inf};
      auto hit_result = HitResult{inf, inf_point, dir_};

      return PointToLightRay{ray, hit_result, color_};
    }

  private:
    Eigen::Vector3f dir_;
    Eigen::Vector3f color_;
  };

  class SphericalLight {
  public:
    // TODO
    __host__ __device__ SphericalLight(Eigen::Vector3f center, float radius, Eigen::Vector3f color)
      : sphere_(center, radius)
      , color_(color)
    { }

    __device__ PointToLightRay ray_from(Eigen::Vector3f object_point) const
    {
      // TODO, deal with point inside the sphere
      auto ray = Ray{object_point, object_point - sphere_.center_};

      auto inf = float(INFINITY);
      auto hit_result = sphere_.hit(ray, Interval{-inf, inf});
      assert(hit_result.is_valid());

      return PointToLightRay{ray, hit_result.value(), color_};
    }
    
  private:
    Sphere sphere_;
    Eigen::Vector3f color_;
  };

  class PanelLight {
  public:
    // TODO
  private:
    Eigen::Vector3f up_left_;
    Eigen::Vector3f down_right_;
  };

  class CylindricalLight {
  public:

  private:
    Eigen::Vector3f bottom_;
    Eigen::Vector3f top_;
    float radius_;
  };

  class DomeLight {
    // TODO
  };


  __device__ inline rt::Optional<Eigen::Vector3f>
  light_color(const rt::HitResult &hit_result,
	      const rt::VectorView<rt::SceneObject> &objects,
	      const rt::VectorView<rt::DirectionalLight> &lights)
  {
    rt::Optional<Eigen::Vector3f> result;
  
    for (auto &light: lights) {
      auto point_to_light_ray = light.ray_from(hit_result.point);

      bool obstructed = false;
      for (auto &object: objects) {
	if (point_to_light_ray.obstructed_by(object)) {
	  obstructed = true;
	  break;
	}
      }

      if (obstructed) {
	continue;
      }

      auto scale = -hit_result.normal.dot(point_to_light_ray.hit_result.normal);
      auto light_color = point_to_light_ray.color;
      if (scale > 0) {
	if (result) {
	  result.value() += scale * light_color;
	} else {
	  result = Eigen::Vector3f{scale * light_color};
	}
      }
    }

    return result;
  }

}
