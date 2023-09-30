#pragma once

#include <cmath>

#include <Eigen/Dense>

#include "ray.cuh"
#include "hit_result.cuh"
#include "sphere.cuh"

namespace rt {

  struct PointToLightRay {
    Ray ray;
    HitResult hit_result;
  };

  class PointLight {
  public:
    __host__ __device__ PointLight(Eigen::Vector3f pos)
      : pos_(pos)
    { }

    __host__ __device__ Eigen::Vector3f &pos() { return pos_; }

    __host__ __device__ const Eigen::Vector3f &pos() const { return pos_; }

    __device__ PointToLightRay ray_from(Eigen::Vector3f object_point) const
    {
      auto ray = Ray{object_point, pos_ - object_point};
      auto hit_result = HitResult{0.0f, 1.0f, pos_, object_point - pos_};

      return PointToLightRay{ray, hit_result};
    }
    
  private:
    Eigen::Vector3f pos_;
  };

  class DirectionalLight {
  public:
    // TODO remove when Array supports initializer list
    __host__ __device__ DirectionalLight() = default;
    
    __host__ __device__ DirectionalLight(Eigen::Vector3f dir)
      : dir_(dir)
    { }

    __device__ Eigen::Vector3f &dir() { return dir_; }

    __device__ const Eigen::Vector3f &dir() const { return dir_; }

    __device__ PointToLightRay ray_from(Eigen::Vector3f object_point) const
    {
      auto ray = Ray{object_point, -dir_};
      auto inf = float(INFINITY);
      auto inf_point = Eigen::Vector3f{inf, inf, inf};
      auto hit_result = HitResult{0.0f, inf, inf_point, dir_};

      return PointToLightRay{ray, hit_result};
    }

  private:
    Eigen::Vector3f dir_;
  };

  class SphericalLight {
  public:
    // TODO
    __host__ __device__ SphericalLight(Eigen::Vector3f center, float radius)
      : sphere_(center, radius)
    { }

    __device__ PointToLightRay ray_from(Eigen::Vector3f object_point) const
    {
      // TODO, deal with point inside the sphere
      auto ray = Ray{sphere_.center_, sphere_.center_ - object_point};
      ray.origin() -= ray.direction() / ray.direction().norm() * sphere_.radius_;

      auto inf = float(INFINITY);
      auto hit_result = sphere_.hit(ray, Interval{-inf, inf});
      assert(hit_result.is_valid());

      return PointToLightRay{ray, hit_result.value()};
    }
    
  private:
    Sphere sphere_;
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
}
