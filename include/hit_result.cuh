#pragma once

#include <Eigen/Dense>

namespace rt {
  struct HitResult {
    float t;
    Eigen::Vector3f point;
    Eigen::Vector3f normal;
  };
}
