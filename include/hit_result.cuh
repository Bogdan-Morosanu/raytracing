#pragma once

#include <Eigen/Dense>

namespace rt {
  struct HitResult {
    float time;
    Eigen::Vector3f point;
    Eigen::Vector3f normal;
  };
}
