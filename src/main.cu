#include <cassert>
#include <cstdio>

#include <Eigen/Dense>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <curand_kernel.h>

#include "alloc.cuh"
#include "check_error.cuh"
#include "light.cuh"
#include "multisample_mesh.cuh"
#include "work_partition.cuh"
#include "ray.cuh"
#include "scene_object.cuh"
#include "variant.cuh"
#include "viewport.cuh"

namespace config {

  void print_help_and_exit(const char *name)
  {
    std::printf("usage %s <image-filename>\n", name);
    std::exit(EXIT_FAILURE);
  }

  rt::Viewport make_viewport()
  {
    return rt::Viewport(Eigen::Vector3f(0.0f,  0.0f,  0.0f),  // origin
			Eigen::Vector3f(4.0f,  0.0f,  0.0f),  // horizontal extent
			Eigen::Vector3f(0.0f, -2.0f,  0.0f),  // vertical extent
			Eigen::Vector3f(-2.0f, 1.0f, -1.0f)); // upper left corner
  }
}

__device__ Eigen::Vector3f simple_light_color(int obj_index,
					      const rt::HitResult &hit_result,
					      const rt::ArrayView<rt::SceneObject, 2> &spheres,
					      const rt::ArrayView<rt::DirectionalLight, 1> &lights)
{
  auto point_color = Eigen::Vector3f{0.0f, 0.0f, 1.0f};
  Eigen::Vector3f result = point_color / 256.0f;
  
  for (auto &light: lights) {
    auto point_to_light_ray = light.ray_from(hit_result.point);

    bool obstructed = false;
    int i = 0;
    for (auto &object: spheres) {
      auto inf = float(INFINITY);
      auto r = point_to_light_ray.ray;
      auto hr = object.hit(r, rt::Interval{-inf, inf});
      if (hr) {
	// std::printf("ray from %i to light ((%f, %f, %f), (%f, %f, %f)) hit obj %i at %f\n",
	// 	    obj_index,
	// 	    r.origin().x(),
	// 	    r.origin().y(),
	// 	    r.origin().z(),
	// 	    r.direction().x(),
	// 	    r.direction().y(),
	// 	    r.direction().z(),
	// 	    i,
	// 	    hr.value().t);
      }
      
      if (point_to_light_ray.obstructed_by(object)) {
	obstructed = true;
	break;
      }
      i++;
    }

    if (obstructed) {
      continue;
    }

    auto scale = -hit_result.normal.dot(point_to_light_ray.hit_result.normal);
    auto light_color = point_to_light_ray.color;
    if (scale > 0) {
      result += scale * Eigen::Vector3f{light_color.x() * point_color.x(),
					light_color.y() * point_color.y(),
					light_color.z() * point_color.z()};
    }
  }

  return result;
}					      

__device__ Eigen::Vector3f color_ray(const rt::Ray &r,
				     const rt::ArrayView<rt::SceneObject, 2> &spheres,
				     const rt::ArrayView<rt::DirectionalLight, 1> &lights)
{
  auto dir = r.direction().normalized();
  auto t_interval = rt::Interval(0.0f, std::numeric_limits<float>::max());
  auto lowest_t = std::numeric_limits<float>::max();
  rt::Optional<rt::HitResult> hit_result;

  auto obj_hit = 0;
  
  // hitting geometry
  auto i = 0;
  for (auto &s : spheres) {
    auto maybe_hit_result = s.hit(r, t_interval);
    if (maybe_hit_result.is_valid()) {
      if (maybe_hit_result.value().t < lowest_t) {
	lowest_t = maybe_hit_result.value().t;
	hit_result = maybe_hit_result;
	obj_hit = i;
      }
    }
    i++;
  }

  if (hit_result) {
    return simple_light_color(obj_hit, hit_result.value(), spheres, lights);
  }
  
  // background
  float y_fraction = 0.5f * (dir[1] + 1.0f);
  return (1.0f - y_fraction) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) + y_fraction * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
}


__global__ void render(rt::Viewport viewport,
		       rt::ArrayView<rt::SceneObject, 2> spheres,
		       rt::ArrayView<rt::DirectionalLight, 1> lights,
		       rt::ImageBufferView img)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  rt::MultisampleMesh msmesh(img.width(), img.height());;
  
  if (row < img.width() && col < img.height()) {
    int pixel_index = col * img.width() + row;
    float u = float(row) / img.width();
    float v = float(col) / img.height();
    Eigen::Vector2f pixel(u, v);
    
    auto samples = msmesh.generate_samples(pixel);
    img.buffer()[pixel_index] = Eigen::Vector3f{0.0f, 0.0f, 0.0f};

    for (auto &s : samples) {
      auto ray = viewport.compute_ray(s.x(), s.y());
      img.buffer()[pixel_index] += color_ray(ray, spheres, lights);
    }
    
    img.buffer()[pixel_index] /= float(samples.size());
  }
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    config::print_help_and_exit(*argv);
  }
  
  rt::ImageBuffer img(2048, 1024);

  int threads_w = 8;
  int threads_h = 8;

  auto w_partition = rt::make_work_partition(img.width(), threads_w);
  auto h_partition = rt::make_work_partition(img.height(), threads_h);
  assert(!w_partition.is_uneven_partition && !h_partition.is_uneven_partition);

  dim3 blocks(w_partition.elements_per_thread, h_partition.elements_per_thread);
  dim3 threads(threads_w, threads_h);

  std::printf("running (%lu, %lu) blocks\n", w_partition.elements_per_thread, h_partition.elements_per_thread);

  auto viewport = config::make_viewport();
  rt::Array<rt::SceneObject, 2> spheres({rt::Sphere(Eigen::Vector3f(3.0f, 0.0f, -8.0f), 4.0f),
					 rt::Triangle(Eigen::Vector3f( 0.0f, 1.0f, -2.0f),
						      Eigen::Vector3f(-1.0f,-0.5f, -2.0f),
						      Eigen::Vector3f( 1.0f,-0.5f, -2.0f))});

  rt::Array<rt::DirectionalLight, 1> lights({rt::DirectionalLight{Eigen::Vector3f(1.0f, 0.0f, -1.0f),
                                                                  Eigen::Vector3f(1.0f, 1.0f, 1.0f)}});
  
  render<<<blocks, threads>>>(viewport,
			      rt::make_view(spheres),
			      rt::make_view(lights),
			      rt::ImageBufferView(img));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stbi_write_hdr(argv[1], img.width(), img.height(), 3, &(img.buffer()[0][0]));
  
  std::printf("Hello, world!\n");
  return 0;
}
