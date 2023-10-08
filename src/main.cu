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
			Eigen::Vector3f(0.0f,  -2.0f,  0.0f),  // vertical extent
			Eigen::Vector3f(-2.0f, 1.0f, -1.0f)); // upper left corner
  }
}

__device__ Eigen::Vector3f simple_light_color(const rt::HitResult &hit_result,
					      const rt::ArrayView<rt::Sphere, 1> &spheres,
					      const rt::ArrayView<rt::DirectionalLight, 1> &lights)
{
  auto point_color = Eigen::Vector3f{0.0f, 0.0f, 1.0f};
  Eigen::Vector3f result = point_color / 256.0f;
  
  for (auto &light: lights) {
    auto point_to_light_ray = light.ray_from(hit_result.point);

    bool obstructed = false;
    for (auto &object: spheres) {
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
      result += scale * Eigen::Vector3f{light_color.x() * point_color.x(),
					light_color.y() * point_color.y(),
					light_color.z() * point_color.z()};
    }
  }

  return result;
}					      

__device__ Eigen::Vector3f color_ray(std::size_t width, std::size_t height,
				     const rt::Ray &r,
				     const rt::ArrayView<rt::Sphere, 1> &spheres,
				     const rt::ArrayView<rt::DirectionalLight, 1> &lights)
{
  auto inf = float(INFINITY);
  auto t_interval = rt::Interval(0, inf);
  auto lowest_t = inf;
  rt::Optional<rt::HitResult> hit_result;

  // hitting geometry
  for (auto &s : spheres) {
    auto maybe_hit_result = s.hit(r, t_interval);
    if (maybe_hit_result.is_valid()) {
      if (maybe_hit_result.value().t < lowest_t) {
	lowest_t = maybe_hit_result.value().t;
	hit_result = maybe_hit_result;
      }
    }
  }

  if (hit_result) {
    return simple_light_color(hit_result.value(), spheres, lights);
  }
  
  // background
  auto dir = r.direction().normalized();
  float y_fraction = 0.5f * (dir[1] + 1.0f);
  return (1.0f - y_fraction) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) + y_fraction * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
}


__global__ void render(rt::Viewport viewport,
		       rt::ArrayView<rt::Sphere, 1> spheres,
		       rt::ArrayView<rt::DirectionalLight, 1> lights,
		       rt::ImageBufferView img)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col == 0 && row == 0) {
    rt::Ray ul = viewport.compute_ray(0.0, 0.0);
    auto dl = viewport.compute_ray(0.0, 1.0);
    auto ur = viewport.compute_ray(1.0, 0.0);
    auto dr = viewport.compute_ray(1.0, 1.0);
    auto center = viewport.compute_ray(0.5, 0.5);
    std::printf("viewport info\n");
    std::printf("up left %f %f %f\n", ul.direction()(0), ul.direction().y(), ul.direction().z());
    std::printf("down left %f %f %f\n", dl.direction().x(), dl.direction().y(), dl.direction().z());
    std::printf("up right %f %f %f\n", ur.direction().x(), ur.direction().y(), ur.direction().z());
    std::printf("down right %f %f %f\n", dr.direction().x(), dr.direction().y(), dr.direction().z());
    std::printf("center %f %f %f\n", center.direction().x(), center.direction().y(), center.direction().z());
  }
  
  rt::MultisampleMesh msmesh(img.width(), img.height());
  
  if (row < img.height() && col < img.width()) {
    int pixel_index = row * img.width() + col;
    float u = float(col) / img.width();
    float v = float(row) / img.height();
    if (int(u * img.width()) % 64 == 0 || int(v * img.height()) % 64 == 0) {
      img.buffer()[pixel_index] = Eigen::Vector3f{0.1f, 0.1f, 0.1f};
      return;
    }
    
    Eigen::Vector2f pixel(u, v);
    
    auto samples = msmesh.generate_samples(pixel);
    img.buffer()[pixel_index] = Eigen::Vector3f{0.0f, 0.0f, 0.0f};

    for (auto &s : samples) {
      auto ray = viewport.compute_ray(s.x(), s.y());
      img.buffer()[pixel_index] += color_ray(img.width(), img.height(), ray, spheres, lights);
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

  auto viewport = config::make_viewport();//.translate(Eigen::Vector3f(1.0, 1.0, 0.0));;
  rt::Array<rt::Sphere, 1> spheres({rt::Sphere(Eigen::Vector3f(1.0f, 1.0f, -2.0f), 0.5f)});
					 // rt::Triangle(Eigen::Vector3f( 0.0f, 1.0f, -2.0f),
					 // 	      Eigen::Vector3f(-1.0f,-0.5f, -2.0f),
					 // 	      Eigen::Vector3f( 1.0f,-0.5f, -2.0f)),
					 // rt::Triangle(Eigen::Vector3f( -2.0f, 1.0f, -2.0f),
					 // 	      Eigen::Vector3f( -1.0f,-0.5f, -2.0f),
					 // 	      Eigen::Vector3f( -3.0f,-0.5f, -2.0f))});

  rt::Array<rt::DirectionalLight, 1> lights({rt::DirectionalLight{Eigen::Vector3f(1.0f, 0.0f, -1.0f),
                                                                  Eigen::Vector3f(1.0f, 1.0f, 1.0f)}});
  
  render<<<blocks, threads>>>(viewport,
			      rt::make_view(spheres),
			      rt::make_view(lights),
			      rt::ImageBufferView(img));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  std::printf("writing %s...\n", argv[1]);
  stbi_write_hdr(argv[1], img.width(), img.height(), 3, &(img.buffer()[0][0]));
  std::printf("done!\n");
  return 0;
}
