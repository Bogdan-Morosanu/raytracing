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
#include "sphere.cuh"
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

__device__ Eigen::Vector3f color_ray(const rt::Ray &r, const rt::ArrayView<rt::Sphere, 2> &spheres)
{
  auto dir = r.direction().normalized();
  auto t_interval = rt::Interval(0.0f, std::numeric_limits<float>::max());
  
  // hitting geometry
  for (auto &s : spheres) {
    auto maybe_hit_result = s.hit(r, t_interval);
    if (maybe_hit_result.is_valid()) {
      auto n = maybe_hit_result.value().normal;
      return 0.5f * Eigen::Vector3f(n.x() + 1.0f, n.y() + 1.0f, n.z() + 1.0f);
    }
  }

  // background
  float y_fraction = 0.5f * (dir[1] + 1.0f);
  return (1.0f - y_fraction) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) + y_fraction * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
}


__global__ void render(rt::Viewport viewport,
		       rt::ArrayView<rt::Sphere, 2> spheres,
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
      img.buffer()[pixel_index] += color_ray(ray, spheres);
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
  rt::Array<rt::Sphere, 2> spheres;
  spheres[0] = rt::Sphere(Eigen::Vector3f(0.0f, 0.0f, -1.0f), 0.5f);
  spheres[1] = rt::Sphere(Eigen::Vector3f(0.0f, -100.5f, -1.0f), 100.0f);

  rt::Array<rt::DirectionalLight, 1> lights;
  lights[1] = rt::DirectionalLight{Eigen::Vector3f(1.0f, -1.0f, -1.0f)};
  
  render<<<blocks, threads>>>(viewport,
			      rt::ArrayView<rt::Sphere, 2>(spheres),
			      rt::ArrayView<rt::DirectionalLight, 1>(lights),
			      rt::ImageBufferView(img));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stbi_write_hdr(argv[1], img.width(), img.height(), 3, &(img.buffer()[0][0]));
  
  std::printf("Hello, world!\n");
  return 0;
}
