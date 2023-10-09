#include <cassert>
#include <cstdio>

#include <Eigen/Dense>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <curand_kernel.h>

#include "check_error.cuh"
#include "container/alloc.cuh"
#include "container/variant.cuh"
#include "container/vector.cuh"
#include "core/light.cuh"
#include "core/ray.cuh"
#include "core/scene_object.cuh"
#include "pipeline/multisample_mesh.cuh"
#include "pipeline/work_partition.cuh"
#include "pipeline/viewport.cuh"

namespace config {

  void print_help_and_exit(const char *name)
  {
    std::printf("usage %s <image-filename>\n", name);
    std::exit(EXIT_FAILURE);
  }

  rt::Viewport make_viewport()
  {
    return rt::Viewport(Eigen::Vector3f(0.0f, 0.0f, 0.0f),
			Eigen::Vector3f(0.0f, 0.0f, -1.0f),
			Eigen::Vector3f(0.0f, 1.0f, 0.0f),
			20.0f,
			2.0f);
  }
}

__device__ Eigen::Vector3f simple_light_color(const rt::HitResult &hit_result,
					      const rt::MaterialColorFunction &color_function,
					      const rt::VectorView<rt::SceneObject> &objects,
					      const rt::ArrayView<rt::DirectionalLight, 1> &lights)
{
  Eigen::Vector3f result{0.01f, 0.01f, 0.01f};
  
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
      result += scale * color_function(light_color);
    }
  }

  return result;
}					      

__device__ Eigen::Vector3f color_ray(std::size_t width, std::size_t height,
				     const rt::Ray &r,
				     const rt::VectorView<rt::SceneObject> &objects,
				     const rt::ArrayView<rt::DirectionalLight, 1> &lights)
{
  auto inf = float(INFINITY);
  auto t_interval = rt::Interval(0, inf);
  auto lowest_t = inf;
  auto hit_index = 0u;
  rt::Optional<rt::HitResult> hit_result;

  // hitting geometry
  auto i = 0u;
  for (auto &object : objects) {
    auto maybe_hit_result = object.hit(r, t_interval);
    if (maybe_hit_result.is_valid()) {
      if (maybe_hit_result.value().t < lowest_t) {
	lowest_t = maybe_hit_result.value().t;
	hit_result = maybe_hit_result;
	hit_index = i;
      }
    }
    i++;
  }

  if (hit_result) {
    auto bounce_result = objects[hit_index].bounce(r,
						   hit_result.value().point,
						   hit_result.value().normal);
    return simple_light_color(hit_result.value(),
			      bounce_result.color_function,
			      objects, lights);
  }
  
  // background
  auto dir = r.direction().normalized();
  float y_fraction = 0.5f * (dir[1] + 1.0f);
  return (1.0f - y_fraction) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) + y_fraction * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
}


__global__ void render(rt::Viewport viewport,
		       rt::VectorView<rt::SceneObject> objects,
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
    // if (int(u * img.width()) % 64 == 0 || int(v * img.height()) % 64 == 0) {
    //   img.buffer()[pixel_index] = Eigen::Vector3f{0.1f, 0.1f, 0.1f};
    //   return;
    // }
    
    Eigen::Vector2f pixel(u, v);
    
    auto samples = msmesh.generate_samples(pixel);
    img.buffer()[pixel_index] = Eigen::Vector3f{0.0f, 0.0f, 0.0f};

    for (auto &s : samples) {
      auto ray = viewport.compute_ray(s.x(), s.y());
      img.buffer()[pixel_index] += color_ray(img.width(), img.height(), ray, objects, lights);
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

  
  auto viewport = config::make_viewport();//.translate(Eigen::Vector3f(0.0, 1.0f, 0.0f));
  auto line_start = Eigen::Vector3f(-2.25f, -1.0f, -10.f);
  auto line_dir = Eigen::Vector3f(2.25f, 1.0f, -3.0f);
  auto line_up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
  auto line_dir_normed = line_dir.normalized();
  line_up = (line_up - line_up.dot(line_dir_normed) * line_dir_normed).normalized();
  Eigen::Vector3f line_horizontal = line_up.cross(line_dir_normed);
  
  Eigen::Vector3f t1_center = line_start + 0.33 * line_dir;
  Eigen::Vector3f t1_a = t1_center + 0.5f * line_up;
  Eigen::Vector3f t1_b = t1_center + 0.5f * line_horizontal - 0.5f * line_up;
  Eigen::Vector3f t1_c = t1_center - 0.5f * line_horizontal - 0.5f * line_up;

  Eigen::Vector3f t2_center = line_start + 1.3f * line_dir;
  Eigen::Vector3f t2_a = t2_center + 0.75f * line_up;
  Eigen::Vector3f t2_b = t2_center + 0.75f * line_horizontal - 0.75f * line_up;
  Eigen::Vector3f t2_c = t2_center - 0.75f * line_horizontal - 0.75f * line_up;
  
  rt::Vector<rt::SceneObject> spheres({rt::SceneObject(rt::Sphere(line_start + 0.0f * line_dir, 0.25f),
						       rt::diffuse_red()),
				       rt::SceneObject(rt::Triangle(t1_a, t1_b, t1_c), rt::diffuse_yellow()),
				       rt::SceneObject(rt::Sphere(line_start + 0.8f * line_dir, 0.5f),
						       rt::diffuse_green()),
				       rt::SceneObject(rt::Triangle(t2_a, t2_b, t2_c), rt::diffuse_yellow()),
				       rt::SceneObject(rt::Sphere(line_start + 2.0f * line_dir, 1.0f),
						       rt::diffuse_blue())});


  rt::Array<rt::DirectionalLight, 1> lights({rt::DirectionalLight{line_dir,
                                                                  Eigen::Vector3f(1.0f, 1.0f, 1.0f)}});
  
  render<<<blocks, threads>>>(viewport,
			      rt::make_view(spheres),
			      rt::make_view(lights),
			      rt::ImageBufferView(img));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  for (auto r = 0; r < (img.height() / 2);  ++r) {
    for (auto c = 0; c < img.width(); ++c) {
      auto lhs = r * img.width() + c;
      auto rhs = (img.height() - r) * img.width() + c;
      std::swap(img.buffer()[lhs], img.buffer()[rhs]);
    }
  }
  std::printf("writing %s...\n", argv[1]);
  stbi_write_hdr(argv[1], img.width(), img.height(), 3, &(img.buffer()[0][0]));
  std::printf("done!\n");
  return 0;
}
