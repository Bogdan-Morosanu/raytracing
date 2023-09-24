#include <cassert>
#include <cstdio>

#include <Eigen/Dense>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "alloc.cuh"
#include "check_error.cuh"
#include "work_partition.cuh"

void print_help_and_exit(const char *name) {
  std::printf("usage %s <image-filename>\n", name);
  std::exit(EXIT_FAILURE);
}

__global__ void render(rt::ImageBufferView img)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < img.width() && col < img.height()) {
    int pixel_index = col * img.width() + row;
    img.buffer()[pixel_index][0] = float(row) / img.width();
    img.buffer()[pixel_index][1] = float(col) / img.height();
    img.buffer()[pixel_index][2] = 0.2f;
  }
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    print_help_and_exit(*argv);
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
  
  render<<<blocks, threads>>>(rt::ImageBufferView(img));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stbi_write_hdr(argv[1], img.width(), img.height(), 3, &(img.buffer()[0][0]));
  
  std::printf("Hello, world!\n");
  return 0;
}
