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

__global__ void render(float *buff, int max_row, int max_col)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < max_row && col < max_col) {
    int pixel_index = (col * max_row + row) * 3;
    buff[pixel_index] = float(row) / max_row;
    buff[pixel_index + 1] = float(col) / max_col;
    buff[pixel_index + 2] = 0.2f;
  }
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    print_help_and_exit(*argv);
  }
  
  rt::ImageBuffer img(256, 128);

  int threads_w = 8;
  int threads_h = 8;

  auto w_partition = rt::make_work_partition(img.width(), threads_w);
  auto h_partition = rt::make_work_partition(img.height(), threads_h);
  assert(!w_partition.is_uneven_partition && !h_partition.is_uneven_partition);

  dim3 blocks(w_partition.elements_per_thread, h_partition.elements_per_thread);
  dim3 threads(threads_w, threads_h);

  std::printf("running (%lu, %lu) blocks\n", w_partition.elements_per_thread, h_partition.elements_per_thread);
  
  render<<<blocks, threads>>>(img.buffer(), img.width(), img.height());

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stbi_write_hdr(argv[1], img.width(), img.height(), 3, img.buffer());
  
  std::printf("Hello, world!\n");
  return 0;
}
