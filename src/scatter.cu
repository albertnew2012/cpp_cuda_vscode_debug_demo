/// Copyright 2024 SafeAI, Inc.
/// \file scatter.cu
/// \brief This file implements function(s) for scatter

// headers in local files
#include <pointpillars_detection/scatter.hpp>
namespace
{
namespace ns = modules::perception::pointpillars_detection;
// namespace nm = modules::map;
}  // anonymous namespace


__global__ void scatter_kernel(int32_t *x_coors, int32_t *y_coors, float32_t *pfe_output,
                               float32_t *scattered_feature, int32_t const grid_x_size,
                               int32_t const grid_y_size)
{
  int32_t i_pillar = blockIdx.x;
  int32_t i_feature = threadIdx.x;
  int32_t x_ind = x_coors[i_pillar];
  int32_t y_ind = y_coors[i_pillar];
  float32_t feature = pfe_output[i_pillar * 64 + i_feature];
  scattered_feature[i_feature * grid_y_size * grid_x_size +
                    y_ind * grid_x_size + x_ind] = feature;
}

::ns::ScatterCuda::ScatterCuda(size_t const num_threads, size_t const grid_x_size,
                         size_t const grid_y_size)
    : num_threads_(num_threads),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size) {}

void ::ns::ScatterCuda::DoScatterCuda(size_t const pillar_count, int32_t *x_coors,
                                int32_t *y_coors, float32_t *pfe_output,
                                float32_t *scattered_feature)
{
  scatter_kernel<<<pillar_count, num_threads_>>>(x_coors, y_coors, pfe_output,
                                                 scattered_feature,
                                                 grid_x_size_, grid_y_size_);
}
