/// Copyright 2024 SafeAI, Inc.
/// \file test_pointpillars_detection.cpp
/// \brief This file implements unit test for scatter cuda class


#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "pointpillars_detection/scatter.hpp"


namespace
{
namespace nsp = modules::perception::pointpillars_detection;
// using msg_type = safeai_interfaces::msg::ChassisState;
}  // anonymous namespace


class ScatterCudaTest : public ::testing::Test
{
protected:
  ::nsp::ScatterCuda * scatter;
  int32_t * d_x_coors, * d_y_coors;
  float32_t * d_pfe_output, * d_scattered_feature;
  static constexpr size_t kPillarCount = 10;
  static constexpr size_t kNumFeatures = 64;
  static constexpr size_t kGridXSize = 10, kGridYSize = 10;

  void SetUp() override
  {
    scatter = new ::nsp::ScatterCuda(1000, kGridXSize, kGridYSize);

    // Allocate device memory
    cudaMalloc(&d_x_coors, kPillarCount * sizeof(int32_t));
    cudaMalloc(&d_y_coors, kPillarCount * sizeof(int32_t));
    cudaMalloc(&d_pfe_output, kPillarCount * kNumFeatures * sizeof(float32_t));
    cudaMalloc(&d_scattered_feature, kGridXSize * kGridYSize * kNumFeatures * sizeof(float32_t));

    // Initialize device memory
    cudaMemset(
      d_scattered_feature, 0,
      kGridXSize * kGridYSize * kNumFeatures * sizeof(float32_t));
  }

  void TearDown() override
  {
    delete scatter;
    cudaFree(d_x_coors);
    cudaFree(d_y_coors);
    cudaFree(d_pfe_output);
    cudaFree(d_scattered_feature);
  }
};

TEST_F(ScatterCudaTest, CorrectScatterOperation) {
  int32_t x_coors[kPillarCount] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int32_t y_coors[kPillarCount] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float32_t pfe_output[kPillarCount * kNumFeatures];
  float32_t scattered_feature[kGridXSize * kGridYSize * kNumFeatures] = {0};

  // Initialize pfe_output with varied test data
  for (size_t i = 0; i < kPillarCount * kNumFeatures; ++i) {
    pfe_output[i] = static_cast<float32_t>(i % kNumFeatures) * 1.5f;
  }


  // Copy inputs to device
  cudaMemcpy(d_x_coors, x_coors, kPillarCount * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_coors, y_coors, kPillarCount * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_pfe_output, pfe_output, kPillarCount * kNumFeatures * sizeof(float32_t),
    cudaMemcpyHostToDevice);

  // After cudaMemcpy
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error after cudaMemcpy: " << cudaGetErrorString(error) << std::endl;
  }

  // Execute the kernel
  scatter->DoScatterCuda(kPillarCount, d_x_coors, d_y_coors, d_pfe_output, d_scattered_feature);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error after kernel: " << cudaGetErrorString(error) << std::endl;
  }
  cudaDeviceSynchronize();    // Ensure all operations are completed


  // Copy data from device to host
  cudaError_t copyErr = cudaMemcpy(
    scattered_feature, d_scattered_feature,
    kGridXSize * kGridYSize * kNumFeatures * sizeof(float32_t),
    cudaMemcpyDeviceToHost);
  if (copyErr != cudaSuccess) {
    std::cerr << "CUDA memcpy error: " << cudaGetErrorString(copyErr) << std::endl;
  }

  // Verify scattered_feature for all features of each pillar
  for (size_t i = 0; i < kPillarCount; ++i) {
    for (size_t f = 0; f < kNumFeatures; ++f) {
      // Adjust index calculation for full feature verification
      size_t idx = f * kGridXSize * kGridYSize + y_coors[i] * kGridXSize + x_coors[i];
      EXPECT_FLOAT_EQ(scattered_feature[idx], pfe_output[i * kNumFeatures + f]);
    }
  }
}


int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
