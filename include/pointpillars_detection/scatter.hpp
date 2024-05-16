#ifndef POINTPILLARS_DETECTION__SCATTER_HPP_
#define POINTPILLARS_DETECTION__SCATTER_HPP_

#include <pointpillars_detection/visibility_control.hpp>
#include <si/defs.h>
#include <cstddef>

/// \namespace modules
/// \brief Namespace containing software implementations for algorithms, sensors and utilities
namespace modules
{
/// \namespace modules::perception
/// \brief Interfaces and implementation for perception algorithms
namespace perception
{
/// \namespace modules::perception::pointpillars_detection
/// \brief Interfaces and implementation for perception algorithms
namespace pointpillars_detection
{


/// \class ScatterCuda
/// \brief This class is used to scatter cuda operation fron conceptual 2d to conceptual 3d
class POINTPILLARS_DETECTION_PUBLIC ScatterCuda
{
private:
  size_t const num_threads_;
  size_t const grid_x_size_;
  size_t const grid_y_size_;

public:
  /// \brief Constructor
  /// \param[in] num_threads The number of threads to launch cuda kernel
  /// \param[in] grid_x_size Number of pillars in x-coordinate
  /// \param[in] grid_y_size Number of pillars in y-coordinate
  ScatterCuda(
    size_t const num_threads, size_t const grid_x_size,
    size_t const grid_y_size);


  /// \brief Call scatter cuda kernel
  /// \param[in] pillar_count The valid number of pillars
  /// \param[in] x_coors X-coordinate indexes for corresponding pillars
  /// \param[in] y_coors Y-coordinate indexes for corresponding pillars
  /// \param[in] pfe_output Output from Pillar Feature Extractor
  /// \param[out] scattered_feature Gridmap representation for pillars' feature
  void DoScatterCuda(
    size_t const pillar_count, int32_t * x_coors, int32_t * y_coors,
    float32_t * pfe_output, float32_t * scattered_feature);
};
}     // namespace pointpillars_detection
}   // namespace perception
}  // namespace modules
#endif   // POINTPILLARS_DETECTION__SCATTER_HPP_
