#ifndef POINTPILLARS_DETECTION__VISIBILITY_CONTROL_HPP_
#define POINTPILLARS_DETECTION__VISIBILITY_CONTROL_HPP_

#include <si/defs.h>

#if defined SI_WINDOWS
#ifdef POINTPILLARS_DETECTION_BUILDING_DLL
#define POINTPILLARS_DETECTION_PUBLIC __declspec(dllexport)
#else
#define POINTPILLARS_DETECTION_PUBLIC _declspec(dllimport)
#endif
#else
// Assume Linux for now
#define POINTPILLARS_DETECTION_PUBLIC __attribute__ ((visibility("default")))
#endif

#endif  // POINTPILLARS_DETECTION__VISIBILITY_CONTROL_HPP_
