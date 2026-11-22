#ifndef DR_DETECTOR_H
#define DR_DETECTOR_H

#include "dr_types.h"

bool evaluate_window_statistics(const ImuSample* ring, size_t count);

void finalize_cluster(StationaryCluster* active, StationaryCluster* best);

#endif // DR_DETECTOR_H
