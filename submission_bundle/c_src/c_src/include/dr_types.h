#ifndef DR_TYPES_H
#define DR_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define MAX_WINDOW_SAMPLES 512
#define NOMINAL_GRAVITY 9.80665
#define GYRO_SATURATION_RAD_S 35.0
#define ACCEL_SATURATION_M_S2 180.0
#define MICRO_GAP_THRESHOLD_NS 25000000LL // 0.025 seconds
#define MIN_STATIONARY_DURATION_NS 2000000000LL // 2.0 seconds

#define MIN_SUPPORTED_TS -2000000000000000000LL // -2.0e18 ns
#define MAX_SUPPORTED_TS  4000000000000000000LL // +4.0e18 ns

typedef struct {
    int64_t timestamp_ns;
    double w_raw[3];
    double a_raw[3];
    double w[3]; // Sanitized estimator-facing
    double a[3]; // Sanitized estimator-facing
    bool valid_timestamp;
    bool valid_gyro;
    bool valid_accel;
    bool is_corrupted;
} ImuSample;

typedef struct {
    int64_t start_idx;
    int64_t end_idx;
    int64_t start_ts_ns;
    int64_t end_ts_ns;
    int64_t duration_ns;
    int64_t sample_count;
    double sum_gyro[3];
    bool is_valid;
} StationaryCluster;

typedef struct {
    int64_t prev_timestamp_ns;
    double q[4];          // [w, x, y, z] canonical internal
    double v[3];          // [vx, vy, vz] in world frame
    double p[3];          // [px, py, pz] in world frame
    double prev_w[3];     // Sanitized w at k-1
    double prev_alin[3];  // Linear world accel at k-1
    bool prev_valid_gyro;
    bool prev_valid_accel;
    bool has_prev;
    double gyro_bias[3];
} DRState;

typedef struct {
    size_t total_samples;
    double median_dt_s;
    size_t window_samples;
    size_t step_samples;
    double q0[4];
    double gyro_bias[3];
    StationaryCluster best_cluster;
    bool bias_confirmed;
} CalibrationResult;

typedef struct {
    double runtime_s;
    double telemetry_duration_s;
    double compute_latency_ms_per_s;
    size_t peak_working_set_bytes;
    size_t peak_pagefile_bytes;
    bool memory_valid;
    uint32_t thread_count;
    size_t input_frames;
    size_t output_poses;
} ExecutionMetrics;

#endif // DR_TYPES_H
