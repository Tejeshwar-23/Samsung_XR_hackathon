#ifndef DR_IO_H
#define DR_IO_H

#include "dr_types.h"
#include <stdio.h>

void safe_timestamp_divmod(int64_t ts_ns, int64_t* sec, int64_t* nsec);

bool parse_csv_line(const char* line, ImuSample* sample);

bool sanitize_sample(ImuSample* sample);

void format_tum_line(char* buffer, size_t buf_len, int64_t ts_ns, const double p[3], const double q[4]);

bool ensure_parent_directory(const char* filepath);

void write_telemetry_json(const char* json_path,
                          const char* input_path,
                          const char* output_path,
                          const char* scenario_id,
                          int seed,
                          const CalibrationResult* calib,
                          const ExecutionMetrics* metrics);

#endif // DR_IO_H
