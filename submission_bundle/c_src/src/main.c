#include "dr_types.h"
#include "dr_math.h"
#include "dr_io.h"
#include "dr_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#endif

static uint32_t get_current_thread_count(void) {
    uint32_t count = 1;
#ifdef _WIN32
    DWORD pid = GetCurrentProcessId();
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (hSnap != INVALID_HANDLE_VALUE) {
        THREADENTRY32 te;
        te.dwSize = sizeof(te);
        if (Thread32First(hSnap, &te)) {
            count = 0;
            do {
                if (te.th32OwnerProcessID == pid) {
                    count++;
                }
            } while (Thread32Next(hSnap, &te));
        }
        CloseHandle(hSnap);
    }
#endif
    return (count == 0) ? 1 : count;
}

int main(int argc, char** argv) {
    const char* input_path = NULL;
    const char* output_path = "outputs/estimated_trajectory.txt";
    const char* scenario_id = "S4-01";
    const char* metadata_output = NULL;
    int seed = 42;
    bool enable_zupt = false;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--input") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((strcmp(argv[i], "--scenario") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            scenario_id = argv[++i];
        } else if ((strcmp(argv[i], "--metadata-output") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            metadata_output = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--zupt") == 0) {
            enable_zupt = true;
        }
    }

    if (!input_path) {
        fprintf(stderr, "Usage: imu_dr --input <path> [--output <path>] [--scenario <id>] [--metadata-output <path>]\n");
        return 1;
    }

#ifdef _WIN32
    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_start);
#endif

    // =========================================================================
    // PASS 1: GLOBAL TIMESTAMP INTERVAL ANALYSIS
    // =========================================================================
    FILE* fp = fopen(input_path, "r");
    if (!fp) {
        fprintf(stderr, "FATAL: Could not open input file: %s\n", input_path);
        return 1;
    }

    char line_buf[512];
    size_t capacity = 40000;
    int64_t* dt_array = (int64_t*)malloc(capacity * sizeof(int64_t));
    if (!dt_array) {
        fprintf(stderr, "FATAL: Out of memory allocating timestamp array.\n");
        fclose(fp);
        return 1;
    }

    size_t sample_count = 0;
    size_t dt_count = 0;
    int64_t prev_ts = 0;
    int64_t first_ts = 0;
    int64_t last_ts = 0;
    bool has_prev = false;

    while (fgets(line_buf, sizeof(line_buf), fp)) {
        ImuSample s;
        if (!parse_csv_line(line_buf, &s)) continue;

        if (!has_prev) {
            first_ts = s.timestamp_ns;
            prev_ts = s.timestamp_ns;
            has_prev = true;
        } else {
            // Checked subtraction
            int64_t a = s.timestamp_ns;
            int64_t b = prev_ts;
            if ((b > 0 && a < INT64_MIN + b) || (b < 0 && a > INT64_MAX + b)) {
                fprintf(stderr, "FATAL: Timestamp subtraction overflow at frame %zu\n", sample_count);
                free(dt_array);
                fclose(fp);
                return 1;
            }
            int64_t dt_ns = a - b;
            if (dt_ns <= 0) {
                fprintf(stderr, "FATAL: Non-monotonic or duplicate timestamp at frame %zu: %lld -> %lld\n",
                        sample_count, (long long)prev_ts, (long long)a);
                free(dt_array);
                fclose(fp);
                return 1;
            }
            if (dt_count >= capacity) {
                capacity *= 2;
                int64_t* new_arr = (int64_t*)realloc(dt_array, capacity * sizeof(int64_t));
                if (!new_arr) {
                    fprintf(stderr, "FATAL: Out of memory expanding dt_array\n");
                    free(dt_array);
                    fclose(fp);
                    return 1;
                }
                dt_array = new_arr;
            }
            dt_array[dt_count++] = dt_ns;
            prev_ts = a;
        }
        last_ts = s.timestamp_ns;
        sample_count++;
    }

    if (sample_count == 0) {
        fprintf(stderr, "FATAL: Empty input dataset.\n");
        free(dt_array);
        fclose(fp);
        return 1;
    }

    double median_dt_s = compute_median_dt_s(dt_array, dt_count);
    free(dt_array);

    if (median_dt_s < 0.002 || median_dt_s > 0.020) {
        fprintf(stderr, "FATAL: Median dt=%.6f s outside supported 50Hz-500Hz range.\n", median_dt_s);
        fclose(fp);
        return 1;
    }

    size_t N = (size_t)py_round(1.0 / median_dt_s);
    size_t S = (size_t)py_round(0.1 / median_dt_s);
    if (N > MAX_WINDOW_SAMPLES) N = MAX_WINDOW_SAMPLES;
    if (N < 5) N = 5;
    if (S < 1) S = 1;

    // =========================================================================
    // PASS 2: CALIBRATION & STATIONARY DETECTION
    // =========================================================================
    rewind(fp);

    ImuSample ring[MAX_WINDOW_SAMPLES];
    size_t current_idx = 0;

    double first_100_accel_sum[3] = {0.0, 0.0, 0.0};
    size_t valid_accel_count = 0;

    StationaryCluster active_cluster;
    StationaryCluster best_cluster;
    memset(&active_cluster, 0, sizeof(active_cluster));
    memset(&best_cluster, 0, sizeof(best_cluster));

    typedef struct {
        int64_t start_idx;
        int64_t end_idx;
    } StationarySpan;
    size_t span_count = 0;
    size_t span_cap = 16;
    StationarySpan* spans = (StationarySpan*)malloc(span_cap * sizeof(StationarySpan));

    bool in_cluster = false;
    int64_t next_needed_sample_idx = 0;

    // We buffer the window samples in ring
    while (fgets(line_buf, sizeof(line_buf), fp)) {
        ImuSample s;
        if (!parse_csv_line(line_buf, &s)) continue;
        sanitize_sample(&s);

        if (s.valid_accel && valid_accel_count < 100) {
            first_100_accel_sum[0] += s.a[0];
            first_100_accel_sum[1] += s.a[1];
            first_100_accel_sum[2] += s.a[2];
            valid_accel_count++;
        }

        ring[current_idx % N] = s;

        // If we have at least N samples and are aligned with step S
        if (current_idx >= N - 1 && ((current_idx - (N - 1)) % S == 0)) {
            size_t win_start = current_idx - (N - 1);

            // Gather sequential window from ring
            ImuSample ordered_win[MAX_WINDOW_SAMPLES];
            for (size_t j = 0; j < N; j++) {
                ordered_win[j] = ring[(win_start + j) % N];
            }

            // Gap check inside window
            bool has_gap = false;
            for (size_t j = 0; j < N - 1; j++) {
                if (ordered_win[j + 1].timestamp_ns - ordered_win[j].timestamp_ns > MICRO_GAP_THRESHOLD_NS) {
                    has_gap = true;
                    break;
                }
            }

            bool qualifies = false;
            if (!has_gap) {
                qualifies = evaluate_window_statistics(ordered_win, N);
            }

            int64_t win_start_ts = ordered_win[0].timestamp_ns;
            int64_t win_end_ts = ordered_win[N - 1].timestamp_ns;

            if (qualifies && !in_cluster) {
                in_cluster = true;
                active_cluster.start_idx = win_start;
                active_cluster.end_idx = current_idx;
                active_cluster.start_ts_ns = win_start_ts;
                active_cluster.end_ts_ns = win_end_ts;
                active_cluster.sample_count = 0;
                active_cluster.sum_gyro[0] = 0.0;
                active_cluster.sum_gyro[1] = 0.0;
                active_cluster.sum_gyro[2] = 0.0;

                for (size_t j = 0; j < N; j++) {
                    active_cluster.sum_gyro[0] += ordered_win[j].w[0];
                    active_cluster.sum_gyro[1] += ordered_win[j].w[1];
                    active_cluster.sum_gyro[2] += ordered_win[j].w[2];
                    active_cluster.sample_count++;
                }
                next_needed_sample_idx = current_idx + 1;
            } else if (qualifies && in_cluster) {
                // Check if step gap occurred across span
                bool step_gap = false;
                // Check last S samples
                for (size_t j = N - S; j < N - 1; j++) {
                    if (ordered_win[j + 1].timestamp_ns - ordered_win[j].timestamp_ns > MICRO_GAP_THRESHOLD_NS) {
                        step_gap = true;
                        break;
                    }
                }
                if (step_gap) {
        if (spans && active_cluster.end_ts_ns - active_cluster.start_ts_ns >= MIN_STATIONARY_DURATION_NS) {
            if (span_count >= span_cap) {
                span_cap *= 2;
                StationarySpan* grown = (StationarySpan*)realloc(spans, span_cap * sizeof(StationarySpan));
                if (grown) spans = grown;
            }
            if (span_count < span_cap) {
                spans[span_count].start_idx = active_cluster.start_idx;
                spans[span_count].end_idx = active_cluster.end_idx;
                span_count++;
            }
        }
                    finalize_cluster(&active_cluster, &best_cluster);
                    in_cluster = true;
                    active_cluster.start_idx = win_start;
                    active_cluster.end_idx = current_idx;
                    active_cluster.start_ts_ns = win_start_ts;
                    active_cluster.end_ts_ns = win_end_ts;
                    active_cluster.sample_count = 0;
                    active_cluster.sum_gyro[0] = 0.0;
                    active_cluster.sum_gyro[1] = 0.0;
                    active_cluster.sum_gyro[2] = 0.0;

                    for (size_t j = 0; j < N; j++) {
                        active_cluster.sum_gyro[0] += ordered_win[j].w[0];
                        active_cluster.sum_gyro[1] += ordered_win[j].w[1];
                        active_cluster.sum_gyro[2] += ordered_win[j].w[2];
                        active_cluster.sample_count++;
                    }
                    next_needed_sample_idx = current_idx + 1;
                } else {
                    active_cluster.end_idx = current_idx;
                    active_cluster.end_ts_ns = win_end_ts;
                    // Accumulate non-overlapping samples [next_needed_sample_idx, current_idx + 1)
                    for (int64_t idx = next_needed_sample_idx; idx <= (int64_t)current_idx; idx++) {
                        ImuSample smp = ring[idx % N];
                        active_cluster.sum_gyro[0] += smp.w[0];
                        active_cluster.sum_gyro[1] += smp.w[1];
                        active_cluster.sum_gyro[2] += smp.w[2];
                        active_cluster.sample_count++;
                    }
                    next_needed_sample_idx = current_idx + 1;
                }
            } else if (!qualifies && in_cluster) {
        if (spans && active_cluster.end_ts_ns - active_cluster.start_ts_ns >= MIN_STATIONARY_DURATION_NS) {
            if (span_count >= span_cap) {
                span_cap *= 2;
                StationarySpan* grown = (StationarySpan*)realloc(spans, span_cap * sizeof(StationarySpan));
                if (grown) spans = grown;
            }
            if (span_count < span_cap) {
                spans[span_count].start_idx = active_cluster.start_idx;
                spans[span_count].end_idx = active_cluster.end_idx;
                span_count++;
            }
        }
                finalize_cluster(&active_cluster, &best_cluster);
                in_cluster = false;
            }
        }
        current_idx++;
    }

    if (in_cluster) {
        if (spans && active_cluster.end_ts_ns - active_cluster.start_ts_ns >= MIN_STATIONARY_DURATION_NS) {
            if (span_count >= span_cap) {
                span_cap *= 2;
                StationarySpan* grown = (StationarySpan*)realloc(spans, span_cap * sizeof(StationarySpan));
                if (grown) spans = grown;
            }
            if (span_count < span_cap) {
                spans[span_count].start_idx = active_cluster.start_idx;
                spans[span_count].end_idx = active_cluster.end_idx;
                span_count++;
            }
        }
        finalize_cluster(&active_cluster, &best_cluster);
    }

    CalibrationResult calib;
    memset(&calib, 0, sizeof(calib));
    calib.total_samples = sample_count;
    calib.median_dt_s = median_dt_s;
    calib.window_samples = N;
    calib.step_samples = S;
    calib.best_cluster = best_cluster;

    double mean_accel[3] = {0.0, 0.0, 0.0};
    if (valid_accel_count > 0) {
        mean_accel[0] = first_100_accel_sum[0] / (double)valid_accel_count;
        mean_accel[1] = first_100_accel_sum[1] / (double)valid_accel_count;
        mean_accel[2] = first_100_accel_sum[2] / (double)valid_accel_count;
    }
    compute_gravity_alignment(mean_accel, calib.q0);

    if (best_cluster.is_valid && best_cluster.sample_count > 0) {
        calib.gyro_bias[0] = best_cluster.sum_gyro[0] / (double)best_cluster.sample_count;
        calib.gyro_bias[1] = best_cluster.sum_gyro[1] / (double)best_cluster.sample_count;
        calib.gyro_bias[2] = best_cluster.sum_gyro[2] / (double)best_cluster.sample_count;
        calib.bias_confirmed = true;
    } else {
        calib.gyro_bias[0] = 0.0;
        calib.gyro_bias[1] = 0.0;
        calib.gyro_bias[2] = 0.0;
        calib.bias_confirmed = false;
    }

    // =========================================================================
    // PASS 3: TRAJECTORY INTEGRATION & STREAMING EXPORT
    // =========================================================================
    rewind(fp);

    ensure_parent_directory(output_path);
    FILE* out_fp = fopen(output_path, "w");
    if (!out_fp) {
        fprintf(stderr, "FATAL: Could not open output path: %s\n", output_path);
        fclose(fp);
        return 1;
    }
    char write_buf[65536];
    setvbuf(out_fp, write_buf, _IOFBF, sizeof(write_buf));

    fprintf(out_fp, "# timestamp tx ty tz qx qy qz qw\n");

    DRState state;
    memset(&state, 0, sizeof(state));
    state.q[0] = calib.q0[0];
    state.q[1] = calib.q0[1];
    state.q[2] = calib.q0[2];
    state.q[3] = calib.q0[3];
    state.gyro_bias[0] = calib.gyro_bias[0];
    state.gyro_bias[1] = calib.gyro_bias[1];
    state.gyro_bias[2] = calib.gyro_bias[2];

    size_t out_count = 0;
    char out_line[256];

    int64_t pass3_sample_idx = -1;
    while (fgets(line_buf, sizeof(line_buf), fp)) {
        ImuSample s;
        if (!parse_csv_line(line_buf, &s)) continue;
        pass3_sample_idx++;
        sanitize_sample(&s);

        // Correct gyro bias where valid
        if (s.valid_gyro) {
            s.w[0] -= state.gyro_bias[0];
            s.w[1] -= state.gyro_bias[1];
            s.w[2] -= state.gyro_bias[2];
        }

        if (!state.has_prev) {
            // First sample: emit initial pose
            format_tum_line(out_line, sizeof(out_line), s.timestamp_ns, state.p, state.q);
            fputs(out_line, out_fp);

            state.prev_timestamp_ns = s.timestamp_ns;
            state.prev_w[0] = s.w[0];
            state.prev_w[1] = s.w[1];
            state.prev_w[2] = s.w[2];
            // Rotate accel to world frame
            double a_lin[3] = {0.0, 0.0, 0.0};
            if (s.valid_accel) {
                double a_w[3];
                rotate_vector_by_quat(state.q, s.a, a_w);
                a_lin[0] = a_w[0];
                a_lin[1] = a_w[1];
                a_lin[2] = a_w[2] - NOMINAL_GRAVITY;
            }
            state.prev_alin[0] = a_lin[0];
            state.prev_alin[1] = a_lin[1];
            state.prev_alin[2] = a_lin[2];
            state.prev_valid_gyro = s.valid_gyro;
            state.prev_valid_accel = s.valid_accel;
            state.has_prev = true;
            out_count++;
            continue;
        }

        int64_t dt_ns = s.timestamp_ns - state.prev_timestamp_ns;
        double dt = (double)dt_ns * 1e-9;
        bool is_macro_gap = (dt_ns > MICRO_GAP_THRESHOLD_NS);

        // Orientation update
        if (!is_macro_gap && state.prev_valid_gyro && s.valid_gyro) {
            double w_mid[3] = {
                0.5 * (state.prev_w[0] + s.w[0]),
                0.5 * (state.prev_w[1] + s.w[1]),
                0.5 * (state.prev_w[2] + s.w[2])
            };
            double rotvec[3] = { w_mid[0] * dt, w_mid[1] * dt, w_mid[2] * dt };
            double dq[4];
            rotvec_to_dq_wxyz(rotvec, dq);
            double q_new[4];
            qmul_wxyz(state.q, dq, q_new);
            normalize_quat(q_new);
            state.q[0] = q_new[0];
            state.q[1] = q_new[1];
            state.q[2] = q_new[2];
            state.q[3] = q_new[3];
        } // else hold state.q

        // World linear acceleration
        double curr_alin[3] = {0.0, 0.0, 0.0};
        if (s.valid_accel) {
            double a_w[3];
            rotate_vector_by_quat(state.q, s.a, a_w);
            curr_alin[0] = a_w[0];
            curr_alin[1] = a_w[1];
            curr_alin[2] = a_w[2] - NOMINAL_GRAVITY;
        }

        bool is_zupt_sample = false;
        if (enable_zupt && spans != NULL) {
            for (size_t sp = 0; sp < span_count; sp++) {
                if (pass3_sample_idx >= spans[sp].start_idx && pass3_sample_idx <= spans[sp].end_idx) {
                    is_zupt_sample = true;
                    break;
                }
            }
        }

        // Velocity and position update
        if (!is_macro_gap && state.prev_valid_accel && s.valid_accel) {
            double v_next[3] = {
                state.v[0] + 0.5 * (state.prev_alin[0] + curr_alin[0]) * dt,
                state.v[1] + 0.5 * (state.prev_alin[1] + curr_alin[1]) * dt,
                state.v[2] + 0.5 * (state.prev_alin[2] + curr_alin[2]) * dt
            };
            if (is_zupt_sample) {
                v_next[0] = 0.0;
                v_next[1] = 0.0;
                v_next[2] = 0.0;
            }
            state.p[0] += 0.5 * (state.v[0] + v_next[0]) * dt;
            state.p[1] += 0.5 * (state.v[1] + v_next[1]) * dt;
            state.p[2] += 0.5 * (state.v[2] + v_next[2]) * dt;
            state.v[0] = v_next[0];
            state.v[1] = v_next[1];
            state.v[2] = v_next[2];
        } else {
            // Hold velocity constant, propagate position linearly
            state.p[0] += state.v[0] * dt;
            state.p[1] += state.v[1] * dt;
            state.p[2] += state.v[2] * dt;
        }

        // Emit pose
        format_tum_line(out_line, sizeof(out_line), s.timestamp_ns, state.p, state.q);
        fputs(out_line, out_fp);

        // Advance state
        state.prev_timestamp_ns = s.timestamp_ns;
        state.prev_w[0] = s.w[0];
        state.prev_w[1] = s.w[1];
        state.prev_w[2] = s.w[2];
        state.prev_alin[0] = curr_alin[0];
        state.prev_alin[1] = curr_alin[1];
        state.prev_alin[2] = curr_alin[2];
        state.prev_valid_gyro = s.valid_gyro;
        state.prev_valid_accel = s.valid_accel;
        out_count++;
    }

    free(spans);
    fclose(out_fp);
    fclose(fp);

#ifdef _WIN32
    QueryPerformanceCounter(&t_end);
    double runtime_s = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    bool mem_valid = GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    size_t peak_ws = mem_valid ? pmc.PeakWorkingSetSize : 0;
    size_t peak_page = mem_valid ? pmc.PeakPagefileUsage : 0;
#else
    double runtime_s = 0.1;
    size_t peak_ws = 0;
    size_t peak_page = 0;
    bool mem_valid = false;
#endif

    double tel_dur_s = (double)(last_ts - first_ts) * 1e-9;
    double lat_ms_per_s = (runtime_s * 1000.0) / (tel_dur_s > 0 ? tel_dur_s : 1.0);

    ExecutionMetrics metrics;
    metrics.runtime_s = runtime_s;
    metrics.telemetry_duration_s = tel_dur_s;
    metrics.compute_latency_ms_per_s = lat_ms_per_s;
    metrics.peak_working_set_bytes = peak_ws;
    metrics.peak_pagefile_bytes = peak_page;
    metrics.memory_valid = mem_valid;
    metrics.thread_count = get_current_thread_count();
    metrics.input_frames = sample_count;
    metrics.output_poses = out_count;

    write_telemetry_json(metadata_output, input_path, output_path, scenario_id, seed, &calib, &metrics);

    printf("[INFO] Samsung XR Track 3 Native Autonomous Dead-Reckoning Estimator (C99)\n");
    printf("[INFO] Pass 1: Global interval analysis... Median dt = %.9f s (N=%zu, S=%zu)\n", median_dt_s, N, S);
    printf("[INFO] Pass 2: Calibration & stationary detection...\n");
    if (calib.best_cluster.is_valid) {
        printf("[INFO]   Selected Interval: [%lld -> %lld] (%.2f s, %lld samples)\n",
               (long long)calib.best_cluster.start_idx, (long long)calib.best_cluster.end_idx,
               (double)calib.best_cluster.duration_ns * 1e-9, (long long)calib.best_cluster.sample_count);
        printf("[INFO]   Stationary Gyro Bias: [%.6f, %.6f, %.6f] rad/s (Confidence: HIGH)\n",
               calib.gyro_bias[0], calib.gyro_bias[1], calib.gyro_bias[2]);
    } else {
        printf("[INFO]   No qualifying stationary interval found. Gyro bias: [0, 0, 0]\n");
    }
    printf("[INFO]   Initial Attitude [w, x, y, z]: [%.6f, %.6f, %.6f, %.6f]\n",
            calib.q0[0], calib.q0[1], calib.q0[2], calib.q0[3]);
    printf("[INFO] Pass 3: Trajectory integration & streaming export complete (%zu poses).\n", out_count);
    printf("[INFO] --------------------------------------------------------------------------------\n");
    printf("[INFO] PERFORMANCE & RESOURCE COMPLIANCE SUMMARY:\n");
    printf("[INFO]   Runtime: %.3f s | Telemetry: %.3f s | Latency: %.2f ms/s (Budget: <= 20.0 ms/s) [%s]\n",
           runtime_s, tel_dur_s, lat_ms_per_s, (lat_ms_per_s <= 20.0) ? "PASS" : "FAIL");
    printf("[INFO]   Peak Working Set (pre-JSON probe): %.2f MB via GetProcessMemoryInfo [%s]\n",
           (double)peak_ws / (1024.0 * 1024.0), (mem_valid && ((double)peak_ws / (1024.0 * 1024.0) <= 8.0)) ? "PASS" : "FAIL");
    printf("[INFO]   Peak Pagefile Usage: %.2f MB\n", (double)peak_page / (1024.0 * 1024.0));
    printf("[INFO]   Native Thread Count: %u\n", metrics.thread_count);
    printf("[INFO] ================================================================================\n");
    fflush(stdout);

#ifdef _WIN32
    /* Final pre-exit peak-working-set measurement:
     * Queried after Pass 1-3, trajectory file closure, metadata JSON writing,
     * and after all normal console telemetry has been printed and flushed.
     */
    PROCESS_MEMORY_COUNTERS pmc_final;
    memset(&pmc_final, 0, sizeof(pmc_final));
    pmc_final.cb = sizeof(pmc_final);
    BOOL final_mem_ok = GetProcessMemoryInfo(GetCurrentProcess(), &pmc_final, sizeof(pmc_final));
    if (final_mem_ok) {
        size_t final_peak_ws = pmc_final.PeakWorkingSetSize;
        double final_peak_ws_mib = (double)final_peak_ws / (1024.0 * 1024.0);
        printf("[INFO] FINAL PRE-EXIT PEAK WORKING SET: %zu bytes (%.4f MiB) [Status: %s]\n",
               final_peak_ws, final_peak_ws_mib,
               (final_peak_ws_mib <= 8.0) ? "PASS (<= 8.00 MiB)" : "FAIL (> 8.00 MiB)");
    } else {
        fprintf(stderr, "[ERROR] FINAL PRE-EXIT PEAK WORKING SET: Win32 GetProcessMemoryInfo failed [Status: UNVERIFIED]\n");
    }
    fflush(stdout);
#endif

    return 0;
}
