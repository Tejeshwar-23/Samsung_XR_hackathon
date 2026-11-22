#include "dr_detector.h"
#include <math.h>

bool evaluate_window_statistics(const ImuSample* ring, size_t count) {
    if (count < 5) return false;
    double N = (double)count;

    double sum_am = 0.0;
    double sum_wm = 0.0;
    double sum_w[3] = {0.0, 0.0, 0.0};
    double a_m[MAX_WINDOW_SAMPLES];
    double w_m[MAX_WINDOW_SAMPLES];

    for (size_t i = 0; i < count; i++) {
        double am = sqrt(ring[i].a[0]*ring[i].a[0] + ring[i].a[1]*ring[i].a[1] + ring[i].a[2]*ring[i].a[2]);
        double wm = sqrt(ring[i].w[0]*ring[i].w[0] + ring[i].w[1]*ring[i].w[1] + ring[i].w[2]*ring[i].w[2]);
        a_m[i] = am;
        w_m[i] = wm;
        sum_am += am;
        sum_wm += wm;
        sum_w[0] += ring[i].w[0];
        sum_w[1] += ring[i].w[1];
        sum_w[2] += ring[i].w[2];
    }

    double mean_am = sum_am / N;
    // Condition 2: Mean accel close to 1g
    if (fabs(mean_am - NOMINAL_GRAVITY) > 0.35) {
        return false;
    }

    // Condition 1: Accel magnitude std
    double var_am = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = a_m[i] - mean_am;
        var_am += diff * diff;
    }
    double std_am = sqrt(var_am / N);
    if (std_am > 0.25) {
        return false;
    }

    // Condition 3: Gyro magnitude std
    double mean_wm = sum_wm / N;
    double var_wm = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = w_m[i] - mean_wm;
        var_wm += diff * diff;
    }
    double std_wm = sqrt(var_wm / N);
    if (std_wm > 0.015) {
        return false;
    }

    // Condition 4: Per-axis gyro std
    double mean_w_ax[3] = { sum_w[0] / N, sum_w[1] / N, sum_w[2] / N };
    double var_w_ax[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < count; i++) {
        for (int c = 0; c < 3; c++) {
            double diff = ring[i].w[c] - mean_w_ax[c];
            var_w_ax[c] += diff * diff;
        }
    }
    for (int c = 0; c < 3; c++) {
        if (sqrt(var_w_ax[c] / N) > 0.015) {
            return false;
        }
    }

    // Condition 5: Gravity direction stability
    double u_acc[3] = {0.0, 0.0, 0.0};
    double u_arr[MAX_WINDOW_SAMPLES][3];
    for (size_t i = 0; i < count; i++) {
        if (a_m[i] < 1e-6) return false;
        u_arr[i][0] = ring[i].a[0] / a_m[i];
        u_arr[i][1] = ring[i].a[1] / a_m[i];
        u_arr[i][2] = ring[i].a[2] / a_m[i];
        u_acc[0] += u_arr[i][0];
        u_acc[1] += u_arr[i][1];
        u_acc[2] += u_arr[i][2];
    }
    double mean_u[3] = { u_acc[0] / N, u_acc[1] / N, u_acc[2] / N };
    double var_u_total = 0.0;
    for (size_t i = 0; i < count; i++) {
        for (int c = 0; c < 3; c++) {
            double diff = u_arr[i][c] - mean_u[c];
            var_u_total += diff * diff;
        }
    }
    double std_dir = sqrt(var_u_total / N);
    if (std_dir > 0.03) {
        return false;
    }

    return true;
}

void finalize_cluster(StationaryCluster* active, StationaryCluster* best) {
    active->duration_ns = active->end_ts_ns - active->start_ts_ns;
    if (active->duration_ns >= MIN_STATIONARY_DURATION_NS) {
        active->is_valid = true;
        if (!best->is_valid || active->duration_ns > best->duration_ns) {
            *best = *active;
        }
    }
    active->is_valid = false;
}
