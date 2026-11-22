#include "dr_math.h"
#include "dr_types.h"
#include <math.h>
#include <stdlib.h>

int64_t py_round(double val) {
    double floor_val = floor(val);
    double diff = val - floor_val;
    if (diff < 0.5) {
        return (int64_t)floor_val;
    } else if (diff > 0.5) {
        return (int64_t)(floor_val + 1.0);
    } else {
        int64_t f_int = (int64_t)floor_val;
        return (f_int % 2 == 0) ? f_int : (f_int + 1);
    }
}

static int compare_int64(const void* a, const void* b) {
    int64_t arg1 = *(const int64_t*)a;
    int64_t arg2 = *(const int64_t*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

double compute_median_dt_s(int64_t* dt_array, size_t count) {
    if (count == 0) return 0.005;
    qsort(dt_array, count, sizeof(int64_t), compare_int64);
    if (count % 2 == 1) {
        size_t mid = count / 2;
        return (double)dt_array[mid] * 1e-9;
    } else {
        size_t mid1 = count / 2 - 1;
        size_t mid2 = count / 2;
        double s1 = (double)dt_array[mid1] * 1e-9;
        double s2 = (double)dt_array[mid2] * 1e-9;
        return 0.5 * (s1 + s2);
    }
}

void rotvec_to_dq_wxyz(const double rv[3], double dq[4]) {
    double angle = sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
    if (angle < 1e-12) {
        dq[0] = 1.0;
        dq[1] = 0.0;
        dq[2] = 0.0;
        dq[3] = 0.0;
        return;
    }
    double half = 0.5 * angle;
    double s = sin(half) / angle;
    dq[0] = cos(half);
    dq[1] = rv[0] * s;
    dq[2] = rv[1] * s;
    dq[3] = rv[2] * s;
}

void qmul_wxyz(const double q1[4], const double q2[4], double q_out[4]) {
    double w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    double w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
    q_out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    q_out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    q_out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    q_out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

void normalize_quat(double q[4]) {
    double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm > 1e-12) {
        double inv = 1.0 / norm;
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
    } else {
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
    }
}

void rotate_vector_by_quat(const double q[4], const double v[3], double v_out[3]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double vx = v[0], vy = v[1], vz = v[2];
    v_out[0] = (1.0 - 2.0 * (y * y + z * z)) * vx
             + 2.0 * (x * y - w * z) * vy
             + 2.0 * (x * z + w * y) * vz;
    v_out[1] = 2.0 * (x * y + w * z) * vx
             + (1.0 - 2.0 * (x * x + z * z)) * vy
             + 2.0 * (y * z - w * x) * vz;
    v_out[2] = 2.0 * (x * z - w * y) * vx
             + 2.0 * (y * z + w * x) * vy
             + (1.0 - 2.0 * (x * x + y * y)) * vz;
}

void compute_gravity_alignment(const double mean_accel[3], double q0[4]) {
    double f_norm = sqrt(mean_accel[0] * mean_accel[0] +
                         mean_accel[1] * mean_accel[1] +
                         mean_accel[2] * mean_accel[2]);
    if (f_norm < 1.0 || isnan(f_norm)) {
        q0[0] = 1.0;
        q0[1] = 0.0;
        q0[2] = 0.0;
        q0[3] = 0.0;
        return;
    }
    double u_S[3] = { mean_accel[0] / f_norm, mean_accel[1] / f_norm, mean_accel[2] / f_norm };
    double dot = u_S[2]; // u_S dot [0, 0, 1]

    if (dot > 1.0 - 1e-12) {
        q0[0] = 1.0; q0[1] = 0.0; q0[2] = 0.0; q0[3] = 0.0;
    } else if (dot < -1.0 + 1e-12) {
        q0[0] = 0.0; q0[1] = 1.0; q0[2] = 0.0; q0[3] = 0.0;
    } else {
        // cross = u_S x z_R = [u_S[1], -u_S[0], 0.0]
        double cross_vec[3] = { u_S[1], -u_S[0], 0.0 };
        double w = 1.0 + dot;
        double norm = sqrt(w * w + cross_vec[0] * cross_vec[0] + cross_vec[1] * cross_vec[1]);
        q0[0] = w / norm;
        q0[1] = cross_vec[0] / norm;
        q0[2] = cross_vec[1] / norm;
        q0[3] = 0.0;
    }
}
