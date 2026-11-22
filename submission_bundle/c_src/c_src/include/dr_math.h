#ifndef DR_MATH_H
#define DR_MATH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

int64_t py_round(double val);

double compute_median_dt_s(int64_t* dt_array, size_t count);

void rotvec_to_dq_wxyz(const double rv[3], double dq[4]);

void qmul_wxyz(const double q1[4], const double q2[4], double q_out[4]);

void normalize_quat(double q[4]);

void rotate_vector_by_quat(const double q[4], const double v[3], double v_out[3]);

void compute_gravity_alignment(const double mean_accel[3], double q0[4]);

#endif // DR_MATH_H
