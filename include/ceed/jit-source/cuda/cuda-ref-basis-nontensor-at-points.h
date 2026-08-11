// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA non-tensor product basis with AtPoints evaluation
#include <ceed/types.h>

#define CEED_BASIS_NONTENSOR_MODAL_LINE 1
#define CEED_BASIS_NONTENSOR_MODAL_TRIANGLE 2
#define CEED_BASIS_NONTENSOR_MODAL_TET 3
#define CEED_BASIS_NONTENSOR_MODAL_PRISM 4
#define CEED_BASIS_NONTENSOR_MODAL_PYRAMID 5

//------------------------------------------------------------------------------
// Non-tensor modal values
//------------------------------------------------------------------------------
template <int DEGREE>
inline __device__ void PowersAtPoint(const CeedScalar x, CeedScalar *x_pow) {
  x_pow[0] = 1.0;
  for (CeedInt i = 1; i <= DEGREE; i++) x_pow[i] = x * x_pow[i - 1];
}

inline __device__ CeedScalar PowerDerivativeAtPoint(const CeedInt degree, const CeedScalar *powers) {
  return degree ? degree * powers[degree - 1] : 0.0;
}

template <int DEGREE, int NUM_MODES>
inline __device__ void NonTensorModesAtPoint(const CeedScalar x, const CeedScalar y, const CeedScalar z, CeedScalar *phi) {
  CeedScalar x_pow[DEGREE + 1], y_pow[DEGREE + 1], z_pow[DEGREE + 1];
  CeedInt    mode = 0;

  PowersAtPoint<DEGREE>(x, x_pow);
  PowersAtPoint<DEGREE>(y, y_pow);
  PowersAtPoint<DEGREE>(z, z_pow);

#if BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_LINE
  for (CeedInt i = 0; i <= DEGREE; i++) phi[mode++] = x_pow[i];
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_TRIANGLE
  for (CeedInt total = 0; total <= DEGREE; total++) {
    for (CeedInt i = 0; i <= total; i++) {
      const CeedInt j = total - i;

      phi[mode++] = x_pow[i] * y_pow[j];
    }
  }
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_TET
  for (CeedInt total = 0; total <= DEGREE; total++) {
    for (CeedInt i = 0; i <= total; i++) {
      for (CeedInt j = 0; j <= total - i; j++) {
        const CeedInt k = total - i - j;

        phi[mode++] = x_pow[i] * y_pow[j] * z_pow[k];
      }
    }
  }
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_PRISM
  for (CeedInt k = 0; k <= DEGREE; k++) {
    for (CeedInt total = 0; total <= DEGREE; total++) {
      for (CeedInt i = 0; i <= total; i++) {
        const CeedInt j = total - i;

        phi[mode++] = x_pow[i] * y_pow[j] * z_pow[k];
      }
    }
  }
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_PYRAMID
  const CeedScalar one_minus_z = 1.0 - z;
  CeedScalar       one_minus_z_pow[DEGREE + 1];

  PowersAtPoint<DEGREE>(one_minus_z, one_minus_z_pow);
  for (CeedInt k = 0; k <= DEGREE; k++) {
    const CeedInt xy_degree = DEGREE - k;

    for (CeedInt i = 0; i <= xy_degree; i++) {
      for (CeedInt j = 0; j <= xy_degree; j++) {
        const CeedInt denominator_degree = i < j ? i : j;

        phi[mode++] =
            one_minus_z == 0.0 ? ((i == 0 && j == 0) ? z_pow[k] : 0.0) : x_pow[i] * y_pow[j] * z_pow[k] / one_minus_z_pow[denominator_degree];
      }
    }
  }
#endif
}

//------------------------------------------------------------------------------
// Non-tensor modal gradients
//------------------------------------------------------------------------------
template <int DEGREE, int NUM_MODES>
inline __device__ void NonTensorModeGradientsAtPoint(const CeedScalar x, const CeedScalar y, const CeedScalar z, CeedScalar *grad_phi) {
  CeedScalar x_pow[DEGREE + 1], y_pow[DEGREE + 1], z_pow[DEGREE + 1];
  CeedInt    mode = 0;

  PowersAtPoint<DEGREE>(x, x_pow);
  PowersAtPoint<DEGREE>(y, y_pow);
  PowersAtPoint<DEGREE>(z, z_pow);

#if BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_LINE
  for (CeedInt i = 0; i <= DEGREE; i++) grad_phi[mode++] = PowerDerivativeAtPoint(i, x_pow);
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_TRIANGLE
  for (CeedInt total = 0; total <= DEGREE; total++) {
    for (CeedInt i = 0; i <= total; i++) {
      const CeedInt j = total - i;

      grad_phi[0 * NUM_MODES + mode] = PowerDerivativeAtPoint(i, x_pow) * y_pow[j];
      grad_phi[1 * NUM_MODES + mode] = x_pow[i] * PowerDerivativeAtPoint(j, y_pow);
      mode++;
    }
  }
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_TET
  for (CeedInt total = 0; total <= DEGREE; total++) {
    for (CeedInt i = 0; i <= total; i++) {
      for (CeedInt j = 0; j <= total - i; j++) {
        const CeedInt k = total - i - j;

        grad_phi[0 * NUM_MODES + mode] = PowerDerivativeAtPoint(i, x_pow) * y_pow[j] * z_pow[k];
        grad_phi[1 * NUM_MODES + mode] = x_pow[i] * PowerDerivativeAtPoint(j, y_pow) * z_pow[k];
        grad_phi[2 * NUM_MODES + mode] = x_pow[i] * y_pow[j] * PowerDerivativeAtPoint(k, z_pow);
        mode++;
      }
    }
  }
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_PRISM
  for (CeedInt k = 0; k <= DEGREE; k++) {
    for (CeedInt total = 0; total <= DEGREE; total++) {
      for (CeedInt i = 0; i <= total; i++) {
        const CeedInt j = total - i;

        grad_phi[0 * NUM_MODES + mode] = PowerDerivativeAtPoint(i, x_pow) * y_pow[j] * z_pow[k];
        grad_phi[1 * NUM_MODES + mode] = x_pow[i] * PowerDerivativeAtPoint(j, y_pow) * z_pow[k];
        grad_phi[2 * NUM_MODES + mode] = x_pow[i] * y_pow[j] * PowerDerivativeAtPoint(k, z_pow);
        mode++;
      }
    }
  }
#elif BASIS_MODAL_TOPOLOGY == CEED_BASIS_NONTENSOR_MODAL_PYRAMID
  const CeedScalar one_minus_z = 1.0 - z;
  CeedScalar       one_minus_z_pow[DEGREE + 1], half_pow[2 * DEGREE + 1];

  PowersAtPoint<DEGREE>(one_minus_z, one_minus_z_pow);
  PowersAtPoint<2 * DEGREE>(0.5, half_pow);
  for (CeedInt k = 0; k <= DEGREE; k++) {
    const CeedInt xy_degree = DEGREE - k;

    for (CeedInt i = 0; i <= xy_degree; i++) {
      for (CeedInt j = 0; j <= xy_degree; j++) {
        const CeedInt denominator_degree = i < j ? i : j, max_xy_degree = i > j ? i : j;

        if (one_minus_z == 0.0) {
          // Match MFEM's conventional centerline value for the direction-dependent apex gradient.
          grad_phi[0 * NUM_MODES + mode] = i && max_xy_degree == 1 ? i * half_pow[i - 1 + j] * z_pow[k] : 0.0;
          grad_phi[1 * NUM_MODES + mode] = j && max_xy_degree == 1 ? j * half_pow[i + j - 1] * z_pow[k] : 0.0;
          grad_phi[2 * NUM_MODES + mode] = k && max_xy_degree == 0 ? k * z_pow[k - 1] : 0.0;
          if (denominator_degree && max_xy_degree == 1) grad_phi[2 * NUM_MODES + mode] += denominator_degree * half_pow[i + j] * z_pow[k];
        } else {
          const CeedScalar denominator = one_minus_z_pow[denominator_degree];

          grad_phi[0 * NUM_MODES + mode] = PowerDerivativeAtPoint(i, x_pow) * y_pow[j] * z_pow[k] / denominator;
          grad_phi[1 * NUM_MODES + mode] = x_pow[i] * PowerDerivativeAtPoint(j, y_pow) * z_pow[k] / denominator;
          grad_phi[2 * NUM_MODES + mode] = x_pow[i] * y_pow[j] * PowerDerivativeAtPoint(k, z_pow) / denominator;
          if (denominator_degree) grad_phi[2 * NUM_MODES + mode] += denominator_degree * x_pow[i] * y_pow[j] * z_pow[k] / (denominator * one_minus_z);
        }
        mode++;
      }
    }
  }
#endif
}

//------------------------------------------------------------------------------
// Non-tensor interpolation at points
//------------------------------------------------------------------------------
extern "C" __global__ void InterpAtPoints(const CeedInt num_elem, const CeedInt points_per_elem_stride, const CeedScalar *__restrict__ d_B,
                                          const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ d_X,
                                          const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  const CeedInt  t_id          = threadIdx.x;
  const CeedSize u_elem_stride = BASIS_P;
  const CeedSize v_elem_stride = points_per_elem_stride;
  const CeedSize u_comp_stride = (CeedSize)num_elem * BASIS_P;
  const CeedSize v_comp_stride = (CeedSize)num_elem * points_per_elem_stride;
  const CeedSize v_q_stride    = (CeedSize)BASIS_NUM_COMP * num_elem * points_per_elem_stride;

  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
    const CeedInt num_points = points_per_elem[elem];

    for (CeedInt p = t_id; p < num_points; p += blockDim.x) {
      const CeedScalar x = d_X[elem * v_elem_stride + 0 * v_comp_stride + p];
#if BASIS_DIM > 1
      const CeedScalar y = d_X[elem * v_elem_stride + 1 * v_comp_stride + p];
#else
      const CeedScalar y = 0.0;
#endif
#if BASIS_DIM > 2
      const CeedScalar z = d_X[elem * v_elem_stride + 2 * v_comp_stride + p];
#else
      const CeedScalar z = 0.0;
#endif
      CeedScalar phi[BASIS_NUM_MODES];

      NonTensorModesAtPoint<BASIS_POLY_DEGREE, BASIS_NUM_MODES>(x, y, z, phi);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        const CeedScalar *U = &d_U[elem * u_elem_stride + comp * u_comp_stride];

        for (CeedInt d = 0; d < BASIS_Q_COMP_INTERP; d++) {
          CeedScalar value = 0.0;

          for (CeedInt node = 0; node < BASIS_P; node++) {
            CeedScalar shape = 0.0;

            for (CeedInt mode = 0; mode < BASIS_NUM_MODES; mode++) {
              shape += d_B[(CeedSize)node + (CeedSize)mode * BASIS_P + (CeedSize)d * BASIS_P * BASIS_NUM_MODES] * phi[mode];
            }
            value += shape * U[node];
          }
          d_V[elem * v_elem_stride + comp * v_comp_stride + d * v_q_stride + p] = value;
        }
      }
    }
  }
}

extern "C" __global__ void InterpTransposeAtPoints(const CeedInt num_elem, const CeedInt points_per_elem_stride, const CeedScalar *__restrict__ d_B,
                                                   const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ d_X,
                                                   const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  const CeedInt  t_id          = threadIdx.x;
  const CeedSize u_elem_stride = points_per_elem_stride;
  const CeedSize v_elem_stride = BASIS_P;
  const CeedSize u_comp_stride = (CeedSize)num_elem * points_per_elem_stride;
  const CeedSize v_comp_stride = (CeedSize)num_elem * BASIS_P;
  const CeedSize u_q_stride    = (CeedSize)BASIS_NUM_COMP * num_elem * points_per_elem_stride;

  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
    const CeedInt num_points = points_per_elem[elem];

    for (CeedInt node = t_id; node < BASIS_P; node += blockDim.x) {
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        CeedScalar value = 0.0;

        for (CeedInt p = 0; p < num_points; p++) {
          const CeedScalar x = d_X[elem * u_elem_stride + 0 * u_comp_stride + p];
#if BASIS_DIM > 1
          const CeedScalar y = d_X[elem * u_elem_stride + 1 * u_comp_stride + p];
#else
          const CeedScalar y = 0.0;
#endif
#if BASIS_DIM > 2
          const CeedScalar z = d_X[elem * u_elem_stride + 2 * u_comp_stride + p];
#else
          const CeedScalar z = 0.0;
#endif
          CeedScalar phi[BASIS_NUM_MODES];

          NonTensorModesAtPoint<BASIS_POLY_DEGREE, BASIS_NUM_MODES>(x, y, z, phi);
          for (CeedInt d = 0; d < BASIS_Q_COMP_INTERP; d++) {
            CeedScalar shape = 0.0;

            for (CeedInt mode = 0; mode < BASIS_NUM_MODES; mode++) {
              shape += d_B[(CeedSize)node + (CeedSize)mode * BASIS_P + (CeedSize)d * BASIS_P * BASIS_NUM_MODES] * phi[mode];
            }
            value += shape * d_U[elem * u_elem_stride + comp * u_comp_stride + d * u_q_stride + p];
          }
        }
        d_V[elem * v_elem_stride + comp * v_comp_stride + node] += value;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Non-tensor H^1 gradient at points
//------------------------------------------------------------------------------
extern "C" __global__ void GradAtPoints(const CeedInt num_elem, const CeedInt points_per_elem_stride, const CeedScalar *__restrict__ d_B,
                                        const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ d_X,
                                        const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  const CeedInt  t_id          = threadIdx.x;
  const CeedSize u_elem_stride = BASIS_P;
  const CeedSize v_elem_stride = points_per_elem_stride;
  const CeedSize u_comp_stride = (CeedSize)num_elem * BASIS_P;
  const CeedSize v_comp_stride = (CeedSize)num_elem * points_per_elem_stride;
  const CeedSize v_dim_stride  = (CeedSize)BASIS_NUM_COMP * num_elem * points_per_elem_stride;

  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
    const CeedInt num_points = points_per_elem[elem];

    for (CeedInt p = t_id; p < num_points; p += blockDim.x) {
      const CeedScalar x = d_X[elem * v_elem_stride + 0 * v_comp_stride + p];
#if BASIS_DIM > 1
      const CeedScalar y = d_X[elem * v_elem_stride + 1 * v_comp_stride + p];
#else
      const CeedScalar y = 0.0;
#endif
#if BASIS_DIM > 2
      const CeedScalar z = d_X[elem * v_elem_stride + 2 * v_comp_stride + p];
#else
      const CeedScalar z = 0.0;
#endif
      CeedScalar grad_phi[BASIS_DIM * BASIS_NUM_MODES];

      NonTensorModeGradientsAtPoint<BASIS_POLY_DEGREE, BASIS_NUM_MODES>(x, y, z, grad_phi);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        const CeedScalar *U = &d_U[elem * u_elem_stride + comp * u_comp_stride];

        for (CeedInt d = 0; d < BASIS_DIM; d++) {
          CeedScalar value = 0.0;

          for (CeedInt node = 0; node < BASIS_P; node++) {
            CeedScalar shape_grad = 0.0;

            for (CeedInt mode = 0; mode < BASIS_NUM_MODES; mode++) {
              shape_grad +=
                  d_B[(CeedSize)node + (CeedSize)mode * BASIS_P + (CeedSize)d * BASIS_P * BASIS_NUM_MODES] * grad_phi[d * BASIS_NUM_MODES + mode];
            }
            value += shape_grad * U[node];
          }
          d_V[elem * v_elem_stride + comp * v_comp_stride + d * v_dim_stride + p] = value;
        }
      }
    }
  }
}

extern "C" __global__ void GradTransposeAtPoints(const CeedInt num_elem, const CeedInt points_per_elem_stride, const CeedScalar *__restrict__ d_B,
                                                 const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ d_X,
                                                 const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  const CeedInt  t_id          = threadIdx.x;
  const CeedSize u_elem_stride = points_per_elem_stride;
  const CeedSize v_elem_stride = BASIS_P;
  const CeedSize u_comp_stride = (CeedSize)num_elem * points_per_elem_stride;
  const CeedSize v_comp_stride = (CeedSize)num_elem * BASIS_P;
  const CeedSize u_dim_stride  = (CeedSize)BASIS_NUM_COMP * num_elem * points_per_elem_stride;

  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
    const CeedInt num_points = points_per_elem[elem];

    for (CeedInt node = t_id; node < BASIS_P; node += blockDim.x) {
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        CeedScalar value = 0.0;

        for (CeedInt p = 0; p < num_points; p++) {
          const CeedScalar x = d_X[elem * u_elem_stride + 0 * u_comp_stride + p];
#if BASIS_DIM > 1
          const CeedScalar y = d_X[elem * u_elem_stride + 1 * u_comp_stride + p];
#else
          const CeedScalar y = 0.0;
#endif
#if BASIS_DIM > 2
          const CeedScalar z = d_X[elem * u_elem_stride + 2 * u_comp_stride + p];
#else
          const CeedScalar z = 0.0;
#endif
          CeedScalar grad_phi[BASIS_DIM * BASIS_NUM_MODES];

          NonTensorModeGradientsAtPoint<BASIS_POLY_DEGREE, BASIS_NUM_MODES>(x, y, z, grad_phi);
          for (CeedInt d = 0; d < BASIS_DIM; d++) {
            CeedScalar shape_grad = 0.0;

            for (CeedInt mode = 0; mode < BASIS_NUM_MODES; mode++) {
              shape_grad +=
                  d_B[(CeedSize)node + (CeedSize)mode * BASIS_P + (CeedSize)d * BASIS_P * BASIS_NUM_MODES] * grad_phi[d * BASIS_NUM_MODES + mode];
            }
            value += shape_grad * d_U[elem * u_elem_stride + comp * u_comp_stride + d * u_dim_stride + p];
          }
        }
        d_V[elem * v_elem_stride + comp * v_comp_stride + node] += value;
      }
    }
  }
}
