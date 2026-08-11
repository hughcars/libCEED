// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <math.h>
#include <string.h>

#ifdef CEED_MAGMA_USE_HIP
#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#else
#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#endif
#include "ceed-magma-common.h"
#include "ceed-magma.h"

#include "ceed-magma-gemm-nontensor.h"
#include "ceed-magma-gemm-selector.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyCore_Magma(CeedBasis basis, bool apply_add, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u,
                                    CeedVector v) {
  Ceed              ceed;
  Ceed_Magma       *data;
  CeedInt           dim, num_comp, num_nodes, P_1d, Q_1d, P, Q;
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  CeedBasis_Magma  *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  P = P_1d;
  Q = Q_1d;
  if (t_mode == CEED_TRANSPOSE) {
    P = Q_1d;
    Q = P_1d;
  }

  // Read vectors
  if (u != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  } else {
    CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  }
  if (apply_add) {
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  } else {
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));
  }

  // Apply basis operation
  switch (e_mode) {
    case CEED_EVAL_INTERP: {
      // Define element sizes for dofs/quad
      CeedInt elem_qpts_size = CeedIntPow(Q_1d, dim);
      CeedInt elem_dofs_size = CeedIntPow(P_1d, dim);

      // E-vector ordering -------------- Q-vector ordering
      //  component                        component
      //    elem                             elem
      //       node                            node

      // ---  Define strides for NOTRANSPOSE mode: ---
      // Input (d_u) is E-vector, output (d_v) is Q-vector

      // Element strides
      CeedInt u_elem_stride = elem_dofs_size;
      CeedInt v_elem_stride = elem_qpts_size;
      // Component strides
      CeedInt u_comp_stride = num_elem * elem_dofs_size;
      CeedInt v_comp_stride = num_elem * elem_qpts_size;
      if (t_mode == CEED_TRANSPOSE) {
        // Input (d_u) is Q-vector, output (d_v) is E-vector
        // Element strides
        v_elem_stride = elem_dofs_size;
        u_elem_stride = elem_qpts_size;
        // Component strides
        v_comp_stride = num_elem * elem_dofs_size;
        u_comp_stride = num_elem * elem_qpts_size;
      }
      CeedInt num_threads = 1;
      CeedInt num_t_col   = 1;
      CeedInt shared_mem  = 0;
      CeedInt max_P_Q     = CeedIntMax(P, Q);

      switch (dim) {
        case 1:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_1D);
          shared_mem += sizeof(CeedScalar) * num_t_col * (num_comp * (1 * P + 1 * Q));
          shared_mem += sizeof(CeedScalar) * (P * Q);
          break;
        case 2:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_2D);
          shared_mem += P * Q * sizeof(CeedScalar);  // for sT
          // for reforming rU we need P x P, and for the intermediate output we need P x Q
          shared_mem += num_t_col * (P * max_P_Q * sizeof(CeedScalar));
          break;
        case 3:
          num_threads = max_P_Q * max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * (P * Q);  // for sT
          // rU needs P^2 x P, the intermediate output needs max(P^2 x Q, P x Q^2)
          shared_mem += sizeof(CeedScalar) * num_t_col * (CeedIntMax(P * P * max_P_Q, P * Q * Q));
          break;
      }
      CeedInt grid   = CeedDivUpInt(num_elem, num_t_col);
      void   *args[] = {&impl->d_interp_1d, &d_u, &u_elem_stride, &u_comp_stride, &d_v, &v_elem_stride, &v_comp_stride, &num_elem};

      if (t_mode == CEED_TRANSPOSE) {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, apply_add ? impl->InterpTransposeAdd : impl->InterpTranspose, NULL, grid, num_threads,
                                                    num_t_col, 1, shared_mem, args));
      } else {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Interp, NULL, grid, num_threads, num_t_col, 1, shared_mem, args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      // Define element sizes for dofs/quad
      CeedInt elem_qpts_size = CeedIntPow(Q_1d, dim);
      CeedInt elem_dofs_size = CeedIntPow(P_1d, dim);

      // In CEED_NOTRANSPOSE mode:
      // d_u is (P^dim x nc), column-major layout (nc = num_comp)
      // d_v is (Q^dim x nc x dim), column-major layout (nc = num_comp)
      // In CEED_TRANSPOSE mode, the sizes of d_u and d_v are switched.

      // E-vector ordering -------------- Q-vector ordering
      //                                  dim
      //  component                        component
      //    elem                              elem
      //       node                            node

      // ---  Define strides for NOTRANSPOSE mode: ---
      // Input (d_u) is E-vector, output (d_v) is Q-vector

      // Element strides
      CeedInt u_elem_stride = elem_dofs_size;
      CeedInt v_elem_stride = elem_qpts_size;
      // Component strides
      CeedInt u_comp_stride = num_elem * elem_dofs_size;
      CeedInt v_comp_stride = num_elem * elem_qpts_size;
      // Dimension strides
      CeedInt u_dim_stride = 0;
      CeedInt v_dim_stride = num_elem * elem_qpts_size * num_comp;
      if (t_mode == CEED_TRANSPOSE) {
        // Input (d_u) is Q-vector, output (d_v) is E-vector
        // Element strides
        v_elem_stride = elem_dofs_size;
        u_elem_stride = elem_qpts_size;
        // Component strides
        v_comp_stride = num_elem * elem_dofs_size;
        u_comp_stride = num_elem * elem_qpts_size;
        // Dimension strides
        v_dim_stride = 0;
        u_dim_stride = num_elem * elem_qpts_size * num_comp;
      }
      CeedInt num_threads = 1;
      CeedInt num_t_col   = 1;
      CeedInt shared_mem  = 0;
      CeedInt max_P_Q     = CeedIntMax(P, Q);

      switch (dim) {
        case 1:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_1D);
          shared_mem += sizeof(CeedScalar) * num_t_col * (num_comp * (1 * P + 1 * Q));
          shared_mem += sizeof(CeedScalar) * (P * Q);
          break;
        case 2:
          num_threads = max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_2D);
          shared_mem += sizeof(CeedScalar) * 2 * P * Q;  // for sTinterp and sTgrad
          // for reforming rU we need P x P, and for the intermediate output we need P x Q
          shared_mem += sizeof(CeedScalar) * num_t_col * (P * max_P_Q);
          break;
        case 3:
          num_threads = max_P_Q * max_P_Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * 2 * P * Q;  // for sTinterp and sTgrad
          // rU needs P^2 x P, the intermediate outputs need (P^2 x Q + P x Q^2)
          shared_mem += sizeof(CeedScalar) * num_t_col * CeedIntMax(P * P * P, (P * P * Q) + (P * Q * Q));
          break;
      }
      CeedInt grid   = CeedDivUpInt(num_elem, num_t_col);
      void   *args[] = {&impl->d_interp_1d, &impl->d_grad_1d, &d_u,          &u_elem_stride, &u_comp_stride, &u_dim_stride, &d_v,
                        &v_elem_stride,     &v_comp_stride,   &v_dim_stride, &num_elem};

      if (t_mode == CEED_TRANSPOSE) {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, apply_add ? impl->GradTransposeAdd : impl->GradTranspose, NULL, grid, num_threads,
                                                    num_t_col, 1, shared_mem, args));
      } else {
        CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Grad, NULL, grid, num_threads, num_t_col, 1, shared_mem, args));
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCheck(t_mode != CEED_TRANSPOSE, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      CeedCheck(impl->d_q_weight_1d, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weight_1d not set", CeedEvalModes[e_mode]);
      CeedInt elem_dofs_size = CeedIntPow(Q, dim);
      CeedInt num_threads    = 1;
      CeedInt num_t_col      = 1;
      CeedInt shared_mem     = 0;

      switch (dim) {
        case 1:
          num_threads = Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_1D);
          shared_mem += sizeof(CeedScalar) * Q;              // for d_q_weight_1d
          shared_mem += sizeof(CeedScalar) * num_t_col * Q;  // for output
          break;
        case 2:
          num_threads = Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_2D);
          shared_mem += sizeof(CeedScalar) * Q;  // for d_q_weight_1d
          break;
        case 3:
          num_threads = Q * Q;
          num_t_col   = MAGMA_BASIS_NTCOL(num_threads, MAGMA_MAXTHREADS_3D);
          shared_mem += sizeof(CeedScalar) * Q;  // for d_q_weight_1d
          break;
      }
      CeedInt grid   = CeedDivUpInt(num_elem, num_t_col);
      void   *args[] = {&impl->d_q_weight_1d, &d_v, &elem_dofs_size, &num_elem};

      CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Weight, NULL, grid, num_threads, num_t_col, 1, shared_mem, args));
    } break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[e_mode]);
    case CEED_EVAL_NONE:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }

  // Must sync to ensure completeness
  ceed_magma_queue_sync(data->queue);

  // Restore vectors
  if (e_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApply_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyCore_Magma(basis, false, num_elem, t_mode, e_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAdd_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyCore_Magma(basis, true, num_elem, t_mode, e_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - tensor AtPoints
//------------------------------------------------------------------------------
int CeedBasisApplyAtPoints_Magma(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                 CeedVector x_ref, CeedVector u, CeedVector v) {
  return CeedError(CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "Backend does not implement CeedBasisApplyAtPoints");
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor AtPoints
//------------------------------------------------------------------------------
static int CeedSizeMultiplyAtPoints_Magma(Ceed ceed, CeedSize factor_1, CeedSize factor_2, CeedSize *product) {
  CeedCheck(factor_1 >= 0 && factor_2 >= 0, ceed, CEED_ERROR_DIMENSION, "Cannot multiply negative vector dimensions");
  CeedCheck(!factor_1 || factor_2 <= PTRDIFF_MAX / factor_1, ceed, CEED_ERROR_DIMENSION, "AtPoints vector dimension exceeds CeedSize range");
  *product = factor_1 * factor_2;
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisUpdatePointsAtPoints_Magma(Ceed ceed, Ceed_Magma *data, CeedBasisNonTensor_Magma *impl, CeedInt num_elem,
                                               const CeedInt *num_points) {
  CeedSize num_bytes;

  CeedCallBackend(CeedSizeMultiplyAtPoints_Magma(ceed, num_elem, sizeof(CeedInt), &num_bytes));
  if (num_elem == impl->num_elem_at_points && !memcmp(impl->h_points_per_elem, num_points, num_bytes)) return CEED_ERROR_SUCCESS;

  CeedInt *d_points_new = NULL, *h_points_new = NULL;
  int      ierr;

  CeedCallBackend(CeedCalloc(num_elem, &h_points_new));
  memcpy(h_points_new, num_points, num_bytes);
  ierr = magma_malloc((void **)&d_points_new, num_bytes);
  if (ierr) {
    (void)CeedFree(&h_points_new);
    return ierr;
  }
  magma_setvector(num_elem, sizeof(CeedInt), num_points, 1, d_points_new, 1, data->queue);
  ceed_magma_queue_sync(data->queue);
  {
    CeedInt *d_points_old = impl->d_points_per_elem, *h_points_old = impl->h_points_per_elem;

    impl->d_points_per_elem  = d_points_new;
    impl->h_points_per_elem  = h_points_new;
    impl->num_elem_at_points = num_elem;
    ierr                     = d_points_old ? magma_free(d_points_old) : 0;
    CeedCallBackend(CeedFree(&h_points_old));
    if (ierr) return ierr;
  }
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAtPointsNonTensorCore_Magma(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
                                                     CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  Ceed        ceed         = CeedBasisReturnCeed(basis);
  const bool  is_transpose = t_mode == CEED_TRANSPOSE;
  Ceed_Magma *data;
  CeedInt     dim, num_nodes, num_comp, q_comp, modal_topology, degree_at_points, num_modes_at_points, max_num_points = 0, storage_points_per_elem;
  CeedSize    u_len, v_len, points_stride, compact_uniform_size, nodes_len;
  const CeedScalar         *d_x = NULL, *d_u = NULL, *modal_at_points;
  CeedScalar               *d_v = NULL;
  CeedBasisNonTensor_Magma *impl;
  bool                      is_padded, x_array_acquired = false, u_array_acquired = false, v_array_acquired = false;
  int                       ierr = CEED_ERROR_SUCCESS;

  if (eval_mode == CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorSetValue(v, 1.0));
    return CEED_ERROR_SUCCESS;
  }

  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
  CeedCheck(eval_mode == CEED_EVAL_INTERP || eval_mode == CEED_EVAL_GRAD, ceed, CEED_ERROR_UNSUPPORTED,
            "Non-tensor AtPoints only supports CEED_EVAL_INTERP and CEED_EVAL_GRAD");

  for (CeedInt i = 0; i < num_elem; i++) max_num_points = CeedIntMax(max_num_points, num_points[i]);
  if (max_num_points == 0) {
    if (is_transpose && !apply_add) CeedCallBackend(CeedVectorSetValue(v, 0.0));
    return CEED_ERROR_SUCCESS;
  }
  CeedCallBackend(CeedBasisGetAtPointsLayout(basis, num_elem, num_points, t_mode, eval_mode, x_ref, u, v, &is_padded, &points_stride));
  CeedCallBackend(CeedSizeMultiplyAtPoints_Magma(ceed, num_elem, max_num_points, &compact_uniform_size));
  CeedCheck(is_padded || points_stride == compact_uniform_size, ceed, CEED_ERROR_BACKEND,
            "MAGMA non-tensor AtPoints requires compact point storage to have a uniform number of points per element");
  CeedCheck(points_stride / num_elem <= INT_MAX, ceed, CEED_ERROR_DIMENSION, "AtPoints padding exceeds CeedInt range");
  storage_points_per_elem = (CeedInt)(points_stride / num_elem);
  CeedCallBackend(CeedVectorGetLength(u, &u_len));
  CeedCallBackend(CeedVectorGetLength(v, &v_len));
  CeedCallBackend(CeedSizeMultiplyAtPoints_Magma(ceed, num_elem, num_nodes, &nodes_len));
  CeedCallBackend(CeedSizeMultiplyAtPoints_Magma(ceed, nodes_len, num_comp, &nodes_len));
  CeedCheck((is_transpose ? v_len : u_len) >= nodes_len, ceed, CEED_ERROR_BACKEND,
            "Vector at nodes incompatible with non-tensor BasisApplyAtPoints. Found %" CeedSize_FMT ", Required %" CeedSize_FMT,
            is_transpose ? v_len : u_len, nodes_len);

  CeedCallBackend(CeedBasisGetNonTensorAtPoints(basis, eval_mode, &modal_topology, &degree_at_points, &num_modes_at_points, &modal_at_points));

  CeedCallBackend(CeedBasisUpdatePointsAtPoints_Magma(ceed, data, impl, num_elem, num_points));

  {
    CeedScalar **d_modal_at_points = eval_mode == CEED_EVAL_INTERP ? &impl->d_interp_at_points : &impl->d_grad_at_points;

    if (!*d_modal_at_points) {
      CeedScalar *d_modal_new = NULL;
      CeedSize    modal_size, modal_bytes;

      CeedCallBackend(CeedSizeMultiplyAtPoints_Magma(ceed, q_comp, num_modes_at_points, &modal_size));
      CeedCallBackend(CeedSizeMultiplyAtPoints_Magma(ceed, modal_size, num_nodes, &modal_size));
      CeedCheck(modal_size <= INT_MAX, ceed, CEED_ERROR_DIMENSION, "Non-tensor AtPoints modal data exceeds MAGMA index range");
      CeedCallBackend(CeedSizeMultiplyAtPoints_Magma(ceed, modal_size, sizeof(CeedScalar), &modal_bytes));
      CeedCallBackend(magma_malloc((void **)&d_modal_new, modal_bytes));
      magma_setvector((magma_int_t)modal_size, sizeof(CeedScalar), modal_at_points, 1, d_modal_new, 1, data->queue);
      *d_modal_at_points = d_modal_new;
    }
  }

  {
    CeedMagmaModule   *module_at_points = eval_mode == CEED_EVAL_INTERP ? &impl->moduleInterpAtPoints : &impl->moduleGradAtPoints;
    CeedMagmaFunction *kernel_at_points = eval_mode == CEED_EVAL_INTERP ? &impl->InterpAtPoints : &impl->GradAtPoints;
    CeedMagmaFunction *kernel_transpose = eval_mode == CEED_EVAL_INTERP ? &impl->InterpTransposeAtPoints : &impl->GradTransposeAtPoints;
    CeedInt           *kernel_degree    = eval_mode == CEED_EVAL_INTERP ? &impl->interp_kernel_degree_at_points : &impl->grad_kernel_degree_at_points;
    CeedInt *kernel_num_modes = eval_mode == CEED_EVAL_INTERP ? &impl->interp_kernel_num_modes_at_points : &impl->grad_kernel_num_modes_at_points;
    CeedInt *kernel_modal_topology =
        eval_mode == CEED_EVAL_INTERP ? &impl->interp_kernel_modal_topology_at_points : &impl->grad_kernel_modal_topology_at_points;

    if (*kernel_degree != degree_at_points || *kernel_num_modes != num_modes_at_points || *kernel_modal_topology != modal_topology) {
      const char basis_kernel_source[] = "// Nontensor basis AtPoints source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor-at-points.h>\n";
      CeedMagmaModule   module_new     = NULL;
      CeedMagmaFunction kernel_new = NULL, kernel_transpose_new = NULL;
      Ceed              ceed_delegate = NULL;
      CeedInt           q_comp_interp;
      int               ierr, ierr_destroy;

      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
      CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));
      ierr = CeedCompileMagma(ceed_delegate, basis_kernel_source, "basis_nontensor_at_points", &module_new, 7, "BASIS_P", num_nodes, "BASIS_NUM_COMP",
                              num_comp, "BASIS_Q_COMP_INTERP", q_comp_interp, "BASIS_POLY_DEGREE", degree_at_points, "BASIS_NUM_MODES",
                              num_modes_at_points, "BASIS_DIM", dim, "BASIS_MODAL_TOPOLOGY", modal_topology);
      ierr_destroy = CeedDestroy(&ceed_delegate);
      if (!ierr) ierr = ierr_destroy;
      if (ierr) goto compile_cleanup;
      ierr = CeedGetKernelMagma(ceed, module_new, eval_mode == CEED_EVAL_INTERP ? "InterpAtPoints" : "GradAtPoints", &kernel_new);
      if (ierr) goto compile_cleanup;
      ierr = CeedGetKernelMagma(ceed, module_new, eval_mode == CEED_EVAL_INTERP ? "InterpTransposeAtPoints" : "GradTransposeAtPoints",
                                &kernel_transpose_new);
      if (ierr) goto compile_cleanup;
      if (*module_at_points) {
#ifdef CEED_MAGMA_USE_HIP
        const hipError_t unload_result = hipModuleUnload(*module_at_points);

        if (unload_result != hipSuccess) {
          (void)hipModuleUnload(module_new);
          CeedCallHip(ceed, unload_result);
        }
#else
        const CUresult unload_result = cuModuleUnload(*module_at_points);

        if (unload_result != CUDA_SUCCESS) {
          (void)cuModuleUnload(module_new);
          CeedChk_Cu(ceed, unload_result);
        }
#endif
      }
      *module_at_points      = module_new;
      *kernel_at_points      = kernel_new;
      *kernel_transpose      = kernel_transpose_new;
      *kernel_degree         = degree_at_points;
      *kernel_num_modes      = num_modes_at_points;
      *kernel_modal_topology = modal_topology;
      goto compile_done;

    compile_cleanup:
      if (module_new) {
#ifdef CEED_MAGMA_USE_HIP
        (void)hipModuleUnload(module_new);
#else
        (void)cuModuleUnload(module_new);
#endif
      }
      return ierr;
    compile_done:;
    }
  }

  ierr = CeedVectorGetArrayRead(x_ref, CEED_MEM_DEVICE, &d_x);
  if (ierr) goto cleanup;
  x_array_acquired = true;
  ierr             = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u);
  if (ierr) goto cleanup;
  u_array_acquired = true;
  if (apply_add) {
    ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v);
  } else {
    if (is_transpose) {
      ierr = CeedVectorSetValue(v, 0.0);
      if (ierr) goto cleanup;
    }
    ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v);
  }
  if (ierr) goto cleanup;
  v_array_acquired = true;
  {
    CeedScalar       *d_B = eval_mode == CEED_EVAL_INTERP ? impl->d_interp_at_points : impl->d_grad_at_points;
    CeedMagmaFunction kernel;
    void             *basis_args[] = {(void *)&num_elem, &storage_points_per_elem, &d_B, &impl->d_points_per_elem, &d_x, &d_u, &d_v};
    CeedInt           block_size;

    if (eval_mode == CEED_EVAL_INTERP) {
      kernel = is_transpose ? impl->InterpTransposeAtPoints : impl->InterpAtPoints;
    } else {
      kernel = is_transpose ? impl->GradTransposeAtPoints : impl->GradAtPoints;
    }
    block_size = CeedIntMin(is_transpose ? num_nodes : max_num_points, 128);
    ierr       = CeedRunKernelMagma(ceed, kernel, num_elem, block_size, basis_args);
    if (ierr) goto cleanup;
  }
  ceed_magma_queue_sync(data->queue);

cleanup:
  if (v_array_acquired) {
    const int cleanup_ierr = CeedVectorRestoreArray(v, &d_v);

    if (!ierr) ierr = cleanup_ierr;
  }
  if (u_array_acquired) {
    const int cleanup_ierr = CeedVectorRestoreArrayRead(u, &d_u);

    if (!ierr) ierr = cleanup_ierr;
  }
  if (x_array_acquired) {
    const int cleanup_ierr = CeedVectorRestoreArrayRead(x_ref, &d_x);

    if (!ierr) ierr = cleanup_ierr;
  }
  return ierr;
}

static int CeedBasisApplyAtPointsNonTensor_Magma(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                                 CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsNonTensorCore_Magma(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddAtPointsNonTensor_Magma(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                                    CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsNonTensorCore_Magma(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorCore_Magma(CeedBasis basis, bool apply_add, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode,
                                             CeedVector u, CeedVector v) {
  Ceed                      ceed;
  Ceed_Magma               *data;
  CeedInt                   num_comp, num_nodes, num_qpts, P, Q, N;
  const CeedScalar         *d_u;
  CeedScalar               *d_v;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  P = num_nodes;
  Q = num_qpts;
  N = num_elem * num_comp;

  // Read vectors
  if (u != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  } else {
    CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  }
  if (apply_add) {
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  } else {
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));
  }

  // Compile kernels for N as needed
  CeedInt iN = 0;
  if (P <= MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P && Q <= MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q && (e_mode != CEED_EVAL_WEIGHT || !impl->Weight)) {
    CeedInt n_array[MAGMA_NONTENSOR_KERNEL_INSTANCES] = {MAGMA_NONTENSOR_KERNEL_N_VALUES};
    CeedInt diff                                      = abs(n_array[iN] - N), idiff;

    for (CeedInt in = iN + 1; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
      idiff = abs(n_array[in] - N);
      if (idiff < diff) {
        iN   = in;
        diff = idiff;
      }
    }

    if (!impl->NB_interp[iN]) {
      CeedFESpace fe_space;
      CeedInt     q_comp_interp, q_comp_deriv;
      Ceed        ceed_delegate;
      magma_int_t arch = magma_getdevice_arch();

      // Tuning parameters for NB
      CeedCallBackend(CeedBasisGetFESpace(basis, &fe_space));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
      switch (fe_space) {
        case CEED_FE_SPACE_H1:
          CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_deriv));
          break;
        case CEED_FE_SPACE_HDIV:
          CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_deriv));
          break;
        case CEED_FE_SPACE_HCURL:
          CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_deriv));
          break;
      }
      impl->NB_interp[iN]   = nontensor_rtc_get_nb(arch, 'n', q_comp_interp, P, Q, n_array[iN]);
      impl->NB_interp_t[iN] = nontensor_rtc_get_nb(arch, 't', q_comp_interp, P, Q, n_array[iN]);
      impl->NB_deriv[iN]    = nontensor_rtc_get_nb(arch, 'n', q_comp_deriv, P, Q, n_array[iN]);
      impl->NB_deriv_t[iN]  = nontensor_rtc_get_nb(arch, 't', q_comp_deriv, P, Q, n_array[iN]);

      // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
      CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

      // Compile kernels
      char basis_kernel_source[] =
          "// Interp and grad basis source\n"
          "#include <ceed/jit-source/magma/magma-basis-interp-deriv-nontensor.h>\n"
          "// Weight basis source\n"
          "#include <ceed/jit-source/magma/magma-basis-weight-nontensor.h>\n";

      CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, "basis_nontensor", &impl->module[iN], 8, "BASIS_Q_COMP_INTERP",
                                       q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_deriv, "BASIS_P", P, "BASIS_Q", Q, "BASIS_NB_INTERP_N",
                                       impl->NB_interp[iN], "BASIS_NB_INTERP_T", impl->NB_interp_t[iN], "BASIS_NB_DERIV_N", impl->NB_deriv[iN],
                                       "BASIS_NB_DERIV_T", impl->NB_deriv_t[iN]));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[iN], "magma_interp_nontensor_n", &impl->Interp[iN]));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[iN], "magma_interp_nontensor_t", &impl->InterpTranspose[iN]));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[iN], "magma_interp_nontensor_ta", &impl->InterpTransposeAdd[iN]));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[iN], "magma_deriv_nontensor_n", &impl->Deriv[iN]));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[iN], "magma_deriv_nontensor_t", &impl->DerivTranspose[iN]));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[iN], "magma_deriv_nontensor_ta", &impl->DerivTransposeAdd[iN]));
      if (!impl->Weight) {
        CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[iN], "magma_weight_nontensor", &impl->Weight));
      }
      CeedCallBackend(CeedDestroy(&ceed_delegate));
    }
  }

  // Apply basis operation
  if (e_mode != CEED_EVAL_WEIGHT) {
    const CeedScalar *d_b = NULL;
    CeedInt           q_comp, NB, M, K;
    CeedMagmaFunction Kernel;

    switch (e_mode) {
      case CEED_EVAL_INTERP:
        d_b = impl->d_interp;
        break;
      case CEED_EVAL_GRAD:
        d_b = impl->d_grad;
        break;
      case CEED_EVAL_DIV:
        d_b = impl->d_div;
        break;
      case CEED_EVAL_CURL:
        d_b = impl->d_curl;
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_NONE:
        return CeedError(ceed, CEED_ERROR_BACKEND, "%s does not make sense in this context", CeedEvalModes[e_mode]);
        // LCOV_EXCL_STOP
    }
    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, e_mode, &q_comp));
    M = (t_mode == CEED_TRANSPOSE) ? P : Q, K = (t_mode == CEED_TRANSPOSE) ? Q : P;

    if (P <= MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P && Q <= MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q) {
      if (e_mode == CEED_EVAL_INTERP) {
        if (t_mode == CEED_TRANSPOSE) {
          Kernel = apply_add ? impl->InterpTransposeAdd[iN] : impl->InterpTranspose[iN];
          NB     = impl->NB_interp_t[iN];
        } else {
          Kernel = impl->Interp[iN];
          NB     = impl->NB_interp[iN];
        }
      } else {
        if (t_mode == CEED_TRANSPOSE) {
          Kernel = apply_add ? impl->DerivTransposeAdd[iN] : impl->DerivTranspose[iN];
          NB     = impl->NB_deriv_t[iN];
        } else {
          Kernel = impl->Deriv[iN];
          NB     = impl->NB_deriv[iN];
        }
      }
      CeedInt num_t_col    = MAGMA_BASIS_NTCOL(M, MAGMA_MAXTHREADS_1D);
      CeedInt grid         = CeedDivUpInt(N, num_t_col * NB);
      CeedInt shared_mem_A = P * Q * sizeof(CeedScalar);
      CeedInt shared_mem_B = num_t_col * K * NB * sizeof(CeedScalar);
      CeedInt shared_mem   = (t_mode != CEED_TRANSPOSE && q_comp > 1) ? (shared_mem_A + shared_mem_B) : CeedIntMax(shared_mem_A, shared_mem_B);
      void   *args[]       = {&N, &d_b, &d_u, &d_v};

      CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, Kernel, NULL, grid, M, num_t_col, 1, shared_mem, args));
    } else {
      for (CeedInt d = 0; d < q_comp; d++) {
        if (t_mode == CEED_TRANSPOSE) {
          const CeedScalar beta = (apply_add || (d > 0)) ? 1.0 : 0.0;
          magma_gemm_nontensor(MagmaNoTrans, MagmaNoTrans, P, N, Q, 1.0, d_b + d * P * Q, P, d_u + d * N * Q, Q, beta, d_v, P, data->queue);
        } else {
          magma_gemm_nontensor(MagmaTrans, MagmaNoTrans, Q, N, P, 1.0, d_b + d * P * Q, P, d_u, P, 0.0, d_v + d * N * Q, Q, data->queue);
        }
      }
    }
  } else {
    CeedCheck(t_mode != CEED_TRANSPOSE, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedCheck(impl->d_q_weight, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weight not set", CeedEvalModes[e_mode]);
    CeedInt num_t_col  = MAGMA_BASIS_NTCOL(Q, MAGMA_MAXTHREADS_1D);
    CeedInt grid       = CeedDivUpInt(num_elem, num_t_col);
    CeedInt shared_mem = Q * sizeof(CeedScalar) + num_t_col * Q * sizeof(CeedScalar);
    void   *args[]     = {&num_elem, &impl->d_q_weight, &d_v};

    CeedCallBackend(CeedRunKernelDimSharedMagma(ceed, impl->Weight, NULL, grid, Q, num_t_col, 1, shared_mem, args));
  }

  // Must sync to ensure completeness
  ceed_magma_queue_sync(data->queue);

  // Restore vectors
  if (e_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyNonTensor_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u,
                                         CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Magma(basis, false, num_elem, t_mode, e_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddNonTensor_Magma(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u,
                                            CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Magma(basis, true, num_elem, t_mode, e_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Magma(CeedBasis basis) {
  Ceed             ceed;
  CeedBasis_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
#ifdef CEED_MAGMA_USE_HIP
  CeedCallHip(ceed, hipModuleUnload(impl->module));
#else
  CeedCallCuda(ceed, cuModuleUnload(impl->module));
#endif
  CeedCallBackend(magma_free(impl->d_interp_1d));
  CeedCallBackend(magma_free(impl->d_grad_1d));
  if (impl->d_q_weight_1d) CeedCallBackend(magma_free(impl->d_q_weight_1d));
  CeedCallBackend(CeedFree(&impl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Magma(CeedBasis basis) {
  Ceed                      ceed;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &impl));
  for (CeedInt in = 0; in < MAGMA_NONTENSOR_KERNEL_INSTANCES; in++) {
    if (impl->module[in]) {
#ifdef CEED_MAGMA_USE_HIP
      CeedCallHip(ceed, hipModuleUnload(impl->module[in]));
#else
      CeedCallCuda(ceed, cuModuleUnload(impl->module[in]));
#endif
    }
  }
  if (impl->moduleInterpAtPoints) {
#ifdef CEED_MAGMA_USE_HIP
    CeedCallHip(ceed, hipModuleUnload(impl->moduleInterpAtPoints));
#else
    CeedCallCuda(ceed, cuModuleUnload(impl->moduleInterpAtPoints));
#endif
  }
  if (impl->moduleGradAtPoints) {
#ifdef CEED_MAGMA_USE_HIP
    CeedCallHip(ceed, hipModuleUnload(impl->moduleGradAtPoints));
#else
    CeedCallCuda(ceed, cuModuleUnload(impl->moduleGradAtPoints));
#endif
  }
  CeedCallBackend(magma_free(impl->d_interp));
  CeedCallBackend(magma_free(impl->d_grad));
  CeedCallBackend(magma_free(impl->d_div));
  CeedCallBackend(magma_free(impl->d_curl));
  if (impl->d_q_weight) CeedCallBackend(magma_free(impl->d_q_weight));
  CeedCallBackend(CeedFree(&impl->h_points_per_elem));
  if (impl->d_points_per_elem) CeedCallBackend(magma_free(impl->d_points_per_elem));
  if (impl->d_interp_at_points) CeedCallBackend(magma_free(impl->d_interp_at_points));
  if (impl->d_grad_at_points) CeedCallBackend(magma_free(impl->d_grad_at_points));
  CeedCallBackend(CeedFree(&impl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                  const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed             ceed, ceed_delegate;
  Ceed_Magma      *data;
  CeedInt          num_comp;
  CeedBasis_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  if (q_weight_1d) {
    CeedCallBackend(magma_malloc((void **)&impl->d_q_weight_1d, Q_1d * sizeof(q_weight_1d[0])));
    magma_setvector(Q_1d, sizeof(q_weight_1d[0]), q_weight_1d, 1, impl->d_q_weight_1d, 1, data->queue);
  }
  CeedCallBackend(magma_malloc((void **)&impl->d_interp_1d, Q_1d * P_1d * sizeof(interp_1d[0])));
  magma_setvector(Q_1d * P_1d, sizeof(interp_1d[0]), interp_1d, 1, impl->d_interp_1d, 1, data->queue);
  CeedCallBackend(magma_malloc((void **)&impl->d_grad_1d, Q_1d * P_1d * sizeof(grad_1d[0])));
  magma_setvector(Q_1d * P_1d, sizeof(grad_1d[0]), grad_1d, 1, impl->d_grad_1d, 1, data->queue);

  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

  // Compile kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  char basis_kernel_source[CEED_MAX_RESOURCE_LEN];

  snprintf(basis_kernel_source, CEED_MAX_RESOURCE_LEN,
           "// Interp basis source\n"
           "#include <ceed/jit-source/magma/magma-basis-interp-%" CeedInt_FMT
           "d.h>\n"
           "// Grad basis source\n"
           "#include <ceed/jit-source/magma/magma-basis-grad-%" CeedInt_FMT
           "d.h>\n"
           "// Weight basis source\n"
           "#include <ceed/jit-source/magma/magma-basis-weight-%" CeedInt_FMT "d.h>\n",
           dim, dim, dim);

  CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, "basis_h1_tensor", &impl->module, 5, "BASIS_DIM", dim, "BASIS_NUM_COMP",
                                   num_comp, "BASIS_P", P_1d, "BASIS_Q", Q_1d, "BASIS_MAX_P_Q", CeedIntMax(P_1d, Q_1d)));
  switch (dim) {
    case 1:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_1d_kernel", &impl->Interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_1d_kernel", &impl->InterpTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpta_1d_kernel", &impl->InterpTransposeAdd));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_1d_kernel", &impl->Grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_1d_kernel", &impl->GradTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradta_1d_kernel", &impl->GradTransposeAdd));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_1d_kernel", &impl->Weight));
      break;
    case 2:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_2d_kernel", &impl->Interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_2d_kernel", &impl->InterpTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpta_2d_kernel", &impl->InterpTransposeAdd));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_2d_kernel", &impl->Grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_2d_kernel", &impl->GradTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradta_2d_kernel", &impl->GradTransposeAdd));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_2d_kernel", &impl->Weight));
      break;
    case 3:
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpn_3d_kernel", &impl->Interp));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpt_3d_kernel", &impl->InterpTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_interpta_3d_kernel", &impl->InterpTransposeAdd));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradn_3d_kernel", &impl->Grad));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradt_3d_kernel", &impl->GradTranspose));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_gradta_3d_kernel", &impl->GradTransposeAdd));
      CeedCallBackend(CeedGetKernelMagma(ceed, impl->module, "magma_weight_3d_kernel", &impl->Weight));
      break;
  }
  CeedCallBackend(CeedBasisSetData(basis, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Magma));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedDestroy(&ceed_delegate));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H^1
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                            const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                      ceed;
  Ceed_Magma               *data;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  if (q_weight) {
    CeedCallBackend(magma_malloc((void **)&impl->d_q_weight, num_qpts * sizeof(q_weight[0])));
    magma_setvector(num_qpts, sizeof(q_weight[0]), q_weight, 1, impl->d_q_weight, 1, data->queue);
  }
  if (interp) {
    CeedInt q_comp_interp;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
    CeedCallBackend(magma_malloc((void **)&impl->d_interp, num_qpts * num_nodes * q_comp_interp * sizeof(interp[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_interp, sizeof(interp[0]), interp, 1, impl->d_interp, 1, data->queue);
  }
  if (grad) {
    CeedInt q_comp_grad;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));
    CeedCallBackend(magma_malloc((void **)&impl->d_grad, num_qpts * num_nodes * q_comp_grad * sizeof(grad[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_grad, sizeof(grad[0]), grad, 1, impl->d_grad, 1, data->queue);
  }

  // Compile the weight kernel if it won't be compiled later on
  if (num_nodes > MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P || num_qpts > MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q) {
    Ceed ceed_delegate;

    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
    CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

    // Compile weight kernel (the remainder of kernel compilation happens at first call to CeedBasisApply)
    const char basis_kernel_source[] = "// Weight basis source\n#include <ceed/jit-source/magma/magma-basis-weight-nontensor.h>\n";

    CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, "basis_h1_nontensor_weight", &impl->module[0], 1, "BASIS_Q", num_qpts));
    CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[0], "magma_weight_nontensor", &impl->Weight));
    CeedCallBackend(CeedDestroy(&ceed_delegate));
  }

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPointsNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPointsNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Magma));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(div)
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                              const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                      ceed;
  Ceed_Magma               *data;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  if (q_weight) {
    CeedCallBackend(magma_malloc((void **)&impl->d_q_weight, num_qpts * sizeof(q_weight[0])));
    magma_setvector(num_qpts, sizeof(q_weight[0]), q_weight, 1, impl->d_q_weight, 1, data->queue);
  }
  if (interp) {
    CeedInt q_comp_interp;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
    CeedCallBackend(magma_malloc((void **)&impl->d_interp, num_qpts * num_nodes * q_comp_interp * sizeof(interp[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_interp, sizeof(interp[0]), interp, 1, impl->d_interp, 1, data->queue);
  }
  if (div) {
    CeedInt q_comp_div;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_div));
    CeedCallBackend(magma_malloc((void **)&impl->d_div, num_qpts * num_nodes * q_comp_div * sizeof(div[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_div, sizeof(div[0]), div, 1, impl->d_div, 1, data->queue);
  }

  // Compile the weight kernel if it won't be compiled later on
  if (num_nodes > MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P || num_qpts > MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q) {
    Ceed ceed_delegate;

    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
    CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

    // Compile weight kernel (the remainder of kernel compilation happens at first call to CeedBasisApply)
    const char basis_kernel_source[] = "// Weight basis source\n#include <ceed/jit-source/magma/magma-basis-weight-nontensor.h>\n";

    CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, "basis_h_div_weight", &impl->module[0], 1, "BASIS_Q", num_qpts));
    CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[0], "magma_weight_nontensor", &impl->Weight));
    CeedCallBackend(CeedDestroy(&ceed_delegate));
  }

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPointsNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPointsNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Magma));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(curl)
//------------------------------------------------------------------------------
int CeedBasisCreateHcurl_Magma(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                               const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                      ceed;
  Ceed_Magma               *data;
  CeedBasisNonTensor_Magma *impl;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));

  // Copy basis data to GPU
  if (q_weight) {
    CeedCallBackend(magma_malloc((void **)&impl->d_q_weight, num_qpts * sizeof(q_weight[0])));
    magma_setvector(num_qpts, sizeof(q_weight[0]), q_weight, 1, impl->d_q_weight, 1, data->queue);
  }
  if (interp) {
    CeedInt q_comp_interp;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
    CeedCallBackend(magma_malloc((void **)&impl->d_interp, num_qpts * num_nodes * q_comp_interp * sizeof(interp[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_interp, sizeof(interp[0]), interp, 1, impl->d_interp, 1, data->queue);
  }
  if (curl) {
    CeedInt q_comp_curl;

    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_curl));
    CeedCallBackend(magma_malloc((void **)&impl->d_curl, num_qpts * num_nodes * q_comp_curl * sizeof(curl[0])));
    magma_setvector(num_qpts * num_nodes * q_comp_curl, sizeof(curl[0]), curl, 1, impl->d_curl, 1, data->queue);
  }

  // Compile the weight kernel if it won't be compiled later on
  if (num_nodes > MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_P || num_qpts > MAGMA_NONTENSOR_CUSTOM_KERNEL_MAX_Q) {
    Ceed ceed_delegate;

    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
    CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));

    // Compile weight kernel (the remainder of kernel compilation happens at first call to CeedBasisApply)
    const char basis_kernel_source[] = "// Weight basis source\n#include <ceed/jit-source/magma/magma-basis-weight-nontensor.h>\n";

    CeedCallBackend(CeedCompileMagma(ceed_delegate, basis_kernel_source, "basis_h_curl_weight", &impl->module[0], 1, "BASIS_Q", num_qpts));
    CeedCallBackend(CeedGetKernelMagma(ceed, impl->module[0], "magma_weight_nontensor", &impl->Weight));
    CeedCallBackend(CeedDestroy(&ceed_delegate));
  }

  CeedCallBackend(CeedBasisSetData(basis, impl));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPointsNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPointsNonTensor_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Magma));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
