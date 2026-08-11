// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <libxsmm.h>

#include "ceed-xsmm.h"

typedef struct {
  CeedInt              A, B, C, J, add;
  CeedTransposeMode    t_mode;
  libxsmm_gemmfunction kernel;
} CeedTensorContractKernel_Xsmm;

static inline libxsmm_gemmfunction CeedTensorContractGetKernel_Xsmm(CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                                                    CeedTransposeMode t_mode, CeedInt add,
                                                                    libxsmm_gemm_shape gemm_shape, int flags) {
#if defined(LIBXSMM_NO_TLS)
  return libxsmm_dispatch_gemm(gemm_shape, (libxsmm_bitfield)flags, (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
#else
  static LIBXSMM_TLS CeedTensorContractKernel_Xsmm cache[16];
  const unsigned int index = ((unsigned int)A * 3u + (unsigned int)B * 5u + (unsigned int)C * 7u +
                              (unsigned int)J * 11u + (unsigned int)t_mode * 13u + (unsigned int)add) & 15u;
  CeedTensorContractKernel_Xsmm *entry = &cache[index];

  if (entry->kernel && entry->A == A && entry->B == B && entry->C == C && entry->J == J && entry->t_mode == t_mode && entry->add == add) {
    return entry->kernel;
  }
  entry->A      = A;
  entry->B      = B;
  entry->C      = C;
  entry->J      = J;
  entry->t_mode = t_mode;
  entry->add    = add;
  entry->kernel = libxsmm_dispatch_gemm(gemm_shape, (libxsmm_bitfield)flags, (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
  return entry->kernel;
#endif
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                        CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  if (C == 1) {
    // The fixed LIBXSMM neov2 ('N', 'N') generator emits unpredicated vector tails for this transpose kernel family and overruns operand
    // bounds (see t366-tensor); dispatch the proven-safe ('T', 'N') kernel family on a transposed copy of the small t operand instead
    if (t_mode == CEED_TRANSPOSE) {
      const int                flags      = LIBXSMM_GEMM_FLAGS('T', 'N') | ((!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE);
      const libxsmm_gemm_shape gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                                ? libxsmm_create_gemm_shape(J, A, B, B, B, J, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
                                                                            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                                : libxsmm_create_gemm_shape(J, A, B, B, B, J, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
      const libxsmm_gemmfunction kernel   = CeedTensorContractGetKernel_Xsmm(A, B, C, J, t_mode, add, gemm_shape, flags);
      CeedScalar                 t_transpose[B * J];
      libxsmm_gemm_param         gemm_param;

      CeedCheck(kernel, CeedTensorContractReturnCeed(contract), CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
      for (CeedInt b = 0; b < B; b++) {
        for (CeedInt j = 0; j < J; j++) t_transpose[j * B + b] = t[b * J + j];
      }

      // Run kernel
      gemm_param.a.primary = (CeedScalar *)&t_transpose[0];
      gemm_param.b.primary = (CeedScalar *)&u[0];
      gemm_param.c.primary = (CeedScalar *)&v[0];
      kernel(&gemm_param);
      return CEED_ERROR_SUCCESS;
    }
    // Build or query the required kernel
    const int                flags_t    = LIBXSMM_GEMM_FLAGS(!t_mode ? 'T' : 'N', 'N');
    const int                flags_ab   = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
    const int                flags      = (flags_t | flags_ab);
    const libxsmm_gemm_shape gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                              ? libxsmm_create_gemm_shape(J, A, B, !t_mode ? B : J, B, J, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
                                                                          LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                              : libxsmm_create_gemm_shape(J, A, B, !t_mode ? B : J, B, J, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                                          LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const libxsmm_gemmfunction kernel   = CeedTensorContractGetKernel_Xsmm(A, B, C, J, t_mode, add, gemm_shape, flags);
    libxsmm_gemm_param         gemm_param;

    CeedCheck(kernel, CeedTensorContractReturnCeed(contract), CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");

    // Run kernel
    gemm_param.a.primary = (CeedScalar *)&t[0];
    gemm_param.b.primary = (CeedScalar *)&u[0];
    gemm_param.c.primary = (CeedScalar *)&v[0];
    kernel(&gemm_param);
  } else {
    // Build or query the required kernel
    const int                flags_t    = LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
    const int                flags_ab   = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
    const int                flags      = (flags_t | flags_ab);
    const libxsmm_gemm_shape gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                              ? libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
                                                                          LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                              : libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                                          LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const libxsmm_gemmfunction kernel   = CeedTensorContractGetKernel_Xsmm(A, B, C, J, t_mode, add, gemm_shape, flags);
    libxsmm_gemm_param         gemm_param;

    CeedCheck(kernel, CeedTensorContractReturnCeed(contract), CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");

    // Run kernel
    gemm_param.b.primary = (CeedScalar *)&t[0];
    for (CeedInt a = 0; a < A; a++) {
      gemm_param.a.primary = (CeedScalar *)&u[a * B * C];
      gemm_param.c.primary = (CeedScalar *)&v[a * J * C];
      kernel(&gemm_param);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Xsmm(CeedTensorContract contract) {
  CeedCallBackend(CeedSetBackendFunction(CeedTensorContractReturnCeed(contract), "TensorContract", contract, "Apply", CeedTensorContractApply_Xsmm));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
