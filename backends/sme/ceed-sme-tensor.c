// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <arm_sme.h>
#include <stdint.h>

#include "ceed-sme.h"

// The contraction v[a,j,c] (+)= sum_b t[j,b] u[a,b,c] is, for each a and b, a rank-one update of the J x C output by the
// vectors t[:,b] and u[a,b,:]. SME computes exactly that with one outer-product instruction per ZA tile, so the kernel
// streams over b and accumulates the output tile in ZA instead of in vector registers. Each ZA tile holds SVL x SVL
// elements, where SVL is the streaming vector length; the kernel is agnostic to SVL.
//
// Tile roles are fixed so that the tile numbers are compile-time constants, as the instructions require:
//   0, 1, 2  accumulators for three adjacent column tiles of the output
//   3        scratch for transposing a row-major operand block into columns
// FP64 has eight tiles and FP32 four; both precisions use the same four so that the two instantiations stay identical.
//
// The strided operand (t[j,b] in CEED_NOTRANSPOSE, u[a,b] in the C == 1 kernel) is loaded one row per horizontal tile
// slice and read back one column per vertical slice. Streaming mode has no gather loads, and this transposes an
// SVL x SVL block in registers at one load per row.

#ifdef CEED_SCALAR_IS_FP64
#define rtype svfloat64_t
#define svl() ((CeedSize)svcntsd())
#define whilelt(i, n) svwhilelt_b64((int64_t)(i), (int64_t)(n))
#define setzero() svdup_f64(0.0)
#define load(pg, a, n) svld1_vnum_f64(pg, a, n)
#define store(pg, a, n, v) svst1_vnum_f64(pg, a, n, v)
#define vadd(pg, a, b) svadd_f64_m(pg, a, b)
#define mopa(tile, pn, pm, zn, zm) svmopa_za64_f64_m(tile, pn, pm, zn, zm)
#define load_row(tile, slice, pg, a) svld1_hor_za64(tile, (uint32_t)(slice), pg, a)
#define read_column(pg, tile, slice) svread_ver_za64_f64_m(setzero(), pg, tile, (uint32_t)(slice))
#define read_row(pg, tile, slice) svread_hor_za64_f64_m(setzero(), pg, tile, (uint32_t)(slice))
#else
#define rtype svfloat32_t
#define svl() ((CeedSize)svcntsw())
#define whilelt(i, n) svwhilelt_b32((int64_t)(i), (int64_t)(n))
#define setzero() svdup_f32(0.0f)
#define load(pg, a, n) svld1_vnum_f32(pg, a, n)
#define store(pg, a, n, v) svst1_vnum_f32(pg, a, n, v)
#define vadd(pg, a, b) svadd_f32_m(pg, a, b)
#define mopa(tile, pn, pm, zn, zm) svmopa_za32_f32_m(tile, pn, pm, zn, zm)
#define load_row(tile, slice, pg, a) svld1_hor_za32(tile, (uint32_t)(slice), pg, a)
#define read_column(pg, tile, slice) svread_ver_za32_f32_m(setzero(), pg, tile, (uint32_t)(slice))
#define read_row(pg, tile, slice) svread_hor_za32_f32_m(setzero(), pg, tile, (uint32_t)(slice))
#endif

//------------------------------------------------------------------------------
// Tensor Contract Slice
//------------------------------------------------------------------------------
// v[j,c] (+)= sum_b t[j,b] u[b,c] for one a; tile rows are j, tile columns are c.
static inline void CeedTensorContract_Sme_Slice(CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t, CeedTransposeMode t_mode,
                                                const CeedInt add_mode, const CeedScalar *restrict u,
                                                CeedScalar *restrict v) __arm_streaming __arm_inout("za") {
  const CeedSize vl = svl();

  for (CeedSize j_0 = 0; j_0 < J; j_0 += vl) {
    const svbool_t pj  = whilelt(j_0, J);
    const CeedSize n_j = J - j_0 < vl ? J - j_0 : vl;

    for (CeedSize c_0 = 0; c_0 < C; c_0 += 3 * vl) {
      const svbool_t pc_0 = whilelt(c_0, C), pc_1 = whilelt(c_0 + vl, C), pc_2 = whilelt(c_0 + 2 * vl, C);

      svzero_za();
      for (CeedSize b_0 = 0; b_0 < B; b_0 += vl) {
        const CeedSize n_b = B - b_0 < vl ? B - b_0 : vl;

        if (t_mode == CEED_TRANSPOSE) {
          // t[j,b] is contiguous in j
          for (CeedSize k = 0; k < n_b; k++) {
            const CeedScalar *u_b = u + (b_0 + k) * C + c_0;
            const rtype       t_b = load(pj, t + (b_0 + k) * J + j_0, 0);

            mopa(0, pj, pc_0, t_b, load(pc_0, u_b, 0));
            mopa(1, pj, pc_1, t_b, load(pc_1, u_b, 1));
            mopa(2, pj, pc_2, t_b, load(pc_2, u_b, 2));
          }
        } else {
          // t[j,b] is contiguous in b: transpose the block through tile 3
          const svbool_t pb = whilelt(b_0, B);

          for (CeedSize r = 0; r < n_j; r++) load_row(3, r, pb, t + (j_0 + r) * B + b_0);
          for (CeedSize k = 0; k < n_b; k++) {
            const CeedScalar *u_b = u + (b_0 + k) * C + c_0;
            const rtype       t_b = read_column(pj, 3, k);

            mopa(0, pj, pc_0, t_b, load(pc_0, u_b, 0));
            mopa(1, pj, pc_1, t_b, load(pc_1, u_b, 1));
            mopa(2, pj, pc_2, t_b, load(pc_2, u_b, 2));
          }
        }
      }
      for (CeedSize r = 0; r < n_j; r++) {
        CeedScalar *v_r = v + (j_0 + r) * C + c_0;
        rtype       v_0 = read_row(pc_0, 0, r), v_1 = read_row(pc_1, 1, r), v_2 = read_row(pc_2, 2, r);

        if (add_mode) v_0 = vadd(pc_0, v_0, load(pc_0, v_r, 0)), v_1 = vadd(pc_1, v_1, load(pc_1, v_r, 1)), v_2 = vadd(pc_2, v_2, load(pc_2, v_r, 2));
        store(pc_0, v_r, 0, v_0), store(pc_1, v_r, 1, v_1), store(pc_2, v_r, 2, v_2);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Tensor Contract C=1
//------------------------------------------------------------------------------
// v[a,j] (+)= sum_b t[j,b] u[a,b]; tile rows are a, tile columns are j.
// u[a,b] is always contiguous in b and is transposed through tile 2; t[j,b] is transposed through tile 3 unless CEED_TRANSPOSE.
static inline void CeedTensorContract_Sme_Single(CeedInt A, CeedInt B, CeedInt J, const CeedScalar *restrict t, CeedTransposeMode t_mode,
                                                 const CeedInt add_mode, const CeedScalar *restrict u,
                                                 CeedScalar *restrict v) __arm_streaming __arm_inout("za") {
  const CeedSize vl = svl();

  for (CeedSize a_0 = 0; a_0 < A; a_0 += vl) {
    const svbool_t pa  = whilelt(a_0, A);
    const CeedSize n_a = A - a_0 < vl ? A - a_0 : vl;

    for (CeedSize j_0 = 0; j_0 < J; j_0 += vl) {
      const svbool_t pj  = whilelt(j_0, J);
      const CeedSize n_j = J - j_0 < vl ? J - j_0 : vl;

      svzero_za();
      for (CeedSize b_0 = 0; b_0 < B; b_0 += vl) {
        const svbool_t pb  = whilelt(b_0, B);
        const CeedSize n_b = B - b_0 < vl ? B - b_0 : vl;

        for (CeedSize r = 0; r < n_a; r++) load_row(2, r, pb, u + (a_0 + r) * B + b_0);
        if (t_mode == CEED_TRANSPOSE) {
          for (CeedSize k = 0; k < n_b; k++) mopa(0, pa, pj, read_column(pa, 2, k), load(pj, t + (b_0 + k) * J + j_0, 0));
        } else {
          for (CeedSize r = 0; r < n_j; r++) load_row(3, r, pb, t + (j_0 + r) * B + b_0);
          for (CeedSize k = 0; k < n_b; k++) mopa(0, pa, pj, read_column(pa, 2, k), read_column(pj, 3, k));
        }
      }
      for (CeedSize r = 0; r < n_a; r++) {
        CeedScalar *v_r = v + (a_0 + r) * J + j_0;
        rtype       v_0 = read_row(pj, 0, r);

        if (add_mode) v_0 = vadd(pj, v_0, load(pj, v_r, 0));
        store(pj, v_r, 0, v_0);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
// Enters streaming mode once per call; the mode switch is not free, so the loop over a runs inside it.
__arm_locally_streaming __arm_new("za") static int CeedTensorContract_Sme_Contract(CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                                                                   const CeedScalar *restrict t, CeedTransposeMode t_mode,
                                                                                   const CeedInt add_mode, const CeedScalar *restrict u,
                                                                                   CeedScalar *restrict v) {
  if (C == 1) {
    CeedTensorContract_Sme_Single(A, B, J, t, t_mode, add_mode, u, v);
  } else {
    for (CeedInt a = 0; a < A; a++) CeedTensorContract_Sme_Slice(B, C, J, t, t_mode, add_mode, &u[(CeedSize)a * B * C], &v[(CeedSize)a * J * C]);
  }
  return CEED_ERROR_SUCCESS;
}

static int CeedTensorContractApply_Sme(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedCallBackend(CeedTensorContract_Sme_Contract(A, B, C, J, t, t_mode, add, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Sme(CeedTensorContract contract) {
  CeedCallBackend(CeedSetBackendFunction(CeedTensorContractReturnCeed(contract), "TensorContract", contract, "Apply", CeedTensorContractApply_Sme));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
