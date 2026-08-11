/// @file
/// Test tensor basis apply bounds and adjoint consistency against guarded buffers
/// \test Test tensor basis apply bounds and adjoint consistency against guarded buffers
//TESTARGS(only="cpu") {ceed_resource}
#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include <ceed.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

typedef struct {
  CeedScalar *data;
#if defined(__linux__)
  void  *mapping;
  size_t mapping_size;
#endif
} GuardedArray;

static int GuardedArrayCreate(CeedSize length, GuardedArray *array) {
  if (length <= 0 || (uintmax_t)length > SIZE_MAX / sizeof(*array->data)) return 1;
#if defined(__linux__)
  const long page_size = sysconf(_SC_PAGESIZE);

  if (page_size <= 0) return 1;
  const size_t bytes = (size_t)length * sizeof(*array->data), num_pages = (bytes + page_size - 1) / page_size;

  if (!bytes || num_pages >= SIZE_MAX / (size_t)page_size) return 1;
  array->mapping_size = (num_pages + 1) * page_size;
  array->mapping      = mmap(NULL, array->mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (array->mapping == MAP_FAILED) return 1;
  void *guard = (char *)array->mapping + num_pages * page_size;

  if (mprotect(guard, page_size, PROT_NONE)) {
    munmap(array->mapping, array->mapping_size);
    array->mapping = NULL;
    return 1;
  }
  array->data = (CeedScalar *)((char *)guard - bytes);
#else
  array->data = malloc((size_t)length * sizeof(*array->data));
  if (!array->data) return 1;
#endif
  return 0;
}

static void GuardedArrayDestroy(GuardedArray *array) {
#if defined(__linux__)
  if (array->mapping && array->mapping != MAP_FAILED) munmap(array->mapping, array->mapping_size);
#else
  free(array->data);
#endif
  array->data = NULL;
}

static CeedScalar GetTolerance(CeedScalarType scalar_type, int dim) {
  CeedScalar tol;
  if (scalar_type == CEED_SCALAR_FP32) {
    if (dim == 3) tol = 1.e-3;
    else tol = 1.e-4;
  } else {
    tol = 1.e-10;
  }
  return tol;
}

static int RunCase(Ceed ceed, CeedInt dim, CeedInt num_comp, CeedInt num_elem, CeedInt p_1d, CeedInt q_1d, CeedQuadMode quad_mode,
                   CeedEvalMode eval_mode) {
  const CeedInt  q_comp = eval_mode == CEED_EVAL_GRAD ? dim : 1;
  const CeedSize u_len = (CeedSize)num_elem * num_comp * CeedIntPow(p_1d, dim), v_len = (CeedSize)q_comp * num_elem * num_comp * CeedIntPow(q_1d, dim);
  GuardedArray   u = {0}, e_u = {0}, w = {0}, v = {0}, v_add = {0};
  CeedVector     u_vec, e_u_vec, w_vec, v_vec, v_add_vec;
  CeedBasis      basis;
  CeedScalar     sum_1 = 0, sum_2 = 0, scale = 0;
  int            failed = 0;

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p_1d, q_1d, quad_mode, &basis);
  if (GuardedArrayCreate(u_len, &u) || GuardedArrayCreate(v_len, &e_u) || GuardedArrayCreate(v_len, &w) || GuardedArrayCreate(u_len, &v) ||
      GuardedArrayCreate(u_len, &v_add)) {
    return 1;
  }
  for (CeedSize i = 0; i < u_len; i++) u.data[i] = 0.5 + (CeedScalar)(i % 7) - 0.25 * (CeedScalar)(i % 3);
  for (CeedSize i = 0; i < v_len; i++) w.data[i] = 1.0 - 0.5 * (CeedScalar)(i % 5) + 0.125 * (CeedScalar)(i % 2);
  for (CeedSize i = 0; i < u_len; i++) v_add.data[i] = 2.0 - (CeedScalar)(i % 4);

  CeedVectorCreate(ceed, u_len, &u_vec);
  CeedVectorSetArray(u_vec, CEED_MEM_HOST, CEED_USE_POINTER, u.data);
  CeedVectorCreate(ceed, v_len, &e_u_vec);
  CeedVectorSetArray(e_u_vec, CEED_MEM_HOST, CEED_USE_POINTER, e_u.data);
  CeedVectorCreate(ceed, v_len, &w_vec);
  CeedVectorSetArray(w_vec, CEED_MEM_HOST, CEED_USE_POINTER, w.data);
  CeedVectorCreate(ceed, u_len, &v_vec);
  CeedVectorSetArray(v_vec, CEED_MEM_HOST, CEED_USE_POINTER, v.data);
  CeedVectorCreate(ceed, u_len, &v_add_vec);
  CeedVectorSetArray(v_add_vec, CEED_MEM_HOST, CEED_USE_POINTER, v_add.data);

  // Adjoint identity with exact-size guarded buffers: w' * (E u) == (E' w)' * u
  CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, eval_mode, u_vec, e_u_vec);
  CeedBasisApply(basis, num_elem, CEED_TRANSPOSE, eval_mode, w_vec, v_vec);
  for (CeedSize i = 0; i < v_len; i++) sum_1 += w.data[i] * e_u.data[i];
  for (CeedSize i = 0; i < u_len; i++) sum_2 += v.data[i] * u.data[i];
  for (CeedSize i = 0; i < v_len; i++) scale += fabs(w.data[i] * e_u.data[i]);

  // ApplyAdd must accumulate into the seeded output
  CeedBasisApplyAdd(basis, num_elem, CEED_TRANSPOSE, eval_mode, w_vec, v_add_vec);
  {
    CeedScalarType scalar_type;

    CeedGetScalarType(&scalar_type);

    const CeedScalar tol = GetTolerance(scalar_type, dim);

    if (fabs(sum_1 - sum_2) > tol * (scale > 1 ? scale : 1)) {
      printf("[dim %" CeedInt_FMT ", comp %" CeedInt_FMT ", elem %" CeedInt_FMT ", P %" CeedInt_FMT ", Q %" CeedInt_FMT
             ", mode %d] adjoint mismatch %f != %f\n",
             dim, num_comp, num_elem, p_1d, q_1d, (int)eval_mode, sum_1, sum_2);
      failed = 1;
    }
    for (CeedSize i = 0; i < u_len; i++) {
      const CeedScalar expected = (2.0 - (CeedScalar)(i % 4)) + v.data[i];

      if (fabs(v_add.data[i] - expected) > tol * (fabs(expected) > 1 ? fabs(expected) : 1)) {
        printf("[dim %" CeedInt_FMT ", comp %" CeedInt_FMT ", elem %" CeedInt_FMT ", P %" CeedInt_FMT ", Q %" CeedInt_FMT
               ", mode %d] ApplyAdd error at %" CeedSize_FMT ": %f != %f\n",
               dim, num_comp, num_elem, p_1d, q_1d, (int)eval_mode, i, v_add.data[i], expected);
        failed = 1;
        break;
      }
    }
  }

  // Quadrature weights write the exact output extent
  if (eval_mode == CEED_EVAL_INTERP && num_comp == 1) {
    GuardedArray q_w = {0};
    CeedVector   q_w_vec;

    if (GuardedArrayCreate((CeedSize)num_elem * CeedIntPow(q_1d, dim), &q_w)) return 1;
    CeedVectorCreate(ceed, (CeedSize)num_elem * CeedIntPow(q_1d, dim), &q_w_vec);
    CeedVectorSetArray(q_w_vec, CEED_MEM_HOST, CEED_USE_POINTER, q_w.data);
    CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_w_vec);
    CeedVectorDestroy(&q_w_vec);
    GuardedArrayDestroy(&q_w);
  }

  CeedVectorDestroy(&u_vec);
  CeedVectorDestroy(&e_u_vec);
  CeedVectorDestroy(&w_vec);
  CeedVectorDestroy(&v_vec);
  CeedVectorDestroy(&v_add_vec);
  CeedBasisDestroy(&basis);
  GuardedArrayDestroy(&u);
  GuardedArrayDestroy(&e_u);
  GuardedArrayDestroy(&w);
  GuardedArrayDestroy(&v);
  GuardedArrayDestroy(&v_add);
  return failed;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  for (CeedInt dim = 1; dim <= 3; dim++) {
    for (CeedInt num_comp = 1; num_comp <= 2; num_comp++) {
      for (CeedInt num_elem = 1; num_elem <= 8; num_elem += 7) {
        for (CeedInt eval = 0; eval <= 1; eval++) {
          const CeedEvalMode eval_mode = eval ? CEED_EVAL_GRAD : CEED_EVAL_INTERP;

          // Gauss with collocated gradient path (Q = P + 1), collocated Lobatto (Q = P), and under-integration (Q = P - 1)
          if (RunCase(ceed, dim, num_comp, num_elem, 4, 5, CEED_GAUSS, eval_mode)) return 1;
          if (RunCase(ceed, dim, num_comp, num_elem, 5, 5, CEED_GAUSS_LOBATTO, eval_mode)) return 1;
          if (RunCase(ceed, dim, num_comp, num_elem, 5, 4, CEED_GAUSS, eval_mode)) return 1;
        }
      }
    }
  }
  CeedDestroy(&ceed);
  return 0;
}
