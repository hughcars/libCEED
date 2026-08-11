/// @file
/// Test tensor interpolation and gradient transpose Apply/ApplyAdd for non-collocated and collocated bases
/// \test Test tensor interpolation and gradient transpose Apply/ApplyAdd for non-collocated and collocated bases
#include <ceed.h>
#include <math.h>
#include <stdio.h>

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

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    for (CeedInt b = 0; b < 2; b++) {
      // b == 0: under-integration (Q_1d < P_1d), ordinary non-collocated basis
      // b == 1: quadrature points collocated with nodes
      CeedBasis basis;
      CeedInt   p_1d = b ? 5 : 6, q_1d = b ? 5 : 4;
      CeedInt   p_dim = CeedIntPow(p_1d, dim), q_dim = CeedIntPow(q_1d, dim);

      if (b) CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_1d, q_1d, CEED_GAUSS_LOBATTO, &basis);
      else CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_1d, q_1d, CEED_GAUSS, &basis);

      for (CeedInt m = 0; m < 2; m++) {
        CeedEvalMode eval_mode = m ? CEED_EVAL_GRAD : CEED_EVAL_INTERP;
        CeedInt      q_comp = m ? dim : 1;
        CeedVector   u, e_u, w, v, v_add;
        CeedScalar   sum_1 = 0, sum_2 = 0, scale = 0;

        CeedVectorCreate(ceed, p_dim, &u);
        CeedVectorCreate(ceed, q_comp * q_dim, &e_u);
        CeedVectorCreate(ceed, q_comp * q_dim, &w);
        CeedVectorCreate(ceed, p_dim, &v);
        CeedVectorCreate(ceed, p_dim, &v_add);
        {
          CeedScalar u_array[p_dim], w_array[q_comp * q_dim], seed_array[p_dim];

          for (CeedInt i = 0; i < p_dim; i++) u_array[i] = 0.5 + (CeedScalar)(i % 7) - 0.25 * (CeedScalar)(i % 3);
          for (CeedInt i = 0; i < q_comp * q_dim; i++) w_array[i] = 1.0 - 0.5 * (CeedScalar)(i % 5) + 0.125 * (CeedScalar)(i % 2);
          for (CeedInt i = 0; i < p_dim; i++) seed_array[i] = 2.0 - (CeedScalar)(i % 4);
          CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
          CeedVectorSetArray(w, CEED_MEM_HOST, CEED_COPY_VALUES, w_array);
          CeedVectorSetArray(v_add, CEED_MEM_HOST, CEED_COPY_VALUES, seed_array);
        }

        // Adjoint identity: w' * (E u) == (E' w)' * u
        CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, eval_mode, u, e_u);
        CeedBasisApply(basis, 1, CEED_TRANSPOSE, eval_mode, w, v);
        {
          const CeedScalar *u_array, *e_u_array, *w_array, *v_array;

          CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
          CeedVectorGetArrayRead(e_u, CEED_MEM_HOST, &e_u_array);
          CeedVectorGetArrayRead(w, CEED_MEM_HOST, &w_array);
          CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
          for (CeedInt i = 0; i < q_comp * q_dim; i++) sum_1 += w_array[i] * e_u_array[i];
          for (CeedInt i = 0; i < p_dim; i++) sum_2 += v_array[i] * u_array[i];
          for (CeedInt i = 0; i < q_comp * q_dim; i++) scale += fabs(w_array[i] * e_u_array[i]);
          CeedVectorRestoreArrayRead(u, &u_array);
          CeedVectorRestoreArrayRead(e_u, &e_u_array);
          CeedVectorRestoreArrayRead(w, &w_array);
          CeedVectorRestoreArrayRead(v, &v_array);
        }

        // ApplyAdd must accumulate into the seeded output
        CeedBasisApplyAdd(basis, 1, CEED_TRANSPOSE, eval_mode, w, v_add);
        {
          CeedScalarType scalar_type;

          CeedGetScalarType(&scalar_type);

          CeedScalar        tol = GetTolerance(scalar_type, dim);
          const CeedScalar *v_array, *v_add_array;
          const char       *label = b ? "collocated" : "non-collocated";

          if (fabs(sum_1 - sum_2) > tol * (scale > 1 ? scale : 1)) {
            printf("[%" CeedInt_FMT ", %s, %s] adjoint mismatch %f != %f\n", dim, label, m ? "grad" : "interp", sum_1, sum_2);
          }
          CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
          CeedVectorGetArrayRead(v_add, CEED_MEM_HOST, &v_add_array);
          for (CeedInt i = 0; i < p_dim; i++) {
            const CeedScalar expected = (2.0 - (CeedScalar)(i % 4)) + v_array[i];

            if (fabs(v_add_array[i] - expected) > tol * (fabs(expected) > 1 ? fabs(expected) : 1)) {
              printf("[%" CeedInt_FMT ", %s, %s] ApplyAdd error at %" CeedInt_FMT ": %f != %f\n", dim, label, m ? "grad" : "interp", i,
                     v_add_array[i], expected);
            }
          }
          CeedVectorRestoreArrayRead(v, &v_array);
          CeedVectorRestoreArrayRead(v_add, &v_add_array);
        }

        CeedVectorDestroy(&u);
        CeedVectorDestroy(&e_u);
        CeedVectorDestroy(&w);
        CeedVectorDestroy(&v);
        CeedVectorDestroy(&v_add);
      }
      CeedBasisDestroy(&basis);
    }
  }
  CeedDestroy(&ceed);
  return 0;
}
