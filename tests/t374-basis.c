/// @file
/// Test non-tensor AtPoints with no points
/// \test Test non-tensor AtPoints with no points
#include <ceed.h>
#include <stdio.h>
#include <string.h>

#define P 4
#define Q 4
#define NUM_ELEM 2

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u, v;

  CeedInit(argv[1], &ceed);
  CeedScalar q_ref[3 * Q] = {0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
  CeedScalar q_weight[Q]  = {1., 1., 1., 1.};
  CeedScalar interp[Q * P], grad[3 * Q * P];
  for (CeedInt i = 0; i < 3 * Q * P; i++) grad[i] = 0.0;
  for (CeedInt q = 0; q < Q; q++) {
    const CeedScalar x = q_ref[0 * Q + q], y = q_ref[1 * Q + q], z = q_ref[2 * Q + q];

    interp[q * P + 0] = 1.0 - x - y - z;
    interp[q * P + 1] = x;
    interp[q * P + 2] = y;
    interp[q * P + 3] = z;
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TET, 1, P, Q, interp, grad, q_ref, q_weight, &basis);

  CeedVectorCreate(ceed, 0, &x_points);
  CeedVectorCreate(ceed, NUM_ELEM * P, &u);
  CeedVectorCreate(ceed, 0, &v);
  {
    CeedScalar u_array[NUM_ELEM * P] = {1., 2., 3., 4., 5., 6., 7., 8.};

    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  {
    const CeedInt num_points[NUM_ELEM] = {0, 0};
    const int     ierr                 = CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);

    if (ierr) printf("Expected zero-point AtPoints apply to succeed\n");
  }
  for (CeedInt mode = 0; mode < 2; mode++) {
    const CeedEvalMode eval_mode            = mode == 0 ? CEED_EVAL_INTERP : CEED_EVAL_GRAD;
    const CeedInt      num_points[NUM_ELEM] = {0, 0};
    const CeedScalar  *u_array;

    CeedVectorSetValue(u, 2.0);
    CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_TRANSPOSE, eval_mode, x_points, v, u);
    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    for (CeedInt i = 0; i < NUM_ELEM * P; i++) {
      if (u_array[i] != 0.0) printf("Zero-point transpose did not zero node %" CeedInt_FMT "\n", i);
    }
    CeedVectorRestoreArrayRead(u, &u_array);

    CeedVectorSetValue(u, 2.0);
    CeedBasisApplyAddAtPoints(basis, NUM_ELEM, num_points, CEED_TRANSPOSE, eval_mode, x_points, v, u);
    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    for (CeedInt i = 0; i < NUM_ELEM * P; i++) {
      if (u_array[i] != 2.0) printf("Zero-point transpose add changed node %" CeedInt_FMT "\n", i);
    }
    CeedVectorRestoreArrayRead(u, &u_array);
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
