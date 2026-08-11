/// @file
/// Test non-tensor H(div) tetrahedron interpolation at arbitrary points
/// \test Test non-tensor H(div) tetrahedron interpolation at arbitrary points
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define P 6
#define Q 5
#define NUM_POINTS 3

static void VectorTetBasis(CeedScalar x, CeedScalar y, CeedScalar z, CeedScalar *interp) {
  for (CeedInt i = 0; i < 3 * P; i++) interp[i] = 0.0;
  interp[0 * P + 0] = 1.0;
  interp[1 * P + 1] = 1.0;
  interp[2 * P + 2] = 1.0;
  interp[0 * P + 3] = x;
  interp[1 * P + 4] = y;
  interp[2 * P + 5] = z;
}

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u, v;

  CeedInit(argv[1], &ceed);
  CeedScalar q_ref[3 * Q] = {0.1, 0.6, 0.1, 0.1, 0.25, 0.2, 0.1, 0.6, 0.1, 0.25, 0.3, 0.1, 0.1, 0.6, 0.25};
  CeedScalar q_weight[Q]  = {1., 1., 1., 1., 1.};
  CeedScalar interp[3 * Q * P], div[Q * P];
  for (CeedInt i = 0; i < Q * P; i++) div[i] = 0.0;
  for (CeedInt q = 0; q < Q; q++) {
    CeedScalar b[3 * P];

    VectorTetBasis(q_ref[0 * Q + q], q_ref[1 * Q + q], q_ref[2 * Q + q], b);
    for (CeedInt d = 0; d < 3; d++) {
      for (CeedInt i = 0; i < P; i++) interp[d * Q * P + q * P + i] = b[d * P + i];
    }
    div[q * P + 3] = 1.0;
    div[q * P + 4] = 1.0;
    div[q * P + 5] = 1.0;
  }
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_TET, 1, P, Q, interp, div, q_ref, q_weight, &basis);

  CeedVectorCreate(ceed, 3 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, P, &u);
  CeedVectorCreate(ceed, 3 * NUM_POINTS, &v);
  {
    CeedScalar x_array[3 * NUM_POINTS] = {0.2, 0.4, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.4};
    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  {
    CeedScalar u_array[P] = {2., -3., 5., 7., -11., 13.};
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  {
    const CeedInt num_points = NUM_POINTS;
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
  }

  {
    const CeedScalar *x_array, *v_array;

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt p = 0; p < NUM_POINTS; p++) {
      const CeedScalar expected[3] = {
          2. + 7. * x_array[0 * NUM_POINTS + p], -3. - 11. * x_array[1 * NUM_POINTS + p], 5. + 13. * x_array[2 * NUM_POINTS + p]
      };
      for (CeedInt d = 0; d < 3; d++) {
        const CeedScalar actual = v_array[d * NUM_POINTS + p];

        if (fabs(actual - expected[d]) > 500. * CEED_EPSILON) {
          printf("%f != %f at dim %" CeedInt_FMT ", point %" CeedInt_FMT "\n", actual, expected[d], d, p);
        }
      }
    }
    CeedVectorRestoreArrayRead(x_points, &x_array);
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
