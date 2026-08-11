/// @file
/// Test non-tensor H^1 prism interpolation and gradient at arbitrary points
/// \test Test non-tensor H^1 prism interpolation and gradient at arbitrary points
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define P 6
#define Q 6
#define NUM_POINTS 3

static void P1PrismBasis(CeedScalar x, CeedScalar y, CeedScalar z, CeedScalar *interp, CeedScalar *grad) {
  const CeedScalar l[3]     = {1. - x - y, x, y};
  const CeedScalar dl[3][2] = {
      {-1., -1.},
      {1.,  0. },
      {0.,  1. }
  };

  for (CeedInt i = 0; i < 3; i++) {
    interp[i]     = l[i] * (1. - z);
    interp[3 + i] = l[i] * z;

    grad[0 * P + i]     = dl[i][0] * (1. - z);
    grad[1 * P + i]     = dl[i][1] * (1. - z);
    grad[2 * P + i]     = -l[i];
    grad[0 * P + 3 + i] = dl[i][0] * z;
    grad[1 * P + 3 + i] = dl[i][1] * z;
    grad[2 * P + 3 + i] = l[i];
  }
}

static CeedScalar PrismPoly(CeedScalar x, CeedScalar y, CeedScalar z) { return 1. + 2. * x - 3. * y + 5. * z + 7. * x * z - 11. * y * z; }

static CeedScalar PrismPolyGrad(CeedInt d, CeedScalar x, CeedScalar y, CeedScalar z) {
  switch (d) {
    case 0:
      return 2. + 7. * z;
    case 1:
      return -3. - 11. * z;
    default:
      return 5. + 7. * x - 11. * y;
  }
}

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u, v_interp, v_grad;

  CeedInit(argv[1], &ceed);
  CeedScalar q_ref[3 * Q] = {0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1.};
  CeedScalar q_weight[Q]  = {1., 1., 1., 1., 1., 1.};
  CeedScalar interp[Q * P], grad[3 * Q * P];
  for (CeedInt q = 0; q < Q; q++) {
    CeedScalar b[P], g[3 * P];

    P1PrismBasis(q_ref[0 * Q + q], q_ref[1 * Q + q], q_ref[2 * Q + q], b, g);
    for (CeedInt i = 0; i < P; i++) interp[q * P + i] = b[i];
    for (CeedInt d = 0; d < 3; d++) {
      for (CeedInt i = 0; i < P; i++) grad[d * Q * P + q * P + i] = g[d * P + i];
    }
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_PRISM, 1, P, Q, interp, grad, q_ref, q_weight, &basis);

  CeedVectorCreate(ceed, 3 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, P, &u);
  CeedVectorCreate(ceed, NUM_POINTS, &v_interp);
  CeedVectorCreate(ceed, 3 * NUM_POINTS, &v_grad);
  {
    CeedScalar x_array[3 * NUM_POINTS] = {0.2, 0.4, 0.1, 0.1, 0.2, 0.3, 0.25, 0.75, 0.5};
    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  {
    CeedScalar u_array[P];

    for (CeedInt i = 0; i < P; i++) u_array[i] = PrismPoly(q_ref[0 * Q + i], q_ref[1 * Q + i], q_ref[2 * Q + i]);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  {
    const CeedInt num_points = NUM_POINTS;

    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v_interp);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v_grad);
  }

  {
    const CeedScalar *x_array, *interp_array, *grad_array;

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(v_interp, CEED_MEM_HOST, &interp_array);
    CeedVectorGetArrayRead(v_grad, CEED_MEM_HOST, &grad_array);
    for (CeedInt p = 0; p < NUM_POINTS; p++) {
      const CeedScalar x               = x_array[0 * NUM_POINTS + p];
      const CeedScalar y               = x_array[1 * NUM_POINTS + p];
      const CeedScalar z               = x_array[2 * NUM_POINTS + p];
      const CeedScalar interp_expected = PrismPoly(x, y, z);

      if (fabs(interp_array[p] - interp_expected) > 500. * CEED_EPSILON) {
        printf("%f != %f at point %" CeedInt_FMT "\n", interp_array[p], interp_expected, p);
      }
      for (CeedInt d = 0; d < 3; d++) {
        const CeedScalar grad_expected = PrismPolyGrad(d, x, y, z);
        const CeedScalar grad_actual   = grad_array[d * NUM_POINTS + p];

        if (fabs(grad_actual - grad_expected) > 500. * CEED_EPSILON) {
          printf("%f != %f at dim %" CeedInt_FMT ", point %" CeedInt_FMT "\n", grad_actual, grad_expected, d, p);
        }
      }
    }
    CeedVectorRestoreArrayRead(x_points, &x_array);
    CeedVectorRestoreArrayRead(v_interp, &interp_array);
    CeedVectorRestoreArrayRead(v_grad, &grad_array);
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v_interp);
  CeedVectorDestroy(&v_grad);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
