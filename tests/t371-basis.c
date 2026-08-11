/// @file
/// Test rational non-tensor H^1 pyramid interpolation and gradient at arbitrary points
/// \test Test rational non-tensor H^1 pyramid interpolation and gradient at arbitrary points
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#define P 5
#define Q 5
#define NUM_POINTS 3

static void P1PyramidBasis(CeedScalar x, CeedScalar y, CeedScalar z, CeedScalar *interp, CeedScalar *grad) {
  const CeedScalar w = 1.0 - z;

  if (w == 0.0) {
    interp[0] = interp[1] = interp[2] = interp[3] = 0.0;
    interp[4]                                     = 1.0;

    grad[0 * P + 0] = -0.5;
    grad[1 * P + 0] = -0.5;
    grad[2 * P + 0] = -0.75;
    grad[0 * P + 1] = 0.5;
    grad[1 * P + 1] = -0.5;
    grad[2 * P + 1] = -0.25;
    grad[0 * P + 2] = 0.5;
    grad[1 * P + 2] = 0.5;
    grad[2 * P + 2] = 0.25;
    grad[0 * P + 3] = -0.5;
    grad[1 * P + 3] = 0.5;
    grad[2 * P + 3] = -0.25;
    grad[0 * P + 4] = 0.0;
    grad[1 * P + 4] = 0.0;
    grad[2 * P + 4] = 1.0;
    return;
  }

  const CeedScalar wx = w - x, wy = w - y, w_inv = 1.0 / w;

  interp[0] = wx * wy * w_inv;
  interp[1] = x * wy * w_inv;
  interp[2] = x * y * w_inv;
  interp[3] = wx * y * w_inv;
  interp[4] = z;

  grad[0 * P + 0] = -wy * w_inv;
  grad[1 * P + 0] = -wx * w_inv;
  grad[2 * P + 0] = x * y * w_inv * w_inv - 1.0;
  grad[0 * P + 1] = wy * w_inv;
  grad[1 * P + 1] = -x * w_inv;
  grad[2 * P + 1] = -x * y * w_inv * w_inv;
  grad[0 * P + 2] = y * w_inv;
  grad[1 * P + 2] = x * w_inv;
  grad[2 * P + 2] = x * y * w_inv * w_inv;
  grad[0 * P + 3] = -y * w_inv;
  grad[1 * P + 3] = wx * w_inv;
  grad[2 * P + 3] = -x * y * w_inv * w_inv;
  grad[0 * P + 4] = 0.0;
  grad[1 * P + 4] = 0.0;
  grad[2 * P + 4] = 1.0;
}

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u, v_interp, v_grad;

  CeedInit(argv[1], &ceed);
  CeedScalar q_ref[3 * Q] = {0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.};
  CeedScalar q_weight[Q]  = {1., 1., 1., 1., 1.};
  CeedScalar interp[Q * P], grad[3 * Q * P];
  for (CeedInt q = 0; q < Q; q++) {
    CeedScalar b[P], g[3 * P];

    P1PyramidBasis(q_ref[0 * Q + q], q_ref[1 * Q + q], q_ref[2 * Q + q], b, g);
    for (CeedInt i = 0; i < P; i++) interp[q * P + i] = b[i];
    for (CeedInt d = 0; d < 3; d++) {
      for (CeedInt i = 0; i < P; i++) grad[d * Q * P + q * P + i] = g[d * P + i];
    }
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_PYRAMID, 1, P, Q, interp, grad, q_ref, q_weight, &basis);

  CeedVectorCreate(ceed, 3 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, P, &u);
  CeedVectorCreate(ceed, NUM_POINTS, &v_interp);
  CeedVectorCreate(ceed, 3 * NUM_POINTS, &v_grad);
  {
    CeedScalar x_array[3 * NUM_POINTS] = {0.1, 0.4, 0.0, 0.1, 0.2, 0.0, 0.5, 0.2, 1.0};
    CeedScalar u_array[P]              = {2.0, -3.0, 5.0, 7.0, -11.0};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  {
    const CeedInt num_points = NUM_POINTS;

    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v_interp);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v_grad);
  }

  {
    const CeedScalar *x_array, *u_array, *interp_array, *grad_array;

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    CeedVectorGetArrayRead(v_interp, CEED_MEM_HOST, &interp_array);
    CeedVectorGetArrayRead(v_grad, CEED_MEM_HOST, &grad_array);
    for (CeedInt p = 0; p < NUM_POINTS; p++) {
      CeedScalar expected_interp = 0.0, expected_grad[3] = {0.0, 0.0, 0.0};

      if (p == NUM_POINTS - 1) {
        // Independent expected values at the apex for the nodal values above.
        expected_interp  = -11.0;
        expected_grad[0] = -3.5;
        expected_grad[1] = 6.5;
        expected_grad[2] = -12.25;
      } else {
        CeedScalar b[P], g[3 * P];

        P1PyramidBasis(x_array[0 * NUM_POINTS + p], x_array[1 * NUM_POINTS + p], x_array[2 * NUM_POINTS + p], b, g);
        for (CeedInt node = 0; node < P; node++) {
          expected_interp += u_array[node] * b[node];
          for (CeedInt d = 0; d < 3; d++) expected_grad[d] += u_array[node] * g[d * P + node];
        }
      }
      if (fabs(interp_array[p] - expected_interp) > 1000. * CEED_EPSILON)
        printf("%f != %f at point %" CeedInt_FMT "\n", interp_array[p], expected_interp, p);
      for (CeedInt d = 0; d < 3; d++) {
        const CeedScalar actual = grad_array[d * NUM_POINTS + p];

        if (fabs(actual - expected_grad[d]) > 2000. * CEED_EPSILON)
          printf("%f != %f at dim %" CeedInt_FMT ", point %" CeedInt_FMT "\n", actual, expected_grad[d], d, p);
      }
    }
    CeedVectorRestoreArrayRead(x_points, &x_array);
    CeedVectorRestoreArrayRead(u, &u_array);
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
