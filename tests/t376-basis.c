/// @file
/// Test non-tensor H1 line and triangle interpolation and gradient at arbitrary points
/// \test Test non-tensor H1 line and triangle interpolation and gradient at arbitrary points
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#define NUM_COMP 2
#define NUM_POINTS 2

static void Check(const CeedScalar *actual, const CeedScalar *expected, CeedInt length, const char *label) {
  for (CeedInt i = 0; i < length; i++) {
    if (fabs(actual[i] - expected[i]) > 100. * CEED_EPSILON) {
      printf("%s: %f != %f at entry %" CeedInt_FMT "\n", label, actual[i], expected[i], i);
    }
  }
}

static void TestLine(Ceed ceed) {
  enum { P = 2, Q = 3 };
  const CeedInt num_points = NUM_POINTS;
  CeedBasis     basis;
  CeedVector    x_points, u, v_interp, v_grad;
  CeedScalar    q_ref[Q], q_weight[Q], interp[Q * P], grad[Q * P];

  for (CeedInt q = 0; q < Q; q++) {
    const CeedScalar x = -1.0 + q;

    q_ref[q]          = x;
    q_weight[q]       = 1.0;
    interp[q * P + 0] = 0.5 * (1.0 - x);
    interp[q * P + 1] = 0.5 * (1.0 + x);
    grad[q * P + 0]   = -0.5;
    grad[q * P + 1]   = 0.5;
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_LINE, NUM_COMP, P, Q, interp, grad, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, NUM_COMP * P, &u);
  CeedVectorCreate(ceed, NUM_COMP * NUM_POINTS, &v_interp);
  CeedVectorCreate(ceed, NUM_COMP * NUM_POINTS, &v_grad);
  {
    CeedScalar x_array[NUM_POINTS]   = {-0.5, 0.25};
    CeedScalar u_array[NUM_COMP * P] = {-1.0, 5.0, -6.0, -2.0};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }
  CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v_interp);
  CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v_grad);
  {
    const CeedScalar  expected_interp[NUM_COMP * NUM_POINTS] = {0.5, 2.75, -5.0, -3.5};
    const CeedScalar  expected_grad[NUM_COMP * NUM_POINTS]   = {3.0, 3.0, 2.0, 2.0};
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v_interp, CEED_MEM_HOST, &v_array);
    Check(v_array, expected_interp, NUM_COMP * NUM_POINTS, "Line interp");
    CeedVectorRestoreArrayRead(v_interp, &v_array);
    CeedVectorGetArrayRead(v_grad, CEED_MEM_HOST, &v_array);
    Check(v_array, expected_grad, NUM_COMP * NUM_POINTS, "Line grad");
    CeedVectorRestoreArrayRead(v_grad, &v_array);
  }
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v_interp);
  CeedVectorDestroy(&v_grad);
  CeedBasisDestroy(&basis);
}

static void TestTriangle(Ceed ceed) {
  enum { P = 3, Q = 4 };
  const CeedInt num_points = NUM_POINTS;
  CeedBasis     basis;
  CeedVector    x_points, u, v_interp, v_grad;
  CeedScalar    q_ref[2 * Q] = {0.0, 1.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 1.0, 1.0 / 3.0};
  CeedScalar    q_weight[Q], interp[Q * P], grad[2 * Q * P];

  for (CeedInt q = 0; q < Q; q++) {
    const CeedScalar x = q_ref[0 * Q + q], y = q_ref[1 * Q + q];

    q_weight[q]                 = 1.0;
    interp[q * P + 0]           = 1.0 - x - y;
    interp[q * P + 1]           = x;
    interp[q * P + 2]           = y;
    grad[0 * Q * P + q * P + 0] = -1.0;
    grad[0 * Q * P + q * P + 1] = 1.0;
    grad[0 * Q * P + q * P + 2] = 0.0;
    grad[1 * Q * P + q * P + 0] = -1.0;
    grad[1 * Q * P + q * P + 1] = 0.0;
    grad[1 * Q * P + q * P + 2] = 1.0;
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, NUM_COMP, P, Q, interp, grad, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, 2 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, NUM_COMP * P, &u);
  CeedVectorCreate(ceed, NUM_COMP * NUM_POINTS, &v_interp);
  CeedVectorCreate(ceed, 2 * NUM_COMP * NUM_POINTS, &v_grad);
  {
    CeedScalar x_array[2 * NUM_POINTS] = {0.2, 0.1, 0.1, 0.3};
    CeedScalar u_array[NUM_COMP * P]   = {1.0, 3.0, 4.0, -2.0, -1.0, -6.0};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }
  CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v_interp);
  CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v_grad);
  {
    const CeedScalar  expected_interp[NUM_COMP * NUM_POINTS]   = {1.7, 2.1, -2.2, -3.1};
    const CeedScalar  expected_grad[2 * NUM_COMP * NUM_POINTS] = {2.0, 2.0, 1.0, 1.0, 3.0, 3.0, -4.0, -4.0};
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v_interp, CEED_MEM_HOST, &v_array);
    Check(v_array, expected_interp, NUM_COMP * NUM_POINTS, "Triangle interp");
    CeedVectorRestoreArrayRead(v_interp, &v_array);
    CeedVectorGetArrayRead(v_grad, CEED_MEM_HOST, &v_array);
    Check(v_array, expected_grad, 2 * NUM_COMP * NUM_POINTS, "Triangle grad");
    CeedVectorRestoreArrayRead(v_grad, &v_array);
  }
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v_interp);
  CeedVectorDestroy(&v_grad);
  CeedBasisDestroy(&basis);
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  TestLine(ceed);
  TestTriangle(ceed);
  CeedDestroy(&ceed);
  return 0;
}
