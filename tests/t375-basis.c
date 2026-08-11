/// @file
/// Test non-tensor AtPoints transpose and ApplyAdd paths
/// \test Test non-tensor AtPoints transpose and ApplyAdd paths
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#define P 4
#define Q 4
#define NUM_POINTS 2

static void CheckVector(CeedVector v, const CeedScalar expected[P], const char *label) {
  const CeedScalar *v_array;

  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
  for (CeedInt i = 0; i < P; i++) {
    if (fabs(v_array[i] - expected[i]) > 100. * CEED_EPSILON) {
      printf("%s: %f != %f at node %" CeedInt_FMT "\n", label, v_array[i], expected[i], i);
    }
  }
  CeedVectorRestoreArrayRead(v, &v_array);
}

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u_interp, u_grad, v_nodes;

  CeedInit(argv[1], &ceed);
  CeedScalar q_ref[3 * Q] = {0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
  CeedScalar q_weight[Q]  = {1., 1., 1., 1.};
  CeedScalar interp[Q * P], grad[3 * Q * P];
  for (CeedInt q = 0; q < Q; q++) {
    const CeedScalar x = q_ref[0 * Q + q], y = q_ref[1 * Q + q], z = q_ref[2 * Q + q];

    interp[q * P + 0]           = 1.0 - x - y - z;
    interp[q * P + 1]           = x;
    interp[q * P + 2]           = y;
    interp[q * P + 3]           = z;
    grad[0 * Q * P + q * P + 0] = -1.0;
    grad[0 * Q * P + q * P + 1] = 1.0;
    grad[0 * Q * P + q * P + 2] = 0.0;
    grad[0 * Q * P + q * P + 3] = 0.0;
    grad[1 * Q * P + q * P + 0] = -1.0;
    grad[1 * Q * P + q * P + 1] = 0.0;
    grad[1 * Q * P + q * P + 2] = 1.0;
    grad[1 * Q * P + q * P + 3] = 0.0;
    grad[2 * Q * P + q * P + 0] = -1.0;
    grad[2 * Q * P + q * P + 1] = 0.0;
    grad[2 * Q * P + q * P + 2] = 0.0;
    grad[2 * Q * P + q * P + 3] = 1.0;
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TET, 1, P, Q, interp, grad, q_ref, q_weight, &basis);

  CeedVectorCreate(ceed, 3 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, NUM_POINTS, &u_interp);
  CeedVectorCreate(ceed, 3 * NUM_POINTS, &u_grad);
  CeedVectorCreate(ceed, P, &v_nodes);
  {
    CeedScalar x_array[3 * NUM_POINTS] = {0.2, 0.4, 0.1, 0.2, 0.1, 0.1};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }

  const CeedInt num_points = NUM_POINTS;
  {
    CeedScalar       u_array[NUM_POINTS] = {5., 6.};
    const CeedScalar expected[P]         = {4.8, 3.4, 1.7, 1.1};
    const CeedScalar expected_add[P]     = {5.8, 4.4, 2.7, 2.1};

    CeedVectorSetArray(u_interp, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_TRANSPOSE, CEED_EVAL_INTERP, x_points, u_interp, v_nodes);
    CheckVector(v_nodes, expected, "Interp transpose");

    CeedVectorSetValue(v_nodes, 1.0);
    CeedBasisApplyAddAtPoints(basis, 1, &num_points, CEED_TRANSPOSE, CEED_EVAL_INTERP, x_points, u_interp, v_nodes);
    CheckVector(v_nodes, expected_add, "Interp transpose add");
  }
  {
    CeedScalar       u_array[3 * NUM_POINTS] = {1., 2., 3., 4., 5., 6.};
    const CeedScalar expected[P]             = {-21., 3., 7., 11.};
    const CeedScalar expected_add[P]         = {-20., 4., 8., 12.};

    CeedVectorSetArray(u_grad, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_TRANSPOSE, CEED_EVAL_GRAD, x_points, u_grad, v_nodes);
    CheckVector(v_nodes, expected, "Grad transpose");

    CeedVectorSetValue(v_nodes, 1.0);
    CeedBasisApplyAddAtPoints(basis, 1, &num_points, CEED_TRANSPOSE, CEED_EVAL_GRAD, x_points, u_grad, v_nodes);
    CheckVector(v_nodes, expected_add, "Grad transpose add");
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u_interp);
  CeedVectorDestroy(&u_grad);
  CeedVectorDestroy(&v_nodes);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
