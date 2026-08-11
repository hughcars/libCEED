/// @file
/// Test tensor AtPoints uniformly padded point storage
/// \test Test tensor AtPoints uniformly padded point storage
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#define P_1D 2
#define Q_1D 2
#define NUM_POINTS 2
#define POINT_STRIDE 4

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u, v, v_grad, point_values, nodes_out;

  CeedInit(argv[1], &ceed);
  CeedBasisCreateTensorH1Lagrange(ceed, 2, 1, P_1D, Q_1D, CEED_GAUSS_LOBATTO, &basis);
  CeedVectorCreate(ceed, 2 * POINT_STRIDE, &x_points);
  CeedVectorCreate(ceed, P_1D * P_1D, &u);
  CeedVectorCreate(ceed, POINT_STRIDE, &v);
  CeedVectorCreate(ceed, 2 * POINT_STRIDE, &v_grad);
  CeedVectorCreate(ceed, POINT_STRIDE, &point_values);
  CeedVectorCreate(ceed, P_1D * P_1D, &nodes_out);
  {
    CeedScalar x_array[2 * POINT_STRIDE] = {0.0, 0.0, 17.5, 19.5, -1.0, 1.0, 31.0, 41.0};
    CeedScalar u_array[P_1D * P_1D]      = {-1.0, -1.0, 1.0, 1.0};
    CeedScalar point_array[POINT_STRIDE] = {2.0, 3.0, -7.0, -11.0};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
    CeedVectorSetArray(point_values, CEED_MEM_HOST, CEED_COPY_VALUES, point_array);
  }
  {
    const CeedInt num_points = NUM_POINTS;

    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v_grad);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_TRANSPOSE, CEED_EVAL_INTERP, x_points, point_values, nodes_out);
  }
  {
    const CeedScalar *array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &array);
    if (fabs(array[0] + 1.0) > 100. * CEED_EPSILON || fabs(array[1] - 1.0) > 100. * CEED_EPSILON)
      printf("Padded tensor interpolation values [%f, %f] != [-1, 1]\n", array[0], array[1]);
    CeedVectorRestoreArrayRead(v, &array);

    CeedVectorGetArrayRead(v_grad, CEED_MEM_HOST, &array);
    for (CeedInt p = 0; p < NUM_POINTS; p++) {
      if (fabs(array[p]) > 100. * CEED_EPSILON || fabs(array[POINT_STRIDE + p] - 1.0) > 100. * CEED_EPSILON)
        printf("Padded tensor gradient mismatch at point %" CeedInt_FMT "\n", p);
    }
    CeedVectorRestoreArrayRead(v_grad, &array);

    CeedVectorGetArrayRead(nodes_out, CEED_MEM_HOST, &array);
    const CeedScalar expected[P_1D * P_1D] = {1.0, 1.0, 1.5, 1.5};
    for (CeedInt i = 0; i < P_1D * P_1D; i++) {
      if (fabs(array[i] - expected[i]) > 100. * CEED_EPSILON)
        printf("Padded tensor transpose value %f != %f at node %" CeedInt_FMT "\n", array[i], expected[i], i);
    }
    CeedVectorRestoreArrayRead(nodes_out, &array);
  }
  {
    const CeedInt num_points = NUM_POINTS;
    CeedVector    x_compact, v_compact;
    CeedScalar    x_array[2 * NUM_POINTS] = {0.0, 0.0, -1.0, 1.0};
    const CeedScalar *array;

    CeedVectorCreate(ceed, 2 * NUM_POINTS, &x_compact);
    CeedVectorCreate(ceed, NUM_POINTS, &v_compact);
    CeedVectorSetArray(x_compact, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_compact, u, v_compact);
    CeedVectorGetArrayRead(v_compact, CEED_MEM_HOST, &array);
    if (fabs(array[0] + 1.0) > 100. * CEED_EPSILON || fabs(array[1] - 1.0) > 100. * CEED_EPSILON)
      printf("Tensor AtPoints cache invalidation failed: [%f, %f] != [-1, 1]\n", array[0], array[1]);
    CeedVectorRestoreArrayRead(v_compact, &array);
    CeedVectorDestroy(&x_compact);
    CeedVectorDestroy(&v_compact);
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&v_grad);
  CeedVectorDestroy(&point_values);
  CeedVectorDestroy(&nodes_out);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
