/// @file
/// Test non-tensor H^1 tetrahedron interpolation at arbitrary points
/// \test Test non-tensor H^1 tetrahedron interpolation at arbitrary points
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define P 4
#define Q 4
#define NUM_ELEM 2
#define MAX_NUM_POINTS 3

static CeedScalar Linear(CeedInt elem, CeedScalar x, CeedScalar y, CeedScalar z) { return 2.0 + elem - 3.0 * x + 5.0 * y - 7.0 * z; }

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

  CeedVectorCreate(ceed, 3 * NUM_ELEM * MAX_NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, NUM_ELEM * P, &u);
  CeedVectorCreate(ceed, NUM_ELEM * MAX_NUM_POINTS, &v);
  {
    // Dimension-major, padded by MAX_NUM_POINTS in each element.
    CeedScalar x_array[3 * NUM_ELEM * MAX_NUM_POINTS] = {
        0.2, 0.4, 0.1, 0.3, 0.2, 0.0,  // x
        0.1, 0.2, 0.3, 0.2, 0.4, 0.0,  // y
        0.1, 0.1, 0.4, 0.1, 0.2, 0.0   // z
    };
    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  {
    CeedScalar u_array[NUM_ELEM * P];

    for (CeedInt elem = 0; elem < NUM_ELEM; elem++) {
      u_array[elem * P + 0] = Linear(elem, 0., 0., 0.);
      u_array[elem * P + 1] = Linear(elem, 1., 0., 0.);
      u_array[elem * P + 2] = Linear(elem, 0., 1., 0.);
      u_array[elem * P + 3] = Linear(elem, 0., 0., 1.);
    }
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  {
    const CeedInt num_points[NUM_ELEM] = {MAX_NUM_POINTS, MAX_NUM_POINTS - 1};
    CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
  }

  {
    const CeedInt     num_points[NUM_ELEM] = {MAX_NUM_POINTS, MAX_NUM_POINTS - 1};
    const CeedScalar *x_array, *v_array;

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt elem = 0; elem < NUM_ELEM; elem++) {
      for (CeedInt p = 0; p < num_points[elem]; p++) {
        const CeedScalar x        = x_array[0 * NUM_ELEM * MAX_NUM_POINTS + elem * MAX_NUM_POINTS + p];
        const CeedScalar y        = x_array[1 * NUM_ELEM * MAX_NUM_POINTS + elem * MAX_NUM_POINTS + p];
        const CeedScalar z        = x_array[2 * NUM_ELEM * MAX_NUM_POINTS + elem * MAX_NUM_POINTS + p];
        const CeedScalar expected = Linear(elem, x, y, z);
        const CeedScalar actual   = v_array[elem * MAX_NUM_POINTS + p];

        if (fabs(actual - expected) > 500. * CEED_EPSILON) {
          printf("%f != %f at elem %" CeedInt_FMT ", point %" CeedInt_FMT "\n", actual, expected, elem, p);
        }
      }
    }
    CeedVectorRestoreArrayRead(x_points, &x_array);
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Reuse the basis with exact storage for a smaller maximum point count.
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&v);
  CeedVectorCreate(ceed, 3 * NUM_ELEM, &x_points);
  CeedVectorCreate(ceed, NUM_ELEM, &v);
  {
    const CeedInt num_points[NUM_ELEM]  = {1, 1};
    CeedScalar    x_array[3 * NUM_ELEM] = {0.2, 0.3, 0.1, 0.2, 0.1, 0.1};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
  }
  {
    const CeedScalar *x_array, *v_array;

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt elem = 0; elem < NUM_ELEM; elem++) {
      const CeedScalar x        = x_array[0 * NUM_ELEM + elem];
      const CeedScalar y        = x_array[1 * NUM_ELEM + elem];
      const CeedScalar z        = x_array[2 * NUM_ELEM + elem];
      const CeedScalar expected = Linear(elem, x, y, z);
      const CeedScalar actual   = v_array[elem];

      if (fabs(actual - expected) > 500. * CEED_EPSILON) {
        printf("%f != %f at elem %" CeedInt_FMT " after point-count update\n", actual, expected, elem);
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
