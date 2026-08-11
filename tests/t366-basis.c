/// @file
/// Test non-tensor H^1 tetrahedron gradient at arbitrary points
/// \test Test non-tensor H^1 tetrahedron gradient at arbitrary points
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define NUM_COMP 3
#define P 10
#define Q 11
#define NUM_POINTS 3

static void P2TetBasis(CeedScalar x, CeedScalar y, CeedScalar z, CeedScalar *interp, CeedScalar *grad) {
  const CeedScalar l[4]     = {1. - x - y - z, x, y, z};
  const CeedScalar dl[4][3] = {
      {-1., -1., -1.},
      {1.,  0.,  0. },
      {0.,  1.,  0. },
      {0.,  0.,  1. }
  };
  const CeedInt edge[6][2] = {
      {0, 1},
      {1, 2},
      {0, 2},
      {0, 3},
      {1, 3},
      {2, 3}
  };

  for (CeedInt i = 0; i < 4; i++) {
    interp[i] = l[i] * (2. * l[i] - 1.);
    for (CeedInt d = 0; d < 3; d++) grad[d * P + i] = (4. * l[i] - 1.) * dl[i][d];
  }
  for (CeedInt e = 0; e < 6; e++) {
    const CeedInt i = edge[e][0], j = edge[e][1], node = 4 + e;

    interp[node] = 4. * l[i] * l[j];
    for (CeedInt d = 0; d < 3; d++) grad[d * P + node] = 4. * (dl[i][d] * l[j] + l[i] * dl[j][d]);
  }
}

static CeedScalar Eval(const CeedScalar c[10], CeedScalar x, CeedScalar y, CeedScalar z) {
  return c[0] + c[1] * x + c[2] * y + c[3] * z + c[4] * x * y + c[5] * x * z + c[6] * y * z + c[7] * x * x + c[8] * y * y + c[9] * z * z;
}

static CeedScalar EvalGrad(const CeedScalar c[10], CeedInt d, CeedScalar x, CeedScalar y, CeedScalar z) {
  switch (d) {
    case 0:
      return c[1] + c[4] * y + c[5] * z + 2. * c[7] * x;
    case 1:
      return c[2] + c[4] * x + c[6] * z + 2. * c[8] * y;
    default:
      return c[3] + c[5] * x + c[6] * y + 2. * c[9] * z;
  }
}

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u, v;

  CeedInit(argv[1], &ceed);
  CeedScalar q_ref[3 * Q] = {0.0, 1.0, 0.0, 0.0, 0.5,  0.5, 0.0, 0.0, 0.5, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5,
                             0.5, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  0.0, 0.5, 0.5, 0.5, 0.25};
  CeedScalar q_weight[Q]  = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
  CeedScalar interp[Q * P], grad[3 * Q * P];
  for (CeedInt q = 0; q < Q; q++) {
    CeedScalar g[3 * P];

    P2TetBasis(q_ref[0 * Q + q], q_ref[1 * Q + q], q_ref[2 * Q + q], &interp[q * P], g);
    for (CeedInt d = 0; d < 3; d++) {
      for (CeedInt i = 0; i < P; i++) grad[d * Q * P + q * P + i] = g[d * P + i];
    }
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TET, NUM_COMP, P, Q, interp, grad, q_ref, q_weight, &basis);

  CeedVectorCreate(ceed, 3 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, NUM_COMP * P, &u);
  CeedVectorCreate(ceed, NUM_COMP * 3 * NUM_POINTS, &v);
  {
    CeedScalar x_array[3 * NUM_POINTS] = {0.2, 0.4, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.4};
    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  {
    const CeedScalar coeffs[NUM_COMP][10] = {
        {1.,  2.,  -1., 3.,  4.,  -2., 5.,  7.,  -3., 6. },
        {-2., 1.,  4.,  -1., 3.,  2.,  -4., 5.,  6.,  -7.},
        {3.,  -5., 2.,  1.,  -6., 4.,  3.,  -2., 8.,  5. }
    };
    const CeedScalar nodes[3 * P] = {0.,  1.,  0., 0., 0.5, 0.5, 0., 0., 0.5, 0., 0., 0., 1.,  0.,  0.,
                                     0.5, 0.5, 0., 0., 0.5, 0.,  0., 0., 1.,  0., 0., 0., 0.5, 0.5, 0.5};
    CeedScalar       u_array[NUM_COMP * P];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      for (CeedInt i = 0; i < P; i++) {
        u_array[comp * P + i] = Eval(coeffs[comp], nodes[0 * P + i], nodes[1 * P + i], nodes[2 * P + i]);
      }
    }
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  {
    const CeedInt num_points = NUM_POINTS;
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v);
  }

  {
    const CeedScalar coeffs[NUM_COMP][10] = {
        {1.,  2.,  -1., 3.,  4.,  -2., 5.,  7.,  -3., 6. },
        {-2., 1.,  4.,  -1., 3.,  2.,  -4., 5.,  6.,  -7.},
        {3.,  -5., 2.,  1.,  -6., 4.,  3.,  -2., 8.,  5. }
    };
    const CeedScalar *x_array, *v_array;

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      for (CeedInt d = 0; d < 3; d++) {
        for (CeedInt p = 0; p < NUM_POINTS; p++) {
          const CeedScalar expected =
              EvalGrad(coeffs[comp], d, x_array[0 * NUM_POINTS + p], x_array[1 * NUM_POINTS + p], x_array[2 * NUM_POINTS + p]);
          const CeedScalar actual = v_array[comp * NUM_POINTS + d * NUM_COMP * NUM_POINTS + p];

          if (fabs(actual - expected) > 500. * CEED_EPSILON)
            printf("%f != %f at comp %" CeedInt_FMT ", dim %" CeedInt_FMT ", point %" CeedInt_FMT "\n", actual, expected, comp, d, p);
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
