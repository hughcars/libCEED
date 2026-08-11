/// @file
/// Test non-H1 non-tensor AtPoints rejects underdetermined tabulation
/// \test Test non-H1 non-tensor AtPoints rejects underdetermined tabulation
#include <ceed.h>
#include <stdio.h>
#include <string.h>

#define P 6
#define Q 3
#define NUM_POINTS 2

static void VectorTetBasis(CeedScalar x, CeedScalar y, CeedScalar z, CeedScalar *interp) {
  for (CeedInt i = 0; i < 3 * P; i++) interp[i] = 0.0;
  interp[0 * P + 0] = 1.0;
  interp[1 * P + 1] = 1.0;
  interp[2 * P + 2] = 1.0;
  interp[0 * P + 3] = x;
  interp[1 * P + 4] = y;
  interp[2 * P + 5] = z;
}

static int CheckUnderdetermined(Ceed ceed, bool hcurl) {
  CeedBasis  basis;
  CeedVector x_points, u, v;
  CeedScalar q_ref[3 * Q] = {0.1, 0.6, 0.1, 0.2, 0.1, 0.6, 0.3, 0.1, 0.1};
  CeedScalar q_weight[Q]  = {1., 1., 1.};
  CeedScalar interp[3 * Q * P], curl[3 * Q * P], div[Q * P];

  for (CeedInt i = 0; i < 3 * Q * P; i++) curl[i] = 0.0;
  for (CeedInt i = 0; i < Q * P; i++) div[i] = 0.0;
  for (CeedInt q = 0; q < Q; q++) {
    CeedScalar b[3 * P];

    VectorTetBasis(q_ref[0 * Q + q], q_ref[1 * Q + q], q_ref[2 * Q + q], b);
    for (CeedInt d = 0; d < 3; d++) {
      for (CeedInt i = 0; i < P; i++) interp[d * Q * P + q * P + i] = b[d * P + i];
    }
  }
  if (hcurl) {
    CeedBasisCreateHcurl(ceed, CEED_TOPOLOGY_TET, 1, P, Q, interp, curl, q_ref, q_weight, &basis);
  } else {
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_TET, 1, P, Q, interp, div, q_ref, q_weight, &basis);
  }

  CeedVectorCreate(ceed, 3 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, P, &u);
  CeedVectorCreate(ceed, 3 * NUM_POINTS, &v);
  {
    CeedScalar x_array[3 * NUM_POINTS] = {0.2, 0.4, 0.1, 0.2, 0.1, 0.1};
    CeedScalar u_array[P]              = {2., -3., 5., 7., -11., 13.};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  {
    const CeedInt num_points = NUM_POINTS;
    const int     ierr       = CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
    const char   *err_msg;

    if (!ierr) printf("Expected underdetermined %s AtPoints error\n", hcurl ? "Hcurl" : "Hdiv");
    CeedGetErrorMessage(ceed, &err_msg);
    if (!strstr(err_msg, "modal reconstruction")) printf("Unexpected error message: %s\n", err_msg);
    CeedResetErrorMessage(ceed, &err_msg);
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  CeedSetErrorHandler(ceed, CeedErrorStore);
  CheckUnderdetermined(ceed, true);
  CheckUnderdetermined(ceed, false);

  CeedDestroy(&ceed);
  return 0;
}
