/// @file
/// Test square non-tensor H(curl)/H(div) prism AtPoints reconstruction
/// \test Test square non-tensor H(curl)/H(div) prism AtPoints reconstruction
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define P 6
#define Q 6
#define NUM_POINTS 3

static void PrismModes(CeedScalar x, CeedScalar y, CeedScalar z, CeedScalar mode[Q], CeedScalar grad[3][Q]) {
  mode[0] = 1.0;
  mode[1] = y;
  mode[2] = x;
  mode[3] = z;
  mode[4] = y * z;
  mode[5] = x * z;

  memset(grad, 0, 3 * Q * sizeof(CeedScalar));
  grad[1][1] = 1.0;
  grad[0][2] = 1.0;
  grad[2][3] = 1.0;
  grad[1][4] = z;
  grad[2][4] = y;
  grad[0][5] = z;
  grad[2][5] = x;
}

static void FillTables(bool hcurl, bool incompatible, const CeedScalar q_ref[3 * Q], CeedScalar interp[3 * Q * P], CeedScalar deriv[3 * Q * P]) {
  memset(interp, 0, 3 * Q * P * sizeof(CeedScalar));
  memset(deriv, 0, 3 * Q * P * sizeof(CeedScalar));
  for (CeedInt q = 0; q < Q; q++) {
    CeedScalar mode[Q], grad[3][Q];

    PrismModes(q_ref[0 * Q + q], q_ref[1 * Q + q], q_ref[2 * Q + q], mode, grad);
    for (CeedInt node = 0; node < P; node++) {
      const CeedInt component = node % 3;

      interp[component * Q * P + q * P + node] = mode[node];
      if (hcurl) {
        const CeedScalar curl[3] = {
            component == 2 ? grad[1][node] : (component == 1 ? -grad[2][node] : 0.0),
            component == 0 ? grad[2][node] : (component == 2 ? -grad[0][node] : 0.0),
            component == 1 ? grad[0][node] : (component == 0 ? -grad[1][node] : 0.0),
        };

        for (CeedInt d = 0; d < 3; d++) deriv[d * Q * P + q * P + node] = curl[d];
      } else {
        deriv[q * P + node] = grad[component][node];
      }
    }
  }
  if (incompatible) deriv[0] += 1.0;
}

static void CheckUnsupportedLineHcurl(Ceed ceed) {
  const CeedInt num_points = 1;
  CeedBasis     basis;
  CeedVector    x_points, u, v;
  CeedScalar    q_ref[2] = {-1.0, 1.0}, q_weight[2] = {1.0, 1.0};
  CeedScalar    interp[4] = {1.0, 0.0, 0.0, 1.0}, curl[4] = {0.0};

  CeedBasisCreateHcurl(ceed, CEED_TOPOLOGY_LINE, 1, 2, 2, interp, curl, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, 1, &x_points);
  CeedVectorCreate(ceed, 2, &u);
  CeedVectorCreate(ceed, 1, &v);
  {
    const int   ierr = CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
    const char *error_message;

    if (!ierr) printf("Expected unsupported line H(curl) AtPoints error\n");
    CeedGetErrorMessage(ceed, &error_message);
    if (!strstr(error_message, "requires dimension 2 or 3")) printf("Unexpected error message: %s\n", error_message);
    CeedResetErrorMessage(ceed, &error_message);
  }
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
}

static void CheckSquarePrism(Ceed ceed, bool hcurl, bool incompatible) {
  CeedBasis  basis;
  CeedVector x_points, u, v;
  CeedScalar q_ref[3 * Q] = {0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1.};
  CeedScalar q_weight[Q]  = {1., 1., 1., 1., 1., 1.};
  CeedScalar interp[3 * Q * P], deriv[3 * Q * P];

  FillTables(hcurl, incompatible, q_ref, interp, deriv);
  if (hcurl) {
    CeedBasisCreateHcurl(ceed, CEED_TOPOLOGY_PRISM, 1, P, Q, interp, deriv, q_ref, q_weight, &basis);
  } else {
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_PRISM, 1, P, Q, interp, deriv, q_ref, q_weight, &basis);
  }

  CeedVectorCreate(ceed, 3 * NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, P, &u);
  CeedVectorCreate(ceed, 3 * NUM_POINTS, &v);
  {
    CeedScalar x_array[3 * NUM_POINTS] = {0.2, 0.4, 0.1, 0.1, 0.2, 0.3, 0.25, 0.75, 0.5};
    CeedScalar u_array[P]              = {2., -3., 5., 7., -11., 13.};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }
  {
    const CeedInt num_points = NUM_POINTS;
    const int     ierr       = CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);

    if (incompatible) {
      const char *error_message;

      if (!ierr) printf("Expected inconsistent square %s prism reconstruction error\n", hcurl ? "H(curl)" : "H(div)");
      CeedGetErrorMessage(ceed, &error_message);
      if (!strstr(error_message, "differential tabulation consistent")) printf("Unexpected error message: %s\n", error_message);
      CeedResetErrorMessage(ceed, &error_message);
      {
        const int repeat_ierr = CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);

        if (!repeat_ierr) printf("Expected repeated inconsistent square %s prism reconstruction error\n", hcurl ? "H(curl)" : "H(div)");
        CeedGetErrorMessage(ceed, &error_message);
        if (!strstr(error_message, "differential tabulation consistent")) printf("Unexpected repeated error message: %s\n", error_message);
        CeedResetErrorMessage(ceed, &error_message);
      }
    } else if (ierr) {
      const char *error_message;

      CeedGetErrorMessage(ceed, &error_message);
      printf("Unexpected square %s prism reconstruction error: %s\n", hcurl ? "H(curl)" : "H(div)", error_message);
      CeedResetErrorMessage(ceed, &error_message);
    } else {
      const CeedScalar *x_array, *u_array, *v_array;

      CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
      CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      for (CeedInt p = 0; p < NUM_POINTS; p++) {
        CeedScalar mode[Q], grad[3][Q], expected[3] = {0.0, 0.0, 0.0};

        PrismModes(x_array[0 * NUM_POINTS + p], x_array[1 * NUM_POINTS + p], x_array[2 * NUM_POINTS + p], mode, grad);
        for (CeedInt node = 0; node < P; node++) expected[node % 3] += u_array[node] * mode[node];
        for (CeedInt d = 0; d < 3; d++) {
          const CeedScalar actual = v_array[d * NUM_POINTS + p];

          if (fabs(actual - expected[d]) > 1000. * CEED_EPSILON)
            printf("%s prism: %f != %f at dim %" CeedInt_FMT ", point %" CeedInt_FMT "\n", hcurl ? "H(curl)" : "H(div)", actual, expected[d], d, p);
        }
      }
      CeedVectorRestoreArrayRead(x_points, &x_array);
      CeedVectorRestoreArrayRead(u, &u_array);
      CeedVectorRestoreArrayRead(v, &v_array);
    }
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  CeedSetErrorHandler(ceed, CeedErrorStore);
  CheckUnsupportedLineHcurl(ceed);
  CheckSquarePrism(ceed, true, false);
  CheckSquarePrism(ceed, false, false);
  CheckSquarePrism(ceed, true, true);
  CheckSquarePrism(ceed, false, true);
  CeedDestroy(&ceed);
  return 0;
}
