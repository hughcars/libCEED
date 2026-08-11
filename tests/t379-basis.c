/// @file
/// Test higher-order non-tensor AtPoints reconstruction and incompatible H^1 tabulation
/// \test Test higher-order non-tensor AtPoints reconstruction and incompatible H^1 tabulation
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define P 5
#define Q 6
#define NUM_POINTS 3

static CeedScalar PowInt(CeedScalar x, CeedInt p) {
  CeedScalar value = 1.0;

  for (CeedInt i = 0; i < p; i++) value *= x;
  return value;
}

static void CheckHigherOrder(Ceed ceed) {
  CeedBasis  basis;
  CeedVector x_points, u, v_interp, v_grad;
  CeedScalar q_ref[Q] = {-1.0, -0.7, -0.2, 0.2, 0.7, 1.0}, q_weight[Q] = {1., 1., 1., 1., 1., 1.};
  CeedScalar interp[Q * P], grad[Q * P];

  for (CeedInt q = 0; q < Q; q++) {
    for (CeedInt node = 0; node < P; node++) {
      interp[q * P + node] = PowInt(q_ref[q], node);
      grad[q * P + node]   = node ? node * PowInt(q_ref[q], node - 1) : 0.0;
    }
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_LINE, 1, P, Q, interp, grad, q_ref, q_weight, &basis);
  {
    CeedSize interp_flops, grad_flops;

    CeedBasisGetFlopsEstimate(basis, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, true, NUM_POINTS, &interp_flops);
    CeedBasisGetFlopsEstimate(basis, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, true, NUM_POINTS, &grad_flops);
    if (interp_flops != 180) printf("Incorrect non-tensor AtPoints interpolation FLOP estimate, %" CeedSize_FMT " != 180\n", interp_flops);
    if (grad_flops != 180) printf("Incorrect non-tensor AtPoints gradient FLOP estimate, %" CeedSize_FMT " != 180\n", grad_flops);
  }
  CeedSetErrorHandler(ceed, CeedErrorStore);
  {
    const CeedEvalMode unsupported_modes[2] = {CEED_EVAL_DIV, CEED_EVAL_CURL};

    for (CeedInt i = 0; i < 2; i++) {
      CeedSize    flops;
      const int   ierr = CeedBasisGetFlopsEstimate(basis, CEED_NOTRANSPOSE, unsupported_modes[i], true, NUM_POINTS, &flops);
      const char *error_message;

      if (ierr != CEED_ERROR_INCOMPATIBLE) printf("Expected incompatible non-tensor AtPoints FLOP estimate error, got %d\n", ierr);
      CeedGetErrorMessage(ceed, &error_message);
      if (!strstr(error_message, "not supported at points")) printf("Unexpected error message: %s\n", error_message);
      CeedResetErrorMessage(ceed, &error_message);
    }
  }
  CeedVectorCreate(ceed, NUM_POINTS, &x_points);
  CeedVectorCreate(ceed, P, &u);
  CeedVectorCreate(ceed, NUM_POINTS, &v_interp);
  CeedVectorCreate(ceed, NUM_POINTS, &v_grad);
  {
    CeedScalar x_array[NUM_POINTS] = {-0.8, 0.1, 0.9};
    CeedScalar u_array[P]          = {1.0, -2.0, 3.0, -4.0, 5.0};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }
  {
    const CeedInt num_points = NUM_POINTS;

    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v_interp);
    CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v_grad);
  }
  {
    const CeedScalar *x_array, *interp_array, *grad_array;
    const CeedScalar  coefficients[P] = {1.0, -2.0, 3.0, -4.0, 5.0};

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(v_interp, CEED_MEM_HOST, &interp_array);
    CeedVectorGetArrayRead(v_grad, CEED_MEM_HOST, &grad_array);
    for (CeedInt point = 0; point < NUM_POINTS; point++) {
      CeedScalar expected_interp = 0.0, expected_grad = 0.0;

      for (CeedInt mode = 0; mode < P; mode++) {
        expected_interp += coefficients[mode] * PowInt(x_array[point], mode);
        if (mode) expected_grad += mode * coefficients[mode] * PowInt(x_array[point], mode - 1);
      }
      if (fabs(interp_array[point] - expected_interp) > 5000. * CEED_EPSILON)
        printf("Higher-order interpolation mismatch at point %" CeedInt_FMT "\n", point);
      if (fabs(grad_array[point] - expected_grad) > 5000. * CEED_EPSILON) printf("Higher-order gradient mismatch at point %" CeedInt_FMT "\n", point);
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
}

static void CheckIncompatibleInterpolation(Ceed ceed) {
  CeedBasis  basis;
  CeedVector x_points, u, v;
  CeedScalar q_ref[4] = {-1.0, -0.4, 0.3, 1.0}, q_weight[4] = {1.0, 1.0, 1.0, 1.0};
  CeedScalar interp[12], grad[12];

  for (CeedInt q = 0; q < 4; q++) {
    interp[q * 3 + 0] = 1.0;
    interp[q * 3 + 1] = q_ref[q];
    interp[q * 3 + 2] = q_ref[q] * q_ref[q] * q_ref[q];
    grad[q * 3 + 0]   = 0.0;
    grad[q * 3 + 1]   = 1.0;
    grad[q * 3 + 2]   = 3.0 * q_ref[q] * q_ref[q];
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_LINE, 1, 3, 4, interp, grad, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, 1, &x_points);
  CeedVectorCreate(ceed, 3, &u);
  CeedVectorCreate(ceed, 1, &v);
  CeedSetErrorHandler(ceed, CeedErrorStore);
  {
    const CeedInt num_points = 1;
    const int     ierr       = CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
    const char   *error_message;

    if (!ierr) printf("Expected incompatible H^1 interpolation tabulation error\n");
    CeedGetErrorMessage(ceed, &error_message);
    if (!strstr(error_message, "consistent with the tabulation")) printf("Unexpected error message: %s\n", error_message);
    CeedResetErrorMessage(ceed, &error_message);
  }
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
}

static void CheckRankDeficientInterpolation(Ceed ceed) {
  CeedBasis        basis;
  CeedVector       x_points, u, v;
  const CeedInt    num_points = 1;
  const CeedScalar q_ref[2] = {0.5, 0.5}, q_weight[2] = {1.0, 1.0};
  const CeedScalar interp[4] = {1.0, 0.5, 1.0, 0.5}, grad[4] = {0.0, 1.0, 0.0, 1.0};

  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_LINE, 1, 2, 2, interp, grad, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, 1, &x_points);
  CeedVectorCreate(ceed, 2, &u);
  CeedVectorCreate(ceed, 1, &v);
  CeedSetErrorHandler(ceed, CeedErrorStore);
  {
    const int   ierr = CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
    const char *error_message;

    if (ierr != CEED_ERROR_UNSUPPORTED) printf("Expected rank-deficient H^1 interpolation error, got %d\n", ierr);
    CeedGetErrorMessage(ceed, &error_message);
    if (!strstr(error_message, "full-rank, well-conditioned")) printf("Unexpected error message: %s\n", error_message);
    CeedResetErrorMessage(ceed, &error_message);
  }
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
}

static void CheckIncompatibleGradient(Ceed ceed) {
  CeedBasis  basis;
  CeedVector x_points, u, v;
  CeedScalar q_ref[3] = {-1.0, 0.0, 1.0}, q_weight[3] = {1.0, 1.0, 1.0};
  CeedScalar interp[9], grad[9]                       = {0.0};

  for (CeedInt q = 0; q < 3; q++) {
    interp[q * 3 + 0] = 1.0;
    interp[q * 3 + 1] = q_ref[q];
    interp[q * 3 + 2] = q_ref[q] * q_ref[q];
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_LINE, 1, 3, 3, interp, grad, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, 1, &x_points);
  CeedVectorCreate(ceed, 3, &u);
  CeedVectorCreate(ceed, 1, &v);
  CeedSetErrorHandler(ceed, CeedErrorStore);
  {
    const CeedInt num_points = 1;
    const int     ierr       = CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, v);
    const char   *error_message;

    if (!ierr) printf("Expected incompatible H^1 gradient tabulation error\n");
    CeedGetErrorMessage(ceed, &error_message);
    if (!strstr(error_message, "gradient tabulation consistent")) printf("Unexpected error message: %s\n", error_message);
    CeedResetErrorMessage(ceed, &error_message);
  }
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  CheckHigherOrder(ceed);
  CheckIncompatibleInterpolation(ceed);
  CheckRankDeficientInterpolation(ceed);
  CheckIncompatibleGradient(ceed);
  CeedDestroy(&ceed);
  return 0;
}
