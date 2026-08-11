/// @file
/// Test AtPoints dimension checks for independently short input and output vectors
/// \test Test AtPoints dimension checks for independently short input and output vectors
#include <ceed.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEM 2
#define P 2
#define Q 2

static void ExpectError(Ceed ceed, int ierr, const char *label) {
  const char *message;

  if (!ierr) printf("Expected dimension error for %s\n", label);
  CeedGetErrorMessage(ceed, &message);
  CeedResetErrorMessage(ceed, &message);
}

static void CheckPseudoinverseOverflow(Ceed ceed) {
  const CeedInt q = 46341, num_points = 1;
  CeedBasis     basis;
  CeedVector    x_points, u, v;
  CeedScalar   *q_ref = calloc(q, sizeof(*q_ref)), *q_weight = calloc(q, sizeof(*q_weight)), *interp = calloc(q, sizeof(*interp)),
             *grad = calloc(q, sizeof(*grad));

  if (!q_ref || !q_weight || !interp || !grad) {
    printf("Unable to allocate pseudoinverse overflow test data\n");
    free(q_ref);
    free(q_weight);
    free(interp);
    free(grad);
    return;
  }
  for (CeedInt i = 0; i < q; i++) {
    q_weight[i] = 1.0;
    interp[i]   = 1.0;
  }
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_LINE, 1, 1, q, interp, grad, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, 1, &x_points);
  CeedVectorCreate(ceed, 1, &u);
  CeedVectorCreate(ceed, 1, &v);
  ExpectError(ceed, CeedBasisApplyAtPoints(basis, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v),
              "pseudoinverse workspace overflow");
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  free(q_ref);
  free(q_weight);
  free(interp);
  free(grad);
}

int main(int argc, char **argv) {
  const CeedInt num_points[NUM_ELEM] = {1, 1};
  Ceed          ceed;
  CeedBasis     basis;
  CeedVector    x_points, oversized_x_points, padded_x_points, short_nodes, full_nodes, short_points, full_points, padded_points;
  CeedScalar    q_ref[Q] = {-1.0, 1.0}, q_weight[Q] = {1.0, 1.0};
  CeedScalar    interp[Q * P] = {1.0, 0.0, 0.0, 1.0};
  CeedScalar    grad[Q * P]   = {-0.5, 0.5, -0.5, 0.5};

  CeedInit(argv[1], &ceed);
  CeedSetErrorHandler(ceed, CeedErrorStore);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_LINE, 1, P, Q, interp, grad, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, NUM_ELEM, &x_points);
  CeedVectorCreate(ceed, NUM_ELEM + 1, &oversized_x_points);
  CeedVectorCreate(ceed, 2 * NUM_ELEM, &padded_x_points);
  CeedVectorCreate(ceed, NUM_ELEM * P - 1, &short_nodes);
  CeedVectorCreate(ceed, NUM_ELEM * P, &full_nodes);
  CeedVectorCreate(ceed, NUM_ELEM - 1, &short_points);
  CeedVectorCreate(ceed, NUM_ELEM, &full_points);
  CeedVectorCreate(ceed, 2 * NUM_ELEM, &padded_points);

  ExpectError(ceed, CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, short_nodes, full_points),
              "short NOTRANSPOSE input");
  ExpectError(ceed, CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, full_nodes, short_points),
              "short NOTRANSPOSE output");
  ExpectError(ceed, CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_TRANSPOSE, CEED_EVAL_INTERP, x_points, short_points, full_nodes),
              "short TRANSPOSE input");
  ExpectError(ceed, CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_TRANSPOSE, CEED_EVAL_INTERP, x_points, full_points, short_nodes),
              "short TRANSPOSE output");
  ExpectError(ceed,
              CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, oversized_x_points, full_nodes, full_points),
              "ambiguous oversized coordinates");
  ExpectError(ceed, CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, padded_x_points, full_nodes, full_points),
              "mismatched compact point values");
  ExpectError(ceed, CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, full_nodes, padded_points),
              "mismatched padded point values");
  {
    const CeedInt invalid_num_points[NUM_ELEM] = {1, -1};

    ExpectError(ceed,
                CeedBasisApplyAtPoints(basis, NUM_ELEM, invalid_num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, full_nodes, full_points),
                "negative point count");
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&oversized_x_points);
  CeedVectorDestroy(&padded_x_points);
  CeedVectorDestroy(&short_nodes);
  CeedVectorDestroy(&full_nodes);
  CeedVectorDestroy(&short_points);
  CeedVectorDestroy(&full_points);
  CeedVectorDestroy(&padded_points);
  CeedBasisDestroy(&basis);
  CheckPseudoinverseOverflow(ceed);
  CeedDestroy(&ceed);
  return 0;
}
