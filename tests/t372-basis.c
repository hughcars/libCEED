/// @file
/// Test CUDA/MAGMA non-tensor AtPoints rejects unpadded coordinate vectors
/// \test Test CUDA/MAGMA non-tensor AtPoints rejects unpadded coordinate vectors
#include <ceed.h>
#include <stdio.h>
#include <string.h>

#define P 4
#define Q 4
#define NUM_ELEM 2
#define MAX_NUM_POINTS 3

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector x_points, u, v;

  CeedInit(argv[1], &ceed);
  {
    const char *resource;

    CeedGetResource(ceed, &resource);
    if (!strstr(resource, "/gpu/cuda/ref") && !strstr(resource, "/gpu/cuda/magma") && !strstr(resource, "/gpu/hip/magma")) {
      CeedDestroy(&ceed);
      return 0;
    }
  }

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

  // Packed length is valid for the interface total-point contract, but GPU AtPoints
  // kernels require element-padded coordinates: dim * NUM_ELEM * MAX_NUM_POINTS.
  CeedVectorCreate(ceed, 3 * (MAX_NUM_POINTS + MAX_NUM_POINTS - 1), &x_points);
  CeedVectorCreate(ceed, NUM_ELEM * P, &u);
  CeedVectorCreate(ceed, MAX_NUM_POINTS + MAX_NUM_POINTS - 1, &v);
  {
    CeedScalar x_array[3 * (MAX_NUM_POINTS + MAX_NUM_POINTS - 1)] = {0.2, 0.4, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.4, 0.1, 0.1, 0.4, 0.1, 0.2};
    CeedScalar u_array[NUM_ELEM * P]                              = {1., 2., 3., 4., 5., 6., 7., 8.};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  CeedSetErrorHandler(ceed, CeedErrorStore);
  {
    const CeedInt num_points[NUM_ELEM] = {MAX_NUM_POINTS, MAX_NUM_POINTS - 1};
    const int     ierr                 = CeedBasisApplyAtPoints(basis, NUM_ELEM, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);
    const char   *err_msg;

    if (!ierr) printf("Expected unpadded coordinate vector error\n");
    CeedGetErrorMessage(ceed, &err_msg);
    if (!strstr(err_msg, "requires compact point storage to have a uniform number of points per element"))
      printf("Unexpected error message: %s\n", err_msg);
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
