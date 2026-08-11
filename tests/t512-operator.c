/// @file
/// Test full mass operator Apply and ApplyAdd against exactly sized guarded L-vectors and passive data
/// \test Test full mass operator Apply and ApplyAdd against exactly sized guarded L-vectors and passive data
//TESTARGS(only="cpu") {ceed_resource}
#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "t500-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

typedef struct {
  CeedScalar *data;
#if defined(__linux__)
  void  *mapping;
  size_t mapping_size;
#endif
} GuardedArray;

static int GuardedArrayCreate(CeedSize length, GuardedArray *array) {
  if (length <= 0 || (uintmax_t)length > SIZE_MAX / sizeof(*array->data)) return 1;
#if defined(__linux__)
  const long page_size = sysconf(_SC_PAGESIZE);

  if (page_size <= 0) return 1;
  const size_t bytes = (size_t)length * sizeof(*array->data), num_pages = (bytes + page_size - 1) / page_size;

  if (!bytes || num_pages >= SIZE_MAX / (size_t)page_size) return 1;
  array->mapping_size = (num_pages + 1) * page_size;
  array->mapping      = mmap(NULL, array->mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (array->mapping == MAP_FAILED) return 1;
  void *guard = (char *)array->mapping + num_pages * page_size;

  if (mprotect(guard, page_size, PROT_NONE)) {
    munmap(array->mapping, array->mapping_size);
    array->mapping = NULL;
    return 1;
  }
  array->data = (CeedScalar *)((char *)guard - bytes);
#else
  array->data = malloc((size_t)length * sizeof(*array->data));
  if (!array->data) return 1;
#endif
  return 0;
}

static void GuardedArrayDestroy(GuardedArray *array) {
#if defined(__linux__)
  if (array->mapping && array->mapping != MAP_FAILED) munmap(array->mapping, array->mapping_size);
#else
  free(array->data);
#endif
  array->data = NULL;
}

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, u, v, v_add;
  GuardedArray        x_g = {0}, u_g = {0}, v_g = {0}, v_add_g = {0}, q_data_g = {0};
  CeedInt             num_elem = 15, p = 5, q = 8;
  CeedInt             num_nodes_x = num_elem + 1, num_nodes_u = num_elem * (p - 1) + 1;
  CeedInt             ind_x[num_elem * 2], ind_u[num_elem * p];

  CeedInit(argv[1], &ceed);
  if (GuardedArrayCreate(num_nodes_x, &x_g) || GuardedArrayCreate(num_nodes_u, &u_g) || GuardedArrayCreate(num_nodes_u, &v_g) ||
      GuardedArrayCreate(num_nodes_u, &v_add_g) || GuardedArrayCreate((CeedSize)q * num_elem, &q_data_g)) {
    return 1;
  }
  for (CeedInt i = 0; i < num_nodes_x; i++) x_g.data[i] = (CeedScalar)i / (num_nodes_x - 1);
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) {
      ind_u[p * i + j] = i * (p - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, 1, 1, num_nodes_u, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_u);
  CeedInt strides_q_data[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_q_data, &elem_restriction_q_data);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);

  CeedVectorCreate(ceed, num_nodes_x, &x);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_g.data);
  CeedVectorCreate(ceed, q * num_elem, &q_data);
  CeedVectorSetArray(q_data, CEED_MEM_HOST, CEED_USE_POINTER, q_data_g.data);

  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, num_nodes_u, &u);
  CeedVectorSetArray(u, CEED_MEM_HOST, CEED_USE_POINTER, u_g.data);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, num_nodes_u, &v);
  CeedVectorSetArray(v, CEED_MEM_HOST, CEED_USE_POINTER, v_g.data);
  CeedVectorCreate(ceed, num_nodes_u, &v_add);
  CeedVectorSetArray(v_add, CEED_MEM_HOST, CEED_USE_POINTER, v_add_g.data);
  for (CeedInt i = 0; i < num_nodes_u; i++) v_add_g.data[i] = 2.0 - (CeedScalar)(i % 4);

  // Mass conservation with guarded active vectors: sum(M 1) == domain length
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.0;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_u; i++) sum += v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
    if (fabs(sum - 1.0) > 1e-10) printf("mass conservation: sum %g != 1.0\n", sum);
  }

  // ApplyAdd must accumulate the same action into the seeded output
  CeedOperatorApplyAdd(op_mass, u, v_add, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *v_array, *v_add_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    CeedVectorGetArrayRead(v_add, CEED_MEM_HOST, &v_add_array);
    for (CeedInt i = 0; i < num_nodes_u; i++) {
      const CeedScalar expected = (2.0 - (CeedScalar)(i % 4)) + v_array[i];

      if (fabs(v_add_array[i] - expected) > 1e-12 * (1 + fabs(expected))) {
        printf("ApplyAdd error at %" CeedInt_FMT ": %g != %g\n", i, v_add_array[i], expected);
        break;
      }
    }
    CeedVectorRestoreArrayRead(v, &v_array);
    CeedVectorRestoreArrayRead(v_add, &v_add_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&v_add);
  CeedVectorDestroy(&q_data);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  GuardedArrayDestroy(&x_g);
  GuardedArrayDestroy(&u_g);
  GuardedArrayDestroy(&v_g);
  GuardedArrayDestroy(&v_add_g);
  GuardedArrayDestroy(&q_data_g);
  CeedDestroy(&ceed);
  return 0;
}
