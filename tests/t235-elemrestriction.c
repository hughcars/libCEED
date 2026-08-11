/// @file
/// Test standard and strided restriction apply against guarded buffers and multiplicity round trips
/// \test Test standard and strided restriction apply against guarded buffers and multiplicity round trips
//TESTARGS(only="cpu") {ceed_resource}
#if defined(__linux__)
#define _GNU_SOURCE
#endif

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

// Round trip y = R^T (R x) for a standard restriction must equal multiplicity .* x
static int RunStandardCase(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, bool interleaved) {
  const CeedInt  num_nodes   = num_elem * (elem_size > 1 ? elem_size - 1 : 1) + 1;  // overlapping chain, multiplicity > 1 at interior joins
  const CeedInt  comp_stride = interleaved ? 1 : num_nodes;
  const CeedSize l_size      = interleaved ? (CeedSize)num_nodes * num_comp : (CeedSize)num_comp * num_nodes;
  CeedSize       e_size;
  CeedInt       *offsets = malloc((size_t)num_elem * elem_size * sizeof(*offsets));
  CeedScalar    *mult    = calloc((size_t)l_size, sizeof(*mult));
  GuardedArray   x = {0}, y = {0}, e = {0};
  CeedVector          x_vec, y_vec, e_vec;
  CeedElemRestriction rstr;
  int                 failed = 0;

  for (CeedInt el = 0; el < num_elem; el++) {
    for (CeedInt i = 0; i < elem_size; i++) {
      const CeedInt node          = elem_size > 1 ? el * (elem_size - 1) + i : el % num_nodes;
      offsets[el * elem_size + i] = interleaved ? node * num_comp : node;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, CEED_MEM_HOST, CEED_COPY_VALUES, offsets, &rstr);
  CeedElemRestrictionGetEVectorSize(rstr, &e_size);
  if (GuardedArrayCreate(l_size, &x) || GuardedArrayCreate(l_size, &y) || GuardedArrayCreate(e_size, &e)) return 1;

  for (CeedSize i = 0; i < l_size; i++) x.data[i] = 1.0 + (CeedScalar)((3 * i) % 7);
  for (CeedInt el = 0; el < num_elem; el++) {
    for (CeedInt i = 0; i < elem_size; i++) {
      for (CeedInt k = 0; k < num_comp; k++) mult[offsets[el * elem_size + i] + k * comp_stride] += 1.0;
    }
  }

  CeedVectorCreate(ceed, l_size, &x_vec);
  CeedVectorSetArray(x_vec, CEED_MEM_HOST, CEED_USE_POINTER, x.data);
  CeedVectorCreate(ceed, l_size, &y_vec);
  CeedVectorSetArray(y_vec, CEED_MEM_HOST, CEED_USE_POINTER, y.data);
  CeedVectorCreate(ceed, e_size, &e_vec);
  CeedVectorSetArray(e_vec, CEED_MEM_HOST, CEED_USE_POINTER, e.data);

  CeedVectorSetValue(y_vec, 0.0);
  CeedElemRestrictionApply(rstr, CEED_NOTRANSPOSE, x_vec, e_vec, CEED_REQUEST_IMMEDIATE);
  CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, e_vec, y_vec, CEED_REQUEST_IMMEDIATE);

  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y_vec, CEED_MEM_HOST, &y_array);
    for (CeedSize i = 0; i < l_size; i++) {
      if (fabs(y_array[i] - mult[i] * x.data[i]) > 1e-12 * (1 + fabs(mult[i] * x.data[i]))) {
        printf("standard num_elem %" CeedInt_FMT " elem_size %" CeedInt_FMT " num_comp %" CeedInt_FMT " interleaved %d: y[%" CeedSize_FMT
               "] = %f != %f\n",
               num_elem, elem_size, num_comp, (int)interleaved, i, y_array[i], mult[i] * x.data[i]);
        failed = 1;
        break;
      }
    }
    CeedVectorRestoreArrayRead(y_vec, &y_array);
  }

  CeedVectorDestroy(&x_vec);
  CeedVectorDestroy(&y_vec);
  CeedVectorDestroy(&e_vec);
  CeedElemRestrictionDestroy(&rstr);
  GuardedArrayDestroy(&x);
  GuardedArrayDestroy(&y);
  GuardedArrayDestroy(&e);
  free(offsets);
  free(mult);
  return failed;
}

// Round trip for strided restriction must return exactly x
static int RunStridedCase(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp) {
  const CeedSize l_size = (CeedSize)num_elem * elem_size * num_comp;
  CeedSize       e_size;
  GuardedArray   x = {0}, y = {0}, e = {0};
  CeedVector          x_vec, y_vec, e_vec;
  CeedElemRestriction rstr;
  int                 failed = 0;

  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, num_comp, l_size, CEED_STRIDES_BACKEND, &rstr);
  CeedElemRestrictionGetEVectorSize(rstr, &e_size);
  if (GuardedArrayCreate(l_size, &x) || GuardedArrayCreate(l_size, &y) || GuardedArrayCreate(e_size, &e)) return 1;
  for (CeedSize i = 0; i < l_size; i++) x.data[i] = 1.0 + (CeedScalar)((5 * i) % 11);

  CeedVectorCreate(ceed, l_size, &x_vec);
  CeedVectorSetArray(x_vec, CEED_MEM_HOST, CEED_USE_POINTER, x.data);
  CeedVectorCreate(ceed, l_size, &y_vec);
  CeedVectorSetArray(y_vec, CEED_MEM_HOST, CEED_USE_POINTER, y.data);
  CeedVectorCreate(ceed, e_size, &e_vec);
  CeedVectorSetArray(e_vec, CEED_MEM_HOST, CEED_USE_POINTER, e.data);

  CeedVectorSetValue(y_vec, 0.0);
  CeedElemRestrictionApply(rstr, CEED_NOTRANSPOSE, x_vec, e_vec, CEED_REQUEST_IMMEDIATE);
  CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, e_vec, y_vec, CEED_REQUEST_IMMEDIATE);

  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y_vec, CEED_MEM_HOST, &y_array);
    for (CeedSize i = 0; i < l_size; i++) {
      if (fabs(y_array[i] - x.data[i]) > 1e-12 * (1 + fabs(x.data[i]))) {
        printf("strided num_elem %" CeedInt_FMT " elem_size %" CeedInt_FMT " num_comp %" CeedInt_FMT ": y[%" CeedSize_FMT "] = %f != %f\n", num_elem,
               elem_size, num_comp, i, y_array[i], x.data[i]);
        failed = 1;
        break;
      }
    }
    CeedVectorRestoreArrayRead(y_vec, &y_array);
  }

  CeedVectorDestroy(&x_vec);
  CeedVectorDestroy(&y_vec);
  CeedVectorDestroy(&e_vec);
  CeedElemRestrictionDestroy(&rstr);
  GuardedArrayDestroy(&x);
  GuardedArrayDestroy(&y);
  GuardedArrayDestroy(&e);
  return failed;
}

// Round trip for oriented restriction must cancel signs: y = R^T (R x) == multiplicity .* x
static int RunOrientedCase(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp) {
  const CeedInt  num_nodes = num_elem * (elem_size - 1) + 1, comp_stride = num_nodes;
  const CeedSize l_size = (CeedSize)num_comp * num_nodes;
  CeedSize       e_size;
  CeedInt       *ind     = malloc((size_t)num_elem * elem_size * sizeof(*ind));
  bool          *orients = malloc((size_t)num_elem * elem_size * sizeof(*orients));
  CeedScalar    *ref     = calloc((size_t)l_size, sizeof(*ref));
  GuardedArray   x = {0}, y = {0}, e = {0};
  CeedVector          x_vec, y_vec, e_vec;
  CeedElemRestriction rstr;
  int                 failed = 0;

  for (CeedInt el = 0; el < num_elem; el++) {
    for (CeedInt i = 0; i < elem_size; i++) {
      ind[el * elem_size + i]     = el * (elem_size - 1) + i;
      orients[el * elem_size + i] = ((3 * el + i) % 3) == 1;
    }
  }
  CeedElemRestrictionCreateOriented(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, CEED_MEM_HOST, CEED_COPY_VALUES, ind, orients, &rstr);
  CeedElemRestrictionGetEVectorSize(rstr, &e_size);
  if (GuardedArrayCreate(l_size, &x) || GuardedArrayCreate(l_size, &y) || GuardedArrayCreate(e_size, &e)) return 1;
  for (CeedSize i = 0; i < l_size; i++) x.data[i] = 1.0 + (CeedScalar)((3 * i) % 7);
  for (CeedInt el = 0; el < num_elem; el++) {
    for (CeedInt i = 0; i < elem_size; i++) {
      for (CeedInt k = 0; k < num_comp; k++) ref[ind[el * elem_size + i] + k * comp_stride] += x.data[ind[el * elem_size + i] + k * comp_stride];
    }
  }

  CeedVectorCreate(ceed, l_size, &x_vec);
  CeedVectorSetArray(x_vec, CEED_MEM_HOST, CEED_USE_POINTER, x.data);
  CeedVectorCreate(ceed, l_size, &y_vec);
  CeedVectorSetArray(y_vec, CEED_MEM_HOST, CEED_USE_POINTER, y.data);
  CeedVectorCreate(ceed, e_size, &e_vec);
  CeedVectorSetArray(e_vec, CEED_MEM_HOST, CEED_USE_POINTER, e.data);

  CeedVectorSetValue(y_vec, 0.0);
  CeedElemRestrictionApply(rstr, CEED_NOTRANSPOSE, x_vec, e_vec, CEED_REQUEST_IMMEDIATE);
  CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, e_vec, y_vec, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y_vec, CEED_MEM_HOST, &y_array);
    for (CeedSize i = 0; i < l_size; i++) {
      if (fabs(y_array[i] - ref[i]) > 1e-11 * (1 + fabs(ref[i]))) {
        printf("oriented num_elem %" CeedInt_FMT " elem_size %" CeedInt_FMT " num_comp %" CeedInt_FMT ": y[%" CeedSize_FMT "] = %f != %f\n",
               num_elem, elem_size, num_comp, i, y_array[i], ref[i]);
        failed = 1;
        break;
      }
    }
    CeedVectorRestoreArrayRead(y_vec, &y_array);
  }

  CeedVectorDestroy(&x_vec);
  CeedVectorDestroy(&y_vec);
  CeedVectorDestroy(&e_vec);
  CeedElemRestrictionDestroy(&rstr);
  GuardedArrayDestroy(&x);
  GuardedArrayDestroy(&y);
  GuardedArrayDestroy(&e);
  free(ind);
  free(orients);
  free(ref);
  return failed;
}

// Curl-oriented round trip must match the explicit tridiagonal reference y = R_std^T T^T T R_std x
static int RunCurlOrientedCase(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp) {
  const CeedInt  num_nodes = num_elem * (elem_size - 1) + 1, comp_stride = num_nodes;
  const CeedSize l_size = (CeedSize)num_comp * num_nodes;
  CeedSize       e_size;
  CeedInt       *ind          = malloc((size_t)num_elem * elem_size * sizeof(*ind));
  CeedInt8      *curl_orients = malloc((size_t)3 * num_elem * elem_size * sizeof(*curl_orients));
  CeedScalar    *ref          = calloc((size_t)l_size, sizeof(*ref));
  GuardedArray   x = {0}, y = {0}, e = {0};
  CeedVector          x_vec, y_vec, e_vec;
  CeedElemRestriction rstr;
  int                 failed = 0;

  for (CeedInt el = 0; el < num_elem; el++) {
    for (CeedInt i = 0; i < elem_size; i++) {
      ind[el * elem_size + i]                       = el * (elem_size - 1) + i;
      curl_orients[3 * (el * elem_size + i) + 0]    = (i == 0) ? 0 : (CeedInt8)(((el + i) % 3) - 1);
      curl_orients[3 * (el * elem_size + i) + 1]    = (CeedInt8)(((2 * el + i) % 3) - 1);
      curl_orients[3 * (el * elem_size + i) + 2]    = (i == elem_size - 1) ? 0 : (CeedInt8)(((el + 2 * i) % 3) - 1);
    }
  }
  CeedElemRestrictionCreateCurlOriented(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, CEED_MEM_HOST, CEED_COPY_VALUES, ind,
                                        curl_orients, &rstr);
  CeedElemRestrictionGetEVectorSize(rstr, &e_size);
  if (GuardedArrayCreate(l_size, &x) || GuardedArrayCreate(l_size, &y) || GuardedArrayCreate(e_size, &e)) return 1;
  for (CeedSize i = 0; i < l_size; i++) x.data[i] = 1.0 + (CeedScalar)((3 * i) % 7);
  for (CeedInt el = 0; el < num_elem; el++) {
    for (CeedInt k = 0; k < num_comp; k++) {
      CeedScalar u_loc[16], w[16], z[16];

      for (CeedInt i = 0; i < elem_size; i++) u_loc[i] = x.data[ind[el * elem_size + i] + k * comp_stride];
      for (CeedInt i = 0; i < elem_size; i++) {
        const CeedInt8 *co = &curl_orients[3 * (el * elem_size + i)];

        w[i] = (i > 0 ? co[0] * u_loc[i - 1] : 0.0) + co[1] * u_loc[i] + (i < elem_size - 1 ? co[2] * u_loc[i + 1] : 0.0);
      }
      for (CeedInt i = 0; i < elem_size; i++) {
        CeedScalar s = curl_orients[3 * (el * elem_size + i) + 1] * w[i];

        if (i > 0) s += curl_orients[3 * (el * elem_size + i - 1) + 2] * w[i - 1];
        if (i < elem_size - 1) s += curl_orients[3 * (el * elem_size + i + 1) + 0] * w[i + 1];
        z[i] = s;
      }
      for (CeedInt i = 0; i < elem_size; i++) ref[ind[el * elem_size + i] + k * comp_stride] += z[i];
    }
  }

  CeedVectorCreate(ceed, l_size, &x_vec);
  CeedVectorSetArray(x_vec, CEED_MEM_HOST, CEED_USE_POINTER, x.data);
  CeedVectorCreate(ceed, l_size, &y_vec);
  CeedVectorSetArray(y_vec, CEED_MEM_HOST, CEED_USE_POINTER, y.data);
  CeedVectorCreate(ceed, e_size, &e_vec);
  CeedVectorSetArray(e_vec, CEED_MEM_HOST, CEED_USE_POINTER, e.data);

  CeedVectorSetValue(y_vec, 0.0);
  CeedElemRestrictionApply(rstr, CEED_NOTRANSPOSE, x_vec, e_vec, CEED_REQUEST_IMMEDIATE);
  CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, e_vec, y_vec, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y_vec, CEED_MEM_HOST, &y_array);
    for (CeedSize i = 0; i < l_size; i++) {
      if (fabs(y_array[i] - ref[i]) > 1e-11 * (1 + fabs(ref[i]))) {
        printf("curl-oriented num_elem %" CeedInt_FMT " elem_size %" CeedInt_FMT " num_comp %" CeedInt_FMT ": y[%" CeedSize_FMT "] = %f != %f\n",
               num_elem, elem_size, num_comp, i, y_array[i], ref[i]);
        failed = 1;
        break;
      }
    }
    CeedVectorRestoreArrayRead(y_vec, &y_array);
  }

  CeedVectorDestroy(&x_vec);
  CeedVectorDestroy(&y_vec);
  CeedVectorDestroy(&e_vec);
  CeedElemRestrictionDestroy(&rstr);
  GuardedArrayDestroy(&x);
  GuardedArrayDestroy(&y);
  GuardedArrayDestroy(&e);
  free(ind);
  free(curl_orients);
  free(ref);
  return failed;
}

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt elem_counts[] = {1, 7, 8, 9, 16, 17, 33};

  CeedInit(argv[1], &ceed);
  for (size_t n = 0; n < sizeof(elem_counts) / sizeof(elem_counts[0]); n++) {
    for (CeedInt elem_size = 1; elem_size <= 8; elem_size += 7) {
      for (CeedInt num_comp = 1; num_comp <= 3; num_comp += 2) {
        if (RunStandardCase(ceed, elem_counts[n], elem_size, num_comp, false)) return 1;
        if (RunStandardCase(ceed, elem_counts[n], elem_size, num_comp, true)) return 1;
        if (RunStridedCase(ceed, elem_counts[n], elem_size, num_comp)) return 1;
      }
    }
    for (CeedInt elem_size = 2; elem_size <= 5; elem_size += 3) {
      for (CeedInt num_comp = 1; num_comp <= 2; num_comp++) {
        if (RunOrientedCase(ceed, elem_counts[n], elem_size, num_comp)) return 1;
        if (RunCurlOrientedCase(ceed, elem_counts[n], elem_size, num_comp)) return 1;
      }
    }
  }
  CeedDestroy(&ceed);
  return 0;
}
