// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Opt(CeedQFunction qf, CeedOperator op, bool is_input, bool *skip_rstr, bool *apply_add_basis,
                                       const CeedInt block_size, CeedElemRestriction *block_rstr, CeedVector *e_vecs_full, CeedVector *e_vecs,
                                       CeedVector *q_vecs, CeedInt start_e, CeedInt num_fields, CeedInt Q) {
  Ceed                ceed;
  CeedSize            e_size, q_size;
  CeedInt             num_comp, size, P;
  CeedQFunctionField *qf_fields;
  CeedOperatorField  *op_fields;

  {
    Ceed ceed_parent;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
    CeedCallBackend(CeedReferenceCopy(ceed_parent, &ceed));
    CeedCallBackend(CeedDestroy(&ceed_parent));
  }
  if (is_input) {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  } else {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  }

  // Loop over fields
  for (CeedInt i = 0; i < num_fields; i++) {
    CeedEvalMode eval_mode;
    CeedBasis    basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {
      Ceed                ceed_rstr;
      CeedSize            l_size;
      CeedInt             num_elem, elem_size, comp_stride;
      CeedRestrictionType rstr_type;
      CeedElemRestriction rstr;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));
      CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed_rstr));
      CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
      CeedCallBackend(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
      CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
      CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));

      CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
      switch (rstr_type) {
        case CEED_RESTRICTION_STANDARD: {
          const CeedInt *offsets = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionCreateBlocked(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, CEED_MEM_HOST,
                                                           CEED_COPY_VALUES, offsets, &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
        } break;
        case CEED_RESTRICTION_ORIENTED: {
          const bool    *orients = NULL;
          const CeedInt *offsets = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionGetOrientations(rstr, CEED_MEM_HOST, &orients));
          CeedCallBackend(CeedElemRestrictionCreateBlockedOriented(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size,
                                                                   CEED_MEM_HOST, CEED_COPY_VALUES, offsets, orients, &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
          CeedCallBackend(CeedElemRestrictionRestoreOrientations(rstr, &orients));
        } break;
        case CEED_RESTRICTION_CURL_ORIENTED: {
          const CeedInt8 *curl_orients = NULL;
          const CeedInt  *offsets      = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionGetCurlOrientations(rstr, CEED_MEM_HOST, &curl_orients));
          CeedCallBackend(CeedElemRestrictionCreateBlockedCurlOriented(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size,
                                                                       CEED_MEM_HOST, CEED_COPY_VALUES, offsets, curl_orients,
                                                                       &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
          CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(rstr, &curl_orients));
        } break;
        case CEED_RESTRICTION_STRIDED: {
          CeedInt strides[3];

          CeedCallBackend(CeedElemRestrictionGetStrides(rstr, strides));
          CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed_rstr, num_elem, elem_size, block_size, num_comp, l_size, strides,
                                                                  &block_rstr[i + start_e]));
        } break;
        // LCOV_EXCL_START
        case CEED_RESTRICTION_POINTS:
          // Empty case - won't occur
          break;
          // LCOV_EXCL_STOP
      }
      CeedCallBackend(CeedDestroy(&ceed_rstr));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr));
      CeedCallBackend(CeedElemRestrictionCreateVector(block_rstr[i + start_e], NULL, &e_vecs_full[i + start_e]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        e_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        CeedCallBackend(CeedBasisDestroy(&basis));
        e_size = (CeedSize)P * num_comp * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)Q * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(CeedBasisApply(basis, block_size, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));
        CeedCallBackend(CeedBasisDestroy(&basis));
        break;
    }
    // Initialize E-vec arrays
    if (e_vecs[i]) CeedCallBackend(CeedVectorSetValue(e_vecs[i], 0.0));
  }
  // Drop duplicate restrictions
  if (is_input) {
    for (CeedInt i = 0; i < num_fields; i++) {
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr_i));
      for (CeedInt j = i + 1; j < num_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[j], &rstr_j));
        if (vec_i == vec_j && rstr_i == rstr_j) {
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs_full[i + start_e], &e_vecs_full[j + start_e]));
          skip_rstr[j] = true;
        }
        CeedCallBackend(CeedVectorDestroy(&vec_j));
        CeedCallBackend(CeedElemRestrictionDestroy(&rstr_j));
      }
      CeedCallBackend(CeedVectorDestroy(&vec_i));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_i));
    }
  } else {
    for (CeedInt i = num_fields - 1; i >= 0; i--) {
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr_i));
      for (CeedInt j = i - 1; j >= 0; j--) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[j], &rstr_j));
        if (vec_i == vec_j && rstr_i == rstr_j) {
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs_full[i + start_e], &e_vecs_full[j + start_e]));
          skip_rstr[j]       = true;
          apply_add_basis[i] = true;
        }
        CeedCallBackend(CeedVectorDestroy(&vec_j));
        CeedCallBackend(CeedElemRestrictionDestroy(&rstr_j));
      }
      CeedCallBackend(CeedVectorDestroy(&vec_i));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_i));
    }
  }
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

static inline int CeedOperatorGetFieldVectors_Opt(CeedInt num_input_fields, CeedOperatorField *op_input_fields,
                                                  CeedInt num_output_fields, CeedOperatorField *op_output_fields,
                                                  bool *is_input_active, bool *is_output_active,
                                                  CeedVector *input_vectors, CeedVector *output_vectors);
static inline int CeedOperatorGetFieldMetadata_Opt(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields,
                                                   CeedOperatorField *op_input_fields, CeedInt num_output_fields,
                                                   CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                                   CeedEvalMode *input_eval_modes, CeedEvalMode *output_eval_modes,
                                                   CeedInt *input_elem_sizes, CeedInt *input_field_sizes,
                                                   CeedInt *input_num_comps, CeedBasis *input_bases, CeedBasis *output_bases);
static inline int CeedOperatorSetBlockSize_Opt(CeedInt num_input_fields, CeedBasis *input_bases,
                                               CeedInt num_output_fields, CeedBasis *output_bases, CeedInt *block_size);

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Opt(CeedOperator op) {
  bool                is_setup_done;
  CeedInt             Q, num_elem, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &impl->is_identity_qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->block_rstr));
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->e_vecs_full));

  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->skip_rstr_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->skip_rstr_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->apply_add_basis_out));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->is_input_active));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->is_output_active));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_eval_modes));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->output_eval_modes));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->active_rstr_input_indices));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_elem_sizes));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_field_sizes));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_num_comps));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_bases));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->output_bases));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_vectors));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->output_vectors));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->input_states));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_out));

  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  CeedCallBackend(CeedOperatorGetFieldVectors_Opt(num_input_fields, op_input_fields, num_output_fields, op_output_fields,
                                                  impl->is_input_active, impl->is_output_active,
                                                  impl->input_vectors, impl->output_vectors));
  CeedCallBackend(CeedOperatorGetFieldMetadata_Opt(num_input_fields, qf_input_fields, op_input_fields, num_output_fields, qf_output_fields,
                                                   op_output_fields, impl->input_eval_modes, impl->output_eval_modes,
                                                   impl->input_elem_sizes, impl->input_field_sizes, impl->input_num_comps,
                                                   impl->input_bases, impl->output_bases));
  CeedCallBackend(CeedOperatorSetBlockSize_Opt(num_input_fields, impl->input_bases, num_output_fields, impl->output_bases,
                                               &impl->block_size));
  const CeedInt block_size = impl->block_size;
  impl->Q                    = Q;
  impl->num_blocks           = (num_elem / block_size) + !!(num_elem % block_size);

  // Set up infield and outfield pointer arrays
  // Infields
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, true, impl->skip_rstr_in, NULL, block_size, impl->block_rstr, impl->e_vecs_full,
                                              impl->e_vecs_in, impl->q_vecs_in, 0, num_input_fields, Q));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, false, impl->skip_rstr_out, impl->apply_add_basis_out, block_size, impl->block_rstr,
                                              impl->e_vecs_full, impl->e_vecs_out, impl->q_vecs_out, num_input_fields, num_output_fields, Q));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    if (impl->is_input_active[i] && impl->block_rstr[i] && !impl->skip_rstr_in[i]) {
      impl->active_rstr_input_indices[impl->num_active_rstr_inputs++] = i;
    }
  }

  // Identity QFunctions
  if (impl->is_identity_qf) {
    CeedEvalMode        in_mode, out_mode;
    CeedQFunctionField *in_fields, *out_fields;

    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &in_fields, NULL, &out_fields));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(in_fields[0], &in_mode));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(out_fields[0], &out_mode));

    if (in_mode == CEED_EVAL_NONE && out_mode == CEED_EVAL_NONE) {
      impl->is_identity_rstr_op = true;
    } else {
      CeedCallBackend(CeedVectorReferenceCopy(impl->q_vecs_in[0], &impl->q_vecs_out[0]));
    }
  }

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get Field Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorGetFieldVectors_Opt(CeedInt num_input_fields, CeedOperatorField *op_input_fields,
                                                  CeedInt num_output_fields, CeedOperatorField *op_output_fields,
                                                  bool *is_input_active, bool *is_output_active,
                                                  CeedVector *input_vectors, CeedVector *output_vectors) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &input_vectors[i]));
    is_input_active[i] = input_vectors[i] == CEED_VECTOR_ACTIVE;
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &output_vectors[i]));
    is_output_active[i] = output_vectors[i] == CEED_VECTOR_ACTIVE;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get Field Metadata
//------------------------------------------------------------------------------
static inline int CeedOperatorGetFieldMetadata_Opt(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields,
                                                   CeedOperatorField *op_input_fields, CeedInt num_output_fields,
                                                   CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                                   CeedEvalMode *input_eval_modes, CeedEvalMode *output_eval_modes,
                                                   CeedInt *input_elem_sizes, CeedInt *input_field_sizes,
                                                   CeedInt *input_num_comps, CeedBasis *input_bases, CeedBasis *output_bases) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedElemRestriction elem_rstr;

    input_bases[i] = CEED_BASIS_NONE;
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &input_eval_modes[i]));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &input_field_sizes[i]));
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &input_elem_sizes[i]));
    CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
    switch (input_eval_modes[i]) {
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &input_bases[i]));
        CeedCallBackend(CeedBasisGetNumComponents(input_bases[i], &input_num_comps[i]));
        break;
      default:
        input_num_comps[i] = 0;
        break;
    }
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    output_bases[i] = CEED_BASIS_NONE;
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &output_eval_modes[i]));
    switch (output_eval_modes[i]) {
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &output_bases[i]));
        break;
      default:
        break;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Set operator block size from tensor contraction geometry
//------------------------------------------------------------------------------
static inline int CeedOperatorSetBlockSize_Opt(CeedInt num_input_fields, CeedBasis *input_bases,
                                               CeedInt num_output_fields, CeedBasis *output_bases, CeedInt *block_size) {
  if (*block_size != 16) return CEED_ERROR_SUCCESS;
  for (CeedInt i = 0; i < num_input_fields + num_output_fields; i++) {
    CeedBasis basis = i < num_input_fields ? input_bases[i] : output_bases[i - num_input_fields];
    bool      is_tensor;

    if (basis == CEED_BASIS_NONE) continue;
    CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
    if (is_tensor) {
      CeedInt P_1d, Q_1d;

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      if (P_1d * Q_1d >= 56) {
        *block_size = 8;
        break;
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedOperatorRestoreFieldBases_Opt(CeedInt num_input_fields, CeedBasis *input_bases, CeedInt num_output_fields,
                                                    CeedBasis *output_bases) {
  for (CeedInt i = 0; i < num_input_fields; i++) CeedCallBackend(CeedBasisDestroy(&input_bases[i]));
  for (CeedInt i = 0; i < num_output_fields; i++) CeedCallBackend(CeedBasisDestroy(&output_bases[i]));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Input Fields
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Opt(CeedInt num_input_fields, const bool *is_input_active,
                                              const CeedEvalMode *input_eval_modes, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                              CeedOperator_Opt *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    if (input_eval_modes[i] == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      if (!is_input_active[i]) {
        uint64_t state;

        CeedCallBackend(CeedVectorGetState(impl->input_vectors[i], &state));
        if (state != impl->input_states[i] && impl->block_rstr[i] && !impl->skip_rstr_in[i]) {
          CeedCallBackend(CeedElemRestrictionApply(impl->block_rstr[i], CEED_NOTRANSPOSE, impl->input_vectors[i], impl->e_vecs_full[i], request));
        }
        impl->input_states[i] = state;
        CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_full[i], CEED_MEM_HOST, (const CeedScalar **)&e_data[i]));
      } else if (input_eval_modes[i] == CEED_EVAL_NONE) {
        CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_in[i], CEED_MEM_HOST, (const CeedScalar **)&e_data[i]));
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, e_data[i]));
        CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_in[i], (const CeedScalar **)&e_data[i]));
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Opt(CeedInt e, CeedInt Q, CeedInt num_input_fields, const bool *is_input_active,
                                             const CeedEvalMode *input_eval_modes, const CeedInt *input_elem_sizes,
                                             const CeedInt *input_field_sizes, const CeedInt *input_num_comps, CeedBasis *input_bases,
                                             CeedInt block_size, bool skip_active,
                                             CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    if (skip_active && is_input_active[i]) continue;

    // Basis action
    switch (input_eval_modes[i]) {
      case CEED_EVAL_NONE:
        if (!is_input_active[i]) {
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER,
                                              &e_data[i][(CeedSize)e * Q * input_field_sizes[i]]));
        }
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        if (!is_input_active[i]) {
          CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER,
                                              &e_data[i][(CeedSize)e * input_elem_sizes[i] * input_num_comps[i]]));
        }
        CeedCallBackend(CeedBasisApply(input_bases[i], block_size, CEED_NOTRANSPOSE, input_eval_modes[i], impl->e_vecs_in[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Output Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputBasisField_Opt(CeedInt i, CeedInt block, CeedInt block_size, const bool *is_output_active,
                                                   const CeedEvalMode *output_eval_modes, CeedBasis *output_bases,
                                                   bool *apply_add_basis, bool *skip_rstr, CeedOperator op, CeedVector out_vec,
                                                   CeedOperator_Opt *impl, CeedRequest *request) {
  CeedVector vec;

  // Basis action
  switch (output_eval_modes[i]) {
    case CEED_EVAL_NONE:
      break;  // No action
    case CEED_EVAL_INTERP:
    case CEED_EVAL_GRAD:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      if (apply_add_basis[i]) {
        CeedCallBackend(CeedBasisApplyAdd(output_bases[i], block_size, CEED_TRANSPOSE, output_eval_modes[i], impl->q_vecs_out[i],
                                           impl->e_vecs_out[i]));
      } else {
        CeedCallBackend(CeedBasisApply(output_bases[i], block_size, CEED_TRANSPOSE, output_eval_modes[i], impl->q_vecs_out[i], impl->e_vecs_out[i]));
      }
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      // LCOV_EXCL_STOP
    }
  }
  // Restrict output block
  if (skip_rstr[i]) return CEED_ERROR_SUCCESS;
  vec = is_output_active[i] ? out_vec : impl->output_vectors[i];
  CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[i + impl->num_inputs], block, CEED_TRANSPOSE, impl->e_vecs_out[i], vec, request));
  return CEED_ERROR_SUCCESS;
}

static inline int CeedOperatorOutputBasis_Opt(CeedInt block, CeedInt block_size, CeedInt num_output_fields,
                                              const bool *is_output_active,
                                              const CeedEvalMode *output_eval_modes, CeedBasis *output_bases,
                                              bool *apply_add_basis, bool *skip_rstr, CeedOperator op, CeedVector out_vec,
                                              CeedOperator_Opt *impl, CeedRequest *request) {
  if (num_output_fields == 1) {
    return CeedOperatorOutputBasisField_Opt(0, block, block_size, is_output_active, output_eval_modes, output_bases, apply_add_basis, skip_rstr, op,
                                            out_vec, impl, request);
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedOperatorOutputBasisField_Opt(i, block, block_size, is_output_active, output_eval_modes, output_bases, apply_add_basis,
                                                     skip_rstr, op, out_vec, impl, request));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Opt(CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl) {
  for (CeedInt i = 0; i < impl->num_inputs; i++) {
    if (impl->input_eval_modes[i] != CEED_EVAL_WEIGHT && !impl->is_input_active[i]) {
      CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_full[i], (const CeedScalar **)&e_data[i]));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt            num_input_fields, num_output_fields;
  CeedScalar       *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunction     qf;
  CeedOperator_Opt *impl;

  // Setup
  CeedCallBackend(CeedOperatorSetup_Opt(op));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  const CeedInt block_size = impl->block_size;
  const CeedInt num_blocks = impl->num_blocks;
  const CeedInt Q          = impl->Q;

  // Restriction only operator
  if (impl->is_identity_rstr_op) {
    for (CeedInt b = 0; b < num_blocks; b++) {
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[0], b, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[0], request));
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[1], b, CEED_TRANSPOSE, impl->e_vecs_in[0], out_vec, request));
    }
    return CEED_ERROR_SUCCESS;
  }

  num_input_fields  = impl->num_inputs;
  num_output_fields = impl->num_outputs;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, impl->is_input_active, impl->input_eval_modes, e_data, impl, request));

  // Output Lvecs, Evecs, and Qvecs
  for (CeedInt i = 0; i < num_output_fields; i++) {
    // Set Qvec if needed
    if (impl->output_eval_modes[i] == CEED_EVAL_NONE) {
      // Set qvec to single block evec
      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_out[i], CEED_MEM_HOST, &e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs_out[i], &e_data[i + num_input_fields]));
    }
  }

  // Loop through elements
  for (CeedInt block = 0; block < num_blocks; block++) {
    const CeedInt e = block * block_size;

    // Active input restrictions
    if (impl->num_active_rstr_inputs == 1) {
      const CeedInt i = impl->active_rstr_input_indices[0];

      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[i], block, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[i], request));
    } else {
      for (CeedInt j = 0; j < impl->num_active_rstr_inputs; j++) {
        const CeedInt i = impl->active_rstr_input_indices[j];

        CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[i], block, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[i], request));
      }
    }

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, num_input_fields, impl->is_input_active, impl->input_eval_modes,
                                               impl->input_elem_sizes, impl->input_field_sizes, impl->input_num_comps, impl->input_bases,
                                               block_size, false, e_data, impl));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply and restriction
    CeedCallBackend(CeedOperatorOutputBasis_Opt(block, block_size, num_output_fields, impl->is_output_active, impl->output_eval_modes,
                                                impl->output_bases, impl->apply_add_basis_out,
                                                impl->skip_rstr_out, op, out_vec, impl, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(e_data, impl));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for linear QFunction assembly
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Opt(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  Ceed                ceed;
  CeedInt             qf_size_in, qf_size_out, Q, num_input_fields, num_output_fields, num_elem;
  CeedScalar         *l_vec_array, *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  CeedCallBackend(CeedOperatorSetup_Opt(op));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  qf_size_in  = impl->qf_size_in;
  qf_size_out = impl->qf_size_out;
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));

  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  const CeedInt       block_size = impl->block_size;
  const CeedInt       num_blocks = (num_elem / block_size) + !!(num_elem % block_size);
  CeedVector          l_vec      = impl->qf_l_vec;
  CeedElemRestriction block_rstr = impl->qf_block_rstr;

  // Check for restriction only operator
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, impl->is_input_active, impl->input_eval_modes, e_data, impl, request));

  // Count number of active input fields
  if (qf_size_in == 0) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      // Check if active input
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
        qf_size_in += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    CeedCheck(qf_size_in > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    impl->qf_size_in = qf_size_in;
  }

  // Count number of active output fields
  if (qf_size_out == 0) {
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      // Check if active output
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &field_size));
        qf_size_out += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    CeedCheck(qf_size_out > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    impl->qf_size_out = qf_size_out;
  }

  // Setup l_vec
  if (!l_vec) {
    const CeedSize l_size = (CeedSize)block_size * Q * qf_size_in * qf_size_out;

    CeedCallBackend(CeedVectorCreate(ceed, l_size, &l_vec));
    CeedCallBackend(CeedVectorSetValue(l_vec, 0.0));
    impl->qf_l_vec = l_vec;
  }

  // Output blocked restriction
  if (!block_rstr) {
    CeedInt strides[3] = {1, Q, qf_size_in * qf_size_out * Q};

    CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, Q, block_size, qf_size_in * qf_size_out,
                                                            qf_size_in * qf_size_out * num_elem * Q, strides, &block_rstr));
    impl->qf_block_rstr = block_rstr;
  }

  // Build objects if needed
  if (build_objects) {
    const CeedSize l_size     = (CeedSize)num_elem * Q * qf_size_in * qf_size_out;
    CeedInt        strides[3] = {1, Q, qf_size_in * qf_size_out * Q};

    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed, num_elem, Q, qf_size_in * qf_size_out,
                                                     (CeedSize)qf_size_in * (CeedSize)qf_size_out * (CeedSize)num_elem * (CeedSize)Q, strides, rstr));
    // Create assembled vector
    CeedCallBackend(CeedVectorCreate(ceed, l_size, assembled));
  }

  // Clear input QFunction buffers
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    // Clear if active input
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  // Clear output vector
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));

  // Loop through elements
  for (CeedInt block = 0; block < num_blocks; block++) {
    const CeedInt e = block * block_size;

    CeedCallBackend(CeedVectorGetArray(l_vec, CEED_MEM_HOST, &l_vec_array));

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, num_input_fields, impl->is_input_active, impl->input_eval_modes,
                                               impl->input_elem_sizes, impl->input_field_sizes, impl->input_num_comps, impl->input_bases,
                                               block_size, true, e_data, impl));

    // Assemble QFunction
    for (CeedInt i = 0; i < num_input_fields; i++) {
      bool       is_active;
      CeedInt    field_size;
      CeedVector vec;

      // Check if active input
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      is_active = vec == CEED_VECTOR_ACTIVE;
      CeedCallBackend(CeedVectorDestroy(&vec));
      if (!is_active) continue;
      CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
      for (CeedInt field = 0; field < field_size; field++) {
        // Set current portion of input to 1.0
        {
          CeedScalar *array;

          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 1.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
        }

        if (!impl->is_identity_qf) {
          // Set Outputs
          for (CeedInt out = 0; out < num_output_fields; out++) {
            CeedVector vec;

            // Check if active output
            CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
            if (vec == CEED_VECTOR_ACTIVE) {
              CeedInt field_size;

              CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, l_vec_array));
              CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &field_size));
              l_vec_array += field_size * Q * block_size;  // Advance the pointer by the size of the output
            }
            CeedCallBackend(CeedVectorDestroy(&vec));
          }
          // Apply QFunction
          CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
        } else {
          CeedInt           field_size;
          const CeedScalar *array;

          // Copy Identity Outputs
          CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &field_size));
          CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < field_size * Q * block_size; j++) l_vec_array[j] = array[j];
          CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &array));
          l_vec_array += field_size * Q * block_size;
        }
        // Reset input to 0.0
        {
          CeedScalar *array;

          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 0.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
        }
      }
    }

    // Un-set output Qvecs to prevent accidental overwrite of Assembled
    if (!impl->is_identity_qf) {
      for (CeedInt out = 0; out < num_output_fields; out++) {
        CeedVector vec;

        // Check if active output
        CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
        if (vec == CEED_VECTOR_ACTIVE && num_elem > 0) {
          CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
        }
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Assemble into assembled vector
    CeedCallBackend(CeedVectorRestoreArray(l_vec, &l_vec_array));
    CeedCallBackend(CeedElemRestrictionApplyBlock(block_rstr, block, CEED_TRANSPOSE, l_vec, *assembled, request));
  }

  // Reset output Qvecs
  for (CeedInt out = 0; out < num_output_fields; out++) {
    CeedVector vec;

    // Initialize array if active output
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_out[out], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(e_data, impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Opt(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Opt(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Opt(CeedOperator op) {
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  for (CeedInt i = 0; i < impl->num_inputs; i++) CeedCallBackend(CeedVectorDestroy(&impl->input_vectors[i]));
  for (CeedInt i = 0; i < impl->num_outputs; i++) CeedCallBackend(CeedVectorDestroy(&impl->output_vectors[i]));
  CeedCallBackend(CeedFree(&impl->input_vectors));
  CeedCallBackend(CeedFree(&impl->output_vectors));
  CeedCallBackend(CeedOperatorRestoreFieldBases_Opt(impl->num_inputs, impl->input_bases, impl->num_outputs, impl->output_bases));
  CeedCallBackend(CeedFree(&impl->input_bases));
  CeedCallBackend(CeedFree(&impl->output_bases));
  CeedCallBackend(CeedFree(&impl->is_input_active));
  CeedCallBackend(CeedFree(&impl->is_output_active));
  CeedCallBackend(CeedFree(&impl->input_eval_modes));
  CeedCallBackend(CeedFree(&impl->output_eval_modes));
  CeedCallBackend(CeedFree(&impl->active_rstr_input_indices));
  CeedCallBackend(CeedFree(&impl->input_elem_sizes));
  CeedCallBackend(CeedFree(&impl->input_field_sizes));
  CeedCallBackend(CeedFree(&impl->input_num_comps));
  for (CeedInt i = 0; i < impl->num_inputs + impl->num_outputs; i++) {
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->block_rstr[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_full[i]));
  }
  CeedCallBackend(CeedFree(&impl->block_rstr));
  CeedCallBackend(CeedFree(&impl->e_vecs_full));
  CeedCallBackend(CeedFree(&impl->input_states));
  CeedCallBackend(CeedFree(&impl->skip_rstr_in));
  CeedCallBackend(CeedFree(&impl->skip_rstr_out));
  CeedCallBackend(CeedFree(&impl->apply_add_basis_out));

  for (CeedInt i = 0; i < impl->num_inputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_in[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs_in));
  CeedCallBackend(CeedFree(&impl->q_vecs_in));

  for (CeedInt i = 0; i < impl->num_outputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_out[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_out[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs_out));
  CeedCallBackend(CeedFree(&impl->q_vecs_out));

  // QFunction assembly data
  CeedCallBackend(CeedVectorDestroy(&impl->qf_l_vec));
  CeedCallBackend(CeedElemRestrictionDestroy(&impl->qf_block_rstr));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Opt(CeedOperator op) {
  Ceed              ceed;
  Ceed_Opt         *ceed_impl;
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  const CeedInt block_size = ceed_impl->block_size;

  CeedCallBackend(CeedCalloc(1, &impl));
  impl->block_size = block_size;
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCheck(block_size == 1 || block_size == 8 || block_size == 16, ceed, CEED_ERROR_BACKEND,
            "Opt backend cannot use blocksize: %" CeedInt_FMT, block_size);

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Opt));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
