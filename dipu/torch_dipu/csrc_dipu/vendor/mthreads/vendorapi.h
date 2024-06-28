// Copyright (c) 2023, DeepLink.

#pragma once
#include <musa.h>
#include <musa_runtime.h>
#include <mccl.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>

namespace dipu {

#define DIPU_CALLMUSA(Expr)                                              \
  {                                                                      \
    musaError_t ret = Expr;                                              \
    TORCH_CHECK(ret == ::musaSuccess, "call musa error, expr = ", #Expr, \
                ", ret = ", ret);                                        \
  }

using deviceStream_t = musaStream_t;
#define deviceDefaultStreamLiteral musaStreamLegacy
using deviceEvent_t = musaEvent_t;

using diclComm_t = mcclComm_t;
using commUniqueId = mcclUniqueId;

}  // namespace dipu
