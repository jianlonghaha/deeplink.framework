#include <cstring>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>
#include <csrc_dipu/vendor/musa/vendorapi.h>


/*** MCCL CAPABILITY CHECK ***/

// from torch/csrc/musa/mccl.h
#if defined(__MUSA_BF16_TYPES_EXIST__)
#define HAS_MCCL_BF16_DATATYPE \
  ((MCCL_MAJOR > 2) || (MCCL_MAJOR == 2) && (MCCL_MINOR >= 10))
#elif defined(USE_ROCM) && (TORCH_HIP_VERSION >= 301)
#define HAS_MCCL_BF16_DATATYPE 1
#else
#define HAS_MCCL_BF16_DATATYPE 0
#endif

// from torch/csrc/distributed/c10d/ProcessGroupMCCL.cpp
#if defined(MCCL_MAJOR) && \
    ((MCCL_MAJOR > 2) || (MCCL_MAJOR == 2) && (MCCL_MINOR >= 10))
#define MCCL_HAS_AVG 1
#endif

/*** MCCL CAPABILITY CHECK END ***/

namespace dipu {

namespace devapis {

// MCCL op mapping
static const std::map<ReduceOp::RedOpType, mcclRedOp_t> mcclOp = {
    {ReduceOp::MIN, mcclMin}, {ReduceOp::MAX, mcclMax},
    {ReduceOp::SUM, mcclSum}, {ReduceOp::PRODUCT, mcclProd},
#ifdef MCCL_HAS_AVG
    {ReduceOp::AVG, mcclAvg},
#endif
};

// MCCL type typing
static const std::map<at::ScalarType, mcclDataType_t> mcclDataType = {
    {at::kChar, mcclInt8},         {at::kByte, mcclUint8},
    {at::kFloat, mcclFloat},       {at::kDouble, mcclDouble},
    {at::kInt, mcclInt32},         {at::kLong, mcclInt64},
    {at::kHalf, mcclHalf},         {at::kBool, mcclUint8},
#if HAS_MCCL_BF16_DATATYPE
    {at::kBFloat16, mcclFloat16},
#endif
};

// Macro to print and abort on a non-successful MCCL return value.
#define MCCL_THROW(cmd)                                                 \
  do {                                                                  \
    mcclResult_t result = cmd;                                          \
    if (result != mcclSuccess) {                                        \
      std::string err = mcclGetErrorString(result);                     \
      fprintf(stderr, "MCCL error in: %s:%d, %s\n", __FILE__, __LINE__, \
              err.c_str());                                             \
      TORCH_CHECK(false, err);                                          \
    }                                                                   \
  } while (0)

const int DICL_UNIQUE_ID_BYTES_SIZE = MCCL_UNIQUE_ID_BYTES;

DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  mcclResult_t mcclAsyncErr_;
  MCCL_THROW(mcclCommGetAsyncError(comm, &mcclAsyncErr_));
  if (mcclAsyncErr_ != mcclSuccess) {
    return DICL_SUCCESS;
  }
  return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  MCCL_THROW(mcclGetUniqueId(uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
                                       commUniqueId uniqueId, int rank,
                                       int localDeviceId) {
  MCCL_THROW(mcclCommInitRank(comm, nranks, uniqueId, rank));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(mcclComm_t comm) {
  MCCL_THROW(mcclCommDestroy(comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllReduce(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    const ReduceOp& reduceOp, diclComm_t comm,
                                    deviceStream_t stream) {
  // TODO(wanglei): add .find() != .end() check.
  MCCL_THROW(mcclAllReduce(sendbuff, recvbuff, count, mcclDataType.at(datatype),
                           mcclOp.at(reduceOp), comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclBroadcast(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    int root, diclComm_t comm,
                                    deviceStream_t stream) {
  MCCL_THROW(mcclBroadcast(sendbuff, recvbuff, count, mcclDataType.at(datatype),
                           root, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void* sendBuf, void* recvBuf,
                                    size_t sendCount, at::ScalarType datatype,
                                    diclComm_t comm, deviceStream_t stream) {
  MCCL_THROW(mcclAllGather(sendBuf, recvBuf, sendCount,
                           mcclDataType.at(datatype), comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff,
                                 size_t count, at::ScalarType datatype,
                                 const ReduceOp& reduceOp, int root,
                                 diclComm_t comm, deviceStream_t stream) {
  MCCL_THROW(mcclReduce(sendbuff, recvbuff, count, mcclDataType.at(datatype),
                        mcclOp.at(reduceOp), root, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduceScatter(
    void* sendBuf, void* recvBuf, size_t recvCount, at::ScalarType datatype,
    const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
  MCCL_THROW(mcclReduceScatter(sendBuf, recvBuf, recvCount,
                               mcclDataType.at(datatype), mcclOp.at(reduceOp),
                               comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclSend(const void* sendbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  MCCL_THROW(
      mcclSend(sendbuff, count, mcclDataType.at(datatype), peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  MCCL_THROW(
      mcclRecv(recvbuff, count, mcclDataType.at(datatype), peer, comm, stream));
  return DICL_SUCCESS;
}

}  // end namespace devapis
}  // end namespace dipu
