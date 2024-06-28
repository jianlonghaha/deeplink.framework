// Copyright (c) 2023, DeepLink.
#include <musa_runtime_api.h>

#include <c10/util/Exception.h>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

#include <csrc_dipu/vendor/mt/vendorapi.h>


namespace dipu {

namespace devapis {

using musa_deviceId = int;
// =====================
//  Device class related
// =====================

void initializeVendor() {}

void finalizeVendor() {}

deviceId_t current_device() {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  musa_deviceId devId_;
  DIPU_CALLMUSA(::musaGetDevice(&devId_))
  return static_cast<deviceId_t>(devId_);
}

DIPUDeviceProperties getDeviceProperties(int32_t device_index) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ::musaDeviceProp device_prop;
  DIPU_CALLMUSA(musaGetDeviceProperties(&device_prop, device_index))

  DIPUDeviceProperties prop;
  prop.name = device_prop.name;
  prop.totalGlobalMem = device_prop.totalGlobalMem;
  prop.major = device_prop.major;
  prop.minor = device_prop.minor;
  prop.multiProcessorCount = device_prop.multiProcessorCount;
  return prop;
}

DIPUDeviceStatus getDeviceStatus(int32_t device_index) {
  DIPUDeviceStatus status;
  DIPU_CALLMUSA(musaMemGetInfo(&status.freeGlobalMem, &status.totalGlobalMem));
  return status;
}

// in musa_runtime_api.h
// set current device given device according to id
void setDevice(deviceId_t devId) {
  musa_deviceId devId_ = static_cast<deviceId_t>(devId);
  DIPU_CALLMUSA(::musaSetDevice(devId_))
}

void resetDevice(deviceId_t devId) { DIPU_CALLMUSA(::musaDeviceReset()) }

void syncDevice() { DIPU_CALLMUSA(::musaDeviceSynchronize()) }

// check last launch succ or not, throw if fail
void checkLastError() { DIPU_CALLMUSA(::musaGetLastError()) }

int getDeviceCount() {
  int num = -1;
  DIPU_CALLMUSA(::musaGetDeviceCount(&num))
  return num;
}

void getDriverVersion(int* version) {
  DIPU_CALLMUSA(::musaDriverGetVersion(version))
}

void getRuntimeVersion(int* version) {
  DIPU_CALLMUSA(::musaRuntimeGetVersion(version))
}

// =====================
//  device stream related
// =====================
void createStream(deviceStream_t* stream, bool prior) {
  if (prior) {
    DIPU_CALLMUSA(::musaStreamCreateWithPriority(stream, musaStreamDefault, -1))
  } else {
    DIPU_CALLMUSA(::musaStreamCreate(stream))
  }
}

void destroyStream(deviceStream_t stream) {
  DIPU_CALLMUSA(::musaStreamDestroy(stream))
}

void destroyStream(deviceStream_t stream, deviceId_t devId) {
  setDevice(devId);
  destroyStream(stream);
}

void releaseStream() {}

bool streamNotNull(deviceStream_t stream) {
  return (stream != nullptr && stream != musaStreamLegacy &&
          stream != musaStreamPerThread);
}

void syncStream(deviceStream_t stream) {
  DIPU_CALLMUSA(::musaStreamSynchronize(stream));
}

void streamWaitEvent(deviceStream_t stream, deviceEvent_t event) {
  DIPU_CALLMUSA(::musaStreamWaitEvent(stream, event, 0))
}

bool isStreamEmpty(deviceStream_t stream) {
  auto err = musaStreamQuery(stream);
  return err == ::musaSuccess;
}

// =====================
//  device event related
// =====================

void createEvent(deviceEvent_t* event) {
  static bool enableTiming = []() {
    const char* env = std::getenv("DIPU_MUSA_EVENT_TIMING");
    if (env) {
      return std::atoi(env) > 0;
    }
    return true;
  }();

  DIPU_CALLMUSA(::musaEventCreateWithFlags(
      event, enableTiming ? musaEventDefault : musaEventDisableTiming))
}

void destroyEvent(deviceEvent_t event) {
  DIPU_CALLMUSA(::musaEventDestroy(event))
}

void waitEvent(deviceEvent_t event) {
  DIPU_CALLMUSA(::musaEventSynchronize(event))
}

void recordEvent(deviceEvent_t event, deviceStream_t stream) {
  DIPU_CALLMUSA(::musaEventRecord(event, stream))
}

void eventElapsedTime(float* time, deviceEvent_t start, deviceEvent_t end){
    DIPU_CALLMUSA(musaEventElapsedTime(time, start, end))}

EventStatus getEventStatus(deviceEvent_t event) {
  ::musaError_t ret = ::musaEventQuery(event);
  if (ret == ::musaSuccess) {
    return devapis::EventStatus::READY;
  }
  if (ret == ::musaErrorNotReady) {
    ::musaGetLastError(); /* reset internal error state*/
    return devapis::EventStatus::PENDING;
  }
  TORCH_CHECK(false, "unexpected event status in getEventStatus, ret = ", ret);
}

// =====================
//  mem related
// =====================
void mallocHost(void** p, size_t nbytes) {
  DIPU_CALLMUSA(::musaMallocHost(p, nbytes))
}

void freeHost(void* p){DIPU_CALLMUSA(::musaFreeHost(p))}

OpStatus mallocDevice(void** p, size_t nbytes, bool throwExcepion) {
  ::musaError_t r = ::musaMalloc(p, nbytes);
  if (r == ::musaSuccess) {
    return OpStatus::SUCCESS;
  }
  if (r == ::musaErrorMemoryAllocation && !throwExcepion) {
    ::musaGetLastError(); /* reset internal error state*/
    return OpStatus::ERR_NOMEM;
  }
  TORCH_CHECK(false, "musaMalloc failed, ret = ", r, " size = ", nbytes);
}

void freeDevice(void* p) { DIPU_CALLMUSA(::musaFree(p)) }

bool isPinnedPtr(const void* p) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ::musaPointerAttributes attr;
  DIPU_CALLMUSA(::musaPointerGetAttributes(&attr, p))
  return attr.type == musaMemoryTypeHost;
}

void memSetAsync(const deviceStream_t stream, void* ptr, int val, size_t size) {
  DIPU_CALLMUSA(::musaMemsetAsync(ptr, val, size, stream))
}

void memCopyD2D(size_t nbytes, deviceId_t dstDevId, void* dst,
                deviceId_t srcDevId, const void* src) {
  if (dstDevId == srcDevId) {
    DIPU_CALLMUSA(::musaMemcpy(dst, src, nbytes, ::musaMemcpyDeviceToDevice))
  } else {
    DIPU_CALLMUSA(::musaMemcpyPeer(dst, dstDevId, src, srcDevId, nbytes))
  }
}

// (synchronous) copy from host to a MUSA device
void memCopyH2D(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLMUSA(::musaMemcpy(dst, src, nbytes, ::musaMemcpyHostToDevice))
}

// (synchronous) copy from a MUSA device to host
void memCopyD2H(size_t nbytes, void* dst, const void* src) {
  DIPU_CALLMUSA(::musaMemcpy(dst, src, nbytes, ::musaMemcpyDeviceToHost))
}

// (asynchronous) copy from device to a device
void memCopyD2DAsync(const deviceStream_t stream, size_t nbytes,
                     deviceId_t dstDevId, void* dst, deviceId_t srcDevId,
                     const void* src) {
  if (dstDevId == srcDevId) {
    DIPU_CALLMUSA(
        ::musaMemcpyAsync(dst, src, nbytes, musaMemcpyDeviceToDevice, stream))
  } else {
    DIPU_CALLMUSA(
        ::musaMemcpyPeerAsync(dst, dstDevId, src, srcDevId, nbytes, stream))
  }
}

// (asynchronous) copy from host to a device
void memCopyH2DAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  DIPU_CALLMUSA(
      ::musaMemcpyAsync(dst, src, nbytes, musaMemcpyHostToDevice, stream))
}

// (asynchronous) copy from a device to host
void memCopyD2HAsync(const deviceStream_t stream, size_t nbytes, void* dst,
                     const void* src) {
  DIPU_CALLMUSA(
      ::musaMemcpyAsync(dst, src, nbytes, musaMemcpyDeviceToHost, stream));
}

}  // end namespace devapis

}  // namespace dipu
