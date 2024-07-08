// Copyright (c) 2023, DeepLink.
#pragma once

// todo:: dev api will remove pytorch dependency
#include <cstdint>

#include <c10/core/Device.h>

// todo: move out deice dir to diopi
namespace dipu {

#define DIPU_API __attribute__((visibility("default")))

#define DIPU_WEAK __attribute__((weak))

// "default", "hidden", "protected" or "internal
#define DIPU_HIDDEN __attribute__((visibility("hidden")))

using enum_t = int32_t;

#define DIPU_STRING(x) #x
#define DIPU_STRINGIFY_AFTER_EXPANSION(x) DIPU_STRING(x)
#define DIPU_CODELOC __FILE__ " (" DIPU_STRING(__LINE__) ")"

#define DIPU_LOGE(fmt, ...)                                              \
  printf("[ERROR]%s:%u:%s: " fmt "\n", __FILE__, __LINE__, __FUNCTION__, \
         ##__VA_ARGS__)

#define DIPU_LOGW(fmt, ...)                                             \
  printf("[WARN]%s:%u:%s: " fmt "\n", __FILE__, __LINE__, __FUNCTION__, \
         ##__VA_ARGS__)

namespace devapis {

enum class VendorDeviceType : enum_t {
  CUDA,     // cuda
  MLU,      // camb
  NPU,      // ascend
  MUXI,     // muxi
  GCU,      // gcu, topsrider
  DROPLET,  // droplet
  SUPA,     // Biren
  KLX,      // Kunlunxin
  MUSA,     // musa
};

enum class EventStatus : enum_t { PENDING, RUNNING, DEFERRED, READY };

enum class OpStatus : enum_t {
  SUCCESS,
  ERR_UNKNOWN,
  ERR_NOMEM,
};

enum class MemCPKind : enum_t {
  D2H,
  H2D,
  D2D,
};

enum diclResult_t {
  /*! The operation was successful. */
  DICL_SUCCESS = 0x0,

  /*! undefined error */
  DICL_ERR_UNDEF = 0x01000,

};

struct DIPUDeviceStatus {
  size_t freeGlobalMem = 0;
  size_t totalGlobalMem = 0;
};

struct DIPUDeviceProperties {
  std::string name;
  size_t totalGlobalMem = 0;
  int32_t major = 0;
  int32_t minor = 0;
  int32_t multiProcessorCount = 0;
};

using deviceId_t = c10::DeviceIndex;

}  // end namespace devapis
}  // end namespace dipu
