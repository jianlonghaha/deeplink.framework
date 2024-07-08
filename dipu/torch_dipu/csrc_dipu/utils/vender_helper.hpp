// Copyright (c) 2023, DeepLink.
#pragma once

#include <stdexcept>

#include "csrc_dipu/runtime/device/basedef.h"

namespace dipu {

// TODO(lilingjie,fandaoyi): use constexpr after 910b CI is ready
// throw in constexpr funtions (c++14) is not supported in gcc-7.5, which is a
// known bug already fixed later (at least in gcc-10.x).
inline const char* VendorDeviceTypeToStr(devapis::VendorDeviceType t) {
  switch (t) {
    case devapis::VendorDeviceType::CUDA:
      return "CUDA";
    case devapis::VendorDeviceType::MLU:
      return "MLU";
    case devapis::VendorDeviceType::NPU:
      return "NPU";
    case devapis::VendorDeviceType::MUXI:
      return "MUXI";
    case devapis::VendorDeviceType::GCU:
      return "GCU";
    case devapis::VendorDeviceType::DROPLET:
      return "DROPLET";
    case devapis::VendorDeviceType::SUPA:
      return "SUPA";
    case devapis::VendorDeviceType::KLX:
      return "KLX";
    case devapis::VendorDeviceType::MUSA:
      return "MUSA";
    default:
      throw std::invalid_argument("Unknown device type");
  }
}

// constexpr version of C-style string comparison
constexpr bool c_string_equal(const char* a, const char* b) noexcept {
  return *a == *b && (*a == '\0' || c_string_equal(a + 1, b + 1));
}

// TODO(lilingjie,fandaoyi): use constexpr after 910b CI is ready
// throw in constexpr funtions (c++14) is not supported in gcc-7.5, which is a
// known bug already fixed later (at least in gcc-10.x).
inline devapis::VendorDeviceType VendorNameToDeviceType(const char* str) {
#define DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(name, type) \
  if (c_string_equal(str, #name)) {                          \
    return devapis::VendorDeviceType::type;                  \
  }
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(camb, MLU);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(cuda, CUDA);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(ascend, NPU);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(muxi, MUXI);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(topsrider, GCU);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(droplet, DROPLET);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(supa, SUPA);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(kunlunxin, KLX);
  DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE(musa, MUSA);


#undef DIPU_MAY_CAST_VENDOR_NAME_TO_DEVICE_TYPE
  throw std::invalid_argument("Unknown device name");
}

}  // namespace dipu
