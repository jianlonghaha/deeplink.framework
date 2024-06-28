include(FindPackageHandleStandardArgs)



# find_package(MUSA "14.0.0" REQUIRED)

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# find_package(XCCL REQUIRED)

# if (CUDA_FOUND)
#     message(STATUS "MUSA Libraries: " ${MUSA_LIBRARIES})
#     message(STATUS "MUSA include: " ${MUSA_INCLUDE_DIRS})
#     message(STATUS "MCC Path: ${MUSA_TOOLKIT_ROOT_DIR}/bin")
#     message(STATUS "MCCL include Path:  ${MCCL_INCLUDE_DIR}")
#     message(STATUS "MCCL lib Path:  ${MCCL_LIBRARIES}")
# endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
# add_executable(myapp test.cpp)

if(NOT DEFINED ENV{MUSA_PATH})
    set(MUSA_PATH /usr/local/musa)
else()
    set(MUSA_PATH $ENV{MUSA_PATH})
endif()

# include
find_path(MUSA_INCLUDE_DIR NAMES musa.h musa_runtime_api.h musa_runtime.h PATHS ${MUSA_PATH}/include)
MESSAGE("MUSA_INCLUDE_DIR:" ${MUSA_INCLUDE_DIR})
IF (NOT MUSA_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR " Could not find musa.h. Make sure fullstack is installed, or MUSA_PATH is set properly.")
ENDIF()

# libmccl.so
find_library(MUSA_MCCL_LIBRARY mccl
  ${MUSA_PATH}/lib 
)
MESSAGE("MUSA_MCCL_LIBRARY:" ${MUSA_MCCL_LIBRARY})
IF (NOT MUSA_MCCL_LIBRARY)
  MESSAGE(FATAL_ERROR " Could not find musa mccl lib. Make sure fullstack is installed, or MUSA_PATH is set properly.")
ENDIF()

# libmusart.so
find_library(MUSART_LIBRARY musart
  ${MUSA_PATH}/lib 
)
MESSAGE("MUSART_LIBRARY:" ${MUSART_LIBRARY})
IF (NOT MUSART_LIBRARY)
  MESSAGE(FATAL_ERROR " Could not find musa runtime lib. Make sure fullstack is installed, or MUSA_PATH is set properly.")
ENDIF()

# 添加头文件搜索路径
# target_include_directories(myapp PRIVATE ${MUSA_PATH}/include)
# include_directories(${MUSA_INCLUDE_DIR})

# 链接库文件
# target_link_libraries(myapp PRIVATE ${MUSA_MCCL_LIBRARY})
# target_link_libraries(myapp PRIVATE ${MUSART_LIBRARY})


find_package_handle_standard_args(MUSA_INCLUDE_DIR DEFAULT_MSG
    MUSA_INCLUDE_DIR
    )
find_package_handle_standard_args(MUSART_LIBRARY DEFAULT_MSG
    MUSART_LIBRARY
    )
find_package_handle_standard_args(MUSA_MCCL_LIBRARY DEFAULT_MSG
    MUSA_MCCL_LIBRARY
    )
mark_as_advanced(MUSA_MCCL_LIBRARY MUSART_LIBRARY MUSA_INCLUDE_DIR   )

# it's incorrect, need enhance find cmake to set lib fullpath
# set(VENDOR_INCLUDE_DIRS "${ASCEND_TOOLKIT_ROOT}/include" PARENT_SCOPE)
# set(VENDOR_LIB_DIRS "${ASCEND_TOOLKIT_ROOT}/lib64" PARENT_SCOPE)
# set(DIPU_VENDOR_LIB ascendcl acl_op_compiler PARENT_SCOPE)
file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)
set(VENDOR_INCLUDE_DIRS ${MUSA_PATH}/include PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${MUSA_PATH}/lib PARENT_SCOPE)
set(DIPU_VENDOR_LIB ${MUSA_MCCL_LIBRARY}  ${MUSART_LIBRARY} PARENT_SCOPE)
