cmake_minimum_required(VERSION 3.20)
project(VapourSynth-EEDI2CUDA CUDA)

include(FetchContent)
FetchContent_Declare(VapourSynth
    URL             https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R54.zip
    URL_HASH    MD5=B869FFB96D423B086C0E19E2210229DF)
FetchContent_MakeAvailable(VapourSynth)
FetchContent_Declare(AviSynthPlus
    URL             https://github.com/AviSynth/AviSynthPlus/archive/refs/tags/v3.7.0.zip
    URL_HASH    MD5=45239C3AB7750730DAB672175B47D357)
FetchContent_GetProperties(AviSynthPlus)
if(NOT AviSynthPlus_POPULATED)
    FetchContent_Populate(AviSynthPlus)
    add_subdirectory(${avisynthplus_SOURCE_DIR} EXCLUDE_FROM_ALL)
endif()

set(CMAKE_CUDA_STANDARD 17 REQUIRED)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
add_library(EEDI2CUDA SHARED
    EEDI2.cu)
target_include_directories(EEDI2CUDA PRIVATE
    "${PROJECT_BINARY_DIR}"
    "${vapoursynth_SOURCE_DIR}/include"
    ${Boost_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(EEDI2CUDA PRIVATE
    --ptxas-options=-v
    --restrict
    --use_fast_math
    --Wdefault-stream-launch
    --Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr
    --no-host-device-initializer-list
    --display-error-number
    --diag-suppress 177,186)
if(MSVC)
    target_compile_options(EEDI2CUDA PRIVATE "-Xcompiler=/W4 /wd4297 /wd4458")
endif()
set_target_properties(EEDI2CUDA PROPERTIES CUDA_ARCHITECTURES "50-virtual")

add_library(EEDI2CUDA_AVS SHARED
    EEDI2_avs.cu)
target_include_directories(EEDI2CUDA_AVS PRIVATE
    "${PROJECT_BINARY_DIR}"
    "${avisynthplus_SOURCE_DIR}/avs_core/include"
    ${Boost_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(EEDI2CUDA_AVS PRIVATE
    --ptxas-options=-v
    --restrict
    --use_fast_math
    --Wdefault-stream-launch
    --Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr
    --no-host-device-initializer-list
    --display-error-number
    --diag-suppress 177,186)
if(MSVC)
    target_compile_options(EEDI2CUDA_AVS PRIVATE "-Xcompiler=/W4 /wd4297 /wd4458")
endif()
set_target_properties(EEDI2CUDA_AVS PROPERTIES CUDA_ARCHITECTURES "50-virtual")

find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --long
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE VCS_TAG)
string(STRIP ${VCS_TAG} VCS_TAG)
get_property(BUILD_OPTIONS TARGET EEDI2CUDA PROPERTY COMPILE_OPTIONS)
string(TIMESTAMP CONFIGURE_TIME)
configure_file(config.h.in config.h)
