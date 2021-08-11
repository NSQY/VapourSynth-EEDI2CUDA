cmake_minimum_required(VERSION 3.20)
project(VapourSynth-EEDI2CUDA CUDA)

include(FetchContent)
FetchContent_Declare(VapourSynth
    URL             https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R54-API4-test2.zip
    URL_HASH    MD5=27606869F37C63EC7B6E10F58F6F0BB7)
FetchContent_MakeAvailable(VapourSynth)

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

find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --long
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE VCS_TAG)
string(STRIP ${VCS_TAG} VCS_TAG)
get_property(BUILD_OPTIONS TARGET EEDI2CUDA PROPERTY COMPILE_OPTIONS)
string(TIMESTAMP CONFIGURE_TIME)
configure_file(config.h.in config.h)
