#pragma once

#ifndef __linux__
    #include <intrin.h>
#endif

bool is_avx2_supported();
bool is_f16c_supported();

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    #ifdef __linux__
        #define AVX2_TARGET __attribute__((target("avx2")))
        #define AVX2_F16C_TARGET __attribute__((target("avx2,f16c")))
        #define AVX2_TARGET_OPTIONAL __attribute__((target_clones("avx2","default")))
    #else
        #define AVX2_TARGET
        #define AVX2_F16C_TARGET
        #define AVX2_TARGET_OPTIONAL
    #endif
#else
    // Non-x86 (ARM): no AVX2 attributes
    #define AVX2_TARGET
    #define AVX2_F16C_TARGET
    #define AVX2_TARGET_OPTIONAL
#endif