#pragma once

#include <cuda_runtime.h>


namespace zhao {


// ==========================================
// 1. 全局配置常量 (Type-safe Constants)
// ==========================================

constexpr int WARP_SIZE = 32;

// Tensor Core Tile Configuration
constexpr int tile_M = 16;
constexpr int tile_K = 8;
constexpr int tile_N = 8;


}   // namespace zhao


// ==========================================
// 2. 辅助宏 (Macros)
// 注意：宏由预处理器处理，无法放入命名空间，但在此文件中定义是安全的。
// ==========================================

// 指针强转辅助
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])

// 类型转换辅助指令
#define cvt_F32_to_TF32(r, addr) asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(r) : "f"(addr))


// ==========================================
// 3. Tensor Core MMA 指令封装 (Inline ASM)
// ==========================================

// 16x8x8 矩阵乘法 (16x8x8)
#define HMMA1688(RC0, RC1, RC2, RC3, RA0, RA1, RA2, RA3, RB0, RB1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
                 : "=f"(RC0), "=f"(RC1), "=f"(RC2), "=f"(RC3)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))

// 16x8x4 矩阵乘法 (16x8x4)
#define HMMA1684(RC0, RC1, RC2, RC3, RA0, RA1, RB0)                                                    \
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %9, %9, %10};\n" \
                 : "=f"(RC0), "=f"(RC1), "=f"(RC2), "=f"(RC3)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RB0), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))

// 16x8x8 矩阵乘法，有 Accumulator Registers (A, B, C) (16x8x8)
#define HMMA1688_D(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                                                    \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
                 : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                                                                               \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))
