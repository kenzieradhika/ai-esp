// Host unit tests for the llm.h numeric kernels, independent of any model file:
// half2float, deq_row (int4 unpack + group scales), and matvec_q vs matvec_q8.
// These are the pieces the whole on-device runtime is built from, so they get a
// check that runs without a 15MB model.bin.
//
// Build: cc -O3 -Wall -Wextra -Werror -DLLM_INT8_ACT -o t firmware/host_verify/test_kernels.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../common/llm.h"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
  fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

// fp16 bit patterns with exact known values.
static void test_half2float(void) {
  CHECK(half2float(0x0000) == 0.0f, "0x0000 -> %g", half2float(0x0000));
  CHECK(half2float(0x3C00) == 1.0f, "0x3C00 -> %g", half2float(0x3C00));
  CHECK(half2float(0xC000) == -2.0f, "0xC000 -> %g", half2float(0xC000));
  CHECK(half2float(0x3800) == 0.5f, "0x3800 -> %g", half2float(0x3800));
  CHECK(half2float(0x7C00) == INFINITY, "0x7C00 -> %g", half2float(0x7C00));
  CHECK(half2float(0x8000) == 0.0f, "0x8000 -> -0 (==0)");
}

// Build a one-row, one-group int4 tensor with codes v[0..cols-1] in [-8,7] and
// scale 1.0, then check deq_row and both matvec paths.
static void test_deq_and_matvec(void) {
  const int cols = 8;
  int8_t vals[8] = {1, -1, 2, -2, 3, -3, 0, 7};
  uint8_t codes[4];
  for (int i = 0; i < 8; i += 2) {
    uint8_t lo = (uint8_t)(vals[i] + 8) & 0xF;
    uint8_t hi = (uint8_t)(vals[i + 1] + 8) & 0xF;
    codes[i >> 1] = (uint8_t)(lo | (hi << 4));
  }
  uint16_t scales[1] = {0x3C00};  // 1.0 in fp16
  QT t = {codes, scales, 1, cols, 8, 1, 4};

  float row[8];
  deq_row(&t, 0, row);
  for (int i = 0; i < 8; i++)
    CHECK(row[i] == (float)vals[i], "deq_row[%d] = %g, want %d", i, row[i], vals[i]);

  float x[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  float yq[1];
  matvec_q(&t, x, yq);
  float want = 0; for (int i = 0; i < 8; i++) want += (float)vals[i];
  CHECK(fabsf(yq[0] - want) < 1e-5f, "matvec_q = %g, want %g", yq[0], want);
#ifdef LLM_INT8_ACT
  // int8 activation quantization of an all-ones vector is exact here.
  float y8[1];
  matvec_q8(&t, x, y8);
  CHECK(fabsf(y8[0] - want) < 1e-4f, "matvec_q8 = %g, want %g", y8[0], want);
#endif
}

// Ragged width (odd cols) must unpack without reading the padded nibble.
static void test_ragged_odd_cols(void) {
  const int cols = 3;
  uint8_t codes[2] = {0, 0};
  // vals = {1,2,3}: lo nibble=1+8, then 2+8 | (3+8)<<4
  codes[0] = (uint8_t)((1 + 8) | ((2 + 8) << 4));
  codes[1] = (uint8_t)((3 + 8) & 0xF);
  uint16_t scales[1] = {0x3C00};
  QT t = {codes, scales, 1, cols, 3, 1, 2};
  float row[3];
  deq_row(&t, 0, row);
  CHECK(row[0] == 1 && row[1] == 2 && row[2] == 3,
        "ragged deq = %g %g %g", row[0], row[1], row[2]);
}

// A ranged matvec (what the dual-core head splits on) must equal the full one.
static void test_matvec_range_matches_full(void) {
  const int rows = 6, cols = 8;
  uint8_t codes[6 * 4];
  for (int r = 0; r < rows; r++)
    for (int i = 0; i < cols; i += 2) {
      int a = ((r + i) % 15) - 7, b = ((r * 2 + i) % 15) - 7;
      codes[r * 4 + (i >> 1)] = (uint8_t)(((a + 8) & 0xF) | (((b + 8) & 0xF) << 4));
    }
  uint16_t scales[6] = {0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00};
  QT t = {codes, scales, rows, cols, 8, 1, 4};
  float x[8] = {0.5f, -1.0f, 2.0f, 0.25f, -0.75f, 1.5f, -2.0f, 0.1f};

  float full[6], ranged[6];
  matvec_q(&t, x, full);
  matvec_q_range(&t, x, ranged, 0, 3);
  matvec_q_range(&t, x, ranged, 3, rows);
  for (int r = 0; r < rows; r++)
    CHECK(fabsf(full[r] - ranged[r]) < 1e-5f,
          "row %d: full=%g ranged=%g", r, full[r], ranged[r]);
}

// rmsnorm/gelu/silu: identities the C runtime relies on to match src/model.py.
static void test_norm_activations(void) {
  float x[4] = {1.f, -2.f, 3.f, -4.f};
  float w[4] = {1.f, 1.f, 1.f, 1.f};
  float out[4];
  rmsnorm(x, w, 4, out);
  float ss = 0.f; for (int i = 0; i < 4; i++) ss += x[i] * x[i];
  float inv = 1.f / sqrtf(ss / 4 + 1e-6f);
  for (int i = 0; i < 4; i++)
    CHECK(fabsf(out[i] - x[i] * inv) < 1e-5f, "rmsnorm[%d]=%g", i, out[i]);

  // Scale invariance: rmsnorm(k*x) == rmsnorm(x) when eps is negligible.
  float y[4]; for (int i = 0; i < 4; i++) y[i] = 10.f * x[i];
  float out2[4]; rmsnorm(y, w, 4, out2);
  for (int i = 0; i < 4; i++)
    CHECK(fabsf(out2[i] - out[i]) < 1e-3f, "rmsnorm scale[%d]=%g vs %g", i, out2[i], out[i]);

  CHECK(gelu(0.f) == 0.f, "gelu(0)=%g", gelu(0.f));
  CHECK(silu(0.f) == 0.f, "silu(0)=%g", silu(0.f));
  // Both are odd-ish monotone; check a known gelu value.
  CHECK(fabsf(gelu(1.f) - 0.8413447f) < 1e-5f, "gelu(1)=%g", gelu(1.f));
  CHECK(fabsf(silu(1.f) - 0.7310586f) < 1e-5f, "silu(1)=%g", silu(1.f));
}

int main(void) {
  // llm_load/llm_forward are unused here (no model file), but referencing them
  // keeps the header warning-free under -Werror without compiling them out.
  (void)&llm_load; (void)&llm_forward;
  test_half2float();
  test_deq_and_matvec();
  test_ragged_odd_cols();
  test_matvec_range_matches_full();
  test_norm_activations();
  if (fails) { printf("%d kernel test(s) FAILED\n", fails); return 1; }
  printf("kernel tests PASS\n");
  return 0;
}
