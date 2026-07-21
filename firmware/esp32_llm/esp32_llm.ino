// PLE TinyLM inference on the ESP32-S3.
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped so the 25M table is read a row at a time from flash; the hot
// tied head plus scratch and KV cache sit in PSRAM. Same llm.h that was verified
// against PyTorch on the host -- only the platform hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/llm.h"
#include "vocab.h"

static const int PROMPT_IDS[] = {433, 447, 259, 405}; // "Once upon a time"
static const int N_GENERATE = 200;

Model model;
Scratch s;

// The output rows are independent. Run half on each LX7 core, preserving the
// exact scalar dot-product order within every row.
static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
// volatile: published to the worker core via the FreeRTOS notify barrier, but
// marking them volatile removes any reliance on that being the only ordering.
static const QT *volatile head_job_t;
static const float *volatile head_job_x;
static float *volatile head_job_y;
static volatile int head_job_split;

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_q_range(head_job_t, head_job_x, head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

static void parallel_head_matvec(const QT *t, const float *x, float *y) {
  head_job_t = t;
  head_job_x = x;
  head_job_y = y;
  head_job_split = t->rows / 2;
  xTaskNotifyGive(head_worker);
  matvec_q_range(t, x, y, head_job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { Serial.printf("PSRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}

// The tied embedding/output head is scanned in full for every token. Stage it
// in PSRAM once at boot; the 25M sparse PLE table and small core stay mapped in
// flash because measured core staging only saved 1.4% while adding complexity.
static void stage_head_in_psram(QT *t) {
  size_t codes_n = (size_t)t->rows * t->row_bytes;
  size_t scales_n = (size_t)t->rows * t->n_groups * sizeof(uint16_t);
  uint8_t *copy = (uint8_t *)ps(codes_n + scales_n);
  memcpy(copy, t->codes, codes_n);
  memcpy(copy + codes_n, t->scales, scales_n);
  t->codes = copy;
  t->scales = (const uint16_t *)(copy + codes_n);
  Serial.printf("head staged in PSRAM: %.2f MB\n", (codes_n + scales_n) / 1e6);
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE TinyLM ===");

  // Map the model partition.
  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: V=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim,
                part->size / 1e6);

  stage_head_in_psram(&model.tok_emb);
  // The tokenizer learned 25,353 entries; the remaining padded model rows can
  // never be emitted, so do not spend a full head dot-product on them.
  model.tok_emb.rows = VOCAB_N;
  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                             &head_worker, 0) != pdPASS) {
    Serial.println("head worker creation failed");
    return;
  }
  model.head_matvec = parallel_head_matvec;

  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, V = c->vocab, S = c->seq_len;
  s.x = (float *)ps(D * 4);
  s.h = (float *)ps((F > D ? F : D) * 4);
  s.qkv = (float *)ps(3 * D * 4);
  s.att = (float *)ps(D * 4);
  s.g1 = (float *)ps(F * 4);
  s.g2 = (float *)ps((P > F ? P : F) * 4);
  s.ple = (float *)ps(L * P * 4);
  s.tmpP = (float *)ps(L * P * 4);
  s.trow = (float *)ps(L * P * 4);
  s.logits = (float *)ps(V * 4);
  s.scores = (float *)ps(S * 4);
  s.kcache = (float *)ps((size_t)L * S * D * 4);
  s.vcache = (float *)ps((size_t)L * S * D * 4);
  Serial.printf("PSRAM free after alloc: %u KB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);

  // ---- generate ----
  Serial.print(">>> ");
  int n_prompt = sizeof(PROMPT_IDS) / sizeof(int);
  int pos = 0, tok = 0;
  int64_t t_start = 0;
  int64_t decode_us = 0;
  int decoded = 0;

  for (int i = 0; i < n_prompt; i++) {  // prime with the prompt
    tok = PROMPT_IDS[i];
    if (tok < VOCAB_N)
      Serial.write(VOCAB_BLOB + VOCAB_OFF[tok], VOCAB_OFF[tok + 1] - VOCAB_OFF[tok]);
    llm_forward(&model, tok, pos++, &s);
  }

  llm_profile_reset(&s);

  t_start = esp_timer_get_time();
  for (int step = 0; step < N_GENERATE && pos < model.c.seq_len; step++) {
    // greedy: argmax over the trained vocab
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < VOCAB_N; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    tok = best;
    Serial.write(VOCAB_BLOB + VOCAB_OFF[tok], VOCAB_OFF[tok + 1] - VOCAB_OFF[tok]);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens (~1.1s), near-free
  }
  int64_t total_us = esp_timer_get_time() - t_start;

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
  blink(0);
}

void loop() { delay(10000); }
