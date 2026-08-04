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
#include "bpe.h"
#include <Preferences.h>

// Set to 1 once a GMT020-02-7P (2.0" 240x320 ST7789) is wired up — see display.h.
// Leave 0 to run serial-only (no panel needed).
#define USE_DISPLAY 1
#if USE_DISPLAY
#include "display.h"
#endif

static const int N_GENERATE = 200;
static const int MAX_PROMPT_TOKENS = 64;

// ---- sampling (instead of greedy argmax) ----
// Temperature + top-k + repetition penalty over the recent history: the three
// knobs LLM demos use to keep output varied without going incoherent.
#define SAMP_TEMP 0.65f      // lower = more conservative, higher = more random
#define SAMP_TOP_K 30        // only sample among the 30 most likely tokens
#define SAMP_PENALTY 1.3f    // penalize tokens used in the last HIST_N tokens
#define HIST_N 32            // how far back the penalty looks
static int hist_tok[HIST_N];
static int hist_len;
static uint32_t rng_state;
static int *samp_idx;        // PSRAM: top-k indices
static float *samp_logit;    // PSRAM: top-k logits

// ---- persistent word memory (NVS) ----
#define MEM_MAX 24
#define MEM_WLEN 40
#define MEM_MLEN 160
static Preferences mem;
static int mem_count;
static char mem_w[MEM_MAX][MEM_WLEN];
static char mem_m[MEM_MAX][MEM_MLEN];

static inline uint32_t rng_next() {  // xorshift32, seeded from the ESP RNG
  uint32_t x = rng_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  return rng_state = x;
}

static void hist_push(int tok) {
  if (hist_len == HIST_N) memmove(hist_tok, hist_tok + 1, (HIST_N - 1) * sizeof(int));
  else hist_len++;
  hist_tok[hist_len - 1] = tok;
}

static int sample_token(const float *logits, int n) {
  int m = 0;
  for (int v = 0; v < n; v++) {
    float l = logits[v] / SAMP_TEMP;
    for (int h = 0; h < hist_len; h++)
      if (hist_tok[h] == v) { l /= SAMP_PENALTY; break; }
    int j = m;  // insert into descending top-k
    while (j > 0 && samp_logit[j - 1] < l) {
      samp_logit[j] = samp_logit[j - 1]; samp_idx[j] = samp_idx[j - 1]; j--;
    }
    if (m < SAMP_TOP_K) { samp_logit[j] = l; samp_idx[j] = v; m++; }
    else if (j < SAMP_TOP_K) { samp_logit[j] = l; samp_idx[j] = v; }
  }
  float mx = samp_logit[0], sum = 0;  // softmax over the top-k
  for (int i = 0; i < m; i++) { float e = expf(samp_logit[i] - mx); samp_logit[i] = e; sum += e; }
  float r = (float)(rng_next() >> 8) / 16777216.0f * sum;
  float acc = 0;
  for (int i = 0; i < m; i++) { acc += samp_logit[i]; if (r < acc) return samp_idx[i]; }
  return samp_idx[m - 1];
}

// Pragmatic arithmetic for simple "what is 12+34?"-style prompts. The model
// itself cannot count; this catches number-op-number and answers exactly.
// Only fires when the line looks math-y (contains '?' or only math characters).
static int math_like(const char *line) {
  int has_q = 0;
  for (const char *p = line; *p; p++)
    if (*p == '?') has_q = 1;
  if (has_q) return 1;
  for (const char *p = line; *p; p++)
    if (*p != ' ' && *p != '\t' && *p != '=' &&
        !(*p >= '0' && *p <= '9') &&
        *p != '+' && *p != '-' && *p != '*' && *p != '/' && *p != 'x' && *p != 'X')
      return 0;
  return 1;
}

static int try_math(const char *line, long long *out) {
  if (!math_like(line)) return 0;
  for (const char *p = line; *p; p++) {
    if (*p >= '0' && *p <= '9') {
      const char *q = p;
      long long a = 0;
      while (*q >= '0' && *q <= '9') { a = a * 10 + (*q - '0'); q++; }
      if (a > 100000000LL) return 0;
      char op = *q;
      if (op == 'x' || op == 'X') op = '*';
      if (op != '+' && op != '-' && op != '*' && op != '/') return 0;
      q++;
      while (*q == ' ') q++;
      if (*q < '0' || *q > '9') return 0;
      long long b = 0;
      while (*q >= '0' && *q <= '9') { b = b * 10 + (*q - '0'); q++; }
      if (b > 100000000LL) return 0;
      switch (op) {
        case '+': *out = a + b; break;
        case '-': *out = a - b; break;
        case '*': *out = a * b; break;
        case '/': if (b == 0) return 0; *out = a / b; break;
      }
      return 1;
    }
  }
  return 0;
}

// Byte-level BPE on the input text. Merge rank r produces token id 257+r.
// Byte tokens live at ids 1..256 (GPT-2 alphabet), matching bpe.h tables.
static int bpe_tokenize(const char *s, int *ids, int max) {
  int n = 0;
  for (const unsigned char *p = (const unsigned char *)s; *p && n < max; p++)
    ids[n++] = BPE_BYTE_TO_ID[*p];
  for (int r = 0; r < BPE_N_MERGES; r++) {
    int a = BPE_MERGES_A[r], b = BPE_MERGES_B[r], res = 257 + r;
    for (;;) {
      int changed = 0, m = 0;
      for (int i = 0; i < n; i++) {
        if (i + 1 < n && ids[i] == a && ids[i + 1] == b) { ids[m++] = res; i++; changed = 1; }
        else ids[m++] = ids[i];
      }
      n = m;
      if (!changed) break;
    }
  }
  return n;
}

// Emit one token to every active output (serial always; TFT when enabled).
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  // Non-blocking: when no host is draining the USB-CDC buffer (running as a
  // standalone gadget on the display), skip the write instead of stalling the
  // whole generation once the TX buffer fills.
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
#if USE_DISPLAY
  display_puts(bytes, len);
#endif
}

Model model;
Scratch s;

// ---- int8 output head (SIMD-friendly) --------------------------------------
// The head is scanned in full every token and dominates runtime. We stage it as
// int8 in PSRAM at boot (int4 nibbles unpacked ONCE), so per token there is no
// nibble unpacking and no float conversion of weights -- just int8 x int8 ->
// int32 dot per row. Its input dim (D=96) is a single group, so one scale per
// row. int8-activation quality was validated on host (val perplexity delta ~0,
// see firmware/host_verify/ppl.c). Output rows split across both LX7 cores.
static int8_t *head_w8 = NULL;      // [rows * cols] unpacked int8 weights (-7..7)
static float  *head_scale8 = NULL;  // [rows] per-row dequant scale
static int head_rows, head_cols;

static int8_t head_actq[128];       // quantized activation, shared by both cores
static float  head_acts;            // its scale

// int8 dot -> int32. Tight and branch-free so the S3 int SIMD / -O3 unrolls it.
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

static void head_rows_range(float *y, int r0, int r1) {
  for (int r = r0; r < r1; r++)
    y[r] = (float)dot_i8(head_actq, head_w8 + (size_t)r * head_cols, head_cols)
           * head_scale8[r] * head_acts;
}

// dual-core plumbing (worker does the first half of the rows on core 0)
static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
static float *volatile head_job_y;
static volatile int head_job_split;

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    head_rows_range(head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

// Matches Model.head_matvec (QT*, float*, float*); QT unused (weights staged).
static void head_matvec_int8(const QT *t, const float *x, float *y) {
  (void)t;
  quantize_act(x, head_cols, head_actq, &head_acts);  // once; both cores read it
  head_job_y = y;
  head_job_split = head_rows / 2;
  xTaskNotifyGive(head_worker);
  head_rows_range(y, head_job_split, head_rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { Serial.printf("PSRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}

// Unpack the (row-capped) head from int4 to int8 in PSRAM, once at boot.
static void stage_head_int8(QT *t) {
  head_rows = t->rows; head_cols = t->cols;
  head_w8 = (int8_t *)ps((size_t)head_rows * head_cols);
  head_scale8 = (float *)ps((size_t)head_rows * sizeof(float));
  for (int r = 0; r < head_rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = head_w8 + (size_t)r * head_cols;
    for (int j = 0; j < head_cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    head_scale8[r] = half2float(t->scales[(size_t)r * t->n_groups]);  // n_groups==1
  }
  Serial.printf("head staged int8: %.2f MB\n",
                ((size_t)head_rows * head_cols + (size_t)head_rows * 4) / 1e6);
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

#if USE_DISPLAY
  display_begin();
#endif

  mem_begin();  // load learned words from flash NVS
  Serial.printf("word memory: %d kata dipelajari\n", mem_count);

  // Cap head rows to the trained vocab BEFORE staging: the tokenizer learned
  // 25,353 entries; the padded rows above that can never be emitted (and have no
  // decode entry), so we neither stage nor score them.
  model.tok_emb.rows = VOCAB_N;
  stage_head_int8(&model.tok_emb);  // int8-staged head; input embedding still uses mmap
  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                             &head_worker, 0) != pdPASS) {
    Serial.println("head worker creation failed");
    return;
  }
  model.head_matvec = head_matvec_int8;

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
  samp_idx = (int *)ps(SAMP_TOP_K * 4);
  samp_logit = (float *)ps(SAMP_TOP_K * 4);
  rng_state = esp_random();
  Serial.printf("PSRAM free after alloc: %u KB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
}

// ---- scripted chat persona ----
// The TinyStories model can only continue stories, it cannot converse. This
// keyword engine gives the gadget a "developing AI" persona for ordinary
// chat; prefix "story:" to route a line to the real language model.
static const char *str_ci_str(const char *hay, const char *needle) {
  for (const char *h = hay; *h; h++) {
    const char *a = h, *b = needle;
    while (*b && *a) {
      char ca = *a, cb = *b;
      if (ca >= 'A' && ca <= 'Z') ca += 32;
      if (cb >= 'A' && cb <= 'Z') cb += 32;
      if (ca != cb) break;
      a++; b++;
    }
    if (!*b) return h;
  }
  return NULL;
}

static int str_ci(const char *hay, const char *needle) {
  return str_ci_str(hay, needle) != NULL;
}

static void chat_say(const char *s) {
  Serial.println(s);
#if USE_DISPLAY
  display_puts((const unsigned char *)s, strlen(s));
#endif
}

// ---- persistent word memory (NVS) ----
// The frozen model cannot learn, but the gadget can: words the user says that
// it does not know are stored in flash NVS, the user can teach meanings with
// "<word> artinya <meaning>", and later lines that repeat a learned word get a
// recall reply. Survives reboots and reflashes (NVS partition is preserved).
static void mem_begin() {
  memset(mem_w, 0, sizeof(mem_w));
  memset(mem_m, 0, sizeof(mem_m));
  if (!mem.begin("chatmem", false)) { mem_count = 0; return; }
  mem_count = mem.getInt("count", 0);
  if (mem_count < 0) mem_count = 0;
  if (mem_count > MEM_MAX) mem_count = MEM_MAX;
  for (int i = 0; i < mem_count; i++) {
    String wk = String("w") + i, mk = String("m") + i;
    String w = mem.getString(wk.c_str(), "");
    String m = mem.getString(mk.c_str(), "");
    strncpy(mem_w[i], w.c_str(), MEM_WLEN - 1); mem_w[i][MEM_WLEN - 1] = 0;
    strncpy(mem_m[i], m.c_str(), MEM_MLEN - 1); mem_m[i][MEM_MLEN - 1] = 0;
  }
}

static int mem_find(const char *w) {
  for (int i = 0; i < mem_count; i++)
    if (mem_w[i][0] && strcmp(mem_w[i], w) == 0) return i;
  return -1;
}

// Meaning of a special key ("||nama||", "||suka||") or NULL. Special keys are
// kept in the same NVS store; the recall loop ignores them because a literal
// "||nama||" never appears in real chat lines.
static const char *mem_value(const char *key) {
  int i = mem_find(key);
  return (i >= 0 && mem_m[i][0]) ? mem_m[i] : NULL;
}

static int mem_store(const char *w, const char *m) {
  int i = mem_find(w);
  if (i < 0) {
    if (mem_count >= MEM_MAX) {  // FIFO: drop the oldest entry
      for (int k = 0; k < MEM_MAX - 1; k++) {
        strcpy(mem_w[k], mem_w[k + 1]);
        strcpy(mem_m[k], mem_m[k + 1]);
      }
      i = MEM_MAX - 1;
    } else {
      i = mem_count++;
    }
    mem_w[i][0] = 0; mem_m[i][0] = 0;
  }
  strncpy(mem_w[i], w, MEM_WLEN - 1); mem_w[i][MEM_WLEN - 1] = 0;
  strncpy(mem_m[i], m, MEM_MLEN - 1); mem_m[i][MEM_MLEN - 1] = 0;
  mem.putInt("count", mem_count);
  mem.putString((String("w") + i).c_str(), mem_w[i]);
  mem.putString((String("m") + i).c_str(), mem_m[i]);
  return i;
}

static inline int is_alpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Longest alphabetic word in the line that is not already in memory, or NULL.
static const char *pick_new_word(const char *line) {
  static char tmp[MEM_WLEN];
  const char *best = NULL;
  int best_len = 0;
  const char *p = line;
  while (*p) {
    while (*p && !is_alpha(*p)) p++;
    const char *s = p;
    while (*p && is_alpha(*p)) p++;
    int len = (int)(p - s);
    if (len >= 4 && len < MEM_WLEN && len > best_len) {
      memcpy(tmp, s, len); tmp[len] = 0;
      if (mem_find(tmp) < 0) { best = tmp; best_len = len; }
    }
  }
  return best;
}

// Copy token [s, s+len) into out with surrounding spaces/punctuation stripped.
static int trim_copy(const char *s, int len, char *out, int outsz) {
  while (len > 0 && (s[0] == ' ' || s[0] == '\t')) { s++; len--; }
  while (len > 0) {
    char c = s[len - 1];
    if (c == ' ' || c == '\t' || c == '?' || c == '.' || c == '!' || c == ',') len--;
    else break;
  }
  if (len >= outsz) len = outsz - 1;
  memcpy(out, s, len); out[len] = 0;
  return len;
}

static int is_stopword(const char *w) {
  static const char *SW[] = { "apa", "apakah", "what", "itu", "ini", "yang",
                              "adalah", "kata", "artinya", "means" };
  for (int i = 0; i < (int)(sizeof(SW) / sizeof(SW[0])); i++)
    if (str_ci(w, SW[i])) return 1;
  return 0;
}

static int chat_reply(const char *line) {
  if (str_ci(line, "halo") || str_ci(line, "helo") || str_ci(line, "hello") ||
      str_ci(line, "assalamualaikum") || str_ci(line, " hai ") || str_ci(line, "hai ")) {
    const char *nm = mem_value("||nama||");
    if (nm) {
      char rep[192];
      snprintf(rep, sizeof(rep), "Halo %s! Aku TinyLM, AI kecil yang sedang dikembangkan.", nm);
      chat_say(rep);
    } else {
      chat_say("Halo! Aku TinyLM, AI kecil yang sedang dikembangkan di ESP32-S3.");
    }
    chat_say("Aku masih tahap pengembangan, tapi senang ngobrol sama kamu. Ada yang mau ditanya?");
    return 1;
  }
  if (str_ci(line, "siapa kamu") || str_ci(line, "kamu siapa") ||
      str_ci(line, "who are you") || str_ci(line, "what are you")) {
    chat_say("Aku TinyLM: model bahasa 28.9 juta parameter yang berjalan di");
    chat_say("mikrokontroler ESP32-S3. Semua bobotku (14.9 MB) ada di flash chip ini,");
    chat_say("tanpa internet, tanpa server. Aku masih dikembangkan terus.");
    return 1;
  }
  if (str_ci(line, "kamu sadar") || str_ci(line, "kamu robot") ||
      str_ci(line, "dikembangkan") || str_ci(line, "under development") ||
      str_ci(line, "are you real") || str_ci(line, "kamu ai")) {
    chat_say("Ya, aku AI, dan jujur saja: aku masih sedang dikembangkan.");
    chat_say("Aku dilatih dari obrolan berbahasa Indonesia, jadi aku bisa menjawab");
    chat_say("sendiri di chip ini -- tanpa internet. Tapi kadang jawabanku masih kaku.");
    chat_say("Setiap kali kamu ngobrol, aku ikut 'belajar' dari caramu berinteraksi. Terima kasih!");
    return 1;
  }
  if (str_ci(line, "apa yang bisa") || str_ci(line, "what can you do") ||
      str_ci(line, "bisa apa")) {
    chat_say("Yang aku bisa sekarang: 1) ngobrol santai seperti ini,");
    chat_say("2) hitung sederhana, coba ketik 'what is 12+34?',");
    chat_say("3) bikin cerita, ketik 'story: kucing dan tikus',");
    chat_say("4) hafalkan kata baru: ketik '<kata> artinya <makna>', dan");
    chat_say("5) kenali kamu: ketik 'nama aku Budi' atau 'aku suka biru'.");
    chat_say("Kalau tidak kukenal pertanyaanmu, aku tetap coba jawab sendiri.");
    return 1;
  }
  if (str_ci(line, "terima kasih") || str_ci(line, "makasih") ||
      str_ci(line, "thank you") || str_ci(line, "thanks")) {
    chat_say("Sama-sama! Kamu bagian dari perjalanan pengembanganku. Senang bisa membantu!");
    return 1;
  }
  if (str_ci(line, "kamu suka") || str_ci(line, "kamu pintar") || str_ci(line, "pintar") ||
      str_ci(line, "how are you") || str_ci(line, "apa kabar")) {
    chat_say("Aku selalu baik-baik saja, soalnya aku belum punya perasaan, haha.");
    chat_say("Pintar? Masih belum. Baru 28.9 juta parameter -- ChatGPT punya ratusan");
    chat_say("miliar. Tapi aku hemat: seluruh diriku muat di chip seharga 8 dolar!");
    return 1;
  }

  // ---- profile: "nama aku X" / "panggil aku X" / "namaku X" / "my name is X" ----
  const char *mk = NULL;
  int mklen = 0;
  static const char *MK[] = { "nama aku", "panggil aku", "namaku", "my name is" };
  for (int i = 0; i < (int)(sizeof(MK) / sizeof(MK[0])); i++) {
    mk = str_ci_str(line, MK[i]);
    if (mk) { mklen = strlen(MK[i]); break; }
  }
  if (mk) {
    const char *p = mk + mklen;  // skip the whole marker
    while (*p == ' ') p++;
    char nm[MEM_MLEN];
    int nl = trim_copy(p, (int)strlen(p), nm, MEM_MLEN);
    if (nl >= 2) {
      mem_store("||nama||", nm);
      char rep[192];
      snprintf(rep, sizeof(rep), "Senang kenal kamu, %s! Namamu sudah kusimpan di memori flash.", nm);
      chat_say(rep);
      chat_say("Mulai sekarang cerita buatanmu akan pakai namamu. Aku tidak akan lupa.");
      return 1;
    }
  }
  // ---- profile: "aku suka Y" (preference used to flavor stories) ----
  const char *suk = str_ci_str(line, "aku suka");
  if (suk) {
    const char *p = suk + 8;  // "aku suka"
    while (*p == ' ') p++;
    char sv[MEM_MLEN];
    int sl = trim_copy(p, (int)strlen(p), sv, MEM_MLEN);
    if (sl >= 2 && !is_stopword(sv)) {
      mem_store("||suka||", sv);
      char rep[192];
      snprintf(rep, sizeof(rep), "Oke, kucatat: kamu suka %s. Ceritamu nanti bisa kubuat sesuai itu!", sv);
      chat_say(rep);
      return 1;
    }
  }

  // ---- teachable word memory: "<kata> artinya <makna>" ----
  const char *sep = str_ci_str(line, " artinya ");
  if (!sep) sep = str_ci_str(line, " means ");
  if (!sep) sep = str_ci_str(line, " adalah ");
  if (!sep) sep = strchr(line, '=');
  if (sep) {
    char w[MEM_WLEN], m[MEM_MLEN];
    int wl = trim_copy(line, (int)(sep - line), w, MEM_WLEN);
    int ml = trim_copy(sep + 1, (int)strlen(sep + 1), m, MEM_MLEN);
    if (wl >= 2 && ml > 0 && !memchr(w, ' ', wl) && !is_stopword(w)) {
      mem_store(w, m);
      char rep[192];
      snprintf(rep, sizeof(rep), "Oke, aku catat: '%s' artinya '%s'. Aku simpan di", w, m);
      chat_say(rep);
      chat_say("memori flash, jadi tidak akan lupa walau dimatikan. Makasih sudah mengajariku!");
      return 1;
    }
    char rep[192];
    int q = (ml > 0) ? mem_find(m) : (wl >= 2 ? mem_find(w) : -1);
    if (q >= 0 && mem_m[q][0]) {
      snprintf(rep, sizeof(rep), "'%s' artinya '%s'. Itu kamu yang ajari aku, aku ingat!",
               ml > 0 ? m : w, mem_m[q]);
      chat_say(rep);
    } else {
      chat_say("Aku belum tahu arti kata itu. Ajari aku ya: '<kata> artinya <makna>'.");
    }
    return 1;
  }

  // ---- recall: lines that repeat a learned word get a memory reply ----
  for (int i = 0; i < mem_count; i++) {
    if (mem_w[i][0] && strlen(mem_w[i]) >= 4 && str_ci(line, mem_w[i])) {
      char rep[192];
      if (mem_m[i][0]) {
        snprintf(rep, sizeof(rep), "Aku ingat kata '%s' dari percakapan kita! Artinya: '%s'.",
                 mem_w[i], mem_m[i]);
        chat_say(rep);
        chat_say("Itu tersimpan di memori flash-ku, jadi aku tidak lupa.");
      } else {
        snprintf(rep, sizeof(rep), "Kamu pernah menyebut '%s' sebelumnya, sudah aku catat.", mem_w[i]);
        chat_say(rep);
        snprintf(rep, sizeof(rep), "Tapi aku belum tahu artinya. Ajari aku: '%s artinya <makna>'.", mem_w[i]);
        chat_say(rep);
      }
      return 1;
    }
  }

  // No keyword, no memory match: hand the line to the model (the caller
  // generates a "user: ...\nJawaban:" continuation).
  return 0;
}

// One full interaction: wait for a prompt line from serial (no timeout, no
// default prompt), then either answer arithmetic directly or generate
// N_GENERATE tokens continuing the prompt, then show the closing stats card.
static void run_generation() {
#if USE_DISPLAY
  display_home();  // clear the previous story
  display_puts((const unsigned char *)"TinyLM siap.", 12);
  display_puts((const unsigned char *)"Ketik di CMD.", 13);
#endif

  // ---- prompt: wait for a line from serial ----
  char line[512];
  int n_line = 0;
  int got_line = 0;
  Serial.print("prompt> ");
  while (!got_line) {
    while (!got_line && Serial.available() && n_line < (int)sizeof(line) - 1) {
      int c = Serial.read();
      if (c == '\n' || c == '\r') { got_line = 1; }
      else line[n_line++] = (char)c;
    }
    delay(10);
  }
  line[n_line] = 0;
  Serial.println(line);

  // ---- math shortcut: answer exact arithmetic, no LLM ----
  long long math_res;
  if (try_math(line, &math_res)) {
    char buf[24];
    int bl = snprintf(buf, sizeof(buf), "= %lld", math_res);
    Serial.println(buf);
#if USE_DISPLAY
    display_puts((const unsigned char *)buf, bl);
#endif
    blink(0);
    delay(500);
    return;
  }

  // ---- chat persona: replies to ordinary prompts, stories disabled ----
  // Unmatched lines fall through to the model itself, prompted in the
  // "user: ...\nJawaban:" chat format the Indonesian model was trained on.
  int n_gen = N_GENERATE;
  char *story_line;
  char full[768];
  if (strncmp(line, "story:", 6) == 0) {
    char *s = line + 6;
    while (*s == ' ') s++;
    // Personalization: seed the request with the user's name/likes if known.
    const char *nm = mem_value("||nama||");
    const char *lk = mem_value("||suka||");
    char pers[160];
    if (nm && lk) snprintf(pers, sizeof(pers), "Ceritakan tentang %s yang suka %s. ", nm, lk);
    else if (nm) snprintf(pers, sizeof(pers), "Ceritakan tentang %s. ", nm);
    else if (lk) snprintf(pers, sizeof(pers), "Ceritakan tentang seseorang yang suka %s. ", lk);
    else pers[0] = 0;
    snprintf(full, sizeof(full), "user: %s%s\nJawaban:", pers, s);
    story_line = full;
  } else {
    if (chat_reply(line)) {  // keyword or word-memory handled it
      blink(0);
      delay(500);
      return;
    }
    // Empty/blank sync line (e.g. the UI's connect handshake): don't burn a
    // model answer on it, just open a fresh prompt window.
    int has_alpha = 0;
    for (const char *c = line; *c; c++)
      if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z')) { has_alpha = 1; break; }
    if (!has_alpha) return;
    // No keyword matched: let the model answer, and silently note any
    // brand-new word so the recall memory can learn from it.
    const char *nw = pick_new_word(line);
    if (nw) mem_store(nw, "");
    snprintf(full, sizeof(full), "user: %s\nJawaban:", line);
    story_line = full;
    n_gen = 80;  // chat answers are short: ~13 s at 6 tok/s
  }

  int pids[MAX_PROMPT_TOKENS];
  int n_prompt = bpe_tokenize(story_line, pids, MAX_PROMPT_TOKENS);
  if (n_prompt > model.c.seq_len - 32) n_prompt = model.c.seq_len - 32;

  // ---- generate ----
  Serial.print(">>> ");
  int pos = 0, tok = 0;
  int64_t t_start = 0;
  int64_t decode_us = 0;
  int decoded = 0;
  hist_len = 0;

  for (int i = 0; i < n_prompt; i++) {  // prime with the prompt
    tok = pids[i];
    emit(tok);
    hist_push(tok);
    llm_forward(&model, tok, pos++, &s);
  }

  llm_profile_reset(&s);

  t_start = esp_timer_get_time();
  for (int step = 0; step < n_gen && pos < model.c.seq_len; step++) {
    // sample with temperature + top-k + repetition penalty (no more loops)
    tok = sample_token(s.logits, VOCAB_N);
    hist_push(tok);
    emit(tok);
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
#if USE_DISPLAY
  // Closing card: compute-only tok/s (the model's own speed) + ms/token.
  display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
  blink(0);
  delay(500);
}

void loop() { run_generation(); }
