# AI 28,9 Juta Parameter yang Ngobrol Bahasa Indonesia di Chip Rp130.000-an

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license">
  <img src="https://github.com/kenzieradhika/ai-esp/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/params-28.9M-purple" alt="28.9M params">
  <img src="https://img.shields.io/badge/speaks-Bahasa%20Indonesia-orange" alt="speaks Bahasa Indonesia">
</p>

![Demo AI berjalan di ESP32-S3](media/esp32-ple-demo.gif)

Ini adalah model bahasa dengan **28,9 juta parameter** yang bisa diajak ngobrol
dalam **Bahasa Indonesia** dan berjalan langsung di chip **ESP32-S3** —
microcontroller murah yang dipakai di papan IoT. Semua diproses di dalam chip.
Tidak ada server, tidak ada internet, tidak ada data yang keluar dari
perangkat. Setiap kata yang dijawab benar-benar "dipikirkan" di chip itu sendiri.

Sebagai pembanding: model bahasa sebelumnya yang pernah berjalan di chip
sekecil ini hanya punya 260 ribu parameter. Model ini kira-kira **110x lebih
besar** — bisa muat karena sebagian besar bobotnya disimpan di flash, bukan di
RAM, memakai ide bernama **Per-Layer Embeddings** (PLE) dari keluarga model
Gemma buatan Google.

## Fitur

- **Ngobrol bahasa Indonesia.** Ketik apa saja lewat serial, model menjawab
  langsung di chip dengan format `user: <kamu>\nJawaban:`.
- **Belajar kata baru dari obrolan.** Kata yang belum dikenal otomatis
  diingat, dan kamu bisa mengajarkan artinya langsung: `kutu artinya louse`,
  `bebek adalah duck`, `sapi means cow`, atau `nyala=on`. Ingatan disimpan di
  flash (NVS) — tetap ada walau dicabut listrik atau reflash firmware.
- **Mengenal kamu.** `nama aku Niko` menyimpan namamu, `aku suka sepeda`
  menyimpan kesukaanmu. Dia menyapa pakai namamu dan membuat cerita yang
  dipersonalisasi: `Ceritakan tentang Niko yang suka sepeda`.
- **Menulis cerita.** `story: kucing yang hilang` → cerita 200 token (~37 detik).
- **Zero server.** Tidak ada koneksi jaringan sama sekali.

## Spesifikasi

| | |
|---|---|
| Parameter | 28,9M tersimpan (25M di antaranya tabel lookup di flash) |
| Chip | ESP32-S3 N16R8 — 512KB SRAM, 8MB PSRAM, 16MB flash |
| Kecepatan | ~9,5 tok/s (model TinyStories), ~5,4-5,7 tok/s end-to-end (model chat) |
| Bahasa model | Bahasa Indonesia (fine-tune chat) |
| Ukuran model | 14,9MB pada 4-bit |
| Konektivitas | tidak ada — semua di perangkat |

## Cara Pakai (Windows)

```bat
cd D:\esp32-ai
.venv\Scripts\python firmware\prompt_ui.py          :: port otomatis terdeteksi
.venv\Scripts\python firmware\prompt_ui.py COM7     :: atau tentukan port manual
```

Lalu langsung ketik. Perintah yang dipahami firmware:

| kamu ketik | yang terjadi |
|---|---|
| `halo` | model menjawab (balasan 80 token ~15 detik) |
| `story: kucing yang hilang` | cerita 200 token yang dipersonalisasi (~37 detik) |
| `nama aku Niko` | mengingat namamu |
| `aku suka sepeda` | mengingat kesukaanmu |
| `kutu artinya louse` | mengajarkan arti kata (bisa juga `adalah` / `means` / `=`) |
| baris lainnya | diteruskan ke model, kata baru otomatis diingat |

Cara pasang firmware + wiring lengkap ada di
[`firmware/esp32_llm/README.md`](firmware/esp32_llm/README.md).

## Kenapa Ini Sulit, dan Kok Bisa Muat

Microcontroller punya memori cepat yang sangat kecil — ESP32-S3 cuma punya
512KB SRAM. Model harus bisa dijangkau dari sana, dan itu yang membatasi model
hanya sampai ratusan ribu parameter.

Cara melewatinya: berhenti menaruh model di memori cepat. Sebagian besar
parameter model bahasa ada di tabel embedding, yang hanya *dibaca* tiap token,
bukan dihitung. Jadi tabel 25 juta parameter itu bisa ditinggal di flash yang
lambat tapi besar — tiap token cukup mengambil beberapa baris saja (±450
byte) — sementara bagian kecil yang bekerja tetap di memori cepat. Hasilnya:
model besar hampir tanpa biaya, karena sebagian besarnya tidak pernah dimuat.

```
  SRAM  (cepat, kecil)   inti "berpikir", dipakai tiap token
  PSRAM (sedang)         output head dan memori kerja
  FLASH (besar, lambat)  tabel 25M-param, ~6 baris dibaca per token (±450 B)
```

## Struktur Repo

- `firmware/esp32_llm/` — sketsa Arduino untuk ESP32-S3 (chat, memori, OLED)
- `firmware/prompt_ui.py` — antarmuka chat serial untuk Windows
- `data/prepare_id.py` — pipeline dataset chat Indonesia + tokenizer BPE 32768
- `src/` — training, kuantisasi, ekspor model, generate aset firmware
- `experiments/` — skrip eksperimen (ablation, sweep)
- `RESULTS.md` — semua hasil terukur: baseline, ablasi, kuantisasi 4-bit, dan hasil model chat Indonesia

## Hasil Terukur

Ringkasan singkat di [`RESULTS.md`](RESULTS.md): PLE mengalahkan baseline sebesar
0,098 nats (2 seed), keunggulan itu bertahan setelah kuantisasi 4-bit, dan
model chat Indonesia (feasibility 1500 langkah) mencapai val ppl 90,81 —
jawabannya koheren dalam Bahasa Indonesia dengan format `Jawaban:`.

## Batasan yang Jujur

- Inti dense-nya cuma 559K parameter — aritmatika tidak bisa diandalkan, dan
  jawaban bisa melenceng ke topik dominan dataset (tips/checklist).
- Ini chatbot mainan yang menarik karena arsitekturnya (model besar di chip
  mungil), bukan karena kepintarannya. Fine-tune penuh ~5000 langkah sedang
  dikerjakan untuk membuatnya lebih fasih.

## Lisensi

MIT — lihat [LICENSE](LICENSE).
