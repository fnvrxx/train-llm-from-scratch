# 7. Konfigurasi, evaluasi, dan batas kode

[Daftar bab](../llm-matematika-2026-09-29.md) · [Berikutnya: glosarium](08-glosarium.md)

## Ada dua jalur konfigurasi

`scripts/train_transformer.py` memakai [config/config.py](../../config/config.py). `scripts/pretrain_base.py` dan trainer post-training memakai dataclass [post_training_config.py](../../config/post_training_config.py) serta JSON di `configs/`.

Pada jalur JSON, prioritas dari rendah ke tinggi adalah default dataclass, `base.json`, JSON tahap, lalu argumen CLI. `load_config` membuang kunci yang tidak dikenal dataclass. Karena ada override, nilai default dalam deklarasi belum tentu nilai yang dipakai saat training.

```bash
rtk proxy env PYTHONPATH=. python scripts/train_sft.py --config configs/sft.json --print-config
```

Perintah itu menampilkan konfigurasi final lalu keluar pada parser, sebelum model training dibuat. Lingkungan Python tetap perlu memiliki dependensi impor skrip.

| Pengaturan | Pertanyaan yang dijawab |
|---|---|
| `vocab_size` | Berapa ID yang bisa dikeluarkan lm_head? |
| `n_embed` | Berapa koordinat fitur per token? |
| `n_head` | Berapa attention head per blok? |
| `n_blocks` | Berapa kali blok ditumpuk? |
| `context_length` | Berapa panjang urutan maksimum model? |
| `batch_size` | Berapa urutan/pasangan diproses bersama per batch? |
| `epochs`, `train_steps`, `iterations` | Berapa lama loop mengulang data atau rollout? |
| `lr`, `min_lr`, `warmup_steps` | Bagaimana ukuran langkah optimizer berubah? |
| `grad_clip` | Seberapa besar norma gradien yang diizinkan? |
| `grad_accum` | Berapa micro-batch dikumpulkan jika trainer memakainya? |
| `beta`, `kl_coef`, `clip` | Bagaimana preferensi, penalti reference, dan clipping ditimbang? |
| `out_ckpt`, `save_every` | Ke mana dan seberapa sering hasil disimpan? |

Kamus mencatat seluruh field konfigurasi, termasuk default pada masing-masing kelas. Dataclass mewarisi field `BaseModelConfig`, jadi `SFTConfig` mempunyai `n_embed` meskipun tidak menuliskannya lagi di tubuh kelas.

Pada konfigurasi awal, `CONTEXT_LENGTH=512` tetapi `T_CONTEXT_LENGTH=16`. Artinya tabel posisi mampu menampung 512 posisi, namun batch training jalur awal memakai 16 token. Kapasitas tabel tidak membuktikan model sudah belajar dependensi sepanjang 512 token.

## Ukuran model dapat dihitung

Untuk implementasi dense ini, $V$ ukuran kosakata, $C$ embedding, $T_{\max}$ kapasitas posisi, dan $N$ banyak blok. Embedding token dan lm_head tidak berbagi bobot.

$$
P_{\mathrm{blok}}=12C^2+10C,
$$
$$
P_{\mathrm{total}}=2VC+T_{\max}C+N(12C^2+10C)+2C+V.
$$

Asalnya: Q/K/V memberi $3C^2$, proyeksi attention $C^2+C$, dua linear MLP $8C^2+5C$, dan dua LayerNorm $4C$. Di luar blok ada dua tabel embedding, lm_head dengan bias, serta final LayerNorm. Buffer causal mask dan indeks posisi tidak termasuk parameter.

Perhitungan ini berlaku bila $C$ habis dibagi jumlah head. Reward/value head menambah parameter sesuai wrapper masing-masing. Mengubah jumlah head dengan $C$ tetap tidak mengubah total bobot Q/K/V pada desain ini.

## Metrik menjawab pertanyaan berbeda

| Metrik | Yang diukur | Yang belum dapat disimpulkan |
|---|---|---|
| Token loss | Kecocokan probabilitas terhadap target data | Kebenaran semua jawaban |
| Perplexity | Eksponensial rata-rata NLL token | Persentase jawaban benar |
| Preference accuracy | Seberapa sering chosen diberi skor lebih tinggi | Akurasi numerik GSM8K |
| Mean reward | Rata-rata skor menurut penilai yang dipakai | Reasoning benar jika penilai hanya memeriksa angka akhir |
| GSM8K accuracy | Proporsi jawaban numerik yang cocok kunci | Kemampuan semua tugas bahasa/matematika |
| Clip fraction/KL | Statistik perubahan policy selama update | Jaminan kualitas atau konvergensi |

$$
\operatorname{PPL}=e^{\overline{\mathrm{NLL}}},\qquad
\operatorname{accuracy}=\frac{\text{jumlah jawaban cocok}}{\text{jumlah soal}}.
$$

Perplexity hanya layak dibandingkan bila tokenizer, data, dan posisi target yang dinilai sama. Loss SFT yang hanya menilai assistant tidak langsung setara loss pretraining yang menilai semua target.

`gsm8k_accuracy` menyiapkan prompt, memanggil `batched_generate`, mengambil `gold`, dan menghitung `correct`. `samples` menyimpan contoh untuk dibaca manusia. `n` adalah jumlah soal, dan `return_samples` membatasi banyak contoh yang disertakan.

Jika set test dipakai berulang untuk memilih hyperparameter, ia ikut memengaruhi keputusan pengembangan. Gunakan validation set untuk keputusan tersebut dan sisakan test untuk laporan akhir. Beberapa trainer lokal memang mengevaluasi split test secara berkala, sehingga jangan menyebut setiap hasil itu evaluasi final yang sepenuhnya tak tersentuh.

## Catatan implementasi yang perlu dibaca bersama tutorial

Temuan berikut berasal dari pembacaan kode lokal. Dokumentasi ini tidak mengubah implementasinya.

| Lokasi | Perilaku aktual dan dampaknya |
|---|---|
| `Transformer.forward_embedding` | Keluaran tengah MLP berlebar $4C$ diteruskan ke blok yang mengharapkan $C$; jalur ini bermasalah untuk banyak blok. Pakai `forward_hidden` untuk representasi backbone normal. |
| `train_sft.py` | Field `grad_accum` tidak dipakai dalam loop. Batch efektif tidak otomatis dikali nilai field tersebut. |
| `train_ppo.py` | Field `ent_coef` tidak digunakan pada total loss. Mengubahnya saja tidak menambahkan entropy bonus. |
| `optim.configure_optimizer` | Semua parameter berdimensi minimal dua, termasuk embedding, masuk kelompok weight decay. |
| `chat_template.encode_chat` | Menambah generation prompt tidak menghapus mask assistant pada riwayat sebelumnya. |
| `pack_examples` | EOT memisahkan teks tetapi tidak mengisolasi attention antar contoh dalam satu baris. |
| `group_advantages` | `std` memakai koreksi sampel; `group_size=1` menghasilkan nilai tak terdefinisi. |
| `grpo_loss` | Rata-rata berbobot token, sehingga panjang jawaban memengaruhi bobot per jawaban. |
| `rollout.generate_with_logprobs` | Log-prob berasal dari distribusi penuh; sampling top-p/top-k yang dipotong memerlukan kehati-hatian saat menginterpretasikan rasio RL. |
| `evaluation.batched_generate` | Mode greedy memakai threshold top-k=1. Jika maksimum logits seri, beberapa kandidat seri dapat tetap lolos. |
| `config.post_training_config.SMOKE` | Kosakata 256 hanya untuk token sintetis; ID EOT r50k_base adalah 50256. Jangan masukkan chat nyata tanpa konfigurasi vocab yang sesuai. |
| Loader SFT/preferensi/prompt | Hanya mengeluarkan batch lengkap. Jika shard rank lebih kecil dari satu batch dan mode infinite aktif, iterator dapat berputar tanpa menghasilkan batch. |
| `inference.load_model_from_ckpt` | Membaca metadata `cfg`, sementara trainer awal menyimpan `config`. Memuat checkpoint awal perlu penyesuaian dimensi/override yang benar. |
| `utils.load_backbone_from_ckpt` | Menghapus `module.` tetapi tidak `transformer.` sebelum filter nama. Checkpoint wrapper reward/value tidak otomatis cocok meskipun komentarnya terdengar umum. |
| `train_ppo.py` saat save | Menyimpan backbone actor, bukan seluruh value head. Hasil cocok untuk inference, tetapi bukan resume actor-critic lengkap. |
| `scripts/chat.py` | Setiap input REPL diteruskan sebagai panggilan baru; percakapan sebelumnya tidak otomatis ditambahkan sebagai history. |

Default `/ephemeral/...` dalam konfigurasi berasal dari lingkungan kerja penulis. Itu string path, bukan variabel matematika dan bukan jaminan direktori tersebut tersedia di komputer Anda.

## Cara belajar dengan kode ini

1. Ikuti `xb` menjadi embedding, attention, hidden state, lalu logits. Tuliskan bentuk tensor setiap perpindahan.
2. Hitung satu loss token dan cocokkan tanda turunannya dengan perubahan logit yang diharapkan.
3. Baca `sft_loss`; geser mask dengan tangan memakai satu contoh pendek.
4. Baca DPO, lalu bandingkan policy, old policy, dan reference pada PPO.
5. Baru baca pembungkus training, DDP, dan checkpoint. Bagian ini mengatur pelaksanaan perhitungan yang sudah Anda pahami.

Untuk variasi arsitektur, lanjutkan ke [panduan notebook dense/sparse](../notebook-dense-sparse-2026-09-27.md) dan [eksperimen evolutionary](../evolutionary-sparse-training-2026-09-23.md). Top-k aktivasi pada eksperimen sparse berbeda dari top-k sampling kosakata saat inference.

## Latihan

Apakah menambah `context_length` pada file JSON sudah cukup untuk memakai checkpoint lama pada model berkapasitas posisi lebih besar?

Jawaban: tidak. Ukuran parameter `position_embed` dan buffer model terkait berubah. Checkpoint lama harus kompatibel dengan arsitektur yang dibuat, atau diperlukan prosedur adaptasi yang jelas. Mengubah sebuah angka konfigurasi tidak otomatis memperluas pengetahuan posisi yang telah dilatih.
