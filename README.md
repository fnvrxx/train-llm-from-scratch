# Small LLM untuk cerita pendek

Repo ini memperlihatkan cara sebuah model bahasa belajar dari cerita pendek Indonesia. Kode utama ada di `small_llm/`: teks menjadi byte, byte menjadi batch prediksi token berikutnya, lalu Transformer diperbarui dengan gradien. Empat varian MLP sparse memakai trainer dan pembagian data yang sama agar hasilnya bisa dibandingkan dengan model dense.

Ukuran bawaan untuk MacBook Air M1 dengan RAM 8 GB adalah embedding 192, empat attention head, empat blok, dan konteks 256 byte. `device: auto` memilih MPS jika PyTorch menyediakannya, lalu CPU jika tidak. Konfigurasi ini adalah titik mulai, bukan janji waktu training atau kualitas cerita.

## Langkah demi langkah menjalankan model

Jalankan perintah berikut dari root repo. Python 3.11 atau lebih baru dan `uv` diperlukan untuk memasang proyek.

1. **Siapkan lingkungan.** Pada mesin baru, buat `.venv` lalu pasang proyek dan PyTorch. Jika `.venv` sudah tersedia, cukup jalankan perintah kedua bila dependensi belum terpasang.

   ```bash
   uv venv .venv
   uv pip install --python .venv/bin/python -e .
   ```

2. **Periksa alur dasar.** Tes ini memeriksa tokenisasi, model, checkpoint, dan satu langkah training kecil untuk tiap varian.

   ```bash
   .venv/bin/python -m unittest discover -s tests
   ```

3. **Coba profil singkat.** Konfigurasi `configs/smoke.json` memakai CPU dan hanya melakukan dua langkah training per varian. Perintah berikut menjalankan lima varian (`dense`, `pruned`, `topk`, `moe`, `set`).

   ```bash
   .venv/bin/python -m small_llm.compare --config configs/smoke.json --overwrite
   ```

   Repo ini sudah memiliki contoh hasil di `artifacts/smoke/`. Opsi `--overwrite` **mengganti hasil lama** agar perintah bisa dijalankan lagi. Jika ingin mempertahankannya, lewati langkah ini dan baca hasil tersimpan pada langkah 4.

4. **Baca hasil evaluasi.** Setelah langkah 3 selesai, buka tabel perbandingan dan riwayat loss salah satu varian:

   ```bash
   cat artifacts/smoke/comparison.csv
   cat artifacts/smoke/dense/metrics.jsonl
   cat artifacts/smoke/dense/summary.json
   ```

   `valid_loss` adalah cross-entropy pada cerita validasi; lebih rendah berarti prediksi token pada data itu lebih baik. Bandingkan `initial_valid_loss` dengan `valid_loss` terakhir untuk melihat perubahan selama training. Jika `train_loss` turun tetapi `valid_loss` naik, model mungkin mulai menghafal data training. `train_tokens_per_second` mengukur kecepatan training dan `peak_process_rss_mb` mengukur puncak memori proses. Hasil smoke hanya membuktikan alur berjalan; dua langkah dan 12 cerita contoh belum cukup untuk menilai kemampuan menulis cerita. Hasil smoke lama yang sudah tersimpan mungkin belum memiliki kolom `initial_valid_loss`; jalankan ulang dengan kode saat ini bila membutuhkan kolom itu.

5. **Latih satu model lebih lama.** Mulai dari varian `dense` dengan profil M1. Konfigurasi ini memakai 300 langkah dan menyimpan hasil di `artifacts/m1/dense/`.

   ```bash
   .venv/bin/python -m small_llm.train --config configs/small_m1.json --variant dense
   ```

   Untuk penilaian yang lebih berguna, siapkan lebih banyak cerita yang boleh Anda gunakan, satu cerita per berkas `.txt`. Salin `configs/small_m1.json`, lalu ubah `train.data_dir` ke folder cerita tersebut dan `train.out_dir` ke folder hasil baru sebelum menjalankan perintah dengan `--config` yang baru. Korpus bawaan terlalu kecil untuk menyimpulkan kualitas model.

6. **Periksa hasil dan coba membuat teks.** Training menghasilkan `metrics.jsonl`, `summary.json`, dan `model.pt` di folder varian.

   ```bash
   cat artifacts/m1/dense/summary.json
   cat artifacts/m1/dense/metrics.jsonl
   .venv/bin/python -m small_llm.generate --checkpoint artifacts/m1/dense/model.pt --prompt "Pagi itu, Mira menemukan "
   ```

   Keluaran generasi sekarang mengikuti urutan byte UTF-8 yang valid, sehingga urutan byte rusak tidak berubah menjadi karakter `�`. Kalimat tetap bisa tidak masuk akal bila data atau training belum cukup; periksa isinya, bukan hanya apakah teks dapat ditampilkan.

7. **Bandingkan semua varian jika diperlukan.** Perintah ini melatih setiap varian dalam proses terpisah dan menulis `artifacts/m1/comparison.csv`. Opsi `--overwrite` diperlukan jika langkah 5 sudah dijalankan; opsi ini juga mengganti hasil `dense` dari langkah tersebut.

   ```bash
   .venv/bin/python -m small_llm.compare --config configs/small_m1.json --overwrite
   cat artifacts/m1/comparison.csv
   ```

   Jika ingin mempertahankan hasil langkah 5, salin konfigurasi dan ubah `train.out_dir` ke folder baru, lalu jalankan tanpa `--overwrite`. Jika lingkungan membatasi proses anak, tambahkan `--in-process`; dalam mode itu puncak RSS tidak dapat dibandingkan antarvarian.

### Percobaan dengan lebih banyak cerita

`data/stories_expanded/` berisi 60 cerita, termasuk 12 cerita demo. Konfigurasi `configs/stories_expanded_m1.json` memakai 45 cerita untuk training dan 15 cerita untuk validasi, menjalankan 600 langkah, serta menyimpan hasil terpisah dari `artifacts/m1/`. Jalankan hanya jika Anda sudah siap melakukan training baru:

```bash
.venv/bin/python -m small_llm.train --config configs/stories_expanded_m1.json --variant dense
cat artifacts/stories_expanded_m1/dense/summary.json
cat artifacts/stories_expanded_m1/dense/metrics.jsonl
.venv/bin/python -m small_llm.generate --checkpoint artifacts/stories_expanded_m1/dense/model.pt --prompt "Pagi itu, Mira menemukan "
```

Perhatikan perubahan `valid_loss` sepanjang training dan baca teks hasil generasi. Nilai loss eksperimen ini tidak langsung sebanding dengan hasil `artifacts/m1/` karena cerita validasinya berbeda. Korpus 60 cerita masih kecil untuk kemampuan menulis cerita yang konsisten; [catatan korpus](data/stories_expanded/README.md) menjelaskan batasnya.

### Jika Python tertutup mendadak

Pop-up “Python quit unexpectedly” saja tidak memastikan penyebabnya. `FileExistsError` berarti folder hasil sudah ada dan bukan tanda kekurangan RAM; gunakan `--overwrite` hanya bila ingin mengganti hasil itu. Catat perintah yang dijalankan dan pesan terakhir di Terminal. Coba profil smoke yang memakai CPU terlebih dahulu. Jika crash hanya muncul saat memakai `configs/small_m1.json`, salin konfigurasi itu dan ubah `train.device` dari `auto` menjadi `cpu`, lalu gunakan file salinan melalui `--config`. Jika masih gagal atau ada pesan memori habis, turunkan `train.batch_size`, `model.context_length`, atau `model.n_embed` di salinan konfigurasi. CPU mungkin lebih lambat. Mengubah ukuran model membuat checkpoint lama tidak cocok dengan arsitektur baru. Simpan bagian `Exception Type` dan `Termination Reason` dari laporan crash macOS untuk diagnosis lebih lanjut.

## Data dan batas percobaan

`data/stories_demo/` berisi 12 cerita contoh yang ditulis untuk repo ini. Tiap berkas `.txt` adalah satu cerita. Loader mengacak nama berkas dengan seed tetap, lalu memisahkan train dan validasi **per cerita**. Token `256` menandai akhir cerita. Model tetap dapat melihat batas antara dua cerita dalam satu jendela training; itu bagian dari format data, bukan kebocoran validasi.

Korpus demo terlalu kecil untuk menguji kemampuan menulis cerita. Model bisa menghafalnya, dan loss validasi dari dua atau tiga cerita memiliki variasi tinggi. Korpus tambahan berisi 60 cerita, tetapi masih kecil untuk kemampuan menulis cerita yang runtut. Untuk percobaan lebih lanjut, kumpulkan lebih banyak cerita yang boleh Anda gunakan, letakkan satu cerita per `.txt`, lalu arahkan `data_dir` ke folder tersebut. Ulangi seluruh varian dengan pembagian dan seed yang sama.

Tokenizer byte memberi kosakata tetap berisi 257 ID. Ia bisa membaca teks UTF-8 apa pun tanpa melatih tokenizer, tetapi satu huruf dapat memakai lebih dari satu byte. Karena itu konteks 256 token hanya memuat potongan pendek. Semua angka loss hanya dapat dibandingkan dengan eksperimen yang memakai tokenizer dan data yang sama.

## Yang dibandingkan

| Varian | Perubahan pada MLP | Hal yang perlu dibaca dari hasil |
|---|---|---|
| `dense` | Semua neuron dan bobot dipakai | Acuan loss, kecepatan, dan memori |
| `pruned` | Bobot kecil dimask sejak awal | Koneksi aktif berkurang, tetapi matriks tetap disimpan penuh |
| `topk` | Tiap token memilih neuron dengan aktivasi terbesar | Lapisan pertama masih menghitung seluruh aktivasi |
| `moe` | Router memilih beberapa expert untuk tiap token | Ada loss tambahan untuk menyeimbangkan pemakaian expert |
| `set` | Mask koneksi diubah selama training | Jumlah koneksi aktif tetap, letaknya berubah |

Trainer menyamakan seed, file train/validasi, batch token, jumlah langkah, optimizer, dan learning rate. Jumlah parameter serta operasi tiap varian tetap berbeda. `mlp_usage` menjelaskan pilihan koneksi, neuron, atau expert; `active_mlp_connections` hanya berlaku untuk mask bobot `pruned` dan `set`. `peak_process_rss_mb` adalah puncak memori seluruh proses Python, termasuk PyTorch dan data. `train_tokens_per_second` hanya menghitung waktu langkah training; evaluasi, penyimpanan, dan generasi tidak masuk. Kecepatan pada korpus contoh hanya mengukur jalur kode kecil ini, bukan kecepatan model pada beban produksi.

## Belajar dari kode

1. [Tokenisasi dan data](docs/01-tokenisasi-data.md): ID byte, penanda akhir cerita, pemisahan data, dan pasangan input-target.
2. [Transformer](docs/02-transformer.md): embedding, causal attention, MLP, residual, dan logits.
3. [Training dan evaluasi](docs/03-training-evaluasi.md): cross-entropy, gradien, akumulasi, checkpoint, dan arti loss validasi.
4. [Sparsity](docs/04-sparsity.md): apa yang berubah pada tiap varian dan cara membaca perbandingan.
5. [Catatan AntiVibe](deep-dive/small-llm-cerita-2026-10-22.md): penjelasan berkas, alasan desain, alternatif, dan latihan.

Pipeline LLM besar sebelumnya, termasuk SFT, DPO, PPO, GRPO, UI Streamlit, serta notebook lama, disimpan di [archive/full_pipeline](archive/full_pipeline/README.md). Catatan pemindahannya ada di [archive/README.md](archive/README.md). Perintah Small LLM di atas tidak mengimpor kode arsip.
