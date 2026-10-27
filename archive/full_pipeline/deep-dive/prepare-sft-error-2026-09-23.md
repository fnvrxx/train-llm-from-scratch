# Deep Dive: kegagalan persiapan data SFT di Colab

**Tanggal:** 2026-09-23  
**Fase:** diagnosis, belum ada perbaikan kode  
**File yang dianalisis:** `notebook/01_baseline.ipynb`, `notebook/build_notebooks.py`, `scripts/prepare_sft_data.py`, `src/post_training/chat_template.py`, `src/post_training/sft.py`

## Gambaran umum

Sel `RUN_PREP_SFT` tidak menyiapkan data secara langsung. Ia memulai proses Python lain untuk menjalankan `scripts/prepare_sft_data.py`. `CalledProcessError` dengan status 1 berarti proses tersebut berhenti dengan kesalahan. Traceback yang ditampilkan pengguna hanya berasal dari proses pemanggil, sehingga belum membuktikan kesalahan mana yang terjadi di dalam skrip SFT.

Skrip ini memang dipisahkan dari notebook supaya logika penyiapan data tetap sama dengan versi repo dan dapat dijalankan melalui CLI. Kelemahannya saat debugging: pesan kesalahan dari proses anak harus dibaca pada output sel sebelum traceback `CalledProcessError`.

## Alur kode, langkah demi langkah

### `notebook/build_notebooks.py` dan sel `RUN_PREP_SFT`

Pada `baseline_runtime()`, generator notebook menulis sel yang melakukan:

```python
dims = json.loads((config_dir/'base.json').read_text())
subprocess.run([sys.executable, 'scripts/prepare_sft_data.py',
                '--context_length', str(dims['context_length']),
                '--out_dir', str(ARTIFACT_ROOT/'data'),
                '--limit_per_set', '2000'], check=True, cwd=PROJECT_ROOT)
```

1. `config_dir/'base.json'` dibaca untuk mengambil `context_length`. Nilai `256` pada traceback konsisten dengan profil `smoke`.
2. `sys.executable` memilih interpreter Python yang sedang dipakai kernel Colab. Pada traceback nilainya `/usr/bin/python3`.
3. `cwd=PROJECT_ROOT` menjalankan skrip dari akar proyek yang dibuat oleh sel `%%writefile`.
4. `check=True` meminta Python melempar `CalledProcessError` jika skrip keluar dengan kode selain nol. Jadi baris yang ditandai traceback adalah tempat kegagalan *dilaporkan*, bukan lokasi akar masalah.

### `scripts/prepare_sft_data.py`

- Baris 23 sampai 27 mengimpor `h5py`, `numpy`, template chat, dan fungsi packing. Kegagalan import akan terjadi sebelum pesan `Loading ...` pertama.
- Baris 29 menetapkan `HF_HOME=/ephemeral/hf_cache` bila variabel itu belum ada. Ini lokasi cache dataset, bukan direktori HDF5 hasil.
- Baris 60 sampai 94 mengambil Alpaca, Dolly, lalu GSM8K. Kesalahan pada Alpaca atau GSM8K tidak ditangkap dan akan menghentikan proses. Kesalahan pada Dolly ditangkap dan dicetak sebagai `skipping dolly`, jadi pesan itu saja tidak berarti proses gagal.
- Baris 68 sampai 73 mengubah percakapan menjadi token dan mengabaikan contoh yang lebih panjang dari `context_length` atau tidak mempunyai token target untuk loss. Dengan `256`, contoh yang lolos dapat lebih sedikit daripada profil penuh. Ini belum terbukti sebagai penyebab status 1.
- Baris 77, 83, dan 90 menerapkan `limit_per_set` **setelah** `load_dataset` berhasil. Nilai `2000` membatasi contoh yang diproses, bukan ukuran awal dataset yang harus diunduh/dimuat.
- Baris 116 sampai 123 mengacak contoh, memisahkan train/dev, mengemas token, lalu menulis `sft_packed.h5` dan `sft_dev_packed.h5` ke `/ephemeral/data`.

### `src/post_training/chat_template.py` dan `src/post_training/sft.py`

`encode_chat` menyusun ID token serta `loss_mask`. Token jawaban asisten diberi mask 1, sedangkan prompt pengguna diberi mask 0. `pack_examples` kemudian menggabungkan contoh menjadi baris dengan panjang tetap. Pemisahan ini memungkinkan SFT menghitung loss hanya pada bagian jawaban, tidak pada prompt. Alternatifnya adalah menyimpan satu contoh per baris dengan padding, tetapi packing mengurangi ruang kosong dan perlu perhatian pada batas antarcontoh.

## Cara membaca error ini

`CalledProcessError: ... exit status 1` hanya menyatakan **skrip anak gagal**. Untuk mengetahui sebabnya, lihat output sel yang sama **sebelum** traceback terakhir, terutama traceback yang menyebut `prepare_sft_data.py`, `datasets`, `tiktoken`, `h5py`, atau HTTP/Hugging Face.

| Pesan terakhir sebelum error | Lokasi yang perlu diperiksa | Makna awal |
|---|---|---|
| Tidak ada `Loading tatsu-lab/alpaca ...` | Import awal skrip | Kemungkinan paket/import atau file sumber belum tertulis |
| `Loading tatsu-lab/alpaca ...` | `load_dataset` Alpaca | Kemungkinan akses, jaringan, cache, atau format dataset |
| `skipping dolly: ...` lalu lanjut | Blok Dolly | Dolly gagal tetapi ditangani; cari error setelahnya |
| `Loading openai/gsm8k ...` | `load_dataset` GSM8K | Kemungkinan akses, jaringan, cache, atau format dataset |
| `Collected ...` lalu gagal | Packing atau penulisan HDF5 | Periksa jumlah contoh, ruang disk, dan pesan HDF5 yang spesifik |

Tabel ini adalah peta diagnosis, bukan penetapan penyebab. Versi Python 3.13 yang terlihat pada traceback juga belum cukup untuk menyimpulkan ada masalah kompatibilitas.

## Konsep yang dipakai

| Konsep | Apa dan mengapa | Kapan dipakai | Alternatif dan konsekuensi |
|---|---|---|---|
| Proses anak (`subprocess`) | Notebook menjalankan skrip terpisah supaya CLI dan notebook memakai alur sama | Skrip sudah ada dan dapat dijalankan mandiri | Memanggil fungsi langsung memberi traceback dalam satu proses, tetapi mengubah cara integrasi dan state kernel |
| `check=True` | Gagal cepat ketika skrip mengembalikan status nonnol | Tahap pipeline tidak boleh diam-diam lanjut setelah gagal | `check=False` memberi kontrol manual, tetapi mudah membuat error terabaikan |
| `load_dataset` | Mengambil dataset dari Hub/cache | Data sumber tersedia sebagai dataset Hugging Face | Data lokal menghindari unduhan, tetapi perlu format dan sumber yang sesuai |
| Loss masking | Hanya bagian jawaban asisten dihitung dalam objective SFT | Pelatihan model instruksi | Melatih semua token lebih sederhana, tetapi model juga belajar meniru prompt |

## Langkah diagnosis berikutnya

1. Jangan jalankan ulang unduhan dulu. Buka output sel Colab yang gagal dan cari traceback atau pesan error **sebelum** `CalledProcessError`.
2. Catat pesan `Loading ...` terakhir dan 10 sampai 30 baris error tepat di atas traceback terakhir. Hindari mengirim token atau kredensial jika kebetulan tercetak.
3. Setelah lokasi gagal diketahui, tentukan perbaikan yang tepat. Tidak ada alasan cukup untuk mengubah kode atau memasang paket secara acak hanya dari exit status 1.

## Referensi untuk belajar

- [Python `subprocess` (dokumentasi resmi)](https://docs.python.org/3/library/subprocess.html): arti `check=True`, status proses, dan `CalledProcessError`.
- [Hugging Face Datasets: Load (dokumentasi resmi)](https://huggingface.co/docs/datasets/en/loading): cara `load_dataset` mengambil dataset dari Hub atau file lokal.
- [Hugging Face Datasets: loading methods](https://huggingface.co/docs/datasets/package_reference/loading_methods): parameter `split`, `cache_dir`, dan opsi pemuatan lain.

## Kode terkait

| File | Hubungannya |
|---|---|
| `notebook/01_baseline.ipynb` | Sel yang Anda jalankan di Colab |
| `notebook/build_notebooks.py` | Sumber pembangkit sel notebook |
| `scripts/prepare_sft_data.py` | Proses anak yang benar-benar menyiapkan HDF5 |
| `src/post_training/chat_template.py` | Tokenisasi percakapan dan mask jawaban |
| `src/post_training/sft.py` | Packing token dan loss SFT |

Dokumen ini dibuat untuk memahami kegagalan, bukan menyatakan bahwa penyebab pastinya sudah diketahui. Tidak ada dataset atau training yang dijalankan secara lokal.
