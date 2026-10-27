# Deep Dive: Pile Uncopyrighted untuk pretraining, bukan pengganti langsung data SFT

**Tanggal:** 2026-09-23  
**Fase:** dokumentasi dan diagnosis, tanpa menjalankan kode  
**File yang dibaca:** `notebook/01_baseline.ipynb`, `notebook/build_notebooks.py`, `scripts/prepare_pretrain_data.py`, `scripts/prepare_sft_data.py`, `src/post_training/chat_template.py`, `src/post_training/sft.py`, `configs/pretrain.json`, `configs/sft.json`

## Jawaban singkat

`CalledProcessError` pada sel `RUN_PREP_SFT` **belum membuktikan bahwa data belum terunduh**. Itu hanya berarti `scripts/prepare_sft_data.py` berhenti dengan status 1. Penyebab sebenarnya ada pada keluaran skrip sebelum traceback `CalledProcessError`, misalnya kesalahan import, akses dataset, koneksi, cache, atau penulisan file. Jangan menebak penyebab dari status 1 saja.

`monology/pile-uncopyrighted` adalah sumber teks untuk **pretraining** dalam repo ini. Skrip `scripts/prepare_pretrain_data.py` sudah mengambil shard dari dataset Hugging Face yang sama, lalu mengubahnya menjadi HDF5 token. Sel SFT yang gagal memakai **Alpaca, Dolly, dan GSM8K**, bukan Pile. Mengunduh Pile tidak akan mengisi `sft_packed.h5` atau memperbaiki kegagalan `RUN_PREP_SFT` secara otomatis.

## Peta data proyek

| Tahap | Sumber | Bentuk yang diperlukan training | File keluaran |
|---|---|---|---|
| Pretraining | Teks mentah Pile Uncopyrighted | Deretan ID token dalam dataset HDF5 `tokens` | `/ephemeral/data/pile_train.h5`, `pile_dev.h5` |
| SFT | Contoh instruksi dan jawaban dari Alpaca, Dolly, GSM8K | Baris HDF5 `tokens` **dan** `loss_mask` yang sejajar | `/ephemeral/data/sft_packed.h5`, `sft_dev_packed.h5` |

Perbedaan ini penting. Pada pretraining, model memprediksi token berikutnya pada teks umum. Pada SFT, model belajar menghasilkan jawaban asisten dari prompt pengguna; `loss_mask` menandai token jawaban yang dihitung dalam loss. Pile berisi kolom `text` dan `meta`, bukan pasangan `instruction`/`output` yang dibutuhkan konversi SFT saat ini. Bentuk kolom itu terlihat di [halaman dataset](https://huggingface.co/datasets/monology/pile-uncopyrighted). Sifat tugas pretraining dan SFT di atas merupakan kesimpulan dari kode repo.

## Walkthrough kode

### 1. Sel `RUN_PREP_SFT` di notebook baseline

Sel membaca `context_length` dari konfigurasi Colab dan memanggil `scripts/prepare_sft_data.py` melalui `subprocess.run(..., check=True)`. Nilai `256` pada traceback berasal dari profil `smoke`. Bila proses anak keluar dengan kode 1, Python melempar `CalledProcessError`; exception itu bukan akar kesalahan. Lihat [dokumentasi resmi `subprocess`](https://docs.python.org/3/library/subprocess.html) untuk arti `check=True`.

### 2. `scripts/prepare_sft_data.py`

Baris 75 memuat `tatsu-lab/alpaca`, baris 82 memuat `databricks/databricks-dolly-15k`, dan baris 89 memuat `openai/gsm8k`. Dolly berada dalam `try/except`, sehingga pesan `skipping dolly` sendiri tidak menghentikan proses. Kesalahan Alpaca atau GSM8K tidak ditangkap oleh blok tersebut. Setelah pemuatan, skrip memakai `encode_chat`, menyaring contoh yang lebih panjang dari `context_length`, membagi train/dev, lalu menulis dua HDF5 SFT.

`--limit_per_set 2000` diterapkan setelah masing-masing pemanggilan `load_dataset`. Jadi batas ini mengurangi contoh yang diproses, tetapi tidak menjamin unduhan awal hanya 2.000 contoh.

### 3. `scripts/prepare_pretrain_data.py`

Baris 32 sampai 41 menyusun alamat shard `monology/pile-uncopyrighted` secara langsung. Skrip mengunduh `val.jsonl.zst` untuk dev dan, secara default, `train/00.jsonl.zst` untuk train. Baris 60 sampai 73 mendekompresi JSONL dan membaca `text`. Baris 76 sampai 117 mengubah teks dengan tokenizer `r50k_base`, menambah token akhir dokumen `50256`, dan menulis dataset HDF5 `tokens` satu dimensi.

Artinya, jika tujuan Anda adalah pretraining memakai Pile, repo **sudah memiliki jalur konversi** lewat sel `RUN_PREP_PRETRAIN`. Batas `--max_tokens` berlaku pada proses tokenisasi setelah file shard diunduh. Daftar file dataset saat ini menunjukkan `val.jsonl.zst` sekitar 338 MB dan `train/00.jsonl.zst` sekitar 11,2 GB; seluruh repositori sekitar 335 GB. Ukuran dapat berubah, jadi cek lagi sebelum memulai unduhan di Colab. [Daftar file dataset](https://huggingface.co/datasets/monology/pile-uncopyrighted/tree/main), [shard train pertama](https://huggingface.co/datasets/monology/pile-uncopyrighted/blob/main/train/00.jsonl.zst).

### 4. `src/post_training/chat_template.py` dan `src/post_training/sft.py`

`encode_chat` menghasilkan dua daftar dengan panjang sama: ID token dan mask. Token prompt pengguna diberi mask 0; token jawaban asisten diberi mask 1. `pack_examples` menggabungkan contoh ke baris tetap sepanjang `context_length`. Data Pile yang hanya berisi teks tidak menghasilkan mask jawaban secara bermakna tanpa keputusan tambahan tentang prompt dan jawaban.

## Jika ingin memakai `load_dataset("monology/pile-uncopyrighted")`

Potongan berikut hanya menunjukkan perbedaan API, **bukan kode yang dijalankan atau ditambahkan ke proyek**:

```python
from datasets import load_dataset

# Permintaan Anda: mode standar. Untuk dataset ini, kebutuhan unduhan/cache dapat sangat besar.
ds = load_dataset("monology/pile-uncopyrighted")

# Opsi untuk membaca bertahap, bila format dan lingkungan mendukungnya.
train_stream = load_dataset(
    "monology/pile-uncopyrighted", split="train", streaming=True
)
```

Tanpa `split`, `load_dataset` biasanya mengembalikan kumpulan split; dengan `split="train"`, hasilnya satu split. Mode standar `streaming=False` mengunduh dan menyiapkan data lokal, sehingga tidak cocok dianggap sebagai unduhan kecil untuk Colab. [Referensi resmi `load_dataset`](https://huggingface.co/docs/datasets/package_reference/loading_methods). Mode `streaming=True` membaca contoh bertahap tanpa mengunduh seluruh file terlebih dahulu, tetapi tetap memakai jaringan dan membutuhkan konversi ke HDF5 sebelum trainer repo dapat menggunakannya. [Panduan resmi streaming](https://huggingface.co/docs/datasets/en/stream).

Ada riwayat masalah streaming pada file `.zst` dataset ini. Dalam [diskusi dataset](https://huggingface.co/datasets/monology/pile-uncopyrighted/discussions/5), seorang pengguna melaporkan error kompresi yang teratasi setelah memasang `zstandard`; pengguna lain masih melaporkan masalah sampai kernel dimulai ulang. Ini catatan kompatibilitas, bukan bukti bahwa Colab Anda mengalami error yang sama. Notebook proyek memang mencantumkan `zstandard` pada sel instalasi paket, tetapi keberhasilan streaming pada runtime Anda belum diuji.

Jika Anda memilih jalur `load_dataset` untuk pretraining, data `text` masih perlu ditokenisasi dengan `r50k_base`, diberi pemisah dokumen, dipisah train/dev tanpa kebocoran data, dan ditulis ke HDF5 `tokens` dengan nama/path yang dibaca `configs/pretrain.json`. Repo saat ini belum menyediakan konverter `load_dataset` langsung ke format HDF5 tersebut; ia memakai jalur unduh shard dan dekompresi di `prepare_pretrain_data.py`. Dokumen ini tidak mengubah pipeline.

## Mengapa Pile tidak langsung menjadi data SFT?

Satu contoh Pile adalah teks umum. Satu contoh SFT pada proyek ini adalah percakapan pengguna dan asisten. Agar Pile dipakai untuk SFT, Anda harus menentukan cara membuat pasangan prompt-jawaban yang layak, memeriksa kualitasnya, menjalankan `encode_chat`, dan menghasilkan `loss_mask`. Sekadar mengubah nama file Pile menjadi `sft_packed.h5` tidak akan bekerja: pembaca SFT mengharapkan dua dataset HDF5 yang sejajar, `tokens` dan `loss_mask`.

Alternatif yang paling dekat dengan tujuan saat ini:

1. **Ingin pretraining pada Pile:** gunakan persiapan pretraining yang sudah ada. Perhitungkan unduhan shard sebelum menjalankan sel di Colab.
2. **Ingin menyelesaikan error SFT:** baca traceback proses anak di atas `CalledProcessError`. Pile bukan pengganti dataset instruksi yang dimuat skrip SFT.
3. **Ingin SFT dari Pile:** ini eksperimen data baru, bukan perubahan sumber unduhan saja. Perlu rancangan prompt-jawaban dan konverter baru yang dapat diuji terpisah.

## Konsep dan trade-off

| Pilihan | Kapan tepat | Kelebihan | Batasan |
|---|---|---|---|
| Unduh shard dengan skrip repo | Ingin memakai pipeline pretraining tanpa mengubah kode | Format HDF5 sudah cocok | Shard train pertama tetap besar; `max_tokens` tidak membatasi unduhan shard |
| `load_dataset` mode standar | Penyimpanan dan waktu cukup untuk dataset besar | API dataset lebih sederhana | Berpotensi mengunduh/menyiapkan ratusan GB |
| `load_dataset(..., streaming=True)` | Ingin membaca contoh bertahap | Tidak perlu menyimpan seluruh dataset dahulu | Tetap perlu jaringan, dukungan `.zst`, dan konversi HDF5 sendiri |
| Data instruksi asli untuk SFT | Ingin melatih perilaku tanya-jawab | Format semantis cocok untuk prompt-masked loss | Sumber dataset SFT tetap perlu diakses dan diproses |

Nama "uncopyrighted" berasal dari penerbit dataset; itu bukan jaminan hukum untuk setiap dokumen. Periksa [dataset card dan metadata lisensi](https://huggingface.co/datasets/monology/pile-uncopyrighted) sebelum penggunaan di luar eksperimen.

## Referensi belajar

- [Hugging Face Datasets: Loading](https://huggingface.co/docs/datasets/en/loading): perbedaan pemuatan Hub, file lokal, dan split.
- [Hugging Face Datasets: Streaming](https://huggingface.co/docs/datasets/en/stream): membaca dataset besar secara bertahap dan batasannya.
- [Hugging Face Dataset Card: Pile Uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted): isi dan metadata sumber data.
- [Python: subprocess](https://docs.python.org/3/library/subprocess.html): alasan `CalledProcessError` muncul di notebook.

## Langkah diagnosis yang belum terjawab

Kirim 10 sampai 30 baris keluaran sel `RUN_PREP_SFT` **sebelum** `CalledProcessError`, terutama pesan `Loading ...` terakhir dan traceback dari skrip anak. Tanpa bagian itu, status unduhan data SFT dan akar kegagalannya belum dapat dipastikan.

Tidak ada kode, dataset, maupun training yang dieksekusi untuk membuat dokumen ini.
