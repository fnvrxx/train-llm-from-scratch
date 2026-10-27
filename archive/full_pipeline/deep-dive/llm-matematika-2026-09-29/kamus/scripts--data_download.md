# scripts/data_download.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/data_download.py)

## Fungsi dan kelas

### `download_file` (baris 12)

Mengunduh byte dataset streaming; versi download memakai file .part sebelum rename.

```python
def download_file(url: str, file_name: str) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `download_dataset` (baris 29)

Mengunduh validasi dan shard train yang belum tersedia lokal.

```python
def download_dataset(val_url: str, train_urls: List[str], val_dir: str, train_dir: str, max_train_files: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `main` (baris 58)

Menyiapkan direktori dan mengunduh shard dataset berdasarkan opsi CLI.

```python
def main() -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `BASE_URL` · 8 | Alamat dasar dataset, berkas validasi, dan daftar shard train. | `'https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main'` |
| `VAL_URL` · 9 | Alamat dasar dataset, berkas validasi, dan daftar shard train. | `f'{BASE_URL}/val.jsonl.zst'` |
| `TRAIN_URLS` · 10 | Alamat dasar dataset, berkas validasi, dan daftar shard train. | `[f'{BASE_URL}/train/{i:02d}.jsonl.zst' for i in range(65)]` |
| `i` · 10 | Nomor shard untuk menyusun daftar 65 URL train. | `iterasi range(65)` |

### Lingkup `download_file`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `url` · 12 | Alamat unduhan tunggal atau kumpulan URL shard. | `argumen str` |
| `file_name` · 12 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `response` · 22 | Teks konteks dan jawaban; response pada download adalah respons HTTP. | `requests.get(url, stream=True)` |
| `total_size` · 23 | Potongan byte unduhan, ukuran potongan, dan ukuran total respons. | `int(response.headers.get('content-length', 0))` |
| `block_size` · 24 | Potongan byte unduhan, ukuran potongan, dan ukuran total respons. | `1024` |
| `f` · 25 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(file_name, 'wb')` |
| `chunk` · 26 | Potongan byte unduhan, ukuran potongan, dan ukuran total respons. | `iterasi tqdm(response.iter_content(block_size), total=total_size // block_size, desc='Downloading', leave=True)` |

### Lingkup `download_dataset`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `val_url` · 29 | Alamat unduhan tunggal atau kumpulan URL shard. | `argumen str` |
| `train_urls` · 29 | Alamat unduhan tunggal atau kumpulan URL shard. | `argumen List[str]` |
| `val_dir` · 29 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `argumen str` |
| `train_dir` · 29 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `argumen str` |
| `max_train_files` · 29 | Batas banyak shard training yang diproses. | `argumen int` |
| `val_file_path` · 41 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `os.path.join(val_dir, 'val.jsonl.zst')` |
| `idx` · 49 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari iterasi enumerate(train_urls[:max_train_files])` |
| `url` · 49 | Alamat unduhan tunggal atau kumpulan URL shard. | `elemen 1 dari iterasi enumerate(train_urls[:max_train_files])` |
| `file_name` · 50 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `f'{idx:02d}.jsonl.zst'` |
| `file_path` · 51 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `os.path.join(train_dir, file_name)` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `parser` · 63 | Objek argparse yang mendefinisikan dan membaca argumen terminal. | `argparse.ArgumentParser(description='Download PILE dataset.')` |
| `args` · 68 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `parser.parse_args()` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `os` | `os` | 1 |
| `argparse` | `argparse` | 2 |
| `requests` | `requests` | 3 |
| `tqdm` | `tqdm.tqdm` | 4 |
| `List` | `typing.List` | 5 |
