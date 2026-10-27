# scripts/prepare_pretrain_data.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/prepare_pretrain_data.py)

## Fungsi dan kelas

### `shard_urls` (baris 38)

Menyusun URL shard train atau satu berkas validasi.

```python
def shard_urls(split: str, num_shards: int) -> list[str]:
```

Keluaran eksplisit: `[f'{BASE_URL}/val.jsonl.zst']`; `[f'{BASE_URL}/train/{i:02d}.jsonl.zst' for i in range(num_shards)]`.

### `download` (baris 44)

Mengunduh byte dataset streaming; versi download memakai file .part sebelum rename.

```python
def download(url: str, dest: str) -> str:
```

Keluaran eksplisit: `dest`.

### `iter_texts` (baris 60)

Mendekompresi JSONL dan mengeluarkan field text dari record yang valid.

```python
def iter_texts(zst_path: str):
```

Keluaran eksplisit: `txt`.

### `tokenize_to_h5` (baris 76)

Men-tokenisasi batch dokumen, menambah EOT, lalu menulis buffer token ke HDF5.

```python
def tokenize_to_h5(zst_paths: list[str], out_path: str, max_tokens: int | None) -> int:
```

Keluaran eksplisit: `total`.

### `tokenize_to_h5.flush` (baris 84)

Memperbesar HDF5, menulis buffer token, memperbarui total, lalu mengosongkan buffer.

```python
def flush():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `main` (baris 121)

Memilih shard, mengunduhnya, dan menulis token corpus ke HDF5.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `BASE_URL` · 32 | Alamat dasar dataset, berkas validasi, dan daftar shard train. | `'https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main'` |
| `EOT_ID` · 33 | Token akhir teks 50256; juga dipakai sebagai stop dan padding post-training. | `50256` |
| `WRITE_CHUNK` · 34 | Ukuran chunk token HDF5 (8 juta) dan banyak dokumen per batch encoding (1024). | `8000000` |
| `ENC_BATCH` · 35 | Ukuran chunk token HDF5 (8 juta) dan banyak dokumen per batch encoding (1024). | `1024` |

### Lingkup `shard_urls`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `split` · 38 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `argumen str` |
| `num_shards` · 38 | Batas banyak shard training yang diproses. | `argumen int` |
| `i` · 41 | Nomor shard train pada alamat unduhan. | `iterasi range(num_shards)` |

### Lingkup `download`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `url` · 44 | Alamat unduhan tunggal atau kumpulan URL shard. | `argumen str` |
| `dest` · 44 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `argumen str` |
| `r` · 50 | Objek respons HTTP streaming untuk berkas dataset. | `requests.get(url, stream=True)` |
| `total` · 52 | Ukuran byte dari header content-length, atau nol bila tidak tersedia. | `int(r.headers.get('content-length', 0))` |
| `f` · 53 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(dest + '.part', 'wb')` |
| `chunk` · 54 | Potongan byte unduhan, ukuran potongan, dan ukuran total respons. | `iterasi tqdm(r.iter_content(1 << 20), total=total >> 20, unit='MB', desc='dl')` |

### Lingkup `iter_texts`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `zst_path` · 60 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `argumen str` |
| `dctx` · 61 | Decompressor zstandard dan stream hasil dekompresinya. | `zstd.ZstdDecompressor()` |
| `fh` · 62 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(zst_path, 'rb')` |
| `reader` · 63 | Decompressor zstandard dan stream hasil dekompresinya. | `dctx.stream_reader(fh)` |
| `line` · 64, 65 | Satu baris teks JSONL. | `iterasi io.TextIOWrapper(reader, encoding='utf-8')`; `line.strip()` |
| `txt` · 69 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `json.loads(line).get('text')` |

### Lingkup `tokenize_to_h5`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `zst_paths` · 76 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `argumen list[str]` |
| `out_path` · 76 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `max_tokens` · 76 | Ambang jumlah token atau record yang diproses; max_tokens diperiksa setelah flush sehingga dapat terlampaui. | `argumen int &#124; None` |
| `enc` · 77 | Tokenizer pada preprocessing; daftar hasil encoding pasangan pada _collate. | `tiktoken.get_encoding('r50k_base')` |
| `total` · 79 | Jumlah token yang sudah ditulis ke HDF5. | `0` |
| `buf` · 80 | Kumpulan dokumen, buffer ID token, dan array NumPy yang ditulis ke HDF5. | `[]` |
| `f` · 81 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `h5py.File(out_path, 'w')` |
| `dset` · 82 | Array dataset/token/mask yang dibaca atau ditulis di HDF5. | `f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i4', chunks=(WRITE_CHUNK,))` |
| `zp` · 94 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `iterasi zst_paths` |
| `docs` · 96, 104 | Kumpulan dokumen, buffer ID token, dan array NumPy yang ditulis ke HDF5. | `[]` |
| `pbar` · 97 | Pembungkus tqdm untuk menampilkan kemajuan loop. | `tqdm(iter_texts(zp), unit='doc', desc='tok')` |
| `txt` · 98 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `iterasi pbar` |
| `ids` · 101, 111 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `iterasi enc.encode_ordinary_batch(docs)` |

### Lingkup `tokenize_to_h5.flush`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `arr` · 88 | Kumpulan dokumen, buffer ID token, dan array NumPy yang ditulis ke HDF5. | `np.asarray(buf, dtype=np.int32)` |
| `total` · 91 | Penghitung token pada enclosing function yang bertambah sebanyak arr.size. | `total += arr.size` |
| `buf` · 92 | Kumpulan dokumen, buffer ID token, dan array NumPy yang ditulis ke HDF5. | `[]` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `p` · 122 | Parser opsi terminal yang sedang ditambahkan field atau dibaca hasilnya. | `argparse.ArgumentParser()` |
| `args` · 128 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `p.parse_args()` |
| `urls` · 130 | Alamat unduhan tunggal atau kumpulan URL shard. | `shard_urls(args.split, args.num_shards)` |
| `local` · 131 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `[]` |
| `u` · 132 | Alamat unduhan tunggal atau kumpulan URL shard. | `iterasi urls` |
| `name` · 133 | Nama parameter model, field konfigurasi, atau berkas yang sedang diperiksa. | `'val.jsonl.zst' if args.split == 'val' else os.path.basename(u)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 18 |
| `argparse` | `argparse` | 20 |
| `io` | `io` | 21 |
| `json` | `json` | 22 |
| `os` | `os` | 23 |
| `h5py` | `h5py` | 25 |
| `np` | `numpy` | 26 |
| `requests` | `requests` | 27 |
| `tiktoken` | `tiktoken` | 28 |
| `zstd` | `zstandard` | 29 |
| `tqdm` | `tqdm.tqdm` | 30 |
