# scripts/data_preprocess.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/data_preprocess.py)

## Fungsi dan kelas

### `process_files` (baris 10)

Preprocessing awal: membaca JSONL zst dan menulis token text+EOT dengan resize HDF5 per dokumen.

```python
def process_files(input_dir: str, output_file: str, tokenizer_name: str, max_data: Optional[int]=None) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `main` (baris 81)

Memproses direktori data train/validasi menjadi HDF5.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `process_files`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `input_dir` · 10 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `argumen str` |
| `output_file` · 10 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `tokenizer_name` · 10 | Nama encoding tiktoken, umumnya r50k_base. | `argumen str` |
| `max_data` · 10 | Ambang jumlah token atau record yang diproses; max_tokens diperiksa setelah flush sehingga dapat terlampaui. | `argumen Optional[int]; default None` |
| `enc` · 29 | Tokenizer pada preprocessing; daftar hasil encoding pasangan pada _collate. | `tiktoken.get_encoding(tokenizer_name)` |
| `out_f` · 32 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `h5py.File(output_file, 'w')` |
| `dataset` · 34 | Array dataset/token/mask yang dibaca atau ditulis di HDF5. | `out_f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i')` |
| `start_index` · 35, 65 | Batas awal/akhir potongan array atau penulisan dataset. | `0`; `end_index` |
| `filename` · 38 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `iterasi sorted(os.listdir(input_dir))` |
| `in_file` · 40 | Path/daftar path data, direktori, atau berkas kompresi lokal. | `os.path.join(input_dir, filename)` |
| `processed_lines` · 43, 76 | Banyak baris diproses, ID hasil tokenisasi, dan panjang hasilnya. | `0`; `processed_lines += 1` |
| `in_f` · 46 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `zstd.open(in_file, 'rt', encoding='utf-8')` |
| `line` · 48 | Satu baris teks JSONL. | `iterasi tqdm(in_f, desc=f'Processing {filename}', total=max_data if max_data is not None else None)` |
| `data` · 51 | Record hasil parsing JSON atau objek data masukan terkait. | `json.loads(line)` |
| `text` · 52 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `data.get('text')` |
| `encoded` · 56 | Banyak baris diproses, ID hasil tokenisasi, dan panjang hasilnya. | `enc.encode(text + '<&#124;endoftext&#124;>', allowed_special={'<&#124;endoftext&#124;>'})` |
| `encoded_len` · 57 | Banyak baris diproses, ID hasil tokenisasi, dan panjang hasilnya. | `len(encoded)` |
| `end_index` · 60 | Batas awal/akhir potongan array atau penulisan dataset. | `start_index + encoded_len` |
| `e` · 72 | Exception yang ditangkap untuk pelaporan kegagalan. | `exception Exception` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `parser` · 86 | Objek argparse yang mendefinisikan dan membaca argumen terminal. | `argparse.ArgumentParser(description='Preprocess PILE dataset files and save tokens to HDF5.')` |
| `args` · 94 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `parser.parse_args()` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `os` | `os` | 1 |
| `json` | `json` | 2 |
| `zstd` | `zstandard` | 3 |
| `tiktoken` | `tiktoken` | 4 |
| `h5py` | `h5py` | 5 |
| `tqdm` | `tqdm.tqdm` | 6 |
| `argparse` | `argparse` | 7 |
| `Optional` | `typing.Optional` | 8 |
