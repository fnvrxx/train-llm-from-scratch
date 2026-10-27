# scripts/prepare_preference_data.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/prepare_preference_data.py)

## Fungsi dan kelas

### `_split_hh` (baris 25)

Memisahkan konteks dan jawaban pada penanda Assistant terakhir dalam format HH-RLHF.

```python
def _split_hh(text: str) -> tuple[str, str] | None:
```

Keluaran eksplisit: `None`; `(prompt, response)`.

### `from_hh` (baris 37)

Memuat sumber preferensi lalu membentuk prompt/chosen/rejected, melewati record kosong atau identik.

```python
def from_hh(max_n: int, split: str) -> list[dict]:
```

Keluaran eksplisit: `out`.

### `from_ultrafeedback` (baris 56)

Memuat sumber preferensi lalu membentuk prompt/chosen/rejected, melewati record kosong atau identik.

```python
def from_ultrafeedback(max_n: int, split: str) -> list[dict]:
```

Keluaran eksplisit: `out`.

### `collect` (baris 74)

Menggabungkan sumber preferensi yang dipilih untuk satu split.

```python
def collect(source: str, max_n: int, split: str) -> list[dict]:
```

Keluaran eksplisit: `rows`.

### `write_jsonl` (baris 88)

Menulis satu dictionary sebagai satu baris JSON untuk setiap record.

```python
def write_jsonl(rows: list[dict], path: str) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `main` (baris 96)

Mengumpulkan preferensi train/test dan menulis JSONL.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `_ASSISTANT_MARKER` · 22 | Penanda Assistant: untuk memisahkan respons terakhir HH-RLHF. | `'\n\nAssistant:'` |

### Lingkup `_split_hh`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 25 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |
| `idx` · 27 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `text.rfind(_ASSISTANT_MARKER)` |
| `prompt` · 30 | Teks konteks dan jawaban; response pada download adalah respons HTTP. | `text[:idx].strip()` |
| `response` · 31 | Teks konteks dan jawaban; response pada download adalah respons HTTP. | `text[idx + len(_ASSISTANT_MARKER):].strip()` |

### Lingkup `from_hh`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `max_n` · 37 | Batas jumlah contoh dataset yang dimuat/diproses. | `argumen int` |
| `split` · 37 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `argumen str` |
| `ds` · 40, 42 | Dataset hasil load_dataset, dapat dipangkas dengan select. | `load_dataset('Anthropic/hh-rlhf', split=split)`; `ds.select(range(min(max_n, len(ds))))` |
| `out` · 43 | Daftar record prompt/chosen/rejected yang lolos pemeriksaan. | `[]` |
| `ex` · 44 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `iterasi ds` |
| `c` · 45 | Tuple konteks dan jawaban chosen dari _split_hh. | `_split_hh(ex['chosen'])` |
| `r` · 45 | Tuple konteks dan jawaban rejected dari _split_hh. | `_split_hh(ex['rejected'])` |
| `prompt` · 48 | Teks konteks dan jawaban; response pada download adalah respons HTTP. | `elemen 0 dari c` |
| `chosen` · 48 | Jawaban lebih disukai/kurang disukai dalam satu pasangan preferensi. | `elemen 1 dari c` |
| `_` · 49 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 0 dari r` |
| `rejected` · 49 | Jawaban lebih disukai/kurang disukai dalam satu pasangan preferensi. | `elemen 1 dari r` |

### Lingkup `from_ultrafeedback`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `max_n` · 56 | Batas jumlah contoh dataset yang dimuat/diproses. | `argumen int` |
| `split` · 56 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `argumen str` |
| `hf_split` · 59 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `'train_prefs' if split == 'train' else 'test_prefs'` |
| `ds` · 60, 62 | Dataset hasil load_dataset, dapat dipangkas dengan select. | `load_dataset('HuggingFaceH4/ultrafeedback_binarized', split=hf_split)`; `ds.select(range(min(max_n, len(ds))))` |
| `out` · 63 | Daftar pasangan preferensi UltraFeedback yang siap ditulis. | `[]` |
| `ex` · 64 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `iterasi ds` |
| `prompt` · 65 | Teks konteks dan jawaban; response pada download adalah respons HTTP. | `ex['prompt'].strip()` |
| `chosen` · 66 | Jawaban lebih disukai/kurang disukai dalam satu pasangan preferensi. | `ex['chosen'][-1]['content'].strip()` |
| `rejected` · 67 | Jawaban lebih disukai/kurang disukai dalam satu pasangan preferensi. | `ex['rejected'][-1]['content'].strip()` |

### Lingkup `collect`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `source` · 74 | Pemilih sumber preferensi hh, ultrafeedback, atau both. | `argumen str` |
| `max_n` · 74 | Batas jumlah contoh dataset yang dimuat/diproses. | `argumen int` |
| `split` · 74 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `argumen str` |
| `rows` · 75, 78, 82 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `[]`; `rows += from_hh(max_n, split)`; `rows += from_ultrafeedback(max_n, split)` |
| `e` · 83 | Exception yang ditangkap untuk pelaporan kegagalan. | `exception Exception` |

### Lingkup `write_jsonl`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rows` · 88 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen list[dict]` |
| `path` · 88 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `f` · 90 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(path, 'w')` |
| `r` · 91 | Satu record dari rows, dengan field sesuai dataset atau hasil evaluasi fungsi ini. | `iterasi rows` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `p` · 97 | Parser opsi terminal yang sedang ditambahkan field atau dibaca hasilnya. | `argparse.ArgumentParser()` |
| `args` · 101 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `p.parse_args()` |
| `train` · 103 | Banyak contoh dev serta koleksi train/test hasil pemisahan atau pengumpulan. | `collect(args.source, args.max_per_source, 'train')` |
| `test` · 104 | Banyak contoh dev serta koleksi train/test hasil pemisahan atau pengumpulan. | `collect(args.source, max(2000, args.max_per_source // 20), 'test')` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 14 |
| `argparse` | `argparse` | 16 |
| `json` | `json` | 17 |
| `os` | `os` | 18 |
| `load_dataset` | `datasets.load_dataset` | 38 |
| `load_dataset` | `datasets.load_dataset` | 57 |
| `random` | `random` | 105 |
