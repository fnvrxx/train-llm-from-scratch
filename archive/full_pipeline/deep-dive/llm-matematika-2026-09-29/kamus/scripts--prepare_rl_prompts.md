# scripts/prepare_rl_prompts.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/prepare_rl_prompts.py)

## Fungsi dan kelas

### `write_jsonl` (baris 24)

Menulis satu dictionary sebagai satu baris JSON untuk setiap record.

```python
def write_jsonl(rows, path):
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `gsm8k_prompts` (baris 32)

Membentuk record prompt dan gold numerik dari GSM8K.

```python
def gsm8k_prompts(split: str, limit: int | None):
```

Keluaran eksplisit: `rows`.

### `arithmetic_prompts` (baris 46)

Membuat soal tambah/kurang/kali dengan operand acak beserta gold hasil operasinya.

```python
def arithmetic_prompts(n: int, max_val: int, seed: int):
```

Keluaran eksplisit: `rows`.

### `main` (baris 57)

Menyiapkan prompt GSM8K dan aritmetika curriculum beserta kunci angka.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `write_jsonl`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rows` · 24 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen tanpa anotasi tipe` |
| `path` · 24 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen tanpa anotasi tipe` |
| `f` · 26 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(path, 'w')` |
| `r` · 27 | Satu record dari rows, dengan field sesuai dataset atau hasil evaluasi fungsi ini. | `iterasi rows` |

### Lingkup `gsm8k_prompts`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `split` · 32 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `argumen str` |
| `limit` · 32 | Batas jumlah contoh dataset yang dimuat/diproses. | `argumen int &#124; None` |
| `ds` · 35, 37 | Dataset hasil load_dataset, dapat dipangkas dengan select. | `load_dataset('openai/gsm8k', 'main', split=split)`; `ds.select(range(min(limit, len(ds))))` |
| `rows` · 38 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `[]` |
| `ex` · 39 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `iterasi ds` |
| `gold` · 40 | Kunci jawaban numerik satu soal atau daftar kunci sejajar dengan prompt. | `gsm8k_gold_answer(ex['answer'])` |

### Lingkup `arithmetic_prompts`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `n` · 46 | Jumlah soal aritmetika yang dibuat. | `argumen int` |
| `max_val` · 46 | Nilai maksimum operand pada soal aritmetika buatan. | `argumen int` |
| `seed` · 46 | Benih generator acak untuk inisialisasi dan pengacakan yang dapat diulang. | `argumen int` |
| `rng` · 47 | Generator acak untuk permutation, shuffle, atau soal aritmetika. | `random.Random(seed)` |
| `ops` · 48 | Daftar operasi aritmetika, simbol terpilih, dan fungsi penghitung gold. | `[('+', lambda a, b: a + b), ('-', lambda a, b: a - b), ('*', lambda a, b: a * b)]` |
| `rows` · 49 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `[]` |
| `_` · 50 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `iterasi range(n)` |
| `a` · 51 | Operand pertama yang diambil acak antara nol dan max_val. | `elemen 0 dari (rng.randint(0, max_val), rng.randint(0, max_val))` |
| `b` · 51 | Operand kedua pada rentang yang sama. | `elemen 1 dari (rng.randint(0, max_val), rng.randint(0, max_val))` |
| `sym` · 52 | Daftar operasi aritmetika, simbol terpilih, dan fungsi penghitung gold. | `elemen 0 dari rng.choice(ops)` |
| `fn` · 52 | Daftar operasi aritmetika, simbol terpilih, dan fungsi penghitung gold. | `elemen 1 dari rng.choice(ops)` |

### Lingkup `arithmetic_prompts.lambda@48`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `a` · 48 | Operand pertama fungsi aritmetika. | `argumen lambda: a + b`; `argumen lambda: a - b`; `argumen lambda: a * b` |
| `b` · 48 | Operand kedua fungsi aritmetika; lambda menghitung tambah, kurang, atau kali. | `argumen lambda: a + b`; `argumen lambda: a - b`; `argumen lambda: a * b` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `p` · 58 | Parser opsi terminal yang sedang ditambahkan field atau dibaca hasilnya. | `argparse.ArgumentParser()` |
| `args` · 64 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `p.parse_args()` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 12 |
| `argparse` | `argparse` | 14 |
| `json` | `json` | 15 |
| `os` | `os` | 16 |
| `random` | `random` | 17 |
| `gsm8k_gold_answer` | `src.post_training.rewards.gsm8k_gold_answer` | 21 |
| `load_dataset` | `datasets.load_dataset` | 33 |
