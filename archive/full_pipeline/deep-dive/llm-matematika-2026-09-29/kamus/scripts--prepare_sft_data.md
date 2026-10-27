# scripts/prepare_sft_data.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/prepare_sft_data.py)

## Fungsi dan kelas

### `gsm8k_to_messages` (baris 34)

Membersihkan anotasi kalkulator dan membuat pesan assistant bertag think/answer dari soal GSM8K.

```python
def gsm8k_to_messages(question: str, answer: str) -> list[dict]:
```

Keluaran eksplisit: `[{'role': 'user', 'content': question.strip()}, {'role': 'assistant', 'content': completion}]`.

### `alpaca_to_messages` (baris 46)

Mengubah instruksi/konteks dan respons dataset menjadi pesan user/assistant.

```python
def alpaca_to_messages(ex: dict) -> list[dict]:
```

Keluaran eksplisit: `[{'role': 'user', 'content': user}, {'role': 'assistant', 'content': ex['output'].strip()}]`.

### `dolly_to_messages` (baris 53)

Mengubah instruksi/konteks dan respons dataset menjadi pesan user/assistant.

```python
def dolly_to_messages(ex: dict) -> list[dict]:
```

Keluaran eksplisit: `[{'role': 'user', 'content': user}, {'role': 'assistant', 'content': ex['response'].strip()}]`.

### `collect_examples` (baris 60)

Memuat sumber SFT dan mengumpulkan contoh yang muat konteks serta mempunyai target assistant.

```python
def collect_examples(context_length: int, limit_per_set: int | None) -> list[tuple[list[int], list[int]]]:
```

Keluaran eksplisit: `examples`.

### `collect_examples.add` (baris 66)

Fungsi lokal penerima satu contoh SFT yang memeriksa panjang/mask dan menghitung kept/skipped.

```python
def add(messages):
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `write_packed` (baris 97)

Memadatkan token/mask ke dua dataset HDF5 dan mengembalikan jumlah baris.

```python
def write_packed(examples, context_length: int, out_path: str) -> int:
```

Keluaran eksplisit: `tokens.shape[0]`.

### `main` (baris 107)

Mengumpulkan demonstrasi, membagi train/dev, dan menulis packed HDF5.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `_CALC_RE` · 30 | Regex angka, tag answer, tag think, penanda ####, atau anotasi kalkulator sesuai nama. | `re.compile('<<[^>]*>>')` |
| `_HASH_RE` · 31 | Regex angka, tag answer, tag think, penanda ####, atau anotasi kalkulator sesuai nama. | `re.compile('####\\s*(.+)\\s*$')` |

### Lingkup `gsm8k_to_messages`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `question` · 34 | Teks soal/jawaban dataset; answer field GSM8K mencakup penjelasan dan jawaban ####. | `argumen str` |
| `answer` · 34, 37 | Teks soal/jawaban dataset; answer field GSM8K mencakup penjelasan dan jawaban ####. | `argumen str`; `_CALC_RE.sub('', answer).strip()` |
| `m` · 38 | Mask versi numerik pada loss/RL; pada chat satu message; pada regex hasil pencocokan pola. | `_HASH_RE.search(answer)` |
| `final` · 39 | Jawaban akhir, teks penjelasan, dan pesan assistant gabungan think/answer. | `m.group(1).strip() if m else answer` |
| `reasoning` · 40 | Jawaban akhir, teks penjelasan, dan pesan assistant gabungan think/answer. | `_HASH_RE.sub('', answer).strip()` |
| `completion` · 41 | Jawaban akhir, teks penjelasan, dan pesan assistant gabungan think/answer. | `f'{THINK_OPEN}{reasoning}{THINK_CLOSE}{ANSWER_OPEN}{final}{ANSWER_CLOSE}'` |

### Lingkup `alpaca_to_messages`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ex` · 46 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen dict` |
| `instr` · 47 | Instruksi, input/konteks tambahan, dan teks user gabungan pada konversi SFT. | `ex['instruction'].strip()` |
| `inp` · 48 | Instruksi, input/konteks tambahan, dan teks user gabungan pada konversi SFT. | `(ex.get('input') or '').strip()` |
| `user` · 49 | Instruksi, input/konteks tambahan, dan teks user gabungan pada konversi SFT. | `f'{instr}\n\n{inp}' if inp else instr` |

### Lingkup `dolly_to_messages`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ex` · 53 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen dict` |
| `instr` · 54 | Instruksi, input/konteks tambahan, dan teks user gabungan pada konversi SFT. | `ex['instruction'].strip()` |
| `ctx` · 55 | Teks konteks tambahan Dolly yang digabung dengan instruksi. | `(ex.get('context') or '').strip()` |
| `user` · 56 | Instruksi beserta konteks jika konteks tidak kosong. | `f'{instr}\n\n{ctx}' if ctx else instr` |

### Lingkup `collect_examples`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `context_length` · 60 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `limit_per_set` · 60 | Batas jumlah contoh dataset yang dimuat/diproses. | `argumen int &#124; None` |
| `examples` · 63 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `[]` |
| `n_kept` · 64 | Jumlah contoh SFT diterima/dilewati karena batas panjang atau mask. | `0` |
| `n_skipped` · 64 | Jumlah contoh SFT diterima/dilewati karena batas panjang atau mask. | `0` |
| `alpaca` · 76 | Dataset publik yang dimuat untuk menyusun contoh SFT. | `load_dataset('tatsu-lab/alpaca', split='train')` |
| `ex` · 77, 83, 90 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `iterasi alpaca.select(range(min(limit_per_set, len(alpaca)))) if limit_per_set else alpaca`; `iterasi dolly.select(range(min(limit_per_set, len(dolly)))) if limit_per_set else dolly`; `iterasi gsm.select(range(min(limit_per_set, len(gsm)))) if limit_per_set else gsm` |
| `dolly` · 82 | Dataset publik yang dimuat untuk menyusun contoh SFT. | `load_dataset('databricks/databricks-dolly-15k', split='train')` |
| `e` · 85 | Exception yang ditangkap untuk pelaporan kegagalan. | `exception Exception` |
| `gsm` · 89 | Dataset publik yang dimuat untuk menyusun contoh SFT. | `load_dataset('openai/gsm8k', 'main', split='train')` |

### Lingkup `collect_examples.add`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `messages` · 66 | Daftar pesan percakapan berupa dictionary role/content. | `argumen tanpa anotasi tipe` |
| `ids` · 68 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari encode_chat(messages)` |
| `mask` · 68 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `elemen 1 dari encode_chat(messages)` |
| `n_kept` · 71 | Jumlah contoh SFT diterima/dilewati karena batas panjang atau mask. | `n_kept += 1` |
| `n_skipped` · 73 | Jumlah contoh SFT diterima/dilewati karena batas panjang atau mask. | `n_skipped += 1` |

### Lingkup `write_packed`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `examples` · 97 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen tanpa anotasi tipe` |
| `context_length` · 97 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `out_path` · 97 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `tokens` · 98 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari pack_examples(examples, context_length)` |
| `masks` · 98 | Array dataset/token/mask yang dibaca atau ditulis di HDF5. | `elemen 1 dari pack_examples(examples, context_length)` |
| `f` · 100 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `h5py.File(out_path, 'w')` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `p` · 108 | Parser opsi terminal yang sedang ditambahkan field atau dibaca hasilnya. | `argparse.ArgumentParser()` |
| `args` · 114 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `p.parse_args()` |
| `examples` · 116 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `collect_examples(args.context_length, args.limit_per_set)` |
| `rng` · 117 | Generator acak untuk permutation, shuffle, atau soal aritmetika. | `np.random.default_rng(args.seed)` |
| `n_dev` · 119 | Banyak contoh dev serta koleksi train/test hasil pemisahan atau pengumpulan. | `max(1, int(len(examples) * args.dev_frac))` |
| `dev` · 120 | Device pada setup DDP; pada trainer/evaluasi dapat berarti data atau loss validasi. | `elemen 0 dari (examples[:n_dev], examples[n_dev:])` |
| `train` · 120 | Banyak contoh dev serta koleksi train/test hasil pemisahan atau pengumpulan. | `elemen 1 dari (examples[:n_dev], examples[n_dev:])` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 17 |
| `argparse` | `argparse` | 19 |
| `os` | `os` | 20 |
| `re` | `re` | 21 |
| `h5py` | `h5py` | 23 |
| `np` | `numpy` | 24 |
| `encode_chat` | `src.post_training.chat_template.encode_chat` | 26 |
| `ANSWER_OPEN` | `src.post_training.chat_template.ANSWER_OPEN` | 26 |
| `ANSWER_CLOSE` | `src.post_training.chat_template.ANSWER_CLOSE` | 26 |
| `THINK_OPEN` | `src.post_training.chat_template.THINK_OPEN` | 26 |
| `THINK_CLOSE` | `src.post_training.chat_template.THINK_CLOSE` | 26 |
| `pack_examples` | `src.post_training.sft.pack_examples` | 27 |
| `load_dataset` | `datasets.load_dataset` | 61 |
