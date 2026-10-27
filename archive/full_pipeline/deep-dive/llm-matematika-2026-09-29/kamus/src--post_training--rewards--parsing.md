# src/post_training/rewards/parsing.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/rewards/parsing.py)

## Fungsi dan kelas

### `parse_number` (baris 27)

Mengambil angka pertama sesuai regex, menghapus koma/dolar, lalu mengubahnya menjadi float atau None.

```python
def parse_number(s: str | None) -> float | None:
```

Keluaran eksplisit: `None`; `float(raw)`.

### `extract_think` (baris 41)

Mengambil isi blok think pertama, atau None.

```python
def extract_think(text: str) -> str | None:
```

Keluaran eksplisit: `m.group(1).strip() if m else None`.

### `extract_answer` (baris 47)

Mencari angka dari tag answer, pola ####, kemudian angka terakhir sebagai fallback.

```python
def extract_answer(text: str) -> float | None:
```

Keluaran eksplisit: `n`; `parse_number(nums[-1])`; `None`.

### `gsm8k_gold_answer` (baris 70)

Mengambil kunci numerik dari answer field GSM8K, mengutamakan penanda ####.

```python
def gsm8k_gold_answer(answer_field: str) -> float | None:
```

Keluaran eksplisit: `parse_number(m.group(1))`; `parse_number(answer_field)`.

### `has_well_formed_answer` (baris 78)

Memeriksa tepat satu blok answer; tidak memvalidasi isi reasoning.

```python
def has_well_formed_answer(text: str) -> bool:
```

Keluaran eksplisit: `len(_ANSWER_RE.findall(text)) == 1`.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `_NUMBER_RE` · 21 | Regex angka, tag answer, tag think, penanda ####, atau anotasi kalkulator sesuai nama. | `re.compile('-?\\$?\\d[\\d,]*(?:\\.\\d+)?')` |
| `_ANSWER_RE` · 22 | Regex angka, tag answer, tag think, penanda ####, atau anotasi kalkulator sesuai nama. | `re.compile(re.escape(ANSWER_OPEN) + '(.*?)' + re.escape(ANSWER_CLOSE), re.DOTALL)` |
| `_THINK_RE` · 23 | Regex angka, tag answer, tag think, penanda ####, atau anotasi kalkulator sesuai nama. | `re.compile(re.escape(THINK_OPEN) + '(.*?)' + re.escape(THINK_CLOSE), re.DOTALL)` |
| `_HASH_RE` · 24 | Regex angka, tag answer, tag think, penanda ####, atau anotasi kalkulator sesuai nama. | `re.compile('####\\s*(-?\\$?\\d[\\d,]*(?:\\.\\d+)?)')` |

### Lingkup `parse_number`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `s` · 27 | String yang akan dicari angka pertamanya; None menghasilkan None. | `argumen str &#124; None` |
| `m` · 31 | Hasil pencocokan _NUMBER_RE pada s. | `_NUMBER_RE.search(s)` |
| `raw` · 34 | Teks angka hasil regex setelah koma ribuan dan dolar dihapus. | `m.group(0).replace(',', '').replace('$', '')` |

### Lingkup `extract_think`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 41 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |
| `m` · 43 | Hasil pencarian blok think pertama oleh regex. | `_THINK_RE.search(text)` |

### Lingkup `extract_answer`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 47 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |
| `m` · 54, 59 | Hasil pencarian tag answer, lalu pola #### jika diperlukan. | `_ANSWER_RE.search(text)`; `_HASH_RE.search(text)` |
| `n` · 56, 61 | Angka yang berhasil diambil dari isi tag/penanda, atau None. | `parse_number(m.group(1))` |
| `nums` · 64 | Daftar kecocokan angka untuk fallback jawaban angka terakhir. | `_NUMBER_RE.findall(text)` |

### Lingkup `gsm8k_gold_answer`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `answer_field` · 70 | Teks soal/jawaban dataset; answer field GSM8K mencakup penjelasan dan jawaban ####. | `argumen str` |
| `m` · 72 | Hasil pencarian angka setelah #### pada answer_field. | `_HASH_RE.search(answer_field)` |

### Lingkup `has_well_formed_answer`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 78 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 12 |
| `re` | `re` | 14 |
| `ANSWER_OPEN` | `src.post_training.chat_template.ANSWER_OPEN` | 16 |
| `ANSWER_CLOSE` | `src.post_training.chat_template.ANSWER_CLOSE` | 16 |
| `THINK_OPEN` | `src.post_training.chat_template.THINK_OPEN` | 16 |
| `THINK_CLOSE` | `src.post_training.chat_template.THINK_CLOSE` | 16 |
