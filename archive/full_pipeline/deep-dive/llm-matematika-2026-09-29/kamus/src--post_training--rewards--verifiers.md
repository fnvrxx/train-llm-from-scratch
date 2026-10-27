# src/post_training/rewards/verifiers.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/rewards/verifiers.py)

## Fungsi dan kelas

### `_answers_match` (baris 24)

Membandingkan angka dengan toleransi absolut FLOAT_TOL; None tidak pernah cocok.

```python
def _answers_match(pred: float | None, gold: float | None) -> bool:
```

Keluaran eksplisit: `False`; `math.isclose(pred, gold, rel_tol=0.0, abs_tol=FLOAT_TOL)`.

### `is_correct` (baris 30)

Mengekstrak angka prediksi lalu membandingkannya dengan gold.

```python
def is_correct(text: str, gold: float | None) -> bool:
```

Keluaran eksplisit: `_answers_match(extract_answer(text), gold)`.

### `reward_gsm8k` (baris 35)

Menjumlahkan bonus benar dan format, dibatasi REWARD_CLIP.

```python
def reward_gsm8k(text: str, gold: float | None) -> float:
```

Keluaran eksplisit: `min(r, REWARD_CLIP)`.

### `reward_format` (baris 49)

Memberi 1 jika ada tepat satu blok answer, selain itu 0.

```python
def reward_format(text: str) -> float:
```

Keluaran eksplisit: `1.0 if has_well_formed_answer(text) else 0.0`.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `CORRECT_BONUS` · 18 | Tambahan reward 1.0 ketika angka jawaban cocok dengan kunci. | `1.0` |
| `FORMAT_BONUS` · 19 | Tambahan reward 0.2 jika tepat satu blok answer ditemukan. | `0.2` |
| `REWARD_CLIP` · 20 | Batas atas skor gabungan verifier, 1.2. | `1.2` |
| `FLOAT_TOL` · 21 | Toleransi absolut 1e-4 untuk perbandingan angka; toleransi relatif nol. | `0.0001` |
| `reward_arithmetic` · 46 | Alias reward_gsm8k karena pemeriksaan jawaban aritmetika memakai logika sama. | `reward_gsm8k` |

### Lingkup `_answers_match`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `pred` · 24 | Angka prediksi yang dibandingkan dengan gold. | `argumen float &#124; None` |
| `gold` · 24 | Kunci jawaban numerik satu soal atau daftar kunci sejajar dengan prompt. | `argumen float &#124; None` |

### Lingkup `is_correct`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 30 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |
| `gold` · 30 | Kunci jawaban numerik satu soal atau daftar kunci sejajar dengan prompt. | `argumen float &#124; None` |

### Lingkup `reward_gsm8k`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 35 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |
| `gold` · 35 | Kunci jawaban numerik satu soal atau daftar kunci sejajar dengan prompt. | `argumen float &#124; None` |
| `r` · 37, 39, 41 | Skor yang dimulai dari nol lalu ditambah bonus benar dan bonus format. | `0.0`; `r += CORRECT_BONUS`; `r += FORMAT_BONUS` |

### Lingkup `reward_format`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 49 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 12 |
| `math` | `math` | 14 |
| `extract_answer` | `src.post_training.rewards.parsing.extract_answer` | 16 |
| `has_well_formed_answer` | `src.post_training.rewards.parsing.has_well_formed_answer` | 16 |
