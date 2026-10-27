# data_loader/preference_dataset.py

[Indeks kamus](README.md) · [Kode sumber](../../../data_loader/preference_dataset.py)

## Fungsi dan kelas

### `_encode_side` (baris 27)

Membentuk token/mask user+assistant untuk satu sisi preferensi, dipotong max_len.

```python
def _encode_side(prompt: str, response: str, max_len: int) -> tuple[list[int], list[int]]:
```

Keluaran eksplisit: `(ids[:max_len], mask[:max_len])`.

### `_collate` (baris 33)

Membentuk tensor ID, mask, dan panjang nyata chosen/rejected dengan padding lebar bersama.

```python
def _collate(rows: list[dict], max_len: int, device: str) -> dict:
```

Keluaran eksplisit: `{'chosen_ids': t(ch_ids, torch.long), 'chosen_mask': t(ch_mask, torch.long), 'chosen_len': t(ch_len, torch.long), 'rejected_ids': t(rj_ids, torch.long), 'rejected_mask': t(rj_mask, torch.long), 'rejected_len': t(rj_len, torch.long)}`.

### `_collate.pad` (baris 39)

Mengisi kanan urutan dengan fill sampai panjang L.

```python
def pad(seq, fill):
```

Keluaran eksplisit: `seq + [fill] * (L - len(seq))`.

### `get_preference_iterator` (baris 55)

Membaca JSONL preferensi, membagi rank, mengacak, dan mengeluarkan batch lengkap.

```python
def get_preference_iterator(path: str, batch_size: int, max_len: int, device: str='cpu', *, rank: int=0, world_size: int=1, shuffle: bool=True, infinite: bool=True) -> Iterator[dict]:
```

Keluaran eksplisit: `_collate(batch, max_len, device)`.

## Variabel dan parameter menurut lingkup

### Lingkup `_encode_side`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `prompt` · 27 | Teks konteks dan jawaban; response pada download adalah respons HTTP. | `argumen str` |
| `response` · 27 | Teks konteks dan jawaban; response pada download adalah respons HTTP. | `argumen str` |
| `max_len` · 27 | Batas panjang token prompt+response setelah encoding dan pemotongan. | `argumen int` |
| `ids` · 28 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari encode_chat([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}])` |
| `mask` · 28 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `elemen 1 dari encode_chat([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}])` |

### Lingkup `_collate`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rows` · 33 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen list[dict]` |
| `max_len` · 33 | Batas panjang token prompt+response setelah encoding dan pemotongan. | `argumen int` |
| `device` · 33 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `enc` · 34 | Tokenizer pada preprocessing; daftar hasil encoding pasangan pada _collate. | `[(_encode_side(r['prompt'], r['chosen'], max_len), _encode_side(r['prompt'], r['rejected'], max_len)) for r in rows]` |
| `r` · 35 | Satu record dari rows, dengan field sesuai dataset atau hasil evaluasi fungsi ini. | `iterasi rows` |
| `L` · 37 | Panjang maksimum kedua sisi seluruh pasangan batch. | `max((max(len(c[0]), len(j[0])) for c, j in enc))` |
| `c` · 37 | Hasil encoding chosen (ids,mask) saat menentukan lebar batch. | `elemen 0 dari iterasi enc` |
| `j` · 37 | Hasil encoding rejected (ids,mask) saat menentukan lebar batch. | `elemen 1 dari iterasi enc` |
| `ch_ids` · 42 | Daftar ID chosen yang sudah di-pad ke lebar batch bersama. | `elemen 0 dari ([], [], [], [], [], [])` |
| `ch_mask` · 42 | Mask target jawaban chosen, nol pada padding. | `elemen 1 dari ([], [], [], [], [], [])` |
| `ch_len` · 42 | Panjang nyata setiap urutan chosen sebelum padding. | `elemen 2 dari ([], [], [], [], [], [])` |
| `rj_ids` · 42 | Daftar ID rejected yang sudah di-pad ke lebar batch bersama. | `elemen 3 dari ([], [], [], [], [], [])` |
| `rj_mask` · 42 | Mask target jawaban rejected, nol pada padding. | `elemen 4 dari ([], [], [], [], [], [])` |
| `rj_len` · 42 | Panjang nyata setiap urutan rejected sebelum padding. | `elemen 5 dari ([], [], [], [], [], [])` |
| `cids` · 43 | ID satu contoh chosen sebelum padding. | `elemen 0 dari elemen 0 dari iterasi enc` |
| `cmask` · 43 | Mask satu contoh chosen sebelum padding. | `elemen 1 dari elemen 0 dari iterasi enc` |
| `jids` · 43 | ID satu contoh rejected sebelum padding. | `elemen 0 dari elemen 1 dari iterasi enc` |
| `jmask` · 43 | Mask satu contoh rejected sebelum padding. | `elemen 1 dari elemen 1 dari iterasi enc` |
| `t` · 48 | Lambda lokal untuk mengubah list menjadi tensor pada dtype/device yang diminta. | `lambda a, dt: torch.tensor(a, dtype=dt, device=device)` |

### Lingkup `_collate.pad`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `seq` · 39 | Urutan untuk padding dan nilai pengisi; seq juga satu potongan rollout. | `argumen tanpa anotasi tipe` |
| `fill` · 39 | Urutan untuk padding dan nilai pengisi; seq juga satu potongan rollout. | `argumen tanpa anotasi tipe` |

### Lingkup `_collate.lambda@48`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `a` · 48 | List angka yang akan diubah menjadi tensor. | `argumen lambda: torch.tensor(a, dtype=dt, device=device)` |
| `dt` · 48 | dtype tensor keluaran, misalnya torch.long. | `argumen lambda: torch.tensor(a, dtype=dt, device=device)` |

### Lingkup `get_preference_iterator`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 56 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `batch_size` · 57 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `argumen int` |
| `max_len` · 58 | Batas panjang token prompt+response setelah encoding dan pemotongan. | `argumen int` |
| `device` · 59 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str; default 'cpu'` |
| `rank` · 61 | Identitas proses global, dimulai dari nol. | `argumen int; default 0` |
| `world_size` · 62 | Banyak proses yang bekerja bersama dalam DDP. | `argumen int; default 1` |
| `shuffle` · 63 | Pilihan mengacak urutan data dan pilihan mengulang iterator setelah akhir data. | `argumen bool; default True` |
| `infinite` · 64 | Pilihan mengacak urutan data dan pilihan mengulang iterator setelah akhir data. | `argumen bool; default True` |
| `f` · 67 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(path)` |
| `rows` · 68, 69 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `[json.loads(line) for line in f if line.strip()]`; `rows[rank::world_size]` |
| `line` · 68 | Satu baris teks JSONL. | `iterasi f` |
| `rng` · 70 | Generator acak untuk permutation, shuffle, atau soal aritmetika. | `np.random.default_rng(7 + rank)` |
| `order` · 72 | Urutan indeks pemilihan/pengacakan record atau baris. | `np.arange(len(rows))` |
| `s` · 75 | Offset awal batch pada urutan indeks preferensi. | `iterasi range(0, len(order) - batch_size + 1, batch_size)` |
| `batch` · 76 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `[rows[i] for i in order[s:s + batch_size]]` |
| `i` · 76 | Indeks record yang dipilih dari order untuk batch ini. | `iterasi order[s:s + batch_size]` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 16 |
| `json` | `json` | 18 |
| `Iterator` | `typing.Iterator` | 19 |
| `np` | `numpy` | 21 |
| `torch` | `torch` | 22 |
| `EOT_ID` | `src.post_training.chat_template.EOT_ID` | 24 |
| `encode_chat` | `src.post_training.chat_template.encode_chat` | 24 |
