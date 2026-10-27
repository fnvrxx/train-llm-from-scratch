# data_loader/prompt_dataset.py

[Indeks kamus](README.md) · [Kode sumber](../../../data_loader/prompt_dataset.py)

## Fungsi dan kelas

### `load_prompt_rows` (baris 15)

Membaca setiap baris JSONL tidak kosong menjadi record prompt/gold.

```python
def load_prompt_rows(path: str) -> list[dict]:
```

Keluaran eksplisit: `[json.loads(line) for line in f if line.strip()]`.

### `get_prompt_iterator` (baris 20)

Mengeluarkan kelompok prompt dari shard rank berulang, dengan shuffle opsional.

```python
def get_prompt_iterator(path: str, prompts_per_iter: int, *, rank: int=0, world_size: int=1, seed: int=0, shuffle: bool=True) -> Iterator[list[dict]]:
```

Keluaran eksplisit: `[rows[i] for i in idx[s:s + prompts_per_iter]]`.

## Variabel dan parameter menurut lingkup

### Lingkup `load_prompt_rows`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 15 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `f` · 16 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(path)` |
| `line` · 17 | Satu baris teks JSONL. | `iterasi f` |

### Lingkup `get_prompt_iterator`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 21 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `prompts_per_iter` · 22 | Banyak prompt unik per iterasi RL per rank. | `argumen int` |
| `rank` · 24 | Identitas proses global, dimulai dari nol. | `argumen int; default 0` |
| `world_size` · 25 | Banyak proses yang bekerja bersama dalam DDP. | `argumen int; default 1` |
| `seed` · 26 | Benih generator acak untuk inisialisasi dan pengacakan yang dapat diulang. | `argumen int; default 0` |
| `shuffle` · 27 | Pilihan mengacak urutan data dan pilihan mengulang iterator setelah akhir data. | `argumen bool; default True` |
| `rows` · 30 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `load_prompt_rows(path)[rank::world_size]` |
| `rng` · 31 | Generator acak untuk permutation, shuffle, atau soal aritmetika. | `np.random.default_rng(seed + rank)` |
| `n` · 32 | Banyak record prompt pada shard rank ini. | `len(rows)` |
| `idx` · 34 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `rng.permutation(n) if shuffle else np.arange(n)` |
| `s` · 35 | Offset awal kelompok prompt pada permutasi data. | `iterasi range(0, n - prompts_per_iter + 1, prompts_per_iter)` |
| `i` · 36 | Indeks record prompt yang dipilih. | `iterasi idx[s:s + prompts_per_iter]` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 7 |
| `json` | `json` | 9 |
| `Iterator` | `typing.Iterator` | 10 |
| `np` | `numpy` | 12 |
