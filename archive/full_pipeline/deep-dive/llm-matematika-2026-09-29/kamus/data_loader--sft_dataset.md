# data_loader/sft_dataset.py

[Indeks kamus](README.md) · [Kode sumber](../../../data_loader/sft_dataset.py)

## Fungsi dan kelas

### `get_sft_batch_iterator` (baris 18)

Mengeluarkan tokens, loss mask, dan epoch dari packed HDF5 yang dibagi antar rank.

```python
def get_sft_batch_iterator(data_path: str, batch_size: int, device: str='cpu', *, rank: int=0, world_size: int=1, shuffle: bool=True, infinite: bool=True) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
```

Keluaran eksplisit: `(tk, mk, epoch)`.

## Variabel dan parameter menurut lingkup

### Lingkup `get_sft_batch_iterator`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `data_path` · 19 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `argumen str` |
| `batch_size` · 20 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `argumen int` |
| `device` · 21 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str; default 'cpu'` |
| `rank` · 23 | Identitas proses global, dimulai dari nol. | `argumen int; default 0` |
| `world_size` · 24 | Banyak proses yang bekerja bersama dalam DDP. | `argumen int; default 1` |
| `shuffle` · 25 | Pilihan mengacak urutan data dan pilihan mengulang iterator setelah akhir data. | `argumen bool; default True` |
| `infinite` · 26 | Pilihan mengacak urutan data dan pilihan mengulang iterator setelah akhir data. | `argumen bool; default True` |
| `f` · 35 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `h5py.File(data_path, 'r')` |
| `tokens` · 36 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `f['tokens']` |
| `masks` · 37 | Array dataset/token/mask yang dibaca atau ditulis di HDF5. | `f['loss_mask']` |
| `n` · 38 | Jumlah baris packed HDF5 sebelum pembagian rank. | `tokens.shape[0]` |
| `idxs` · 39 | Urutan indeks pemilihan/pengacakan record atau baris. | `np.arange(rank, n, world_size)` |
| `epoch` · 40, 50 | Nomor putaran data atau jumlah putaran yang diminta. | `0`; `epoch += 1` |
| `rng` · 41 | Generator acak untuk permutation, shuffle, atau soal aritmetika. | `np.random.default_rng(1234 + rank)` |
| `start` · 45 | Batas awal/akhir potongan array atau penulisan dataset. | `iterasi range(0, len(idxs) - batch_size + 1, batch_size)` |
| `batch` · 46 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `np.sort(idxs[start:start + batch_size])` |
| `tk` · 47 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `torch.tensor(np.asarray(tokens[batch]), dtype=torch.long, device=device)` |
| `mk` · 48 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `torch.tensor(np.asarray(masks[batch]), dtype=torch.long, device=device)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 9 |
| `Iterator` | `typing.Iterator` | 11 |
| `h5py` | `h5py` | 13 |
| `np` | `numpy` | 14 |
| `torch` | `torch` | 15 |
