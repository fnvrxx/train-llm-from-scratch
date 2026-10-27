# data_loader/data_loader.py

[Indeks kamus](README.md) · [Kode sumber](../../../data_loader/data_loader.py)

## Fungsi dan kelas

### `get_batch_iterator` (baris 6)

Mengeluarkan batch jendela corpus beserta target bergeser satu posisi secara berulang.

```python
def get_batch_iterator(data_path: str, batch_size: int, context_length: int, device: str='cpu') -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
```

Keluaran eksplisit: `(xb, yb)`.

## Variabel dan parameter menurut lingkup

### Lingkup `get_batch_iterator`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `data_path` · 6 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `argumen str` |
| `batch_size` · 6 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `argumen int` |
| `context_length` · 6 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `device` · 6 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str; default 'cpu'` |
| `hdf5_file` · 20 | Handle berkas dalam with; f dalam parsing konfigurasi dapat menjadi objek field dataclass. | `h5py.File(data_path, 'r')` |
| `dataset` · 23 | Array dataset/token/mask yang dibaca atau ditulis di HDF5. | `hdf5_file['tokens']` |
| `dataset_size` · 26 | Jumlah token pada array HDF5 pretraining satu dimensi. | `dataset.shape[0]` |
| `n_examples` · 29 | Banyak jendela input-target penuh yang tersedia dalam corpus. | `(dataset_size - 1) // context_length` |
| `example_idxs` · 32 | Nomor jendela data yang diacak sebelum batching. | `np.arange(n_examples)` |
| `epochs` · 36, 46 | Nomor putaran data atau jumlah putaran yang diminta. | `0`; `epochs += 1` |
| `counter` · 37, 44, 59 | Posisi pada daftar jendela untuk memilih batch berikutnya. | `0`; `counter += batch_size` |
| `random_indices` · 49 | Offset token awal jendela yang dipilih untuk batch. | `example_idxs[counter:counter + batch_size] * context_length` |
| `random_samples` · 52 | Potongan sepanjang context_length+1 sebelum dipisah menjadi xb dan yb. | `torch.tensor(np.array([dataset[idx:idx + context_length + 1] for idx in random_indices]))` |
| `idx` · 52 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `iterasi random_indices` |
| `xb` · 55 | Batch token input pretraining (B,T). | `random_samples[:, :context_length].to(device)` |
| `yb` · 56 | Token target; yb pretraining sudah bergeser satu token terhadap xb. | `random_samples[:, 1:context_length + 1].to(device)` |

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `dummy_data_path` · 68 | Path HDF5 sintetis untuk demo loader. | `'dummy_data.h5'` |
| `f` · 70 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `h5py.File(dummy_data_path, 'w')` |
| `batch_size` · 73 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `4` |
| `context_length` · 74 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `10` |
| `xb` · 75 | Batch token input pretraining (B,T). | `elemen 0 dari iterasi get_batch_iterator(dummy_data_path, batch_size, context_length)` |
| `yb` · 75 | Token target; yb pretraining sudah bergeser satu token terhadap xb. | `elemen 1 dari iterasi get_batch_iterator(dummy_data_path, batch_size, context_length)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `torch` | `torch` | 1 |
| `np` | `numpy` | 2 |
| `h5py` | `h5py` | 3 |
| `Iterator` | `typing.Iterator` | 4 |
| `Tuple` | `typing.Tuple` | 4 |
| `os` | `os` | 67 |
