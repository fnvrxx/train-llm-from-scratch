# src/post_training/distributed.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/distributed.py)

## Fungsi dan kelas

### `DDPContext` (baris 21)

Dataclass identitas proses dan device untuk training terdistribusi.

### `DDPContext.is_main` (baris 28)

Memeriksa rank nol untuk tugas log/save utama.

```python
def is_main(self) -> bool:
```

Keluaran eksplisit: `self.rank == 0`.

### `DDPContext.enabled` (baris 32)

Property yang memeriksa apakah world_size lebih dari satu.

```python
def enabled(self) -> bool:
```

Keluaran eksplisit: `self.world_size > 1`.

### `ddp_setup` (baris 36)

Menyiapkan identitas rank, device, dan komunikasi DDP, atau konteks satu proses.

```python
def ddp_setup(device: str='cuda') -> DDPContext:
```

Keluaran eksplisit: `DDPContext(rank=0, local_rank=0, world_size=1, device=dev)`; `DDPContext(rank=rank, local_rank=local_rank, world_size=world_size, device=dev)`.

### `ddp_wrap` (baris 61)

Membungkus model trainable dalam DistributedDataParallel bila world_size lebih dari satu.

```python
def ddp_wrap(model: torch.nn.Module, ctx: DDPContext, find_unused_parameters: bool=False) -> torch.nn.Module:
```

Keluaran eksplisit: `model`; `DDP(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)`.

### `is_main_process` (baris 78)

Memeriksa rank nol untuk tugas log/save utama.

```python
def is_main_process(ctx: DDPContext) -> bool:
```

Keluaran eksplisit: `ctx.is_main`.

### `reduce_scalar` (baris 82)

Melakukan all-reduce skalar dan opsional merata-ratakan antar rank.

```python
def reduce_scalar(value: float, ctx: DDPContext, average: bool=True) -> float:
```

Keluaran eksplisit: `value`; `t.item()`.

### `barrier` (baris 93)

Menunggu semua proses pada titik sinkronisasi jika DDP aktif.

```python
def barrier(ctx: DDPContext) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `cleanup` (baris 98)

Mengakhiri process group terdistribusi yang aktif.

```python
def cleanup(ctx: DDPContext) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `shard_indices` (baris 103)

Menghasilkan potongan indeks milik satu rank dengan stride world_size.

```python
def shard_indices(num_items: int, ctx: DDPContext) -> range:
```

Keluaran eksplisit: `range(num_items)`; `range(ctx.rank, num_items, ctx.world_size)`.

## Variabel dan parameter menurut lingkup

### Lingkup `DDPContext`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rank` · 22 | Identitas proses global, dimulai dari nol. | `deklarasi tipe int` |
| `local_rank` · 23 | Nomor device atau nomor proses lokal pada satu mesin. | `deklarasi tipe int` |
| `world_size` · 24 | Banyak proses yang bekerja bersama dalam DDP. | `deklarasi tipe int` |
| `device` · 25 | Lokasi komputasi, misalnya cpu atau cuda. | `deklarasi tipe str` |

### Lingkup `DDPContext.is_main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 28 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |

### Lingkup `DDPContext.enabled`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 32 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |

### Lingkup `ddp_setup`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `device` · 36 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str; default 'cuda'` |
| `world_size` · 42 | Banyak proses yang bekerja bersama dalam DDP. | `int(os.environ.get('WORLD_SIZE', '1'))` |
| `dev` · 44, 55, 57 | Device pada setup DDP; pada trainer/evaluasi dapat berarti data atau loss validasi. | `device if device == 'cuda' and torch.cuda.is_available() else 'cpu'`; `f'cuda:{local_rank}'`; `'cpu'` |
| `rank` · 49 | Identitas proses global, dimulai dari nol. | `int(os.environ['RANK'])` |
| `local_rank` · 50 | Nomor device atau nomor proses lokal pada satu mesin. | `int(os.environ['LOCAL_RANK'])` |
| `backend` · 51 | Backend komunikasi DDP, nccl untuk CUDA atau gloo untuk CPU. | `'nccl' if device == 'cuda' and torch.cuda.is_available() else 'gloo'` |

### Lingkup `ddp_wrap`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 61 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen torch.nn.Module` |
| `ctx` · 61 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen DDPContext` |
| `find_unused_parameters` · 61 | Opsi DDP untuk mendeteksi parameter yang tidak mendapat gradien pada forward tertentu. | `argumen bool; default False` |
| `device_ids` · 74 | GPU lokal yang diberikan ke wrapper DDP. | `[ctx.local_rank] if ctx.device.startswith('cuda') else None` |

### Lingkup `is_main_process`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ctx` · 78 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen DDPContext` |

### Lingkup `reduce_scalar`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `value` · 82 | Linear pembentuk values jika atribut Head; pada utilitas konversi/reduksi berupa nilai skalar masukan. | `argumen float` |
| `ctx` · 82 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen DDPContext` |
| `average` · 82 | Pilihan membagi hasil all-reduce menjadi rerata antar proses. | `argumen bool; default True` |
| `t` · 86, 89 | Tensor float32 berisi satu skalar, dijumlahkan antar rank lalu opsional dibagi world_size. | `torch.tensor([value], device=ctx.device, dtype=torch.float32)`; `t /= ctx.world_size` |

### Lingkup `barrier`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ctx` · 93 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen DDPContext` |

### Lingkup `cleanup`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ctx` · 98 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen DDPContext` |

### Lingkup `shard_indices`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `num_items` · 103 | Jumlah item yang dibagi menjadi shard indeks antar rank. | `argumen int` |
| `ctx` · 103 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen DDPContext` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 10 |
| `os` | `os` | 12 |
| `dataclass` | `dataclasses.dataclass` | 13 |
| `torch` | `torch` | 15 |
| `dist` | `torch.distributed` | 16 |
| `DDP` | `torch.nn.parallel.DistributedDataParallel` | 17 |
