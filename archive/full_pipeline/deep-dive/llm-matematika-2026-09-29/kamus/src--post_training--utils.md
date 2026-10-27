# src/post_training/utils.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/utils.py)

## Fungsi dan kelas

### `amp_autocast` (baris 23)

Membuat context autocast bila opsi dan device mendukung, atau context kosong.

```python
def amp_autocast(amp_dtype: str | None, device: str):
```

Keluaran eksplisit: `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`; `contextlib.nullcontext()`.

### `set_seed` (baris 35)

Mengatur seed Python, NumPy, PyTorch, dan CUDA; tidak menjamin semua kernel deterministik.

```python
def set_seed(seed: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `_cfg_get` (baris 45)

Membaca satu field dari dictionary atau atribut objek konfigurasi.

```python
def _cfg_get(cfg: Any, key: str) -> Any:
```

Keluaran eksplisit: `cfg[key]`; `getattr(cfg, key)`.

### `build_model_from_config` (baris 52)

Membuat Transformer baru menurut dimensi arsitektur dalam cfg.

```python
def build_model_from_config(cfg: Any) -> Transformer:
```

Keluaran eksplisit: `Transformer(n_head=_cfg_get(cfg, 'n_head'), n_embed=_cfg_get(cfg, 'n_embed'), context_length=_cfg_get(cfg, 'context_length'), vocab_size=_cfg_get(cfg, 'vocab_size'), N_BLOCKS=_cfg_get(cfg, 'n_blocks'))`.

### `_strip_ddp_prefix` (baris 67)

Menghapus awalan module. pada nama bobot checkpoint DDP.

```python
def _strip_ddp_prefix(state_dict: dict) -> dict:
```

Keluaran eksplisit: `{k.removeprefix('module.'): v for k, v in state_dict.items()}`; `state_dict`.

### `load_backbone_from_ckpt` (baris 74)

Membuat backbone dan memuat bobot dengan nama yang cocok; hanya menghapus prefix module., bukan transformer.

```python
def load_backbone_from_ckpt(cfg: Any, ckpt_path: str, device: str) -> Transformer:
```

Keluaran eksplisit: `model.to(device)`.

### `unwrap` (baris 94)

Mengambil model.module jika ada wrapper DDP; selain itu mengembalikan model.

```python
def unwrap(model: nn.Module) -> nn.Module:
```

Keluaran eksplisit: `model.module if hasattr(model, 'module') else model`.

### `make_frozen_copy` (baris 99)

Membuat deep copy model, mode eval, dan parameter tanpa gradien sebagai reference.

```python
def make_frozen_copy(model: nn.Module, device: str | None=None) -> nn.Module:
```

Keluaran eksplisit: `ref`.

### `masked_mean` (baris 116)

Menghitung sum(values*mask)/max(1,sum(mask)) untuk seluruh posisi.

```python
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `total / count`.

### `masked_mean_per_row` (baris 124)

Menghitung rerata bermask per baris sehingga hasil (B,).

```python
def masked_mean_per_row(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `total / count`.

### `gather_last` (baris 132)

Memilih values pada seq_lengths-1 agar padding kanan tidak ikut menjadi skor akhir.

```python
def gather_last(values: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `values[torch.arange(values.size(0), device=values.device), idx]`.

### `_cfg_to_dict` (baris 144)

Mengubah dataclass menjadi dictionary sebelum save; objek lainnya diteruskan.

```python
def _cfg_to_dict(cfg: Any) -> Any:
```

Keluaran eksplisit: `asdict(cfg) if is_dataclass(cfg) else cfg`.

### `save_stage_ckpt` (baris 148)

Menyimpan bobot, optimizer, konfigurasi, stage, step, dan metrik ke checkpoint.

```python
def save_stage_ckpt(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None, *, stage: str, cfg: Any, step: int, metrics: dict | None=None, extra: dict | None=None) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `amp_autocast`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `amp_dtype` · 23 | Tipe atau nama presisi autocast, biasanya bf16/fp16; None berarti tanpa autocast. | `argumen str &#124; None` |
| `device` · 23 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |

### Lingkup `set_seed`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `seed` · 35 | Benih generator acak untuk inisialisasi dan pengacakan yang dapat diulang. | `argumen int` |

### Lingkup `_cfg_get`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 45 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Any` |
| `key` · 45 | Linear dari n_embed ke head_size untuk membentuk keys; key pada utilitas cfg adalah nama field yang dicari. | `argumen str` |

### Lingkup `build_model_from_config`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 52 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Any` |

### Lingkup `_strip_ddp_prefix`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `state_dict` · 67 | Pemetaan nama parameter/buffer ke tensor bobot. | `argumen dict` |
| `k` · 69, 70 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `iterasi state_dict`; `elemen 0 dari iterasi state_dict.items()` |
| `v` · 70 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi state_dict.items()` |

### Lingkup `load_backbone_from_ckpt`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 74 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Any` |
| `ckpt_path` · 74 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `argumen str` |
| `device` · 74 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `model` · 81 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `build_model_from_config(cfg)` |
| `ckpt` · 82 | Isi berkas checkpoint, termasuk bobot dan metadata bila tersedia. | `torch.load(ckpt_path, map_location='cpu', weights_only=False)` |
| `state` · 83, 84 | Pemetaan nama parameter/buffer ke tensor bobot. | `ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt`; `_strip_ddp_prefix(state)` |
| `backbone_keys` · 86 | Himpunan nama parameter yang diterima model atau kumpulan kunci objek yang diperiksa. | `set(model.state_dict().keys())` |
| `filtered` · 87 | Bobot yang lolos pemilihan nama sebelum dimuat ke model. | `{k: v for k, v in state.items() if k in backbone_keys}` |
| `k` · 87 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `elemen 0 dari iterasi state.items()` |
| `v` · 87 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi state.items()` |
| `missing` · 88 | Daftar nama bobot yang dibutuhkan model tetapi tidak ditemukan saat loading. | `elemen 0 dari model.load_state_dict(filtered, strict=False)` |
| `unexpected` · 88 | Daftar nama bobot checkpoint yang tidak diharapkan model. | `elemen 1 dari model.load_state_dict(filtered, strict=False)` |

### Lingkup `unwrap`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 94 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen nn.Module` |

### Lingkup `make_frozen_copy`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 99 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen nn.Module` |
| `device` · 99 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str &#124; None; default None` |
| `ref` · 105, 107 | Salinan policy/reference yang dibekukan sebagai pembanding. | `copy.deepcopy(unwrap(model))`; `ref.to(device)` |
| `p` · 109 | Satu tensor parameter model yang dihitung jumlah elemennya, dikelompokkan optimizer, atau dibekukan sesuai fungsi. | `iterasi ref.parameters()` |

### Lingkup `masked_mean`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `values` · 116 | Estimasi value pada state baru/lama/berikutnya di PPO; values pada utilitas adalah tensor angka umum. | `argumen torch.Tensor` |
| `mask` · 116, 118 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen torch.Tensor`; `mask.to(values.dtype)` |
| `total` · 119 | Jumlah seluruh values*mask; numerator satu skalar. | `(values * mask).sum()` |
| `count` · 120 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `mask.sum().clamp(min=1.0)` |

### Lingkup `masked_mean_per_row`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `values` · 124 | Estimasi value pada state baru/lama/berikutnya di PPO; values pada utilitas adalah tensor angka umum. | `argumen torch.Tensor` |
| `mask` · 124, 126 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen torch.Tensor`; `mask.to(values.dtype)` |
| `total` · 127 | Jumlah values*mask per baris, bentuk (B,). | `(values * mask).sum(dim=-1)` |
| `count` · 128 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `mask.sum(dim=-1).clamp(min=1.0)` |

### Lingkup `gather_last`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `values` · 132 | Estimasi value pada state baru/lama/berikutnya di PPO; values pada utilitas adalah tensor angka umum. | `argumen torch.Tensor` |
| `seq_lengths` · 132 | Banyak token nyata setiap urutan, sebelum right padding. | `argumen torch.Tensor` |
| `idx` · 138 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `(seq_lengths - 1).clamp(min=0).long()` |

### Lingkup `_cfg_to_dict`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 144 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Any` |

### Lingkup `save_stage_ckpt`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 149 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `model` · 150 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen nn.Module` |
| `optimizer` · 151 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `argumen torch.optim.Optimizer &#124; None` |
| `stage` · 153 | Nama tahap seperti pretrain/sft/dpo/ppo/grpo untuk log dan checkpoint. | `argumen str` |
| `cfg` · 154 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Any` |
| `step` · 155 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `argumen int` |
| `metrics` · 156 | Dictionary metrik; record menambahkan step dan waktu pencatatan. | `argumen dict &#124; None; default None` |
| `extra` · 157 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `argumen dict &#124; None; default None` |
| `payload` · 166 | Dictionary bobot, optimizer, konfigurasi, step, dan metrik yang akan disimpan. | `{'model_state_dict': unwrap(model).state_dict(), 'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None, 'stage': stage, 'cfg': _cfg_to_dict(cfg), 'step': step, 'metrics': metrics or {}, 'pytorch_version': torch.__version__, 'cuda_version': torch.version.cuda}` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 7 |
| `contextlib` | `contextlib` | 9 |
| `copy` | `copy` | 10 |
| `os` | `os` | 11 |
| `random` | `random` | 12 |
| `asdict` | `dataclasses.asdict` | 13 |
| `is_dataclass` | `dataclasses.is_dataclass` | 13 |
| `Any` | `typing.Any` | 14 |
| `np` | `numpy` | 16 |
| `torch` | `torch` | 17 |
| `nn` | `torch.nn` | 18 |
| `Transformer` | `src.models.transformer.Transformer` | 20 |
