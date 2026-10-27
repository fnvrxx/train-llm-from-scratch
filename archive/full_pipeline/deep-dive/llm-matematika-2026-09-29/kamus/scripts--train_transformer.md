# scripts/train_transformer.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/train_transformer.py)

## Fungsi dan kelas

### `bytes_to_gib` (baris 27)

Mengonversi byte menjadi GiB dengan membagi 1024 pangkat tiga.

```python
def bytes_to_gib(num_bytes: int) -> float:
```

Keluaran eksplisit: `num_bytes / 1024 ** 3`.

### `get_device_report` (baris 32)

Membentuk teks versi PyTorch/CUDA, device, dan properti GPU.

```python
def get_device_report(device: str) -> str:
```

Keluaran eksplisit: `'\n'.join(lines)`.

### `get_peak_memory_report` (baris 60)

Melaporkan puncak memori CUDA allocated/reserved atau N/A pada CPU.

```python
def get_peak_memory_report(device: str) -> str:
```

Keluaran eksplisit: `f'Peak VRAM allocated: {peak_allocated:.2f} GiB &#124; Peak VRAM reserved: {peak_reserved:.2f} GiB'`; `'Peak VRAM allocated: N/A &#124; Peak VRAM reserved: N/A'`.

### `estimate_memory_budget` (baris 72)

Memperkirakan memori state 16 byte per parameter; aktivasi belum dihitung dan use_amp tidak memengaruhi rumus.

```python
def estimate_memory_budget(num_params: int, device: str, use_amp: bool) -> str:
```

Keluaran eksplisit: `'VRAM budget: N/A (no CUDA device)'`; `f'VRAM budget: ~{state_gib:.2f} GiB params+optimizer state vs {total_gib:.2f} GiB total on {torch.cuda.get_device_name()}{note}'`.

### `load_checkpoint_file` (baris 99)

Memuat checkpoint dengan fallback argumen torch.load versi lama.

```python
def load_checkpoint_file(path: str, device: str) -> Dict[str, Any]:
```

Keluaran eksplisit: `torch.load(path, map_location=torch.device(device), weights_only=False)`; `torch.load(path, map_location=torch.device(device))`.

### `default_checkpoint_dir` (baris 107)

Menyusun direktori checkpoint dari nama output tanpa ekstensi dan suffix _checkpoints.

```python
def default_checkpoint_dir(out_path: str) -> str:
```

Keluaran eksplisit: `str(model_path.with_suffix('')) + '_checkpoints'`.

### `checkpoint_path` (baris 113)

Menyusun path checkpoint dengan nomor step delapan digit.

```python
def checkpoint_path(checkpoint_dir: str, step: int) -> str:
```

Keluaran eksplisit: `os.path.join(checkpoint_dir, f'checkpoint_step_{step:08d}.pt')`.

### `checkpoint_step` (baris 118)

Membaca step dari nama checkpoint; -1 jika pola tidak cocok.

```python
def checkpoint_step(path: str) -> int:
```

Keluaran eksplisit: `-1`; `int(match.group(1))`.

### `list_checkpoints` (baris 126)

Mengumpulkan checkpoint periodik dan mengurutkannya menurut step.

```python
def list_checkpoints(checkpoint_dir: str) -> List[str]:
```

Keluaran eksplisit: `[]`; `sorted(paths, key=checkpoint_step)`.

### `resolve_resume_path` (baris 138)

Memilih path resume eksplisit/latest/None; latest gagal jika tidak ada checkpoint.

```python
def resolve_resume_path(resume: Optional[str], checkpoint_dir: str) -> Optional[str]:
```

Keluaran eksplisit: `None`; `checkpoints[-1]`; `resume`.

### `current_lr` (baris 155)

Membaca learning rate kelompok optimizer pertama.

```python
def current_lr(optimizer: torch.optim.Optimizer) -> float:
```

Keluaran eksplisit: `float(optimizer.param_groups[0]['lr'])`.

### `lr_for_step` (baris 160)

Memilih learning rate awal atau decayed berdasarkan step trainer awal.

```python
def lr_for_step(train_config: Dict[str, Any], step: int) -> float:
```

Keluaran eksplisit: `float(train_config['t_lr_decayed'])`; `float(train_config['t_lr'])`.

### `set_optimizer_lr` (baris 167)

Mengganti learning rate semua kelompok optimizer.

```python
def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `save_training_checkpoint` (baris 173)

Menyimpan bobot, optimizer, history, dan step ke file sementara lalu rename atomik.

```python
def save_training_checkpoint(path: str, model: Transformer, optimizer: torch.optim.Optimizer, train_config: Dict[str, Any], losses: List[float], *, step: int, train_loss: Optional[float]=None, dev_loss: Optional[float]=None, is_final: bool=False) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `restore_training_checkpoint` (baris 230)

Memulihkan model/optimizer dan mengembalikan step berikutnya serta history loss.

```python
def restore_training_checkpoint(path: str, model: Transformer, optimizer: torch.optim.Optimizer, train_config: Dict[str, Any], device: str) -> Tuple[int, List[float]]:
```

Keluaran eksplisit: `(next_step, losses)`.

### `prune_old_checkpoints` (baris 268)

Menghapus checkpoint periodik lama sesuai keep_last jika nilainya positif.

```python
def prune_old_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `unique_output_path` (baris 277)

Menambahkan suffix angka sampai nama output belum dipakai.

```python
def unique_output_path(out_path: str) -> str:
```

Keluaran eksplisit: `modified_model_out_path`.

### `as_float` (baris 288)

Mengubah tensor skalar/angka ke float Python, dengan None tetap None.

```python
def as_float(value: Any) -> Optional[float]:
```

Keluaran eksplisit: `None`; `float(value.item())`; `float(value)`.

### `estimate_loss` (baris 300)

Mengukur rerata loss token train/dev dalam eval tanpa backward, lalu mengembalikan mode train.

```python
def estimate_loss(model: Transformer, train_config: Dict[str, Any], steps: int) -> Dict[str, float]:
```

Keluaran eksplisit: `out`.

### `parse_args` (baris 348)

Membaca opsi CLI trainer awal untuk resume, checkpoint, dan pengaturan memori.

```python
def parse_args() -> argparse.Namespace:
```

Keluaran eksplisit: `parser.parse_args()`.

### `main` (baris 416)

Menjalankan trainer awal beserta resume, opsi memori, save periodik, dan evaluasi final.

```python
def main() -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `main.autocast_ctx` (baris 455)

Membuat context autocast bila opsi dan device mendukung, atau context kosong.

```python
def autocast_ctx():
```

Keluaran eksplisit: `torch.autocast(device_type='cuda', dtype=amp_dtype)`; `contextlib.nullcontext()`.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ROOT_DIR` · 17 | Akar proyek yang ditambahkan ke sys.path trainer awal. | `Path(__file__).resolve().parents[1]` |
| `CHECKPOINT_RE` · 96 | Regex nama checkpoint_step_<angka>.pt untuk mengambil nomor step. | `re.compile('checkpoint_step_(\\d+)\\.pt$')` |

### Lingkup `bytes_to_gib`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `num_bytes` · 27 | Ukuran memori dalam byte sebelum dikonversi menjadi GiB. | `argumen int` |

### Lingkup `get_device_report`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `device` · 32 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `lines` · 38 | Bagian teks laporan diagnostik runtime atau memori. | `[f'PyTorch version: {torch.__version__}', f'Configured device: {device}', f'CUDA available: {torch.cuda.is_available()}', f'CUDA version: {torch.version.cuda}']` |
| `device_index` · 46 | Nomor device atau nomor proses lokal pada satu mesin. | `torch.cuda.current_device()` |
| `props` · 47 | Properti GPU, termasuk memori dan compute capability. | `torch.cuda.get_device_properties(device_index)` |
| `total_vram_gib` · 48 | Kapasitas total memori GPU dalam GiB. | `bytes_to_gib(props.total_memory)` |

### Lingkup `get_peak_memory_report`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `device` · 60 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `peak_allocated` · 63 | Puncak memori yang dialokasikan untuk tensor CUDA dalam GiB. | `bytes_to_gib(torch.cuda.max_memory_allocated())` |
| `peak_reserved` · 64 | Puncak memori yang dicadangkan allocator CUDA dalam GiB. | `bytes_to_gib(torch.cuda.max_memory_reserved())` |

### Lingkup `estimate_memory_budget`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `num_params` · 72 | Jumlah elemen parameter model yang dihitung dengan numel, bukan jumlah layer. | `argumen int` |
| `device` · 72 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `use_amp` · 72 | Sakelar mixed precision pada trainer awal. | `argumen bool` |
| `state_gib` · 84 | Perkiraan state bobot, gradien, dan AdamW dengan asumsi 16 byte per parameter, belum termasuk aktivasi. | `bytes_to_gib(num_params * 16)` |
| `props` · 85 | Properti GPU, termasuk memori dan compute capability. | `torch.cuda.get_device_properties(torch.cuda.current_device())` |
| `total_gib` · 86 | Kapasitas total memori GPU dalam GiB. | `bytes_to_gib(props.total_memory)` |
| `note` · 87 | Bagian teks laporan diagnostik runtime atau memori. | `' (+ activations; reduce with --grad-checkpointing / --grad-accum)'` |

### Lingkup `load_checkpoint_file`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 99 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `device` · 99 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |

### Lingkup `default_checkpoint_dir`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `out_path` · 107 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `model_path` · 109 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `Path(out_path)` |

### Lingkup `checkpoint_path`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `checkpoint_dir` · 113 | Direktori penyimpanan checkpoint. | `argumen str` |
| `step` · 113 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `argumen int` |

### Lingkup `checkpoint_step`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 118 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `match` · 120 | Hasil regex nama checkpoint, None jika tidak cocok. | `CHECKPOINT_RE.search(os.path.basename(path))` |

### Lingkup `list_checkpoints`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `checkpoint_dir` · 126 | Direktori penyimpanan checkpoint. | `argumen str` |
| `paths` · 130 | Daftar path; checkpoints khusus daftar checkpoint periodik terurut. | `[os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir) if CHECKPOINT_RE.search(name)]` |
| `name` · 132 | Nama parameter model, field konfigurasi, atau berkas yang sedang diperiksa. | `iterasi os.listdir(checkpoint_dir)` |

### Lingkup `resolve_resume_path`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `resume` · 138 | Pilihan/path checkpoint untuk melanjutkan training atau latest pada trainer awal. | `argumen Optional[str]` |
| `checkpoint_dir` · 138 | Direktori penyimpanan checkpoint. | `argumen str` |
| `checkpoints` · 148 | Daftar path; checkpoints khusus daftar checkpoint periodik terurut. | `list_checkpoints(checkpoint_dir)` |

### Lingkup `current_lr`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `optimizer` · 155 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `argumen torch.optim.Optimizer` |

### Lingkup `lr_for_step`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `train_config` · 160 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Dict[str, Any]` |
| `step` · 160 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `argumen int` |

### Lingkup `set_optimizer_lr`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `optimizer` · 167 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `argumen torch.optim.Optimizer` |
| `lr` · 167 | Learning rate, ukuran langkah pembaruan parameter. | `argumen float` |
| `group` · 169 | Kelompok parameter optimizer, dibedakan menurut penerapan weight decay; group/g satu kelompok yang sedang diproses. | `iterasi optimizer.param_groups` |

### Lingkup `save_training_checkpoint`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 174 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `model` · 175 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen Transformer` |
| `optimizer` · 176 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `argumen torch.optim.Optimizer` |
| `train_config` · 177 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Dict[str, Any]` |
| `losses` · 178 | Riwayat/daftar nilai loss untuk rata-rata, laporan, atau checkpoint. | `argumen List[float]` |
| `step` · 180 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `argumen int` |
| `train_loss` · 181 | Nilai loss train untuk laporan/checkpoint. | `argumen Optional[float]; default None` |
| `dev_loss` · 182 | Nilai loss validasi untuk laporan/checkpoint. | `argumen Optional[float]; default None` |
| `is_final` · 183 | Penanda bahwa berkas checkpoint merupakan hasil akhir, bukan checkpoint periodik. | `argumen bool; default False` |
| `payload` · 192 | Dictionary bobot, optimizer, konfigurasi, step, dan metrik yang akan disimpan. | `{'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'losses': losses, 'train_loss': train_loss, 'dev_loss': dev_loss, 'step': step, 'last_completed_step': step, 'steps': step + 1, 'is_final': is_final, 'config': dict(train_config), 'device': train_config['device'], 'pytorch_version': torch.__version__, 'cuda_ver ...` |
| `target_dir` · 213 | Direktori tujuan, handle file sementara, dan path sementara untuk save atomik. | `os.path.dirname(path) or '.'` |
| `tmp_file` · 214 | Direktori tujuan, handle file sementara, dan path sementara untuk save atomik. | `tempfile.NamedTemporaryFile(dir=target_dir, prefix=f'.{os.path.basename(path)}.', suffix='.tmp', delete=False)` |
| `tmp_path` · 220 | Direktori tujuan, handle file sementara, dan path sementara untuk save atomik. | `tmp_file.name` |

### Lingkup `restore_training_checkpoint`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 231 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `model` · 232 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen Transformer` |
| `optimizer` · 233 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `argumen torch.optim.Optimizer` |
| `train_config` · 234 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Dict[str, Any]` |
| `device` · 235 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `checkpoint` · 243 | Isi berkas checkpoint, termasuk bobot dan metadata bila tersedia. | `load_checkpoint_file(path, device)` |
| `optimizer_state` · 246 | State optimizer dari checkpoint, termasuk momen dan pengaturan kelompok parameter. | `checkpoint.get('optimizer_state_dict')` |
| `last_completed_step` · 251, 255 | Posisi training: awal, berikutnya, terakhir selesai, atau yang dicatat saat save final. | `int(checkpoint['last_completed_step'])`; `next_step - 1` |
| `next_step` · 252, 254 | Posisi training: awal, berikutnya, terakhir selesai, atau yang dicatat saat save final. | `last_completed_step + 1`; `int(checkpoint.get('steps', 0))` |
| `losses` · 260 | Riwayat/daftar nilai loss untuk rata-rata, laporan, atau checkpoint. | `[float(loss) for loss in checkpoint.get('losses', [])]` |
| `loss` · 260 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `iterasi checkpoint.get('losses', [])` |

### Lingkup `prune_old_checkpoints`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `checkpoint_dir` · 268 | Direktori penyimpanan checkpoint. | `argumen str` |
| `keep_last` · 268 | Banyak checkpoint terbaru yang dipertahankan; nol menyimpan semua. | `argumen int` |
| `checkpoints` · 272 | Daftar path; checkpoints khusus daftar checkpoint periodik terurut. | `list_checkpoints(checkpoint_dir)` |
| `old_path` · 273 | Path checkpoint lama yang dipilih untuk dibuang oleh keep_last. | `iterasi checkpoints[:-keep_last]` |

### Lingkup `unique_output_path`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `out_path` · 277 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `modified_model_out_path` · 279, 284 | Nama output alternatif dan penghitung suffix agar tidak menimpa hasil lama. | `out_path`; `model_out_name + f'_{save_tries}' + '.pt'` |
| `save_tries` · 280, 282 | Nama output alternatif dan penghitung suffix agar tidak menimpa hasil lama. | `0`; `save_tries += 1` |
| `model_out_name` · 283 | Nama output alternatif dan penghitung suffix agar tidak menimpa hasil lama. | `os.path.splitext(out_path)[0]` |

### Lingkup `as_float`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `value` · 288 | Linear pembentuk values jika atribut Head; pada utilitas konversi/reduksi berupa nilai skalar masukan. | `argumen Any` |

### Lingkup `estimate_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 300 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen Transformer` |
| `train_config` · 300 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen Dict[str, Any]` |
| `steps` · 300 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `argumen int` |
| `out` · 312 | Dictionary rerata loss train dan dev dari sampel batch evaluasi. | `{}` |
| `split` · 316 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `iterasi ['train', 'dev']` |
| `data_path` · 318 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `train_config['train_path'] if split == 'train' else train_config['dev_path']` |
| `batch_iterator_eval` · 321 | Sumber batch yang dilanjutkan dengan next; akhiran eval khusus evaluasi. | `get_batch_iterator(data_path, train_config['t_batch_size'], train_config['t_context_length'], device=train_config['device'])` |
| `losses_eval` · 329 | Riwayat/daftar nilai loss untuk rata-rata, laporan, atau checkpoint. | `[]` |
| `_` · 330, 334 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `iterasi range(steps)`; `elemen 0 dari model(xb, yb)` |
| `xb` · 333 | Batch token input pretraining (B,T). | `elemen 0 dari next(batch_iterator_eval)` |
| `yb` · 333 | Token target; yb pretraining sudah bergeser satu token terhadap xb. | `elemen 1 dari next(batch_iterator_eval)` |
| `loss` · 334 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 1 dari model(xb, yb)` |

### Lingkup `parse_args`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `parser` · 349 | Objek argparse yang mendefinisikan dan membaca argumen terminal. | `argparse.ArgumentParser(description='Train the Transformer model from scratch.')` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `args` · 417 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `parse_args()` |
| `train_config` · 418 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `dict(config)` |
| `checkpoint_every` · 419 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `args.checkpoint_every if args.checkpoint_every is not None else train_config.get('t_checkpoint_steps', 0)` |
| `keep_last` · 424 | Banyak checkpoint terbaru yang dipertahankan; nol menyimpan semua. | `args.keep_last if args.keep_last is not None else train_config.get('t_keep_last_checkpoints', 0)` |
| `checkpoint_dir` · 429 | Direktori penyimpanan checkpoint. | `args.checkpoint_dir or train_config.get('t_checkpoint_dir') or default_checkpoint_dir(train_config['t_out_path'])` |
| `use_amp` · 436, 452 | Sakelar mixed precision pada trainer awal. | `args.amp if args.amp is not None else bool(train_config.get('use_amp', False))`; `False` |
| `amp_dtype_name` · 437 | Tipe atau nama presisi autocast, biasanya bf16/fp16; None berarti tanpa autocast. | `args.amp_dtype or train_config.get('amp_dtype', 'bf16')` |
| `use_grad_ckpt` · 438 | Sakelar menghitung ulang aktivasi blok saat backward untuk mengurangi memori. | `args.grad_checkpointing if args.grad_checkpointing is not None else bool(train_config.get('use_gradient_checkpointing', False))` |
| `grad_accum` · 442 | Jumlah micro-batch per optimizer step jika loop trainer mengimplementasikannya. | `max(1, args.grad_accum if args.grad_accum is not None else int(train_config.get('grad_accum_steps', 1)))` |
| `report_memory` · 444 | Sakelar pelaporan perkiraan kebutuhan memori. | `args.report_memory if args.report_memory is not None else bool(train_config.get('report_memory_budget', False))` |
| `device_is_cuda` · 449 | Apakah device dan ketersediaan CUDA mendukung jalur komputasi CUDA. | `train_config['device'].startswith('cuda') and torch.cuda.is_available()` |
| `amp_dtype` · 453 | Tipe atau nama presisi autocast, biasanya bf16/fp16; None berarti tanpa autocast. | `torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16` |
| `use_scaler` · 462 | Apakah autocast fp16 membutuhkan loss scaling; bf16 tidak memakainya pada trainer ini. | `use_amp and amp_dtype == torch.float16` |
| `scaler` · 463 | Objek GradScaler untuk scale/unscale gradien dan update optimizer pada fp16. | `torch.amp.GradScaler('cuda', enabled=use_scaler)` |
| `model` · 472 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `Transformer(n_head=train_config['n_head'], n_embed=train_config['n_embed'], context_length=train_config['context_length'], vocab_size=train_config['vocab_size'], N_BLOCKS=train_config['n_blocks']).to(train_config['device'])` |
| `total_params` · 481 | Jumlah elemen parameter model yang dihitung dengan numel, bukan jumlah layer. | `sum((p.numel() for p in model.parameters()))` |
| `p` · 481 | Satu tensor parameter model yang dihitung jumlah elemennya, dikelompokkan optimizer, atau dibekukan sesuai fungsi. | `iterasi model.parameters()` |
| `optimizer` · 498 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `torch.optim.AdamW(model.parameters(), lr=train_config['t_lr'])` |
| `losses` · 501, 506 | Riwayat/daftar nilai loss untuk rata-rata, laporan, atau checkpoint. | `[]`; `elemen 1 dari restore_training_checkpoint(resume_path, model, optimizer, train_config, train_config['device'])` |
| `start_step` · 502, 506 | Posisi training: awal, berikutnya, terakhir selesai, atau yang dicatat saat save final. | `0`; `elemen 0 dari restore_training_checkpoint(resume_path, model, optimizer, train_config, train_config['device'])` |
| `last_completed_step` · 503, 513, 565 | Posisi training: awal, berikutnya, terakhir selesai, atau yang dicatat saat save final. | `-1`; `start_step - 1`; `step` |
| `resume_path` · 504 | Pilihan/path checkpoint untuk melanjutkan training atau latest pada trainer awal. | `resolve_resume_path(args.resume, checkpoint_dir)` |
| `avg_window` · 516 | Banyak loss terbaru untuk moving average tampilan trainer awal. | `64` |
| `batch_iterator` · 523 | Sumber batch yang dilanjutkan dengan next; akhiran eval khusus evaluasi. | `get_batch_iterator(train_config['train_path'], train_config['t_batch_size'], train_config['t_context_length'], device=train_config['device'])` |
| `tokens_per_step` · 531 | Jumlah token nominal yang diproses per optimizer step. | `train_config['t_batch_size'] * train_config['t_context_length'] * grad_accum` |
| `last_eval_time` · 532, 579 | Penanda waktu untuk mengukur durasi step atau interval evaluasi. | `time.perf_counter()`; `now` |
| `latest_train_loss` · 533, 574 | Nilai loss train untuk laporan/checkpoint. | `None`; `evaluation_losses['train']` |
| `latest_dev_loss` · 534, 575 | Nilai loss validasi untuk laporan/checkpoint. | `None`; `evaluation_losses['dev']` |
| `pbar` · 537 | Pembungkus tqdm untuk menampilkan kemajuan loop. | `tqdm(range(start_step, train_config['t_train_steps']))` |
| `step` · 538 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `iterasi pbar` |
| `step_start_time` · 541 | Penanda waktu untuk mengukur durasi step atau interval evaluasi. | `time.perf_counter()` |
| `step_loss` · 545, 553 | Akumulasi loss micro-batch yang telah dibagi faktor accumulation. | `0.0`; `step_loss += float(loss.item())` |
| `_` · 546, 549 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `iterasi range(grad_accum)`; `elemen 0 dari model(xb, yb)` |
| `xb` · 547 | Batch token input pretraining (B,T). | `elemen 0 dari next(batch_iterator)` |
| `yb` · 547 | Token target; yb pretraining sudah bergeser satu token terhadap xb. | `elemen 1 dari next(batch_iterator)` |
| `loss` · 549, 551 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 1 dari model(xb, yb)`; `loss / grad_accum` |
| `step_time` · 568 | Durasi komputasi/interval; dt pada lambda collate adalah dtype tensor. | `time.perf_counter() - step_start_time` |
| `tokens_per_second` · 569 | Throughput token per detik. | `tokens_per_step / step_time if step_time > 0 else float('inf')` |
| `evaluation_losses` · 573, 614 | Dictionary hasil evaluasi loss train/dev. | `estimate_loss(model, train_config, train_config['t_eval_iters'])`; `estimate_loss(model, train_config, 200)` |
| `now` · 577 | Penanda waktu untuk mengukur durasi step atau interval evaluasi. | `time.perf_counter()` |
| `elapsed_since_eval` · 578 | Durasi komputasi/interval; dt pada lambda collate adalah dtype tensor. | `now - last_eval_time` |
| `path` · 593 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `checkpoint_path(checkpoint_dir, step)` |
| `train_loss` · 615 | Nilai loss train untuk laporan/checkpoint. | `evaluation_losses['train']` |
| `dev_loss` · 616 | Nilai loss validasi untuk laporan/checkpoint. | `evaluation_losses['dev']` |
| `final_step` · 618 | Posisi training: awal, berikutnya, terakhir selesai, atau yang dicatat saat save final. | `max(last_completed_step, start_step - 1)` |
| `modified_model_out_path` · 619 | Nama output alternatif dan penghitung suffix agar tidak menimpa hasil lama. | `unique_output_path(train_config['t_out_path'])` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 1 |
| `argparse` | `argparse` | 3 |
| `contextlib` | `contextlib` | 4 |
| `os` | `os` | 5 |
| `re` | `re` | 6 |
| `sys` | `sys` | 7 |
| `tempfile` | `tempfile` | 8 |
| `time` | `time` | 9 |
| `Path` | `pathlib.Path` | 10 |
| `Any` | `typing.Any` | 11 |
| `Dict` | `typing.Dict` | 11 |
| `List` | `typing.List` | 11 |
| `Optional` | `typing.Optional` | 11 |
| `Tuple` | `typing.Tuple` | 11 |
| `np` | `numpy` | 13 |
| `torch` | `torch` | 14 |
| `tqdm` | `tqdm.tqdm` | 15 |
| `config` | `config.config.default_config` | 21 |
| `Transformer` | `src.models.transformer.Transformer` | 22 |
| `get_batch_iterator` | `data_loader.data_loader.get_batch_iterator` | 314 |
| `get_batch_iterator` | `data_loader.data_loader.get_batch_iterator` | 520 |
