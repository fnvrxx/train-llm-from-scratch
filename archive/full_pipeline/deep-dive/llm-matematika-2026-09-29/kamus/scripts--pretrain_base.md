# scripts/pretrain_base.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/pretrain_base.py)

## Fungsi dan kelas

### `estimate_loss` (baris 38)

Mengukur rerata loss token train/dev dalam eval tanpa backward, lalu mengembalikan mode train.

```python
def estimate_loss(model, cfg, ctx, iters: int) -> dict[str, float]:
```

Keluaran eksplisit: `out`.

### `main` (baris 56)

Menjalankan pretraining dengan DDP, accumulation, jadwal LR, evaluasi, dan checkpoint.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `_nullcm` (baris 153)

Membuat context manager kosong ketika no_sync tidak diperlukan.

```python
def _nullcm():
```

Keluaran eksplisit: `contextlib.nullcontext()`.

## Variabel dan parameter menurut lingkup

### Lingkup `estimate_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 38 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `cfg` · 38 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen tanpa anotasi tipe` |
| `ctx` · 38 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen tanpa anotasi tipe` |
| `iters` · 38 | Jumlah atau batas batch untuk estimasi metrik evaluasi. | `argumen int` |
| `out` · 40 | Dictionary rerata loss untuk split train dan dev yang tersedia. | `{}` |
| `split` · 41 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `elemen 0 dari iterasi [('train', cfg.train_path), ('dev', cfg.dev_path)]` |
| `path` · 41 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `elemen 1 dari iterasi [('train', cfg.train_path), ('dev', cfg.dev_path)]` |
| `it` · 44 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `get_batch_iterator(path, cfg.batch_size, cfg.context_length, device=ctx.device)` |
| `losses` · 45 | Riwayat/daftar nilai loss untuk rata-rata, laporan, atau checkpoint. | `torch.zeros(iters)` |
| `k` · 46 | Nomor batch evaluasi, dipakai mengisi losses[k]. | `iterasi range(iters)` |
| `xb` · 47 | Batch token input pretraining (B,T). | `elemen 0 dari next(it)` |
| `yb` · 47 | Token target; yb pretraining sudah bergeser satu token terhadap xb. | `elemen 1 dari next(it)` |
| `_` · 49 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 0 dari model(xb, yb)` |
| `loss` · 49 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 1 dari model(xb, yb)` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 57 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `elemen 0 dari parse_config_with_json(PretrainConfig, 'configs/pretrain.json', extra={'--resume': dict(type=str, default=None, help='checkpoint to resume from')})` |
| `extras` · 57 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `elemen 1 dari parse_config_with_json(PretrainConfig, 'configs/pretrain.json', extra={'--resume': dict(type=str, default=None, help='checkpoint to resume from')})` |
| `resume` · 60 | Pilihan/path checkpoint untuk melanjutkan training atau latest pada trainer awal. | `extras.resume` |
| `ctx` · 61 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `ddp_setup(cfg.device)` |
| `model` · 65, 75, 76 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `build_model_from_config(cfg).to(ctx.device)`; `torch.compile(model)`; `ddp_wrap(model, ctx)` |
| `start_step` · 66, 70 | Posisi training: awal, berikutnya, terakhir selesai, atau yang dicatat saat save final. | `0`; `ck.get('step', 0)` |
| `ck` · 68, 80 | Isi berkas checkpoint, termasuk bobot dan metadata bila tersedia. | `torch.load(resume, map_location='cpu', weights_only=False)` |
| `optimizer` · 78 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `configure_optimizer(unwrap(model), cfg.lr, cfg.weight_decay)` |
| `logger` · 84, 90 | Pencatat metrik JSONL dan opsional W&B. | `None`; `MetricsLogger('pretrain', cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project, config=vars(cfg).copy() if hasattr(cfg, '__dict__') else None)` |
| `n_params` · 86 | Jumlah elemen parameter model yang dihitung dengan numel, bukan jumlah layer. | `sum((p.numel() for p in unwrap(model).parameters()))` |
| `p` · 86 | Satu tensor parameter model yang dihitung jumlah elemennya, dikelompokkan optimizer, atau dibekukan sesuai fungsi. | `iterasi unwrap(model).parameters()` |
| `batch_iter` · 93 | Sumber batch yang dilanjutkan dengan next; akhiran eval khusus evaluasi. | `get_batch_iterator(cfg.train_path, cfg.batch_size, cfg.context_length, device=ctx.device)` |
| `tokens_per_step` · 94 | Jumlah token nominal yang diproses per optimizer step. | `cfg.batch_size * cfg.context_length * cfg.grad_accum * ctx.world_size` |
| `t0` · 97, 123 | Penanda waktu untuk mengukur durasi step atau interval evaluasi. | `time.perf_counter()` |
| `step` · 98 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `iterasi range(start_step, cfg.train_steps)` |
| `lr` · 99 | Learning rate, ukuran langkah pembaruan parameter. | `cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=cfg.train_steps, lr=cfg.lr, min_lr=cfg.min_lr)` |
| `g` · 101 | Kelompok parameter optimizer, dibedakan menurut penerapan weight decay; group/g satu kelompok yang sedang diproses. | `iterasi optimizer.param_groups` |
| `accum_loss` · 105, 115 | Akumulasi loss micro-batch yang telah dibagi faktor accumulation. | `0.0`; `accum_loss += loss.item()` |
| `micro` · 106 | Indeks micro-batch pada gradient accumulation. | `iterasi range(cfg.grad_accum)` |
| `xb` · 107 | Batch token input pretraining (B,T). | `elemen 0 dari next(batch_iter)` |
| `yb` · 107 | Token target; yb pretraining sudah bergeser satu token terhadap xb. | `elemen 1 dari next(batch_iter)` |
| `sync` · 109 | Penanda perlu sinkronisasi gradien dan context manager no_sync/context kosong. | `micro == cfg.grad_accum - 1 or not ctx.enabled` |
| `cm` · 110 | Penanda perlu sinkronisasi gradien dan context manager no_sync/context kosong. | `model.no_sync() if ctx.enabled and (not sync) else _nullcm()` |
| `_` · 112 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 0 dari model(xb, yb)` |
| `loss` · 112, 113 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 1 dari model(xb, yb)`; `loss / cfg.grad_accum` |
| `dt` · 121 | Durasi komputasi/interval; dt pada lambda collate adalah dtype tensor. | `time.perf_counter() - t0` |
| `tok_s` · 122 | Throughput token per detik. | `tokens_per_step * 20 / dt if step > start_step else 0.0` |
| `ev` · 129, 130 | Dictionary hasil evaluasi loss train/dev. | `estimate_loss(model, cfg, ctx, cfg.eval_iters)`; `{k: reduce_scalar(v, ctx) for k, v in ev.items()}` |
| `k` · 130, 132, 134 | Nama split/metrik pada dictionary ev, misalnya train atau dev. | `elemen 0 dari iterasi ev.items()` |
| `v` · 130, 132, 134 | Nilai loss split pada ev yang direduksi antar proses. | `elemen 1 dari iterasi ev.items()` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 18 |
| `os` | `os` | 20 |
| `time` | `time` | 21 |
| `np` | `numpy` | 23 |
| `torch` | `torch` | 24 |
| `PretrainConfig` | `config.post_training_config.PretrainConfig` | 26 |
| `get_batch_iterator` | `data_loader.data_loader.get_batch_iterator` | 27 |
| `parse_config_with_json` | `src.post_training.cli.parse_config_with_json` | 28 |
| `ddp_setup` | `src.post_training.distributed.ddp_setup` | 29 |
| `ddp_wrap` | `src.post_training.distributed.ddp_wrap` | 29 |
| `cleanup` | `src.post_training.distributed.cleanup` | 29 |
| `reduce_scalar` | `src.post_training.distributed.reduce_scalar` | 29 |
| `MetricsLogger` | `src.post_training.logging_utils.MetricsLogger` | 30 |
| `configure_optimizer` | `src.post_training.optim.configure_optimizer` | 31 |
| `cosine_lr` | `src.post_training.optim.cosine_lr` | 31 |
| `amp_autocast` | `src.post_training.utils.amp_autocast` | 32 |
| `build_model_from_config` | `src.post_training.utils.build_model_from_config` | 32 |
| `save_stage_ckpt` | `src.post_training.utils.save_stage_ckpt` | 32 |
| `set_seed` | `src.post_training.utils.set_seed` | 32 |
| `unwrap` | `src.post_training.utils.unwrap` | 32 |
| `contextlib` | `contextlib` | 150 |
