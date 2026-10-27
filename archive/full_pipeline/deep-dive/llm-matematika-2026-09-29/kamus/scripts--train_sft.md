# scripts/train_sft.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/train_sft.py)

## Fungsi dan kelas

### `eval_dev` (baris 34)

Mengukur rerata batch loss SFT bermask pada validasi.

```python
def eval_dev(model, cfg, ctx, dev_path: str, max_batches: int=50) -> float:
```

Keluaran eksplisit: `total / max(1, n)`.

### `main` (baris 50)

Memuat base checkpoint dan melatih loss token assistant pada demonstrasi.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `DEV_PATH` · 30 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `'/ephemeral/data/sft_dev_packed.h5'` |

### Lingkup `eval_dev`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 34 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `cfg` · 34 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen tanpa anotasi tipe` |
| `ctx` · 34 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen tanpa anotasi tipe` |
| `dev_path` · 34 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `argumen str` |
| `max_batches` · 34 | Jumlah atau batas batch untuk estimasi metrik evaluasi. | `argumen int; default 50` |
| `it` · 36 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `get_sft_batch_iterator(dev_path, cfg.batch_size, device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, shuffle=False, infinite=False)` |
| `total` · 38, 43 | Jumlah loss skalar dari batch validasi yang sudah dibaca. | `elemen 0 dari (0.0, 0)`; `total += loss.item()` |
| `n` · 38, 43 | Jumlah batch validasi yang sudah dinilai. | `elemen 1 dari (0.0, 0)`; `n += 1` |
| `tokens` · 39 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari iterasi it` |
| `mask` · 39 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `elemen 1 dari iterasi it` |
| `_` · 39, 41 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 2 dari iterasi it`; `elemen 1 dari model(tokens)` |
| `logits` · 41 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `elemen 0 dari model(tokens)` |
| `loss` · 42 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `sft_loss(logits, tokens, mask)` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 51 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `elemen 0 dari parse_config_with_json(SFTConfig, 'configs/sft.json')` |
| `_` · 51, 90 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari parse_config_with_json(SFTConfig, 'configs/sft.json')`; `elemen 1 dari model(tokens)` |
| `ctx` · 52 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `ddp_setup(cfg.device)` |
| `model` · 55, 57, 58 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `load_backbone_from_ckpt(cfg, cfg.pretrained_ckpt, ctx.device)`; `torch.compile(model)`; `ddp_wrap(model, ctx)` |
| `optimizer` · 59 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `configure_optimizer(unwrap(model), cfg.lr, cfg.weight_decay)` |
| `logger` · 61, 64 | Pencatat metrik JSONL dan opsional W&B. | `None`; `MetricsLogger('sft', cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project)` |
| `train_it` · 66 | Iterator data training untuk tahap saat ini. | `get_sft_batch_iterator(cfg.data_path, cfg.batch_size, device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, shuffle=True, infinite=True)` |
| `f` · 71 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `h5py.File(cfg.data_path, 'r')` |
| `n_rows` · 72 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `f['tokens'].shape[0]` |
| `steps_per_epoch` · 73 | Perkiraan batch lengkap per putaran data setelah pembagian batch size dan world size. | `max(1, n_rows // (cfg.batch_size * ctx.world_size))` |
| `total_steps` · 74 | Batas update; max_steps negatif pada konfigurasi SFT berarti gunakan batas epochs. | `cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.epochs` |
| `t0` · 79, 97 | Penanda waktu untuk mengukur durasi step atau interval evaluasi. | `time.perf_counter()` |
| `step` · 80 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `iterasi range(total_steps)` |
| `lr` · 81 | Learning rate, ukuran langkah pembaruan parameter. | `cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=total_steps, lr=cfg.lr, min_lr=cfg.min_lr)` |
| `g` · 82 | Kelompok parameter optimizer, dibedakan menurut penerapan weight decay; group/g satu kelompok yang sedang diproses. | `iterasi optimizer.param_groups` |
| `tokens` · 85 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari next(train_it)` |
| `mask` · 85 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `elemen 1 dari next(train_it)` |
| `epoch` · 85 | Nomor putaran data atau jumlah putaran yang diminta. | `elemen 2 dari next(train_it)` |
| `logits` · 90 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `elemen 0 dari model(tokens)` |
| `loss` · 91 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `sft_loss(logits, tokens, mask)` |
| `dt` · 97 | Durasi komputasi/interval; dt pada lambda collate adalah dtype tensor. | `time.perf_counter() - t0` |
| `dev` · 103, 117 | Device pada setup DDP; pada trainer/evaluasi dapat berarti data atau loss validasi. | `reduce_scalar(eval_dev(model, cfg, ctx, DEV_PATH), ctx)`; `eval_dev(unwrap(model), cfg, ctx, DEV_PATH)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 13 |
| `contextlib` | `contextlib` | 15 |
| `math` | `math` | 16 |
| `time` | `time` | 17 |
| `torch` | `torch` | 19 |
| `SFTConfig` | `config.post_training_config.SFTConfig` | 21 |
| `get_sft_batch_iterator` | `data_loader.sft_dataset.get_sft_batch_iterator` | 22 |
| `parse_config_with_json` | `src.post_training.cli.parse_config_with_json` | 23 |
| `ddp_setup` | `src.post_training.distributed.ddp_setup` | 24 |
| `ddp_wrap` | `src.post_training.distributed.ddp_wrap` | 24 |
| `cleanup` | `src.post_training.distributed.cleanup` | 24 |
| `reduce_scalar` | `src.post_training.distributed.reduce_scalar` | 24 |
| `MetricsLogger` | `src.post_training.logging_utils.MetricsLogger` | 25 |
| `configure_optimizer` | `src.post_training.optim.configure_optimizer` | 26 |
| `cosine_lr` | `src.post_training.optim.cosine_lr` | 26 |
| `sft_loss` | `src.post_training.sft.sft_loss` | 27 |
| `amp_autocast` | `src.post_training.utils.amp_autocast` | 28 |
| `load_backbone_from_ckpt` | `src.post_training.utils.load_backbone_from_ckpt` | 28 |
| `save_stage_ckpt` | `src.post_training.utils.save_stage_ckpt` | 28 |
| `set_seed` | `src.post_training.utils.set_seed` | 28 |
| `unwrap` | `src.post_training.utils.unwrap` | 28 |
| `h5py` | `h5py` | 70 |
