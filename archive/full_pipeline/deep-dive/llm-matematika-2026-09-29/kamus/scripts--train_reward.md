# scripts/train_reward.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/train_reward.py)

## Fungsi dan kelas

### `_pair_rewards` (baris 32)

Menggabungkan dua sisi pasangan untuk satu forward RM, lalu memisahkan skor kembali.

```python
def _pair_rewards(rm, batch, cfg, ctx):
```

Keluaran eksplisit: `(rewards[:B], rewards[B:])`.

### `eval_accuracy` (baris 43)

Mengukur akurasi preferensi dan margin reward model pada data evaluasi.

```python
def eval_accuracy(rm, cfg, ctx, max_batches: int=100) -> tuple[float, float]:
```

Keluaran eksplisit: `(acc / max(1, n), marg / max(1, n))`.

### `main` (baris 59)

Melatih backbone SFT dan reward head pada pasangan preferensi.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `TEST_PATH` · 29 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `'/ephemeral/data/preferences_test.jsonl'` |

### Lingkup `_pair_rewards`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rm` · 32 | Model reward yang menghasilkan skor jawaban. | `argumen tanpa anotasi tipe` |
| `batch` · 32 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen tanpa anotasi tipe` |
| `cfg` · 32 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen tanpa anotasi tipe` |
| `ctx` · 32 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen tanpa anotasi tipe` |
| `B` · 34 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `batch['chosen_ids'].size(0)` |
| `ids` · 35 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `torch.cat([batch['chosen_ids'], batch['rejected_ids']], dim=0)` |
| `lens` · 36 | Banyak token nyata setiap urutan, sebelum right padding. | `torch.cat([batch['chosen_len'], batch['rejected_len']], dim=0)` |
| `rewards` · 38 | Reward per urutan/action/kelompok sesuai fungsi; berbeda dari peluang token. | `rm(ids, seq_lengths=lens).float()` |

### Lingkup `eval_accuracy`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rm` · 43 | Model reward yang menghasilkan skor jawaban. | `argumen tanpa anotasi tipe` |
| `cfg` · 43 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen tanpa anotasi tipe` |
| `ctx` · 43 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen tanpa anotasi tipe` |
| `max_batches` · 43 | Jumlah atau batas batch untuk estimasi metrik evaluasi. | `argumen int; default 100` |
| `it` · 45 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `get_preference_iterator(TEST_PATH, cfg.batch_size, cfg.max_len, device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, shuffle=False, infinite=False)` |
| `acc` · 47, 50 | Indikator, jumlah, atau proporsi prediksi/preferensi benar pada evaluator terkait. | `elemen 0 dari (0.0, 0.0, 0)`; `acc += preference_accuracy(cr, rr).item()` |
| `marg` · 47, 51 | Akumulasi/rerata reward chosen minus rejected. | `elemen 1 dari (0.0, 0.0, 0)`; `marg += reward_margin(cr, rr).item()` |
| `n` · 47, 52 | Jumlah batch evaluasi yang telah dihitung; pembagi akurasi/margin rata-rata. | `elemen 2 dari (0.0, 0.0, 0)`; `n += 1` |
| `batch` · 48 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `iterasi it` |
| `cr` · 49 | Skor reward model untuk jawaban chosen. | `elemen 0 dari _pair_rewards(rm, batch, cfg, ctx)` |
| `rr` · 49 | Skor reward model untuk jawaban rejected. | `elemen 1 dari _pair_rewards(rm, batch, cfg, ctx)` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 60 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `elemen 0 dari parse_config_with_json(RewardConfig, 'configs/reward.json')` |
| `_` · 60 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari parse_config_with_json(RewardConfig, 'configs/reward.json')` |
| `ctx` · 61 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `ddp_setup(cfg.device)` |
| `backbone` · 64 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)` |
| `rm` · 65, 69 | Model reward yang menghasilkan skor jawaban. | `RewardModel(backbone).to(ctx.device)`; `ddp_wrap(rm, ctx, find_unused_parameters=True)` |
| `optimizer` · 70 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `configure_optimizer(unwrap(rm), cfg.lr, cfg.weight_decay)` |
| `f` · 73 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(cfg.pref_path)` |
| `n_rows` · 74 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `sum((1 for line in f if line.strip()))` |
| `line` · 74 | Satu baris teks JSONL. | `iterasi f` |
| `total_steps` · 75 | Batas update; max_steps negatif pada konfigurasi SFT berarti gunakan batas epochs. | `max(1, n_rows // (cfg.batch_size * ctx.world_size) * cfg.epochs)` |
| `logger` · 77, 80 | Pencatat metrik JSONL dan opsional W&B. | `None`; `MetricsLogger('reward', cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project)` |
| `train_it` · 82 | Iterator data training untuk tahap saat ini. | `get_preference_iterator(cfg.pref_path, cfg.batch_size, cfg.max_len, device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, shuffle=True, infinite=True)` |
| `t0` · 86, 102 | Penanda waktu untuk mengukur durasi step atau interval evaluasi. | `time.perf_counter()` |
| `step` · 87 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `iterasi range(total_steps)` |
| `lr` · 88 | Learning rate, ukuran langkah pembaruan parameter. | `cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=total_steps, lr=cfg.lr, min_lr=cfg.lr * 0.1)` |
| `g` · 89 | Kelompok parameter optimizer, dibedakan menurut penerapan weight decay; group/g satu kelompok yang sedang diproses. | `iterasi optimizer.param_groups` |
| `batch` · 92 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `next(train_it)` |
| `cr` · 93 | Skor reward model untuk jawaban chosen. | `elemen 0 dari _pair_rewards(rm, batch, cfg, ctx)` |
| `rr` · 93 | Skor reward model untuk jawaban rejected. | `elemen 1 dari _pair_rewards(rm, batch, cfg, ctx)` |
| `loss` · 94 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `bradley_terry_loss(cr, rr)` |
| `acc` · 101, 108, 109, 122 | Indikator, jumlah, atau proporsi prediksi/preferensi benar pada evaluator terkait. | `preference_accuracy(cr, rr).item()`; `elemen 0 dari eval_accuracy(rm, cfg, ctx)`; `elemen 0 dari (reduce_scalar(acc, ctx), reduce_scalar(marg, ctx))` |
| `dt` · 102 | Durasi komputasi/interval; dt pada lambda collate adalah dtype tensor. | `time.perf_counter() - t0` |
| `marg` · 108, 109, 122 | Akumulasi/rerata reward chosen minus rejected. | `elemen 1 dari eval_accuracy(rm, cfg, ctx)`; `elemen 1 dari (reduce_scalar(acc, ctx), reduce_scalar(marg, ctx))`; `elemen 1 dari eval_accuracy(unwrap(rm), cfg, ctx)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 13 |
| `time` | `time` | 15 |
| `torch` | `torch` | 17 |
| `RewardConfig` | `config.post_training_config.RewardConfig` | 19 |
| `get_preference_iterator` | `data_loader.preference_dataset.get_preference_iterator` | 20 |
| `parse_config_with_json` | `src.post_training.cli.parse_config_with_json` | 21 |
| `ddp_setup` | `src.post_training.distributed.ddp_setup` | 22 |
| `ddp_wrap` | `src.post_training.distributed.ddp_wrap` | 22 |
| `cleanup` | `src.post_training.distributed.cleanup` | 22 |
| `reduce_scalar` | `src.post_training.distributed.reduce_scalar` | 22 |
| `MetricsLogger` | `src.post_training.logging_utils.MetricsLogger` | 23 |
| `configure_optimizer` | `src.post_training.optim.configure_optimizer` | 24 |
| `cosine_lr` | `src.post_training.optim.cosine_lr` | 24 |
| `RewardModel` | `src.post_training.reward_model.RewardModel` | 25 |
| `bradley_terry_loss` | `src.post_training.reward_train.bradley_terry_loss` | 26 |
| `preference_accuracy` | `src.post_training.reward_train.preference_accuracy` | 26 |
| `reward_margin` | `src.post_training.reward_train.reward_margin` | 26 |
| `amp_autocast` | `src.post_training.utils.amp_autocast` | 27 |
| `load_backbone_from_ckpt` | `src.post_training.utils.load_backbone_from_ckpt` | 27 |
| `save_stage_ckpt` | `src.post_training.utils.save_stage_ckpt` | 27 |
| `set_seed` | `src.post_training.utils.set_seed` | 27 |
| `unwrap` | `src.post_training.utils.unwrap` | 27 |
| `json` | `json` | 72 |
