# scripts/train_dpo.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/train_dpo.py)

## Fungsi dan kelas

### `_logps` (baris 33)

Meneruskan pemanggilan ke sequence_logprobs untuk trainer preferensi.

```python
def _logps(model, ids, mask, requires_grad):
```

Keluaran eksplisit: `sequence_logprobs(model, ids, mask, requires_grad=requires_grad)`.

### `_compute_losses` (baris 37)

Menggabungkan chosen/rejected, menghitung log-prob dan panjang, lalu memilih DPO/ORPO/KTO.

```python
def _compute_losses(policy, ref, batch, cfg, ctx):
```

Keluaran eksplisit: `orpo_loss(pc, pr, ncn, nrn, orpo_lambda=cfg.orpo_lambda)`; `kto_loss(pc, pr, rc, rr, beta=cfg.beta)`; `dpo_loss(pc, pr, rc, rr, beta=cfg.beta)`.

### `eval_implicit_acc` (baris 58)

Mengukur akurasi dan margin reward implisit pada data preferensi evaluasi.

```python
def eval_implicit_acc(policy, ref, cfg, ctx, max_batches: int=100) -> tuple[float, float]:
```

Keluaran eksplisit: `(acc / max(1, n), marg / max(1, n))`.

### `main` (baris 74)

Melatih policy dengan objective DPO/ORPO/KTO dan reference bila diperlukan.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `TEST_PATH` · 30 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `'/ephemeral/data/preferences_test.jsonl'` |

### Lingkup `_logps`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 33 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `ids` · 33 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen tanpa anotasi tipe` |
| `mask` · 33 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen tanpa anotasi tipe` |
| `requires_grad` · 33 | Apakah recomputation perlu graph gradien; benar untuk policy yang sedang dilatih. | `argumen tanpa anotasi tipe` |

### Lingkup `_compute_losses`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `policy` · 37 | Model distribusi token yang dilatih; akhiran ddp berarti pembungkus terdistribusi. | `argumen tanpa anotasi tipe` |
| `ref` · 37 | Salinan policy/reference yang dibekukan sebagai pembanding. | `argumen tanpa anotasi tipe` |
| `batch` · 37 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen tanpa anotasi tipe` |
| `cfg` · 37 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen tanpa anotasi tipe` |
| `ctx` · 37 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen tanpa anotasi tipe` |
| `B` · 38 | Jumlah pasangan sebelum chosen/rejected digabung menjadi 2B baris. | `batch['chosen_ids'].size(0)` |
| `ids` · 39 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `torch.cat([batch['chosen_ids'], batch['rejected_ids']], dim=0)` |
| `mask` · 40 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `torch.cat([batch['chosen_mask'], batch['rejected_mask']], dim=0)` |
| `psum` · 43 | Jumlah log-prob policy, jumlah log-prob reference, dan jumlah token tiap jawaban sebelum split dua sisi. | `elemen 0 dari _logps(policy, ids, mask, requires_grad=True)` |
| `pn` · 43 | Jumlah log-prob policy, jumlah log-prob reference, dan jumlah token tiap jawaban sebelum split dua sisi. | `elemen 1 dari _logps(policy, ids, mask, requires_grad=True)` |
| `pc` · 44 | Jumlah log-prob policy untuk chosen. | `elemen 0 dari (psum[:B], psum[B:], pn[:B], pn[B:])` |
| `pr` · 44 | Jumlah log-prob policy untuk rejected. | `elemen 1 dari (psum[:B], psum[B:], pn[:B], pn[B:])` |
| `ncn` · 44 | Jumlah token jawaban aktif pada sisi chosen. | `elemen 2 dari (psum[:B], psum[B:], pn[:B], pn[B:])` |
| `nrn` · 44 | Jumlah token jawaban aktif pada sisi rejected. | `elemen 3 dari (psum[:B], psum[B:], pn[:B], pn[B:])` |
| `rsum` · 50 | Jumlah log-prob policy, jumlah log-prob reference, dan jumlah token tiap jawaban sebelum split dua sisi. | `elemen 0 dari _logps(ref, ids, mask, requires_grad=False)` |
| `_` · 50 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari _logps(ref, ids, mask, requires_grad=False)` |
| `rc` · 51 | Jumlah log-prob reference untuk chosen. | `elemen 0 dari (rsum[:B], rsum[B:])` |
| `rr` · 51 | Jumlah log-prob reference untuk rejected. | `elemen 1 dari (rsum[:B], rsum[B:])` |

### Lingkup `eval_implicit_acc`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `policy` · 58 | Model distribusi token yang dilatih; akhiran ddp berarti pembungkus terdistribusi. | `argumen tanpa anotasi tipe` |
| `ref` · 58 | Salinan policy/reference yang dibekukan sebagai pembanding. | `argumen tanpa anotasi tipe` |
| `cfg` · 58 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen tanpa anotasi tipe` |
| `ctx` · 58 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `argumen tanpa anotasi tipe` |
| `max_batches` · 58 | Jumlah atau batas batch untuk estimasi metrik evaluasi. | `argumen int; default 100` |
| `it` · 60 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `get_preference_iterator(TEST_PATH, cfg.batch_size, cfg.max_len, device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, shuffle=False, infinite=False)` |
| `acc` · 62, 65 | Indikator, jumlah, atau proporsi prediksi/preferensi benar pada evaluator terkait. | `elemen 0 dari (0.0, 0.0, 0)`; `acc += implicit_accuracy(cr, rr).item()` |
| `marg` · 62, 66 | Akumulasi/rerata reward chosen minus rejected. | `elemen 1 dari (0.0, 0.0, 0)`; `marg += (cr - rr).mean().item()` |
| `n` · 62, 67 | Jumlah batch evaluasi yang telah dihitung; pembagi akurasi/margin rata-rata. | `elemen 2 dari (0.0, 0.0, 0)`; `n += 1` |
| `batch` · 63 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `iterasi it` |
| `loss` · 64 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 0 dari _compute_losses(policy, ref, batch, cfg, ctx)` |
| `cr` · 64 | Reward implisit chosen dari objective preferensi yang dipilih. | `elemen 1 dari _compute_losses(policy, ref, batch, cfg, ctx)` |
| `rr` · 64 | Reward implisit rejected dari objective preferensi yang dipilih. | `elemen 2 dari _compute_losses(policy, ref, batch, cfg, ctx)` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 75 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `elemen 0 dari parse_config_with_json(DPOConfig, 'configs/dpo.json')` |
| `_` · 75 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari parse_config_with_json(DPOConfig, 'configs/dpo.json')` |
| `ctx` · 76 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `ddp_setup(cfg.device)` |
| `policy` · 79, 81 | Model distribusi token yang dilatih; akhiran ddp berarti pembungkus terdistribusi. | `load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)`; `ddp_wrap(policy, ctx)` |
| `ref` · 80 | Salinan policy/reference yang dibekukan sebagai pembanding. | `make_frozen_copy(policy, device=ctx.device) if cfg.loss_type != 'orpo' else None` |
| `optimizer` · 82 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `configure_optimizer(unwrap(policy), cfg.lr, cfg.weight_decay)` |
| `f` · 84 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(cfg.pref_path)` |
| `n_rows` · 85 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `sum((1 for line in f if line.strip()))` |
| `line` · 85 | Satu baris teks JSONL. | `iterasi f` |
| `total_steps` · 86 | Batas update; max_steps negatif pada konfigurasi SFT berarti gunakan batas epochs. | `max(1, n_rows // (cfg.batch_size * ctx.world_size) * cfg.epochs)` |
| `logger` · 88, 91 | Pencatat metrik JSONL dan opsional W&B. | `None`; `MetricsLogger(f'dpo_{cfg.loss_type}', cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project)` |
| `train_it` · 93 | Iterator data training untuk tahap saat ini. | `get_preference_iterator(cfg.pref_path, cfg.batch_size, cfg.max_len, device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, shuffle=True, infinite=True)` |
| `t0` · 97, 112 | Penanda waktu untuk mengukur durasi step atau interval evaluasi. | `time.perf_counter()` |
| `step` · 98 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `iterasi range(total_steps)` |
| `lr` · 99 | Learning rate, ukuran langkah pembaruan parameter. | `cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=total_steps, lr=cfg.lr, min_lr=cfg.lr * 0.1)` |
| `g` · 100 | Kelompok parameter optimizer, dibedakan menurut penerapan weight decay; group/g satu kelompok yang sedang diproses. | `iterasi optimizer.param_groups` |
| `batch` · 103 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `next(train_it)` |
| `loss` · 104 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 0 dari _compute_losses(policy, ref, batch, cfg, ctx)` |
| `cr` · 104 | Reward implisit chosen dari objective preferensi yang dipilih. | `elemen 1 dari _compute_losses(policy, ref, batch, cfg, ctx)` |
| `rr` · 104 | Reward implisit rejected dari objective preferensi yang dipilih. | `elemen 2 dari _compute_losses(policy, ref, batch, cfg, ctx)` |
| `acc` · 111, 120, 121, 134 | Indikator, jumlah, atau proporsi prediksi/preferensi benar pada evaluator terkait. | `implicit_accuracy(cr, rr).item()`; `elemen 0 dari eval_implicit_acc(policy, ref, cfg, ctx)`; `elemen 0 dari (reduce_scalar(acc, ctx), reduce_scalar(marg, ctx))` |
| `dt` · 112 | Durasi komputasi/interval; dt pada lambda collate adalah dtype tensor. | `time.perf_counter() - t0` |
| `marg` · 120, 121, 134 | Akumulasi/rerata reward chosen minus rejected. | `elemen 1 dari eval_implicit_acc(policy, ref, cfg, ctx)`; `elemen 1 dari (reduce_scalar(acc, ctx), reduce_scalar(marg, ctx))`; `elemen 1 dari eval_implicit_acc(unwrap(policy), ref, cfg, ctx)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 12 |
| `time` | `time` | 14 |
| `torch` | `torch` | 16 |
| `DPOConfig` | `config.post_training_config.DPOConfig` | 18 |
| `get_preference_iterator` | `data_loader.preference_dataset.get_preference_iterator` | 19 |
| `parse_config_with_json` | `src.post_training.cli.parse_config_with_json` | 20 |
| `ddp_setup` | `src.post_training.distributed.ddp_setup` | 21 |
| `ddp_wrap` | `src.post_training.distributed.ddp_wrap` | 21 |
| `cleanup` | `src.post_training.distributed.cleanup` | 21 |
| `reduce_scalar` | `src.post_training.distributed.reduce_scalar` | 21 |
| `dpo_loss` | `src.post_training.dpo.dpo_loss` | 22 |
| `orpo_loss` | `src.post_training.dpo.orpo_loss` | 22 |
| `kto_loss` | `src.post_training.dpo.kto_loss` | 22 |
| `implicit_accuracy` | `src.post_training.dpo.implicit_accuracy` | 22 |
| `MetricsLogger` | `src.post_training.logging_utils.MetricsLogger` | 23 |
| `configure_optimizer` | `src.post_training.optim.configure_optimizer` | 24 |
| `cosine_lr` | `src.post_training.optim.cosine_lr` | 24 |
| `sequence_logprobs` | `src.post_training.rollout.sequence_logprobs` | 25 |
| `amp_autocast` | `src.post_training.utils.amp_autocast` | 26 |
| `load_backbone_from_ckpt` | `src.post_training.utils.load_backbone_from_ckpt` | 26 |
| `make_frozen_copy` | `src.post_training.utils.make_frozen_copy` | 26 |
| `save_stage_ckpt` | `src.post_training.utils.save_stage_ckpt` | 26 |
| `set_seed` | `src.post_training.utils.set_seed` | 26 |
| `unwrap` | `src.post_training.utils.unwrap` | 26 |
