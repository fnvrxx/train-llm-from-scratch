# scripts/train_ppo.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/train_ppo.py)

## Fungsi dan kelas

### `actor_logp_values` (baris 38)

Menghitung logits/value actor sekali, lalu memilih log-prob token dan values pada frame action (B,T-1).

```python
def actor_logp_values(actor, seqs, temperature):
```

Keluaran eksplisit: `(logp, values[:, :-1])`.

### `seq_lengths_from_mask` (baris 47)

Mengambil posisi jawaban terakhir ditambah satu; fallback panjang prompt bila respons kosong.

```python
def seq_lengths_from_mask(response_mask, prompt_lens):
```

Keluaran eksplisit: `torch.where(last >= 0, last + 1, prompt_lens)`.

### `main` (baris 55)

Mengulang rollout, reward, GAE, dan update policy/value clipped pada minibatch.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `actor_logp_values`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `actor` · 38 | Policy dengan value head untuk PPO; ddp menyinkronkan gradien antar proses. | `argumen tanpa anotasi tipe` |
| `seqs` · 38 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen tanpa anotasi tipe` |
| `temperature` · 38 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen tanpa anotasi tipe` |
| `logits` · 40, 41 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `elemen 0 dari actor(seqs)`; `logits[:, :-1, :]` |
| `values` · 40 | Estimasi value pada state baru/lama/berikutnya di PPO; values pada utilitas adalah tensor angka umum. | `elemen 1 dari actor(seqs)` |
| `logp_all` · 42 | Log-softmax setiap kandidat kosakata sebelum gather token target. | `F.log_softmax(logits.float() / max(temperature, 1e-06), dim=-1)` |
| `logp` · 43 | Log-prob token yang benar-benar dipilih/diberikan setelah gather dari kosakata. | `logp_all.gather(-1, seqs[:, 1:, None]).squeeze(-1)` |

### Lingkup `seq_lengths_from_mask`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `response_mask` · 47 | Mask posisi jawaban nyata, bukan prompt/padding. | `argumen tanpa anotasi tipe` |
| `prompt_lens` · 47 | Panjang prompt bersama atau panjang prompt masing-masing baris. | `argumen tanpa anotasi tipe` |
| `N` · 49 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `elemen 0 dari response_mask.shape` |
| `T` · 49 | Banyak posisi token pada urutan yang diproses. | `elemen 1 dari response_mask.shape` |
| `pos` · 50 | Nomor posisi token untuk mencari posisi respons terakhir dari mask. | `torch.arange(T, device=response_mask.device)` |
| `last` · 51 | Lapisan terakhir value head atau indeks token respons terakhir, sesuai asal. | `torch.where(response_mask, pos[None, :], torch.full_like(response_mask, -1, dtype=torch.long)).max(dim=1).values` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 56 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `elemen 0 dari parse_config_with_json(PPOConfig, 'configs/ppo.json')` |
| `_` · 56, 100, 117 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari parse_config_with_json(PPOConfig, 'configs/ppo.json')`; `elemen 1 dari compute_logprobs(ref, seqs, rmask, temperature=cfg.temperature, requires_grad=False)`; `iterasi range(cfg.ppo_epochs)` |
| `ctx` · 57 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `ddp_setup(cfg.device)` |
| `backbone` · 60 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)` |
| `ref` · 61 | Salinan policy/reference yang dibekukan sebagai pembanding. | `make_frozen_copy(backbone, device=ctx.device)` |
| `actor` · 62 | Policy dengan value head untuk PPO; ddp menyinkronkan gradien antar proses. | `TransformerWithValueHead(backbone).to(ctx.device)` |
| `actor_ddp` · 63 | Policy dengan value head untuk PPO; ddp menyinkronkan gradien antar proses. | `ddp_wrap(actor, ctx)` |
| `optimizer` · 64 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `configure_optimizer(unwrap(actor_ddp), cfg.lr, weight_decay=0.0)` |
| `rm` · 66 | Model reward yang menghasilkan skor jawaban. | `load_reward_model(cfg, cfg.reward_ckpt, ctx.device) if cfg.reward_source == 'rm' else None` |
| `eval_set` · 68, 149 | Data evaluasi yang dimuat saat pertama diperlukan. | `None`; `load_gsm8k_eval('test', limit=200)` |
| `logger` · 69 | Pencatat metrik JSONL dan opsional W&B. | `MetricsLogger('ppo', cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project) if ctx.is_main else None` |
| `prompt_it` · 73 | Iterator prompt dan gold untuk rollout. | `get_prompt_iterator(cfg.prompt_path, cfg.prompts_per_iter, rank=ctx.rank, world_size=ctx.world_size, seed=cfg.seed)` |
| `it` · 76 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `iterasi range(cfg.iterations)` |
| `rows` · 77 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `next(prompt_it)` |
| `prompts` · 78 | Daftar prompt bertoken sebelum/untuk rollout; GRPO mengulang setiap base prompt G kali. | `[encode_prompt([{'role': 'user', 'content': r['prompt']}]) for r in rows]` |
| `r` · 78, 79 | Record prompt/gold untuk satu soal. | `iterasi rows` |
| `golds` · 79 | Kunci jawaban numerik satu soal atau daftar kunci sejajar dengan prompt. | `[r.get('gold') for r in rows]` |
| `seqs` · 84 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari rollout_prompts(actor, prompts, cfg.rollout_len, device=ctx.device, temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)` |
| `rmask` · 84 | Mask posisi jawaban nyata, bukan prompt/padding. | `elemen 1 dari rollout_prompts(actor, prompts, cfg.rollout_len, device=ctx.device, temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)` |
| `plens` · 84 | Panjang prompt bersama atau panjang prompt masing-masing baris. | `elemen 2 dari rollout_prompts(actor, prompts, cfg.rollout_len, device=ctx.device, temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)` |
| `resp` · 86 | Mask jawaban pada frame action di trainer; teks jawaban pada evaluator GSM8K. | `rmask[:, 1:]` |
| `seq_lens` · 89 | Banyak token nyata setiap urutan, sebelum right padding. | `seq_lengths_from_mask(rmask, plens)` |
| `responses` · 90 | Hasil teks generation atau contoh jawaban beserta penilaiannya. | `[decode(seqs[i, plens[i]:seq_lens[i]].tolist()) for i in range(len(rows))]` |
| `i` · 90, 95 | Indeks satu prompt/jawaban dalam batch rollout. | `iterasi range(len(rows))` |
| `task_r` · 93, 95 | List Python reward tugas per jawaban, dari verifier atau reward model. | `rm(seqs, seq_lengths=seq_lens).float().tolist()`; `[reward_gsm8k(responses[i], golds[i]) for i in range(len(rows))]` |
| `old_logp` · 99, 101 | Log-prob action saat rollout dikumpulkan, disimpan tetap selama update atas rollout itu. | `elemen 0 dari actor_logp_values(actor, seqs, cfg.temperature)`; `elemen 0 dari (old_logp.float(), old_values.float(), ref_logp.float())` |
| `old_values` · 99, 101 | Prediksi value state yang disimpan sebelum update pada rollout. | `elemen 1 dari actor_logp_values(actor, seqs, cfg.temperature)`; `elemen 1 dari (old_logp.float(), old_values.float(), ref_logp.float())` |
| `ref_logp` · 100, 101 | Log-prob action pada reference model yang dibekukan di awal tahap. | `elemen 0 dari compute_logprobs(ref, seqs, rmask, temperature=cfg.temperature, requires_grad=False)`; `elemen 2 dari (old_logp.float(), old_values.float(), ref_logp.float())` |
| `rewards` · 103 | Reward per urutan/action/kelompok sesuai fungsi; berbeda dari peluang token. | `-cfg.kl_coef * (old_logp - ref_logp) * resp.float()` |
| `last_idx` · 104, 105 | Indeks action jawaban terakhir, panjang nyata urutan dikurangi dua. | `seq_lens - 2`; `last_idx.clamp(min=0)` |
| `task_t` · 106 | Tensor float32 dari task_r untuk ditambahkan pada action terakhir. | `torch.tensor(task_r, device=ctx.device, dtype=torch.float32)` |
| `values_next` · 109 | Prediksi value state setelah action sekarang, sejajar dengan rewards (B,T-1). | `torch.cat([old_values[:, 1:], torch.zeros_like(old_values[:, :1])], dim=1)` |
| `adv` · 110, 111 | Advantage per token pada PPO atau per jawaban pada GRPO. | `elemen 0 dari compute_gae(rewards, old_values, values_next, resp, gamma=cfg.gamma, lam=cfg.gae_lambda)`; `whiten(adv, resp)` |
| `returns` · 110 | Target return untuk value head, dihitung sebagai advantage GAE ditambah value lama. | `elemen 1 dari compute_gae(rewards, old_values, values_next, resp, gamma=cfg.gamma, lam=cfg.gae_lambda)` |
| `N` · 115 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `seqs.size(0)` |
| `stats` · 116 | Metrik loss, KL, clip fraction, dan hitungan untuk agregasi. | `{'policy_loss': 0.0, 'value_loss': 0.0, 'clipfrac': 0.0, 'kl': 0.0, 'n': 0}` |
| `perm` · 118 | Urutan indeks pemilihan/pengacakan record atau baris. | `torch.randperm(N, device=ctx.device)` |
| `s` · 119 | Offset awal minibatch update PPO. | `iterasi range(0, N, cfg.minibatch_size)` |
| `mb` · 120 | Jumlah baris per update PPO dan indeks baris minibatch terpilih. | `perm[s:s + cfg.minibatch_size]` |
| `new_logp` · 122 | Log-prob action pada policy yang sedang di-update, bentuk (B,T-1). | `elemen 0 dari actor_logp_values(actor_ddp, seqs[mb], cfg.temperature)` |
| `new_values` · 122 | Prediksi value state dari actor saat update sekarang. | `elemen 1 dari actor_logp_values(actor_ddp, seqs[mb], cfg.temperature)` |
| `m` · 123 | Response mask pada baris minibatch yang sedang di-update. | `resp[mb]` |
| `p_loss` · 124 | Skalar loss policy PPO, negative clipped surrogate bermask. | `elemen 0 dari ppo_policy_loss(new_logp.float(), old_logp[mb], adv[mb], m, clip=cfg.clip)` |
| `clipf` · 124 | Proporsi token bermask yang rasionya berada di luar interval clipping. | `elemen 1 dari ppo_policy_loss(new_logp.float(), old_logp[mb], adv[mb], m, clip=cfg.clip)` |
| `v_loss` · 125 | Skalar loss value PPO dari maksimum galat kuadrat biasa dan clipped. | `ppo_value_loss(new_values.float(), old_values[mb], returns[mb], m, vf_clip=cfg.vf_clip)` |
| `loss` · 126 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `p_loss + cfg.vf_coef * v_loss` |
| `mean_reward` · 135 | Rerata skor jawaban dan panjang respons untuk monitoring. | `reduce_scalar(float(sum(task_r) / max(1, len(task_r))), ctx)` |
| `kl_ref` · 136 | Rerata perbedaan log-prob old policy versus reference untuk log. | `reduce_scalar(masked_mean(old_logp - ref_logp, resp).item(), ctx)` |
| `resp_len` · 137 | Rerata skor jawaban dan panjang respons untuk monitoring. | `reduce_scalar(resp.float().sum(1).mean().item(), ctx)` |
| `n` · 138 | Jumlah minibatch update pada stats, minimal satu untuk averaging. | `max(1, stats['n'])` |
| `res` · 150 | Representasi residual yang disimpan untuk penjumlahan atau inspeksi; res pada evaluator juga dictionary hasil. | `gsm8k_accuracy(unwrap(actor_ddp), eval_set, device=ctx.device, max_new_tokens=cfg.rollout_len)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 13 |
| `time` | `time` | 15 |
| `torch` | `torch` | 17 |
| `F` | `torch.nn.functional` | 18 |
| `PPOConfig` | `config.post_training_config.PPOConfig` | 20 |
| `get_prompt_iterator` | `data_loader.prompt_dataset.get_prompt_iterator` | 21 |
| `parse_config_with_json` | `src.post_training.cli.parse_config_with_json` | 22 |
| `EOT_ID` | `src.post_training.chat_template.EOT_ID` | 23 |
| `decode` | `src.post_training.chat_template.decode` | 23 |
| `encode_prompt` | `src.post_training.chat_template.encode_prompt` | 23 |
| `ddp_setup` | `src.post_training.distributed.ddp_setup` | 24 |
| `ddp_wrap` | `src.post_training.distributed.ddp_wrap` | 24 |
| `cleanup` | `src.post_training.distributed.cleanup` | 24 |
| `reduce_scalar` | `src.post_training.distributed.reduce_scalar` | 24 |
| `gsm8k_accuracy` | `src.post_training.evaluation.gsm8k_accuracy` | 25 |
| `load_gsm8k_eval` | `src.post_training.evaluation.load_gsm8k_eval` | 25 |
| `MetricsLogger` | `src.post_training.logging_utils.MetricsLogger` | 26 |
| `configure_optimizer` | `src.post_training.optim.configure_optimizer` | 27 |
| `compute_gae` | `src.post_training.ppo.compute_gae` | 28 |
| `whiten` | `src.post_training.ppo.whiten` | 28 |
| `ppo_policy_loss` | `src.post_training.ppo.ppo_policy_loss` | 28 |
| `ppo_value_loss` | `src.post_training.ppo.ppo_value_loss` | 28 |
| `approx_kl` | `src.post_training.ppo.approx_kl` | 28 |
| `load_reward_model` | `src.post_training.reward_model.load_reward_model` | 29 |
| `reward_gsm8k` | `src.post_training.rewards.reward_gsm8k` | 30 |
| `compute_logprobs` | `src.post_training.rollout.compute_logprobs` | 31 |
| `rollout_prompts` | `src.post_training.rollout.rollout_prompts` | 31 |
| `amp_autocast` | `src.post_training.utils.amp_autocast` | 32 |
| `load_backbone_from_ckpt` | `src.post_training.utils.load_backbone_from_ckpt` | 32 |
| `make_frozen_copy` | `src.post_training.utils.make_frozen_copy` | 32 |
| `masked_mean` | `src.post_training.utils.masked_mean` | 32 |
| `save_stage_ckpt` | `src.post_training.utils.save_stage_ckpt` | 32 |
| `set_seed` | `src.post_training.utils.set_seed` | 32 |
| `unwrap` | `src.post_training.utils.unwrap` | 32 |
| `TransformerWithValueHead` | `src.post_training.value_head.TransformerWithValueHead` | 35 |
