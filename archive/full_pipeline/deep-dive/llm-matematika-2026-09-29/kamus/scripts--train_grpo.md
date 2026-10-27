# scripts/train_grpo.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/train_grpo.py)

## Fungsi dan kelas

### `seq_lengths_from_mask` (baris 36)

Mengambil posisi jawaban terakhir ditambah satu; fallback panjang prompt bila respons kosong.

```python
def seq_lengths_from_mask(response_mask, prompt_lens):
```

Keluaran eksplisit: `torch.where(last >= 0, last + 1, prompt_lens)`.

### `main` (baris 43)

Mengulang generation kelompok, verifier, advantage relatif, dan update GRPO.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `seq_lengths_from_mask`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `response_mask` · 36 | Mask posisi jawaban nyata, bukan prompt/padding. | `argumen tanpa anotasi tipe` |
| `prompt_lens` · 36 | Panjang prompt bersama atau panjang prompt masing-masing baris. | `argumen tanpa anotasi tipe` |
| `N` · 37 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `elemen 0 dari response_mask.shape` |
| `T` · 37 | Banyak posisi token pada urutan yang diproses. | `elemen 1 dari response_mask.shape` |
| `pos` · 38 | Nomor posisi token untuk mencari posisi respons terakhir dari mask. | `torch.arange(T, device=response_mask.device)` |
| `last` · 39 | Lapisan terakhir value head atau indeks token respons terakhir, sesuai asal. | `torch.where(response_mask, pos[None, :], torch.full_like(response_mask, -1, dtype=torch.long)).max(dim=1).values` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 44 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `elemen 0 dari parse_config_with_json(GRPOConfig, 'configs/grpo.json')` |
| `_` · 44, 68, 69, 84, 85, 91, 96 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari parse_config_with_json(GRPOConfig, 'configs/grpo.json')`; `iterasi range(G)`; `elemen 1 dari compute_logprobs(policy, seqs, rmask, temperature=cfg.temperature, requires_grad=False)` |
| `ctx` · 45 | Konteks DDP pada trainer; context manager autograd pada log-prob; teks konteks pada konversi Dolly. | `ddp_setup(cfg.device)` |
| `policy` · 48 | Model distribusi token yang dilatih; akhiran ddp berarti pembungkus terdistribusi. | `load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)` |
| `ref` · 49 | Salinan policy/reference yang dibekukan sebagai pembanding. | `make_frozen_copy(policy, device=ctx.device)` |
| `policy_ddp` · 50 | Model distribusi token yang dilatih; akhiran ddp berarti pembungkus terdistribusi. | `ddp_wrap(policy, ctx)` |
| `optimizer` · 51 | AdamW yang menyimpan momen gradien dan memperbarui parameter. | `configure_optimizer(unwrap(policy_ddp), cfg.lr, weight_decay=0.0)` |
| `eval_set` · 53, 121 | Data evaluasi yang dimuat saat pertama diperlukan. | `None`; `load_gsm8k_eval('test', limit=200)` |
| `logger` · 54 | Pencatat metrik JSONL dan opsional W&B. | `MetricsLogger('grpo', cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project) if ctx.is_main else None` |
| `warm_it` · 58 | Iterator soal aritmetika curriculum sebelum fase utama GRPO. | `get_prompt_iterator(cfg.curriculum_path, cfg.prompts_per_iter, rank=ctx.rank, world_size=ctx.world_size, seed=cfg.seed)` |
| `main_it` · 60 | Iterator prompt tugas utama GRPO. | `get_prompt_iterator(cfg.prompt_path, cfg.prompts_per_iter, rank=ctx.rank, world_size=ctx.world_size, seed=cfg.seed)` |
| `G` · 63 | Jumlah jawaban per prompt pada GRPO. | `cfg.group_size` |
| `it` · 64 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `iterasi range(cfg.iterations)` |
| `rows` · 65 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `next(warm_it if it < cfg.curriculum_iters else main_it)` |
| `base_prompts` · 67 | Prompt unik yang sudah ditokenisasi sebelum diulang G kali pada GRPO. | `[encode_prompt([{'role': 'user', 'content': r['prompt']}]) for r in rows]` |
| `r` · 67, 69 | Record prompt/gold dari iterator tugas. | `iterasi rows` |
| `prompts` · 68 | Daftar prompt bertoken sebelum/untuk rollout; GRPO mengulang setiap base prompt G kali. | `[p for p in base_prompts for _ in range(G)]` |
| `p` · 68 | Satu prompt bertoken yang diulang G kali. | `iterasi base_prompts` |
| `golds` · 69 | Kunci jawaban numerik satu soal atau daftar kunci sejajar dengan prompt. | `[r.get('gold') for r in rows for _ in range(G)]` |
| `seqs` · 73 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari rollout_prompts(policy, prompts, cfg.rollout_len, device=ctx.device, temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)` |
| `rmask` · 73 | Mask posisi jawaban nyata, bukan prompt/padding. | `elemen 1 dari rollout_prompts(policy, prompts, cfg.rollout_len, device=ctx.device, temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)` |
| `plens` · 73 | Panjang prompt bersama atau panjang prompt masing-masing baris. | `elemen 2 dari rollout_prompts(policy, prompts, cfg.rollout_len, device=ctx.device, temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)` |
| `resp` · 75 | Mask jawaban pada frame action di trainer; teks jawaban pada evaluator GSM8K. | `rmask[:, 1:]` |
| `seq_lens` · 77 | Banyak token nyata setiap urutan, sebelum right padding. | `seq_lengths_from_mask(rmask, plens)` |
| `responses` · 78 | Hasil teks generation atau contoh jawaban beserta penilaiannya. | `[decode(seqs[i, plens[i]:seq_lens[i]].tolist()) for i in range(len(prompts))]` |
| `i` · 78, 79 | Indeks jawaban dalam batch prompt yang sudah diulang per kelompok. | `iterasi range(len(prompts))` |
| `rewards` · 79 | Reward per urutan/action/kelompok sesuai fungsi; berbeda dari peluang token. | `torch.tensor([reward_gsm8k(responses[i], golds[i]) for i in range(len(prompts))], device=ctx.device, dtype=torch.float32)` |
| `adv` · 81 | Advantage per token pada PPO atau per jawaban pada GRPO. | `group_advantages(rewards, G)` |
| `old_logp` · 84, 86 | Log-prob action saat rollout dikumpulkan, disimpan tetap selama update atas rollout itu. | `elemen 0 dari compute_logprobs(policy, seqs, rmask, temperature=cfg.temperature, requires_grad=False)`; `elemen 0 dari (old_logp.float(), ref_logp.float())` |
| `ref_logp` · 85, 86 | Log-prob action pada reference model yang dibekukan di awal tahap. | `elemen 0 dari compute_logprobs(ref, seqs, rmask, temperature=cfg.temperature, requires_grad=False)`; `elemen 1 dari (old_logp.float(), ref_logp.float())` |
| `N` · 89 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `seqs.size(0)` |
| `agg` · 90 | Metrik loss, KL, clip fraction, dan hitungan untuk agregasi. | `{'loss': 0.0, 'kl': 0.0, 'clipfrac': 0.0, 'n': 0}` |
| `perm` · 92 | Urutan indeks pemilihan/pengacakan record atau baris. | `torch.randperm(N, device=ctx.device)` |
| `s` · 93 | Offset awal minibatch update, dengan langkah sebesar G. | `iterasi range(0, N, max(1, G))` |
| `mb` · 94 | Jumlah baris per update PPO dan indeks baris minibatch terpilih. | `perm[s:s + max(1, G)]` |
| `new_logp` · 96 | Log-prob action pada policy yang sedang di-update, bentuk (B,T-1). | `elemen 0 dari compute_logprobs(policy_ddp, seqs[mb], rmask[mb], temperature=cfg.temperature, requires_grad=True)` |
| `loss` · 97 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 0 dari grpo_loss(new_logp.float(), old_logp[mb], ref_logp[mb], adv[mb], resp[mb], clip=cfg.clip, kl_coef=cfg.kl_coef)` |
| `st` · 97 | Metrik loss, KL, clip fraction, dan hitungan untuk agregasi. | `elemen 1 dari grpo_loss(new_logp.float(), old_logp[mb], ref_logp[mb], adv[mb], resp[mb], clip=cfg.clip, kl_coef=cfg.kl_coef)` |
| `mean_reward` · 105 | Rerata skor jawaban dan panjang respons untuk monitoring. | `reduce_scalar(rewards.mean().item(), ctx)` |
| `grp_std` · 107 | Simpangan baku reward setiap kelompok untuk mengukur variasi sinyal. | `rewards.view(-1, G).std(dim=1)` |
| `informative` · 108 | Proporsi kelompok dengan simpangan baku reward melebihi ambang 1e-6. | `reduce_scalar((grp_std > 1e-06).float().mean().item(), ctx)` |
| `resp_len` · 109 | Rerata skor jawaban dan panjang respons untuk monitoring. | `reduce_scalar(resp.float().sum(1).mean().item(), ctx)` |
| `n` · 110 | Jumlah minibatch update yang diakumulasikan pada agg, minimal satu untuk pembagian. | `max(1, agg['n'])` |
| `phase` · 112 | Nama fase curriculum untuk laporan. | `'warmup' if it < cfg.curriculum_iters else 'gsm8k'` |
| `res` · 122 | Representasi residual yang disimpan untuk penjumlahan atau inspeksi; res pada evaluator juga dictionary hasil. | `gsm8k_accuracy(unwrap(policy_ddp), eval_set, device=ctx.device, max_new_tokens=cfg.rollout_len)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 14 |
| `time` | `time` | 16 |
| `torch` | `torch` | 18 |
| `GRPOConfig` | `config.post_training_config.GRPOConfig` | 20 |
| `get_prompt_iterator` | `data_loader.prompt_dataset.get_prompt_iterator` | 21 |
| `parse_config_with_json` | `src.post_training.cli.parse_config_with_json` | 22 |
| `decode` | `src.post_training.chat_template.decode` | 23 |
| `encode_prompt` | `src.post_training.chat_template.encode_prompt` | 23 |
| `ddp_setup` | `src.post_training.distributed.ddp_setup` | 24 |
| `ddp_wrap` | `src.post_training.distributed.ddp_wrap` | 24 |
| `cleanup` | `src.post_training.distributed.cleanup` | 24 |
| `reduce_scalar` | `src.post_training.distributed.reduce_scalar` | 24 |
| `gsm8k_accuracy` | `src.post_training.evaluation.gsm8k_accuracy` | 25 |
| `load_gsm8k_eval` | `src.post_training.evaluation.load_gsm8k_eval` | 25 |
| `group_advantages` | `src.post_training.grpo.group_advantages` | 26 |
| `grpo_loss` | `src.post_training.grpo.grpo_loss` | 26 |
| `MetricsLogger` | `src.post_training.logging_utils.MetricsLogger` | 27 |
| `configure_optimizer` | `src.post_training.optim.configure_optimizer` | 28 |
| `reward_gsm8k` | `src.post_training.rewards.reward_gsm8k` | 29 |
| `compute_logprobs` | `src.post_training.rollout.compute_logprobs` | 30 |
| `rollout_prompts` | `src.post_training.rollout.rollout_prompts` | 30 |
| `amp_autocast` | `src.post_training.utils.amp_autocast` | 31 |
| `load_backbone_from_ckpt` | `src.post_training.utils.load_backbone_from_ckpt` | 31 |
| `make_frozen_copy` | `src.post_training.utils.make_frozen_copy` | 31 |
| `save_stage_ckpt` | `src.post_training.utils.save_stage_ckpt` | 31 |
| `set_seed` | `src.post_training.utils.set_seed` | 31 |
| `unwrap` | `src.post_training.utils.unwrap` | 31 |
