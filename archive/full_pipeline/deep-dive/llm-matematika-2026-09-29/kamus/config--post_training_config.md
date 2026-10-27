# config/post_training_config.py

[Indeks kamus](README.md) · [Kode sumber](../../../config/post_training_config.py)

## Fungsi dan kelas

### `BaseModelConfig` (baris 28)

Field arsitektur dan runtime yang diwarisi konfigurasi tahap.

### `PretrainConfig` (baris 48)

Field data corpus, optimizer, accumulation, dan checkpoint pretraining.

### `SFTConfig` (baris 67)

Field checkpoint awal, data demonstrasi, epochs, dan optimizer SFT.

### `RewardConfig` (baris 85)

Field training penilai reward dari pasangan preferensi.

### `DPOConfig` (baris 101)

Field objective DPO/ORPO/KTO dan data preferensi.

### `PPOConfig` (baris 120)

Field rollout, scoring, GAE, clipping, dan update policy/value.

### `GRPOConfig` (baris 148)

Field kelompok jawaban, curriculum, clipping, dan penalti reference.

### `smoke` (baris 176)

Membuat instance konfigurasi dengan dimensi sintetis kecil dari SMOKE.

```python
def smoke(cfg_cls):
```

Keluaran eksplisit: `replace(cfg_cls(), **SMOKE)`.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `EPHEMERAL` · 21 | Akar direktori data/artefak; /ephemeral adalah default lingkungan penulis. | `'/ephemeral'` |
| `CKPT_DIR` · 22 | Direktori penyimpanan checkpoint. | `f'{EPHEMERAL}/ckpts'` |
| `DATA_DIR` · 23 | Akar direktori data/artefak; /ephemeral adalah default lingkungan penulis. | `f'{EPHEMERAL}/data'` |
| `LOG_DIR` · 24 | Direktori penyimpanan log training. | `f'{EPHEMERAL}/logs'` |
| `SMOKE` · 171 | Konfigurasi dimensi kecil untuk token sintetis; vocab 256 tidak cukup untuk token chat r50k_base nyata. | `dict(vocab_size=256, context_length=64, n_embed=64, n_head=4, n_blocks=2, device='cpu', amp_dtype=None)` |

### Lingkup `BaseModelConfig`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `vocab_size` · 30 | Jumlah ID kandidat token, yaitu lebar keluaran lm_head. | `50304` |
| `context_length` · 31 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `1024` |
| `n_embed` · 32 | Banyak koordinat vektor representasi satu token. | `1024` |
| `n_head` · 33 | Jumlah attention head dalam setiap blok. | `16` |
| `n_blocks` · 34 | Jumlah blok Transformer yang disusun berurutan. | `24` |
| `device` · 37 | Lokasi komputasi, misalnya cpu atau cuda. | `'cuda'` |
| `amp_dtype` · 38 | Tipe atau nama presisi autocast, biasanya bf16/fp16; None berarti tanpa autocast. | `'bf16'` |
| `seed` · 39 | Benih generator acak untuk inisialisasi dan pengacakan yang dapat diulang. | `1337` |
| `compile` · 40 | Opsi torch.compile; hanya berlaku jika trainer benar-benar memakai field ini. | `False` |
| `ckpt_dir` · 41 | Direktori penyimpanan checkpoint. | `CKPT_DIR` |
| `log_dir` · 42 | Direktori penyimpanan log training. | `LOG_DIR` |
| `use_wandb` · 43 | Sakelar W&B, nama proyek W&B, dan nama eksperimen. | `False` |
| `wandb_project` · 44 | Sakelar W&B, nama proyek W&B, dan nama eksperimen. | `'train-llm-from-scratch-posttrain'` |

### Lingkup `PretrainConfig`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `train_path` · 50 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `'data/train/pile_train.h5'` |
| `dev_path` · 51 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `'data/val/pile_dev.h5'` |
| `batch_size` · 52 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `24` |
| `grad_accum` · 53 | Jumlah micro-batch per optimizer step jika loop trainer mengimplementasikannya. | `8` |
| `train_steps` · 54 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `200000` |
| `eval_steps` · 55 | Interval step/iterasi antar evaluasi. | `1000` |
| `eval_iters` · 56 | Jumlah atau batas batch untuk estimasi metrik evaluasi. | `100` |
| `warmup_steps` · 57 | Jumlah step awal untuk kenaikan learning rate linear. | `2000` |
| `lr` · 58 | Learning rate, ukuran langkah pembaruan parameter. | `0.0003` |
| `min_lr` · 59 | Learning rate minimum/akhir atau nilai sesudah ambang decay trainer awal. | `3e-05` |
| `weight_decay` · 60 | Koefisien penyusutan bobot pada kelompok parameter yang menerapkannya. | `0.1` |
| `grad_clip` · 61 | Batas norma gabungan gradien sebelum update. | `1.0` |
| `out_ckpt` · 62 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `f'{CKPT_DIR}/base_pretrained.pt'` |
| `save_every` · 63 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `2000` |

### Lingkup `SFTConfig`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `pretrained_ckpt` · 68 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `f'{CKPT_DIR}/base_pretrained.pt'` |
| `data_path` · 69 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/sft_packed.h5'` |
| `out_ckpt` · 70 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `f'{CKPT_DIR}/sft.pt'` |
| `batch_size` · 71 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `16` |
| `grad_accum` · 72 | Jumlah micro-batch per optimizer step jika loop trainer mengimplementasikannya. | `2` |
| `epochs` · 73 | Nomor putaran data atau jumlah putaran yang diminta. | `3` |
| `max_steps` · 74 | Batas update; max_steps negatif pada konfigurasi SFT berarti gunakan batas epochs. | `-1` |
| `eval_steps` · 75 | Interval step/iterasi antar evaluasi. | `200` |
| `warmup_steps` · 76 | Jumlah step awal untuk kenaikan learning rate linear. | `100` |
| `lr` · 77 | Learning rate, ukuran langkah pembaruan parameter. | `1e-05` |
| `min_lr` · 78 | Learning rate minimum/akhir atau nilai sesudah ambang decay trainer awal. | `1e-06` |
| `weight_decay` · 79 | Koefisien penyusutan bobot pada kelompok parameter yang menerapkannya. | `0.0` |
| `grad_clip` · 80 | Batas norma gabungan gradien sebelum update. | `1.0` |
| `save_every` · 81 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `500` |

### Lingkup `RewardConfig`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `sft_ckpt` · 86 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `f'{CKPT_DIR}/sft.pt'` |
| `pref_path` · 87 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/preferences.jsonl'` |
| `out_ckpt` · 88 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `f'{CKPT_DIR}/reward.pt'` |
| `batch_size` · 89 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `8` |
| `epochs` · 90 | Nomor putaran data atau jumlah putaran yang diminta. | `1` |
| `eval_steps` · 91 | Interval step/iterasi antar evaluasi. | `200` |
| `warmup_steps` · 92 | Jumlah step awal untuk kenaikan learning rate linear. | `50` |
| `lr` · 93 | Learning rate, ukuran langkah pembaruan parameter. | `1e-05` |
| `weight_decay` · 94 | Koefisien penyusutan bobot pada kelompok parameter yang menerapkannya. | `0.0` |
| `grad_clip` · 95 | Batas norma gabungan gradien sebelum update. | `1.0` |
| `max_len` · 96 | Batas panjang token prompt+response setelah encoding dan pemotongan. | `768` |
| `save_every` · 97 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `500` |

### Lingkup `DPOConfig`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `sft_ckpt` · 102 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `f'{CKPT_DIR}/sft.pt'` |
| `pref_path` · 103 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/preferences.jsonl'` |
| `out_ckpt` · 104 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `f'{CKPT_DIR}/dpo.pt'` |
| `loss_type` · 105 | Pemilih objective dpo, orpo, atau kto. | `'dpo'` |
| `beta` · 106 | Skala objective preferensi DPO/KTO, bukan learning rate atau sampling temperature. | `0.1` |
| `orpo_lambda` · 107 | Bobot penalti preferensi ORPO yang ditambahkan ke NLL. | `1.0` |
| `batch_size` · 108 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `8` |
| `epochs` · 109 | Nomor putaran data atau jumlah putaran yang diminta. | `1` |
| `eval_steps` · 110 | Interval step/iterasi antar evaluasi. | `200` |
| `warmup_steps` · 111 | Jumlah step awal untuk kenaikan learning rate linear. | `50` |
| `lr` · 112 | Learning rate, ukuran langkah pembaruan parameter. | `5e-07` |
| `weight_decay` · 113 | Koefisien penyusutan bobot pada kelompok parameter yang menerapkannya. | `0.0` |
| `grad_clip` · 114 | Batas norma gabungan gradien sebelum update. | `1.0` |
| `max_len` · 115 | Batas panjang token prompt+response setelah encoding dan pemotongan. | `768` |
| `save_every` · 116 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `500` |

### Lingkup `PPOConfig`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `sft_ckpt` · 121 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `f'{CKPT_DIR}/sft.pt'` |
| `reward_ckpt` · 122 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `f'{CKPT_DIR}/reward.pt'` |
| `prompt_path` · 123 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/rl_prompts_train.jsonl'` |
| `eval_prompt_path` · 124 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/rl_prompts_test.jsonl'` |
| `out_ckpt` · 125 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `f'{CKPT_DIR}/ppo.pt'` |
| `reward_source` · 126 | Pemilih penilai PPO: verifier numerik atau rm yang dimuat dari checkpoint reward. | `'verifier'` |
| `iterations` · 127 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `1000` |
| `prompts_per_iter` · 128 | Banyak prompt unik per iterasi RL per rank. | `32` |
| `rollout_len` · 129 | Batas jumlah token tambahan di luar prompt. | `300` |
| `temperature` · 130 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `1.0` |
| `top_p` · 131 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `1.0` |
| `ppo_epochs` · 132 | Banyak putaran update atas kumpulan rollout yang sama. | `4` |
| `minibatch_size` · 133 | Jumlah baris per update PPO dan indeks baris minibatch terpilih. | `16` |
| `clip` · 134 | Epsilon lebar interval rasio [1-clip,1+clip] pada PPO/GRPO. | `0.2` |
| `vf_clip` · 135 | Batas perubahan prediksi value pada cabang clipped value loss. | `0.2` |
| `vf_coef` · 136 | Bobot pengali v_loss ketika ditambahkan ke p_loss. | `0.5` |
| `ent_coef` · 137 | Field bonus entropy; trainer PPO lokal belum menggunakannya dalam total loss. | `0.0` |
| `gamma` · 138 | Discount factor untuk reward/value mendatang. | `1.0` |
| `gae_lambda` · 139 | Koefisien jejak temporal difference pada GAE. | `0.95` |
| `kl_coef` · 140 | Bobot penalti penyimpangan terhadap reference. | `0.05` |
| `lr` · 141 | Learning rate, ukuran langkah pembaruan parameter. | `1e-06` |
| `grad_clip` · 142 | Batas norma gabungan gradien sebelum update. | `1.0` |
| `eval_every` · 143 | Interval step/iterasi antar evaluasi. | `50` |
| `save_every` · 144 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `100` |

### Lingkup `GRPOConfig`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `sft_ckpt` · 149 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `f'{CKPT_DIR}/sft.pt'` |
| `prompt_path` · 150 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/rl_prompts_train.jsonl'` |
| `eval_prompt_path` · 151 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/rl_prompts_test.jsonl'` |
| `curriculum_path` · 152 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `f'{DATA_DIR}/arithmetic_prompts.jsonl'` |
| `curriculum_iters` · 153 | Jumlah iterasi memakai aritmetika sebelum pindah ke tugas utama. | `100` |
| `out_ckpt` · 154 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `f'{CKPT_DIR}/grpo.pt'` |
| `iterations` · 155 | Banyak/indeks iterasi RL; it juga iterator data jika berasal dari get_*_iterator. | `1000` |
| `prompts_per_iter` · 156 | Banyak prompt unik per iterasi RL per rank. | `8` |
| `group_size` · 157 | Jumlah jawaban per prompt pada GRPO. | `8` |
| `rollout_len` · 158 | Batas jumlah token tambahan di luar prompt. | `300` |
| `temperature` · 159 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `1.0` |
| `top_p` · 160 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `1.0` |
| `grpo_epochs` · 161 | Banyak putaran update atas kumpulan rollout yang sama. | `1` |
| `clip` · 162 | Epsilon lebar interval rasio [1-clip,1+clip] pada PPO/GRPO. | `0.2` |
| `kl_coef` · 163 | Bobot penalti penyimpangan terhadap reference. | `0.04` |
| `lr` · 164 | Learning rate, ukuran langkah pembaruan parameter. | `1e-06` |
| `grad_clip` · 165 | Batas norma gabungan gradien sebelum update. | `1.0` |
| `eval_every` · 166 | Interval step/iterasi antar evaluasi. | `50` |
| `save_every` · 167 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `100` |

### Lingkup `smoke`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg_cls` · 176 | Kelas dataclass konfigurasi yang akan dibuat instance-nya. | `argumen tanpa anotasi tipe` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 15 |
| `dataclass` | `dataclasses.dataclass` | 17 |
| `field` | `dataclasses.field` | 17 |
| `replace` | `dataclasses.replace` | 17 |
