# Kamus fungsi dan variabel pipeline LLM

[Daftar bab](../../llm-matematika-2026-09-29.md)

Cari berkas yang sedang Anda baca, lalu cari nama fungsi atau variabelnya. Setiap nama dikelompokkan menurut lingkup dan diberi arti, baris, serta ekspresi asal. Nama pendek yang dipakai ulang perlu dibaca bersama lingkup dan asal nilainya. Bab 1 sampai 7 menjelaskan rumus, bentuk tensor, alasan desain, dan alternatifnya.

Cakupan: semua berkas Python dalam src (termasuk models), data_loader, config, dan scripts. Impor dicatat sebagai alias. Parameter lambda dicatat pada lingkup lambda. UI, generator gambar, notebook, eksperimen alternatif, dan test bukan bagian kamus pipeline ini. Tautan berkas membuka sumber; nomor baris tertulis merujuk snapshot 29 September 2026.

Jika nama diberi nilai berkali-kali, seluruh lokasi binding dicatat dan maksimal tiga ekspresi pertama ditampilkan. Kolom asal merupakan cuplikan kode, bukan rumus pengganti bab utama. Atribut konfigurasi yang hanya dibaca mengacu deklarasi dataclass-nya. Assignment ke elemen array/dictionary memperbarui isi, bukan mendeklarasikan nama baru.

| Berkas | Fungsi/kelas | Nama dalam lingkup |
|---|---:|---:|
| [config/__init__.py](config--__init__.md) | 0 | 0 |
| [config/config.py](config--config.md) | 0 | 26 |
| [config/loader.py](config--loader.md) | 3 | 20 |
| [config/post_training_config.py](config--post_training_config.md) | 8 | 117 |
| [data_loader/__init__.py](data_loader--__init__.md) | 0 | 0 |
| [data_loader/data_loader.py](data_loader--data_loader.md) | 1 | 22 |
| [data_loader/preference_dataset.py](data_loader--preference_dataset.md) | 4 | 44 |
| [data_loader/prompt_dataset.py](data_loader--prompt_dataset.md) | 2 | 15 |
| [data_loader/sft_dataset.py](data_loader--sft_dataset.md) | 1 | 18 |
| [scripts/chat.py](scripts--chat.md) | 2 | 6 |
| [scripts/data_download.py](scripts--data_download.md) | 3 | 23 |
| [scripts/data_preprocess.py](scripts--data_preprocess.md) | 2 | 21 |
| [scripts/eval_post_training.py](scripts--eval_post_training.md) | 3 | 22 |
| [scripts/generate_text.py](scripts--generate_text.md) | 3 | 16 |
| [scripts/prepare_preference_data.py](scripts--prepare_preference_data.md) | 6 | 38 |
| [scripts/prepare_pretrain_data.py](scripts--prepare_pretrain_data.md) | 6 | 41 |
| [scripts/prepare_rl_prompts.py](scripts--prepare_rl_prompts.md) | 4 | 25 |
| [scripts/prepare_sft_data.py](scripts--prepare_sft_data.md) | 7 | 44 |
| [scripts/pretrain_base.py](scripts--pretrain_base.md) | 3 | 44 |
| [scripts/train_dpo.py](scripts--train_dpo.md) | 4 | 59 |
| [scripts/train_grpo.py](scripts--train_grpo.md) | 2 | 51 |
| [scripts/train_ppo.py](scripts--train_ppo.md) | 3 | 64 |
| [scripts/train_reward.py](scripts--train_reward.md) | 3 | 43 |
| [scripts/train_sft.py](scripts--train_sft.md) | 2 | 36 |
| [scripts/train_transformer.py](scripts--train_transformer.md) | 22 | 130 |
| [src/__init__.py](src--__init__.md) | 0 | 0 |
| [src/models/__init__.py](src--models--__init__.md) | 0 | 0 |
| [src/models/attention.py](src--models--attention.md) | 6 | 38 |
| [src/models/mlp.py](src--models--mlp.md) | 5 | 17 |
| [src/models/transformer.py](src--models--transformer.md) | 7 | 62 |
| [src/models/transformer_block.py](src--models--transformer_block.md) | 4 | 21 |
| [src/post_training/__init__.py](src--post_training--__init__.md) | 0 | 0 |
| [src/post_training/chat_template.py](src--post_training--chat_template.md) | 7 | 29 |
| [src/post_training/cli.py](src--post_training--cli.md) | 3 | 32 |
| [src/post_training/distributed.py](src--post_training--distributed.md) | 10 | 25 |
| [src/post_training/dpo.py](src--post_training--dpo.md) | 5 | 38 |
| [src/post_training/evaluation.py](src--post_training--evaluation.md) | 4 | 48 |
| [src/post_training/grpo.py](src--post_training--grpo.md) | 3 | 26 |
| [src/post_training/inference.py](src--post_training--inference.md) | 2 | 24 |
| [src/post_training/logging_utils.py](src--post_training--logging_utils.md) | 4 | 18 |
| [src/post_training/optim.py](src--post_training--optim.md) | 2 | 16 |
| [src/post_training/ppo.py](src--post_training--ppo.md) | 5 | 41 |
| [src/post_training/reward_model.py](src--post_training--reward_model.md) | 6 | 23 |
| [src/post_training/reward_train.py](src--post_training--reward_train.md) | 3 | 6 |
| [src/post_training/rewards/__init__.py](src--post_training--rewards--__init__.md) | 0 | 1 |
| [src/post_training/rewards/parsing.py](src--post_training--rewards--parsing.md) | 5 | 16 |
| [src/post_training/rewards/verifiers.py](src--post_training--rewards--verifiers.md) | 4 | 13 |
| [src/post_training/rollout.py](src--post_training--rollout.md) | 9 | 103 |
| [src/post_training/sft.py](src--post_training--sft.md) | 2 | 16 |
| [src/post_training/utils.py](src--post_training--utils.md) | 13 | 47 |
| [src/post_training/value_head.py](src--post_training--value_head.md) | 5 | 15 |

Total: 51 berkas, 208 deklarasi fungsi/kelas, dan 1600 nama dalam lingkup. Nama yang sama pada dua fungsi dihitung dua kali; deklarasi mencakup parameter, variabel, atribut, loop, dan field dataclass.
