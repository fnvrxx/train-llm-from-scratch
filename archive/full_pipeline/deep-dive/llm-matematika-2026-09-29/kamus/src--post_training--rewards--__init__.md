# src/post_training/rewards/__init__.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/rewards/__init__.py)

## Fungsi dan kelas

Berkas ini berisi konfigurasi, ekspor, atau penanda paket; tidak mendefinisikan fungsi/kelas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `__all__` · 16 | Daftar simbol yang diekspor paket pada wildcard import. | `['extract_answer', 'extract_think', 'parse_number', 'gsm8k_gold_answer', 'reward_gsm8k', 'reward_arithmetic', 'reward_format', 'is_correct']` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `extract_answer` | `src.post_training.rewards.parsing.extract_answer` | 3 |
| `extract_think` | `src.post_training.rewards.parsing.extract_think` | 3 |
| `parse_number` | `src.post_training.rewards.parsing.parse_number` | 3 |
| `gsm8k_gold_answer` | `src.post_training.rewards.parsing.gsm8k_gold_answer` | 3 |
| `reward_gsm8k` | `src.post_training.rewards.verifiers.reward_gsm8k` | 9 |
| `reward_arithmetic` | `src.post_training.rewards.verifiers.reward_arithmetic` | 9 |
| `reward_format` | `src.post_training.rewards.verifiers.reward_format` | 9 |
| `is_correct` | `src.post_training.rewards.verifiers.is_correct` | 9 |
