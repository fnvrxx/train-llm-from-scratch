# src/post_training/evaluation.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/evaluation.py)

## Fungsi dan kelas

### `_model_context_length` (baris 18)

Membaca kapasitas konteks backbone di balik wrapper, atau fallback bila tidak ada.

```python
def _model_context_length(model) -> int:
```

Keluaran eksplisit: `getattr(inner, 'context_length', 1024)`.

### `batched_generate` (baris 24)

Generation dalam bucket prompt sama panjang; greedy memakai top_k=1; hasil dikembalikan menurut urutan input.

```python
def batched_generate(model, prompts: list[list[int]], max_new_tokens: int, *, device: str, temperature: float=1.0, top_k: int | None=None, top_p: float | None=None, greedy: bool=False, micro_batch: int=32) -> list[str]:
```

Keluaran eksplisit: `[r or '' for r in results]`.

### `gsm8k_accuracy` (baris 77)

Menghitung proporsi jawaban numerik benar dan opsional mengumpulkan contoh hasil.

```python
def gsm8k_accuracy(model, qa_pairs: list[tuple[str, str]], *, device: str, max_new_tokens: int=300, greedy: bool=True, return_samples: int=0) -> dict:
```

Keluaran eksplisit: `{'accuracy': correct / max(1, n), 'n': n, 'correct': correct, 'samples': samples}`.

### `load_gsm8k_eval` (baris 107)

Memuat pasangan pertanyaan/answer field GSM8K dengan batas jumlah contoh.

```python
def load_gsm8k_eval(split: str='test', limit: int | None=200) -> list[tuple[str, str]]:
```

Keluaran eksplisit: `[(ex['question'], ex['answer']) for ex in ds]`.

## Variabel dan parameter menurut lingkup

### Lingkup `_model_context_length`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 18 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `inner` · 19 | Backbone yang diperoleh dari wrapper model. | `getattr(model, 'transformer', model)` |

### Lingkup `batched_generate`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 25 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `prompts` · 26 | Daftar prompt bertoken sebelum/untuk rollout; GRPO mengulang setiap base prompt G kali. | `argumen list[list[int]]` |
| `max_new_tokens` · 27 | Batas jumlah token tambahan di luar prompt. | `argumen int` |
| `device` · 29 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `temperature` · 30, 44 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 1.0`; `elemen 0 dari (1.0, 1, None)` |
| `top_k` · 31, 44 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen int &#124; None; default None`; `elemen 1 dari (1.0, 1, None)` |
| `top_p` · 32, 44 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen float &#124; None; default None`; `elemen 2 dari (1.0, 1, None)` |
| `greedy` · 33 | Mode top_k=1; beberapa kandidat maksimum seri dapat tetap lolos implementasi threshold. | `argumen bool; default False` |
| `micro_batch` · 34 | Banyak prompt sama panjang maksimum dalam satu panggilan generation. | `argumen int; default 32` |
| `cap` · 46 | Kapasitas konteks model dan ruang token tambahan yang tersisa. | `_model_context_length(model)` |
| `order` · 47 | Urutan indeks pemilihan/pengacakan record atau baris. | `sorted(range(len(prompts)), key=lambda i: len(prompts[i]))` |
| `results` · 48 | Hasil teks generation atau contoh jawaban beserta penilaiannya. | `[None] * len(prompts)` |
| `i` · 50, 58 | Posisi awal bucket pada daftar indeks prompt yang telah diurutkan. | `0`; `j` |
| `j` · 52, 56 | Batas akhir bucket prompt sama panjang, dibatasi micro_batch. | `i`; `j += 1` |
| `L` · 53 | Lebar batch padding atau panjang frame action T-1 pada PPO, sesuai fungsi. | `len(prompts[order[i]])` |
| `idxs` · 57 | Urutan indeks pemilihan/pengacakan record atau baris. | `order[i:j]` |
| `budget` · 59 | Kapasitas konteks model dan ruang token tambahan yang tersisa. | `min(max_new_tokens, cap - L)` |
| `k` · 61, 64, 68 | Indeks asli prompt yang dipulihkan ketika menyimpan hasil. | `iterasi idxs`; `elemen 1 dari iterasi enumerate(idxs)` |
| `batch` · 64 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `torch.tensor([prompts[k] for k in idxs], dtype=torch.long, device=device)` |
| `rb` · 65 | RolloutBatch: sequences, response_mask, gen_logprobs, dan prompt_len. | `generate_with_logprobs(model, batch, budget, temperature=temperature, top_k=top_k, top_p=top_p, stop_tokens=(EOT_ID,), pad_id=EOT_ID)` |
| `gen` · 67 | Bagian token jawaban saja dari hasil rollout, sesudah panjang prompt. | `rb.sequences[:, L:]` |
| `row` · 68 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `elemen 0 dari iterasi enumerate(idxs)` |
| `toks` · 69, 71 | Daftar token satu jawaban yang dipotong pada EOT sebelum decode. | `gen[row].tolist()`; `toks[:toks.index(EOT_ID)]` |
| `r` · 73 | Satu hasil teks, diganti string kosong jika belum berisi teks. | `iterasi results` |

### Lingkup `batched_generate.lambda@47`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `i` · 47 | Indeks prompt; panjang prompts[i] menjadi kunci pengurutan. | `argumen lambda: len(prompts[i])` |

### Lingkup `gsm8k_accuracy`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 78 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `qa_pairs` · 79 | Pasangan pertanyaan dan answer field untuk evaluasi GSM8K. | `argumen list[tuple[str, str]]` |
| `device` · 81 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `max_new_tokens` · 82 | Batas jumlah token tambahan di luar prompt. | `argumen int; default 300` |
| `greedy` · 83 | Mode top_k=1; beberapa kandidat maksimum seri dapat tetap lolos implementasi threshold. | `argumen bool; default True` |
| `return_samples` · 84 | Jumlah contoh jawaban yang dikembalikan bersama hasil evaluasi. | `argumen int; default 0` |
| `prompts` · 92 | Daftar prompt bertoken sebelum/untuk rollout; GRPO mengulang setiap base prompt G kali. | `[encode_prompt([{'role': 'user', 'content': q}]) for q, _ in qa_pairs]` |
| `q` · 92, 97 | Teks pertanyaan satu pasangan evaluasi. | `elemen 0 dari iterasi qa_pairs`; `elemen 0 dari elemen 0 dari iterasi zip(qa_pairs, responses)` |
| `_` · 92 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari iterasi qa_pairs` |
| `responses` · 93 | Hasil teks generation atau contoh jawaban beserta penilaiannya. | `batched_generate(model, prompts, max_new_tokens, device=device, greedy=greedy)` |
| `correct` · 95, 100 | Jumlah jawaban yang cocok dengan gold. | `0`; `correct += int(ok)` |
| `samples` · 96 | Hasil teks generation atau contoh jawaban beserta penilaiannya. | `[]` |
| `ans` · 97 | Answer field dataset untuk mengambil gold. | `elemen 1 dari elemen 0 dari iterasi zip(qa_pairs, responses)` |
| `resp` · 97 | Teks jawaban model yang dinilai. | `elemen 1 dari iterasi zip(qa_pairs, responses)` |
| `gold` · 98 | Kunci jawaban numerik satu soal atau daftar kunci sejajar dengan prompt. | `gsm8k_gold_answer(ans)` |
| `ok` · 99 | Boolean kecocokan jawaban model dengan gold. | `is_correct(resp, gold)` |
| `n` · 103 | Jumlah soal evaluasi. | `len(qa_pairs)` |

### Lingkup `load_gsm8k_eval`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `split` · 107 | Nama bagian dataset seperti train, val, test, atau train_prefs. | `argumen str; default 'test'` |
| `limit` · 107 | Batas jumlah contoh dataset yang dimuat/diproses. | `argumen int &#124; None; default 200` |
| `ds` · 113, 115 | Dataset hasil load_dataset, dapat dipangkas dengan select. | `load_dataset('openai/gsm8k', 'main', split=split)`; `ds.select(range(min(limit, len(ds))))` |
| `ex` · 116 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `iterasi ds` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 9 |
| `torch` | `torch` | 11 |
| `EOT_ID` | `src.post_training.chat_template.EOT_ID` | 13 |
| `decode` | `src.post_training.chat_template.decode` | 13 |
| `encode_prompt` | `src.post_training.chat_template.encode_prompt` | 13 |
| `generate_with_logprobs` | `src.post_training.rollout.generate_with_logprobs` | 14 |
| `gsm8k_gold_answer` | `src.post_training.rewards.gsm8k_gold_answer` | 15 |
| `is_correct` | `src.post_training.rewards.is_correct` | 15 |
| `os` | `os` | 109 |
| `load_dataset` | `datasets.load_dataset` | 111 |
