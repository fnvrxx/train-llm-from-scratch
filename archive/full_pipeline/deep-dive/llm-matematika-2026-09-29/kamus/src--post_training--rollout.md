# src/post_training/rollout.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/rollout.py)

## Fungsi dan kelas

### `RolloutBatch` (baris 33)

Dataclass hasil rollout: sequences, response_mask, gen_logprobs, prompt_len.

### `_logits_from` (baris 52)

Mengambil logits dari keluaran model tensor langsung atau tuple logits dengan loss/value.

```python
def _logits_from(model, idx: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `out[0] if isinstance(out, tuple) else out`.

### `filter_logits` (baris 59)

Menerapkan suhu, top-k, dan top-p; kandidat terhapus diberi minus tak hingga.

```python
def filter_logits(logits: torch.Tensor, temperature: float=1.0, top_k: int | None=None, top_p: float | None=None) -> torch.Tensor:
```

Keluaran eksplisit: `logits`.

### `generate_with_logprobs` (baris 94)

Sampling autoregresif dengan stop per baris, padding sesudah stop, dan pencatatan mask/log-prob.

```python
def generate_with_logprobs(model, prompt_ids: torch.Tensor, max_new_tokens: int, *, temperature: float=1.0, top_k: int | None=None, top_p: float | None=None, stop_tokens: tuple[int, ...]=(EOT_ID,), pad_id: int=EOT_ID, context_length: int | None=None) -> RolloutBatch:
```

Keluaran eksplisit: `RolloutBatch(sequences=sequences, response_mask=response_mask, gen_logprobs=gen_logprobs, prompt_len=P)`.

### `_cap_of` (baris 174)

Membaca kapasitas konteks backbone di balik wrapper, atau fallback bila tidak ada.

```python
def _cap_of(model) -> int:
```

Keluaran eksplisit: `getattr(inner, 'context_length', 1024)`.

### `rollout_prompts` (baris 180)

Mengelompokkan prompt sama panjang, menggenerasikan jawaban, lalu menggabungkan hasil dalam urutan input.

```python
def rollout_prompts(model, prompts: list[list[int]], max_new_tokens: int, *, device: str, temperature: float=1.0, top_k: int | None=None, top_p: float | None=None, pad_id: int=EOT_ID, micro_batch: int=64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(sequences, response_mask, prompt_lens)`.

### `compute_logprobs` (baris 233)

Menghitung ulang log-prob token teramati dengan teacher forcing dan menggeser response mask ke frame action.

```python
def compute_logprobs(model, sequences: torch.Tensor, response_mask: torch.Tensor, *, temperature: float=1.0, requires_grad: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(logprobs, mask)`.

### `sequence_logprobs` (baris 268)

Menjumlahkan log-prob token jawaban dan mengembalikan jumlah token aktif per urutan.

```python
def sequence_logprobs(model, sequences: torch.Tensor, response_mask: torch.Tensor, *, temperature: float=1.0, requires_grad: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `((lp * m).sum(dim=-1), m.sum(dim=-1))`.

### `sequence_entropy` (baris 287)

Menghitung entropi distribusi kosakata tiap posisi tanpa gradien; berguna untuk diagnostik.

```python
def sequence_entropy(model, sequences: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
```

Keluaran eksplisit: `-(logp.exp() * logp).sum(dim=-1)`.

## Variabel dan parameter menurut lingkup

### Lingkup `RolloutBatch`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `sequences` · 46 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `deklarasi tipe torch.Tensor` |
| `response_mask` · 47 | Mask posisi jawaban nyata, bukan prompt/padding. | `deklarasi tipe torch.Tensor` |
| `gen_logprobs` · 48 | Log-prob token generated yang disimpan per step, nol setelah baris selesai. | `deklarasi tipe torch.Tensor` |
| `prompt_len` · 49 | Panjang prompt bersama atau panjang prompt masing-masing baris. | `deklarasi tipe int` |

### Lingkup `_logits_from`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 52 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `idx` · 52 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `out` · 55 | Keluaran model(idx), berupa logits langsung atau tuple logits dan loss/value. | `model(idx)` |

### Lingkup `filter_logits`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `logits` · 60, 73, 78, 88 | Skor next-token (B,V), sesudah suhu dan penghapusan kandidat diberi -inf. | `argumen torch.Tensor`; `logits / max(temperature, 1e-06)`; `logits.masked_fill(logits < kth, float('-inf'))` |
| `temperature` · 61 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 1.0` |
| `top_k` · 62 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen int &#124; None; default None` |
| `top_p` · 63 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen float &#124; None; default None` |
| `k` · 76 | Jumlah kandidat top-k efektif, dibatasi ukuran kosakata. | `min(top_k, logits.size(-1))` |
| `kth` · 77 | Skor kandidat peringkat k, menjadi ambang penghapusan logit. | `torch.topk(logits, k, dim=-1).values[..., -1, None]` |
| `sorted_logits` · 81 | Logits diurutkan menurun untuk membentuk kandidat nucleus/top-p. | `elemen 0 dari torch.sort(logits, descending=True, dim=-1)` |
| `sorted_idx` · 81 | Indeks kosakata asal dari logits yang diurutkan. | `elemen 1 dari torch.sort(logits, descending=True, dim=-1)` |
| `cumprobs` · 82 | Peluang kumulatif kandidat sesudah diurutkan menurut skor. | `sorted_logits.softmax(dim=-1).cumsum(dim=-1)` |
| `remove` · 83, 87 | Mask kandidat yang dihapus oleh top-p, lalu dipulihkan ke urutan kosakata semula. | `cumprobs > top_p`; `remove.scatter(-1, sorted_idx, remove)` |

### Lingkup `generate_with_logprobs`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 95 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `prompt_ids` · 96 | Batch ID prompt (B,P), dengan panjang P sama sebelum token jawaban ditambahkan. | `argumen torch.Tensor` |
| `max_new_tokens` · 97, 122 | Batas jumlah token tambahan di luar prompt. | `argumen int`; `min(max_new_tokens, cap - P)` |
| `temperature` · 99 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 1.0` |
| `top_k` · 100 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen int &#124; None; default None` |
| `top_p` · 101 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen float &#124; None; default None` |
| `stop_tokens` · 102 | ID penghentian generation sebagai tuple atau tensor. | `argumen tuple[int, ...]; default (EOT_ID,)` |
| `pad_id` · 103 | ID pengisi setelah sebuah baris berhenti; mask mengecualikannya dari loss. | `argumen int; default EOT_ID` |
| `context_length` · 104 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int &#124; None; default None` |
| `device` · 117 | Lokasi komputasi, misalnya cpu atau cuda. | `prompt_ids.device` |
| `B` · 118 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `elemen 0 dari prompt_ids.shape` |
| `P` · 118 | Panjang prompt bersama atau panjang prompt masing-masing baris. | `elemen 1 dari prompt_ids.shape` |
| `cap` · 120, 121 | Kapasitas konteks model dan ruang token tambahan yang tersisa. | `context_length if context_length is not None else getattr(model, 'context_length', P + max_new_tokens)`; `getattr(getattr(model, 'transformer', model), 'context_length', cap)` |
| `stop_set` · 126 | ID penghentian generation sebagai tuple atau tensor. | `torch.tensor(stop_tokens, device=device)` |
| `sequences` · 127, 149, 158 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `prompt_ids.clone()`; `torch.cat([sequences, next_tok[:, None]], dim=1)`; `torch.cat([sequences, torch.full((B, rem), pad_id, dtype=sequences.dtype, device=device)], dim=1)` |
| `gen_logprobs` · 128 | Log-prob token generated yang disimpan per step, nol setelah baris selesai. | `torch.zeros(B, max_new_tokens, device=device)` |
| `gen_mask` · 129 | Mask token generated nyata sebelum digabung dengan mask prompt. | `torch.zeros(B, max_new_tokens, dtype=torch.bool, device=device)` |
| `finished` · 130, 152 | Boolean per baris, benar jika baris telah mengeluarkan stop token. | `torch.zeros(B, dtype=torch.bool, device=device)`; `finished &#124; active & hit_stop` |
| `t` · 132 | Nomor token baru yang sedang dihasilkan, mulai nol. | `iterasi range(max_new_tokens)` |
| `idx_cond` · 133 | Potongan konteks terakhir yang masih muat untuk prediksi berikutnya. | `sequences[:, -cap:]` |
| `logits` · 134 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `_logits_from(model, idx_cond)[:, -1, :]` |
| `full_logprobs` · 137 | Log-softmax seluruh kosakata pada suhu sampling sebelum penyaringan top-k/top-p. | `F.log_softmax(logits.float() / max(temperature, 1e-06), dim=-1)` |
| `probs` · 139 | Peluang token setelah softmax, dipakai multinomial untuk sampling. | `F.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)` |
| `next_tok` · 140, 145 | Token berikutnya hasil sampling distribusi model. | `torch.multinomial(probs, num_samples=1).squeeze(-1)`; `torch.where(active, next_tok, torch.full_like(next_tok, pad_id))` |
| `active` · 142 | Kebalikan finished, baris yang masih boleh menghasilkan token nyata. | `~finished` |
| `token_lp` · 143 | Log-prob token yang terpilih dari gather distribusi penuh. | `full_logprobs.gather(-1, next_tok[:, None]).squeeze(-1)` |
| `hit_stop` · 151 | Apakah token terbaru tiap baris cocok dengan salah satu stop_tokens. | `(next_tok[:, None] == stop_set).any(dim=-1)` |
| `rem` · 157 | Jumlah kolom padding tersisa jika semua baris berhenti lebih awal. | `max_new_tokens - (t + 1)` |
| `response_mask` · 163 | Mask posisi jawaban nyata, bukan prompt/padding. | `torch.cat([torch.zeros(B, P, dtype=torch.bool, device=device), gen_mask], dim=1)` |

### Lingkup `_cap_of`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 174 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `inner` · 175 | Backbone yang diperoleh dari wrapper model. | `getattr(model, 'transformer', model)` |

### Lingkup `rollout_prompts`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 181 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `prompts` · 182 | Daftar prompt bertoken sebelum/untuk rollout; GRPO mengulang setiap base prompt G kali. | `argumen list[list[int]]` |
| `max_new_tokens` · 183 | Batas jumlah token tambahan di luar prompt. | `argumen int` |
| `device` · 185 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `temperature` · 186 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 1.0` |
| `top_k` · 187 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen int &#124; None; default None` |
| `top_p` · 188 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen float &#124; None; default None` |
| `pad_id` · 189 | ID pengisi setelah sebuah baris berhenti; mask mengecualikannya dari loss. | `argumen int; default EOT_ID` |
| `micro_batch` · 190 | Banyak prompt sama panjang maksimum dalam satu panggilan generation. | `argumen int; default 64` |
| `cap` · 202 | Kapasitas konteks model dan ruang token tambahan yang tersisa. | `_cap_of(model)` |
| `order` · 203 | Urutan indeks pemilihan/pengacakan record atau baris. | `sorted(range(len(prompts)), key=lambda i: len(prompts[i]))` |
| `pieces` · 204 | Hasil rollout sementara per bucket panjang prompt sebelum penggabungan. | `[]` |
| `i` · 205, 212 | Posisi awal bucket pada urutan indeks prompt. | `0`; `j` |
| `L` · 207, 224 | Lebar batch padding atau panjang frame action T-1 pada PPO, sesuai fungsi. | `len(prompts[order[i]])`; `elemen 3 dari iterasi pieces` |
| `j` · 208, 210 | Batas akhir bucket panjang sama, dibatasi micro_batch. | `i`; `j += 1` |
| `idxs` · 211, 224 | Urutan indeks pemilihan/pengacakan record atau baris. | `order[i:j]`; `elemen 0 dari iterasi pieces` |
| `budget` · 213 | Kapasitas konteks model dan ruang token tambahan yang tersisa. | `max(1, min(max_new_tokens, cap - L))` |
| `batch` · 214 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `torch.tensor([prompts[k] for k in idxs], dtype=torch.long, device=device)` |
| `k` · 214, 226 | Indeks asli prompt untuk penempatan hasil rollout kembali. | `iterasi idxs`; `elemen 1 dari iterasi enumerate(idxs)` |
| `rb` · 215 | RolloutBatch: sequences, response_mask, gen_logprobs, dan prompt_len. | `generate_with_logprobs(model, batch, budget, temperature=temperature, top_k=top_k, top_p=top_p, stop_tokens=(EOT_ID,), pad_id=pad_id)` |
| `N` · 219 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `len(prompts)` |
| `Tmax` · 220 | Lebar maksimum hasil prompt+response sesudah seluruh bucket digabung. | `max((seq.size(1) for _, seq, _, _ in pieces))` |
| `_` · 220 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 0 dari iterasi pieces`; `elemen 2 dari iterasi pieces`; `elemen 3 dari iterasi pieces` |
| `seq` · 220, 224 | Urutan untuk padding dan nilai pengisi; seq juga satu potongan rollout. | `elemen 1 dari iterasi pieces` |
| `sequences` · 221 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `torch.full((N, Tmax), pad_id, dtype=torch.long, device=device)` |
| `response_mask` · 222 | Mask posisi jawaban nyata, bukan prompt/padding. | `torch.zeros((N, Tmax), dtype=torch.bool, device=device)` |
| `prompt_lens` · 223 | Panjang prompt bersama atau panjang prompt masing-masing baris. | `torch.zeros(N, dtype=torch.long, device=device)` |
| `rmask` · 224 | Mask posisi jawaban nyata, bukan prompt/padding. | `elemen 2 dari iterasi pieces` |
| `t` · 225 | Lebar urutan hasil bucket ini ketika disalin ke output global. | `seq.size(1)` |
| `row` · 226 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `elemen 0 dari iterasi enumerate(idxs)` |

### Lingkup `rollout_prompts.lambda@203`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `i` · 203 | Indeks prompt yang panjangnya dipakai untuk pengurutan bucket. | `argumen lambda: len(prompts[i])` |

### Lingkup `compute_logprobs`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 234 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `sequences` · 235 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `response_mask` · 236 | Mask posisi jawaban nyata, bukan prompt/padding. | `argumen torch.Tensor` |
| `temperature` · 238 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 1.0` |
| `requires_grad` · 239 | Apakah recomputation perlu graph gradien; benar untuk policy yang sedang dilatih. | `argumen bool; default True` |
| `ctx` · 258 | Context autograd aktif/nonaktif menurut requires_grad. | `torch.enable_grad() if requires_grad else torch.no_grad()` |
| `logits` · 260 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `_logits_from(model, sequences)[:, :-1, :]` |
| `logprobs_all` · 261 | Log-softmax setiap kandidat kosakata sebelum gather token target. | `F.log_softmax(logits.float() / max(temperature, 1e-06), dim=-1)` |
| `targets` · 262 | ID sequences[:,1:] dengan sumbu tambahan agar gather memilih satu token per posisi. | `sequences[:, 1:].unsqueeze(-1)` |
| `logprobs` · 263 | Log-prob token teramati, bentuk (B,T-1). | `logprobs_all.gather(-1, targets).squeeze(-1)` |
| `mask` · 264 | Response mask digeser ke target positions 1..T-1. | `response_mask[:, 1:].to(torch.bool)` |

### Lingkup `sequence_logprobs`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 269 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `sequences` · 270 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `response_mask` · 271 | Mask posisi jawaban nyata, bukan prompt/padding. | `argumen torch.Tensor` |
| `temperature` · 273 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 1.0` |
| `requires_grad` · 274 | Apakah recomputation perlu graph gradien; benar untuk policy yang sedang dilatih. | `argumen bool; default True` |
| `lp` · 282 | Log-prob tiap target token hasil compute_logprobs. | `elemen 0 dari compute_logprobs(model, sequences, response_mask, temperature=temperature, requires_grad=requires_grad)` |
| `mask` · 282 | Boolean target jawaban, bentuk (B,T-1). | `elemen 1 dari compute_logprobs(model, sequences, response_mask, temperature=temperature, requires_grad=requires_grad)` |
| `m` · 283 | Mask jawaban dikonversi ke dtype lp. | `mask.to(lp.dtype)` |

### Lingkup `sequence_entropy`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 287 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `sequences` · 287 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `temperature` · 287 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 1.0` |
| `logits` · 291 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `_logits_from(model, sequences)[:, :-1, :]` |
| `logp` · 292 | Log-prob token yang benar-benar dipilih/diberikan setelah gather dari kosakata. | `F.log_softmax(logits.float() / max(temperature, 1e-06), dim=-1)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 22 |
| `dataclass` | `dataclasses.dataclass` | 24 |
| `torch` | `torch` | 26 |
| `F` | `torch.nn.functional` | 27 |
| `EOT_ID` | `src.post_training.chat_template.EOT_ID` | 29 |
