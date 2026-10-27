# src/post_training/inference.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/inference.py)

## Fungsi dan kelas

### `load_model_from_ckpt` (baris 19)

Membangun Transformer dari cfg/override checkpoint dan memuat bobot dengan normalisasi prefix.

```python
def load_model_from_ckpt(ckpt_path: str, device: str, overrides: dict | None=None) -> Transformer:
```

Keluaran eksplisit: `model.to(device).eval()`.

### `generate_reply` (baris 37)

Membentuk prompt raw/chat dan menghasilkan satu jawaban, dengan system opsional.

```python
def generate_reply(model, user_text: str, *, device: str, system: str | None=None, raw: bool=False, max_new_tokens: int=256, temperature: float=0.8, top_k: int | None=None, top_p: float | None=0.95, greedy: bool=False) -> str:
```

Keluaran eksplisit: `out[0]`.

## Variabel dan parameter menurut lingkup

### Lingkup `load_model_from_ckpt`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ckpt_path` · 19 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `argumen str` |
| `device` · 19 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `overrides` · 19 | Nilai pengganti konfigurasi dengan prioritas di atas default/JSON. | `argumen dict &#124; None; default None` |
| `ck` · 22 | Isi berkas checkpoint, termasuk bobot dan metadata bila tersedia. | `torch.load(ckpt_path, map_location='cpu', weights_only=False)` |
| `cfg` · 23 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `{**(ck.get('cfg') or {}), **(overrides or {})}` |
| `model` · 24 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `Transformer(n_head=cfg.get('n_head', 16), n_embed=cfg.get('n_embed', 1024), context_length=cfg.get('context_length', 1024), vocab_size=cfg.get('vocab_size', 50304), N_BLOCKS=cfg.get('n_blocks', 24))` |
| `state` · 29, 30 | Pemetaan nama parameter/buffer ke tensor bobot. | `ck['model_state_dict'] if 'model_state_dict' in ck else ck`; `{k.removeprefix('module.').removeprefix('transformer.'): v for k, v in state.items()}` |
| `k` · 30, 32 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `elemen 0 dari iterasi state.items()` |
| `v` · 30, 32 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi state.items()` |
| `keys` · 31 | Himpunan nama parameter yang diterima model atau kumpulan kunci objek yang diperiksa. | `set(model.state_dict().keys())` |

### Lingkup `generate_reply`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 38 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen tanpa anotasi tipe` |
| `user_text` · 39 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |
| `device` · 41 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `system` · 42 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str &#124; None; default None` |
| `raw` · 43 | Sakelar melanjutkan teks mentah pada inference; string angka bersih pada parse_number. | `argumen bool; default False` |
| `max_new_tokens` · 44 | Batas jumlah token tambahan di luar prompt. | `argumen int; default 256` |
| `temperature` · 45 | Pembagi logits sebelum softmax/log-softmax saat sampling dan recomputation. | `argumen float; default 0.8` |
| `top_k` · 46 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen int &#124; None; default None` |
| `top_p` · 47 | Batas kandidat berdasarkan peringkat atau massa probabilitas kumulatif. | `argumen float &#124; None; default 0.95` |
| `greedy` · 48 | Mode top_k=1; beberapa kandidat maksimum seri dapat tetap lolos implementasi threshold. | `argumen bool; default False` |
| `ids` · 59 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `get_tokenizer().encode_ordinary(user_text)` |
| `out` · 60, 69 | List hasil batched_generate untuk satu prompt; out[0] adalah teks balasan. | `batched_generate(model, [ids], max_new_tokens, device=device, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy)`; `batched_generate(model, [prompt_ids], max_new_tokens, device=device, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy)` |
| `messages` · 64 | Daftar pesan percakapan berupa dictionary role/content. | `[]` |
| `prompt_ids` · 68 | Batch ID prompt (B,P), dengan panjang P sama sebelum token jawaban ditambahkan. | `encode_prompt(messages)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 10 |
| `torch` | `torch` | 12 |
| `Transformer` | `src.models.transformer.Transformer` | 14 |
| `EOT_ID` | `src.post_training.chat_template.EOT_ID` | 15 |
| `decode` | `src.post_training.chat_template.decode` | 15 |
| `encode_prompt` | `src.post_training.chat_template.encode_prompt` | 15 |
| `get_tokenizer` | `src.post_training.chat_template.get_tokenizer` | 15 |
| `batched_generate` | `src.post_training.evaluation.batched_generate` | 16 |
