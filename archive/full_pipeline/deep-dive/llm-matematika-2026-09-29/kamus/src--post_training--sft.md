# src/post_training/sft.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/sft.py)

## Fungsi dan kelas

### `sft_loss` (baris 18)

Menggeser target dan mask, menghitung cross-entropy per posisi, lalu merata-ratakan hanya target assistant.

```python
def sft_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `ce.sum() / mask.sum().clamp(min=1.0)`.

### `pack_examples` (baris 41)

Menyambung token/mask contoh, membentuk baris context_length, dan membuang sisa tak penuh.

```python
def pack_examples(examples: list[tuple[list[int], list[int]]], context_length: int) -> tuple[np.ndarray, np.ndarray]:
```

Keluaran eksplisit: `(ids_arr, mask_arr)`.

## Variabel dan parameter menurut lingkup

### Lingkup `sft_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `logits` · 18, 31 | Skor token (B,T,V) yang dipotong menjadi (B,T-1,V). | `argumen torch.Tensor`; `logits[:, :-1, :]` |
| `tokens` · 18 | ID seluruh urutan (B,T), mencakup prompt dan jawaban. | `argumen torch.Tensor` |
| `loss_mask` · 18 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen torch.Tensor` |
| `targets` · 32 | tokens[:,1:], target token berikutnya (B,T-1). | `tokens[:, 1:]` |
| `mask` · 33 | loss_mask[:,1:] dalam dtype logits, sejajar dengan target. | `loss_mask[:, 1:].to(logits.dtype)` |
| `V` · 35 | Jumlah kandidat token pada sumbu terakhir logits. | `logits.size(-1)` |
| `ce` · 36, 37 | Loss tiap target yang diratakan, lalu dibentuk kembali seperti targets dan dikalikan mask. | `F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1).long(), reduction='none')`; `ce.view(targets.shape) * mask` |

### Lingkup `pack_examples`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `examples` · 42 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `argumen list[tuple[list[int], list[int]]]` |
| `context_length` · 43 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `flat_ids` · 54 | Token semua contoh yang telah disambung sebelum dipotong menjadi baris packed. | `[]` |
| `flat_mask` · 55 | Mask semua contoh yang disambung dengan susunan sejajar flat_ids. | `[]` |
| `ids` · 56 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari iterasi examples` |
| `mask` · 56 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `elemen 1 dari iterasi examples` |
| `n_rows` · 60 | Jumlah elemen atau total akumulasi; satuannya ditentukan len/shape/sum pada ekspresi asal. | `len(flat_ids) // context_length` |
| `ids_arr` · 61 | Array int32 ID packed dengan bentuk (n_rows,context_length). | `np.asarray(flat_ids[:n_rows * context_length], dtype=np.int32).reshape(n_rows, context_length)` |
| `mask_arr` · 62 | Array int8 mask packed dengan bentuk yang sama dengan ids_arr. | `np.asarray(flat_mask[:n_rows * context_length], dtype=np.int8).reshape(n_rows, context_length)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 11 |
| `np` | `numpy` | 13 |
| `torch` | `torch` | 14 |
| `F` | `torch.nn.functional` | 15 |
