# scripts/generate_text.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/generate_text.py)

## Fungsi dan kelas

### `load_checkpoint` (baris 8)

Memuat checkpoint dengan fallback argumen torch.load versi lama.

```python
def load_checkpoint(model_path: str, device: str):
```

Keluaran eksplisit: `torch.load(model_path, map_location=torch.device(device), weights_only=False)`; `torch.load(model_path, map_location=torch.device(device))`.

### `generate_text` (baris 14)

Memuat model konfigurasi awal, melanjutkan token prefix, dan mendekode seluruh hasil.

```python
def generate_text(model_path: str, input_text: str, max_new_tokens: int=100, device: str='cuda') -> str:
```

Keluaran eksplisit: `output_text`.

### `main` (baris 56)

Membaca opsi terminal dan mencetak kelanjutan teks model awal.

```python
def main() -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `load_checkpoint`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model_path` · 8 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `argumen str` |
| `device` · 8 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |

### Lingkup `generate_text`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model_path` · 14 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `argumen str` |
| `input_text` · 14 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |
| `max_new_tokens` · 14 | Batas jumlah token tambahan di luar prompt. | `argumen int; default 100` |
| `device` · 14 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str; default 'cuda'` |
| `checkpoint` · 28 | Isi berkas checkpoint, termasuk bobot dan metadata bila tersedia. | `load_checkpoint(model_path, device)` |
| `model` · 31 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `Transformer(n_head=config['n_head'], n_embed=config['n_embed'], context_length=config['context_length'], vocab_size=config['vocab_size'], N_BLOCKS=config['n_blocks'])` |
| `enc` · 42 | Tokenizer pada preprocessing; daftar hasil encoding pasangan pada _collate. | `tiktoken.get_encoding('r50k_base')` |
| `start_ids` · 44 | ID token awal sebagai prefix generation. | `enc.encode_ordinary(input_text)` |
| `context` · 45 | Tensor batch prefix pada generate_text, berbeda dari context_length. | `torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)` |
| `generated_tokens` · 49 | Token hasil generation; jalur dasar menyertakan prefix input. | `model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()` |
| `output_text` · 52 | Teks hasil generation yang ditampilkan skrip. | `enc.decode(generated_tokens)` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `parser` · 57 | Objek argparse yang mendefinisikan dan membaca argumen terminal. | `argparse.ArgumentParser(description='Generate text using a pre-trained Transformer model.')` |
| `args` · 62 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `parser.parse_args()` |
| `generated` · 64 | Teks hasil generation yang ditampilkan skrip. | `generate_text(args.model_path, args.input_text, args.max_new_tokens, config['device'])` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `torch` | `torch` | 1 |
| `tiktoken` | `tiktoken` | 2 |
| `argparse` | `argparse` | 3 |
| `config` | `config.config.default_config` | 4 |
| `Transformer` | `src.models.transformer.Transformer` | 5 |
