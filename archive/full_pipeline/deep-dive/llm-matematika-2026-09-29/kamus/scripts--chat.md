# scripts/chat.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/chat.py)

## Fungsi dan kelas

### `main` (baris 24)

Memuat model dan menjawab prompt sekali atau melalui REPL; riwayat turn tidak disimpan otomatis.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `main.reply` (baris 43)

Fungsi lokal chat yang meneruskan teks dan opsi CLI ke generate_reply.

```python
def reply(text):
```

Keluaran eksplisit: `generate_reply(model, text, device=args.device, system=args.system, raw=args.raw, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p if args.top_p < 1 else None, top_k=args.top_k, greedy=args.greedy)`.

## Variabel dan parameter menurut lingkup

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `p` · 25, 39 | Parser CLI; nama p pada comprehension model.parameters() terpisah dan berarti satu tensor parameter. | `argparse.ArgumentParser()`; `iterasi model.parameters()` |
| `args` · 36 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `p.parse_args()` |
| `model` · 38 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `load_model_from_ckpt(args.ckpt, args.device)` |
| `n` · 39 | Jumlah parameter model yang ditampilkan dalam satuan juta. | `sum((p.numel() for p in model.parameters()))` |
| `text` · 55 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `input('\nyou> ').strip()` |

### Lingkup `main.reply`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 43 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen tanpa anotasi tipe` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 15 |
| `argparse` | `argparse` | 17 |
| `torch` | `torch` | 19 |
| `generate_reply` | `src.post_training.inference.generate_reply` | 21 |
| `load_model_from_ckpt` | `src.post_training.inference.load_model_from_ckpt` | 21 |
