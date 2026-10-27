# scripts/eval_post_training.py

[Indeks kamus](README.md) · [Kode sumber](../../../scripts/eval_post_training.py)

## Fungsi dan kelas

### `model_from_ckpt` (baris 28)

Membangun Transformer dari cfg/override checkpoint dan memuat bobot dengan normalisasi prefix.

```python
def model_from_ckpt(ckpt_path: str, device: str, overrides: dict | None=None) -> Transformer:
```

Keluaran eksplisit: `model.to(device).eval()`.

### `print_table` (baris 45)

Membaca JSONL metrik dan mencetak label tahap, akurasi, dan jumlah soal.

```python
def print_table(path: str):
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `main` (baris 53)

Mengukur akurasi checkpoint, mencetak sampel, dan opsional menulis/menampilkan tabel hasil.

```python
def main():
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `model_from_ckpt`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ckpt_path` · 28 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `argumen str` |
| `device` · 28 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `overrides` · 28 | Nilai pengganti konfigurasi dengan prioritas di atas default/JSON. | `argumen dict &#124; None; default None` |
| `ck` · 29 | Isi berkas checkpoint, termasuk bobot dan metadata bila tersedia. | `torch.load(ckpt_path, map_location='cpu', weights_only=False)` |
| `cfg` · 30, 31 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `ck.get('cfg', {}) or {}`; `{**cfg, **(overrides or {})}` |
| `model` · 32 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `Transformer(n_head=cfg.get('n_head', 16), n_embed=cfg.get('n_embed', 1024), context_length=cfg.get('context_length', 1024), vocab_size=cfg.get('vocab_size', 50304), N_BLOCKS=cfg.get('n_blocks', 24))` |
| `state` · 37, 38 | Pemetaan nama parameter/buffer ke tensor bobot. | `ck['model_state_dict'] if 'model_state_dict' in ck else ck`; `{k.removeprefix('module.').removeprefix('transformer.'): v for k, v in state.items()}` |
| `k` · 38, 40 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `elemen 0 dari iterasi state.items()` |
| `v` · 38, 40 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi state.items()` |
| `backbone_keys` · 39 | Himpunan nama parameter yang diterima model atau kumpulan kunci objek yang diperiksa. | `set(model.state_dict().keys())` |
| `filtered` · 40 | Bobot yang lolos pemilihan nama sebelum dimuat ke model. | `{k: v for k, v in state.items() if k in backbone_keys}` |

### Lingkup `print_table`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `path` · 45 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `argumen str` |
| `rows` · 46 | Record/contoh atau kumpulan contoh yang sedang diproses; struktur terlihat pada ekspresi sumber. | `[json.loads(l) for l in open(path) if l.strip()]` |
| `l` · 46 | Satu baris JSONL metrik yang dibaca dari file. | `iterasi open(path)` |
| `r` · 49 | Satu record hasil evaluasi berisi label, accuracy, dan n. | `iterasi rows` |

### Lingkup `main`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `p` · 54 | Parser opsi terminal yang sedang ditambahkan field atau dibaca hasilnya. | `argparse.ArgumentParser()` |
| `args` · 64 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `p.parse_args()` |
| `model` · 70 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `model_from_ckpt(args.ckpt, args.device)` |
| `qa` · 71 | Pasangan pertanyaan dan answer field untuk evaluasi GSM8K. | `load_gsm8k_eval(args.split, limit=args.limit)` |
| `res` · 72 | Representasi residual yang disimpan untuk penjumlahan atau inspeksi; res pada evaluator juga dictionary hasil. | `gsm8k_accuracy(model, qa, device=args.device, max_new_tokens=args.max_new_tokens, greedy=True, return_samples=args.samples)` |
| `s` · 75 | Satu contoh jawaban dari res[samples] untuk dicetak. | `iterasi res['samples']` |
| `f` · 80 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(args.append, 'a')` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 16 |
| `argparse` | `argparse` | 18 |
| `json` | `json` | 19 |
| `os` | `os` | 20 |
| `torch` | `torch` | 22 |
| `Transformer` | `src.models.transformer.Transformer` | 24 |
| `gsm8k_accuracy` | `src.post_training.evaluation.gsm8k_accuracy` | 25 |
| `load_gsm8k_eval` | `src.post_training.evaluation.load_gsm8k_eval` | 25 |
