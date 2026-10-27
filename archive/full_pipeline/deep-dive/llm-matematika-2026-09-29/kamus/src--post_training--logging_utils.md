# src/post_training/logging_utils.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/logging_utils.py)

## Fungsi dan kelas

### `MetricsLogger` (baris 18)

Pencatat metrik JSONL dan opsional W&B.

### `MetricsLogger.__init__` (baris 19)

Membuat komponen dan keadaan objek MetricsLogger. Pencatat metrik JSONL dan opsional W&B.

```python
def __init__(self, stage: str, log_dir: str, *, use_wandb: bool=False, wandb_project: str='train-llm-from-scratch-posttrain', config: dict | None=None, run_name: str | None=None) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `MetricsLogger.log` (baris 46)

Menulis satu record metrik JSONL dan opsional W&B, lalu flush.

```python
def log(self, step: int, metrics: dict[str, Any]) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `MetricsLogger.close` (baris 53)

Menutup berkas log serta sesi W&B bila ada.

```python
def close(self) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

## Variabel dan parameter menurut lingkup

### Lingkup `MetricsLogger.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 20 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `stage` · 21 | Nama tahap seperti pretrain/sft/dpo/ppo/grpo untuk log dan checkpoint. | `argumen str` |
| `log_dir` · 22 | Direktori penyimpanan log training. | `argumen str` |
| `use_wandb` · 24 | Sakelar W&B, nama proyek W&B, dan nama eksperimen. | `argumen bool; default False` |
| `wandb_project` · 25 | Sakelar W&B, nama proyek W&B, dan nama eksperimen. | `argumen str; default 'train-llm-from-scratch-posttrain'` |
| `config` · 26 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen dict &#124; None; default None` |
| `run_name` · 27 | Sakelar W&B, nama proyek W&B, dan nama eksperimen. | `argumen str &#124; None; default None` |
| `self.stage` · 30 | Nama tahap seperti pretrain/sft/dpo/ppo/grpo untuk log dan checkpoint. | `stage` |
| `stamp` · 32 | Waktu Unix integer untuk membedakan nama berkas log. | `int(time.time())` |
| `self.path` · 33 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `os.path.join(log_dir, f'{stage}_{stamp}.jsonl')` |
| `self._fh` · 34 | Handle berkas JSONL dan modul W&B jika aktif. | `open(self.path, 'a')` |
| `self._wandb` · 35, 41 | Handle berkas JSONL dan modul W&B jika aktif. | `None`; `wandb` |
| `e` · 42 | Exception yang ditangkap untuk pelaporan kegagalan. | `exception Exception` |

### Lingkup `MetricsLogger.log`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 46 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `step` · 46 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `argumen int` |
| `metrics` · 46 | Dictionary metrik; record menambahkan step dan waktu pencatatan. | `argumen dict[str, Any]` |
| `record` · 47 | Dictionary metrik; record menambahkan step dan waktu pencatatan. | `{'step': step, 'wall': time.time(), **metrics}` |

### Lingkup `MetricsLogger.close`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 53 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 10 |
| `json` | `json` | 12 |
| `os` | `os` | 13 |
| `time` | `time` | 14 |
| `Any` | `typing.Any` | 15 |
| `wandb` | `wandb` | 38 |
