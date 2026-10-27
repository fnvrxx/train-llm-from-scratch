# src/post_training/optim.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/optim.py)

## Fungsi dan kelas

### `configure_optimizer` (baris 17)

Membuat AdamW dengan weight decay untuk parameter minimal dua dimensi, termasuk embedding.

```python
def configure_optimizer(model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]=(0.9, 0.95)) -> torch.optim.AdamW:
```

Keluaran eksplisit: `torch.optim.AdamW(groups, lr=lr, betas=betas)`.

### `cosine_lr` (baris 40)

Menghasilkan learning rate satu step melalui warmup linear dan cosine decay menuju min_lr.

```python
def cosine_lr(step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
```

Keluaran eksplisit: `lr * (step + 1) / max(1, warmup_steps)`; `min_lr`; `min_lr + coeff * (lr - min_lr)`.

## Variabel dan parameter menurut lingkup

### Lingkup `configure_optimizer`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `model` · 18 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen nn.Module` |
| `lr` · 19 | Learning rate, ukuran langkah pembaruan parameter. | `argumen float` |
| `weight_decay` · 20 | Koefisien penyusutan bobot pada kelompok parameter yang menerapkannya. | `argumen float` |
| `betas` · 21 | Koefisien moving average gradien dan kuadrat gradien pada AdamW. | `argumen tuple[float, float]; default (0.9, 0.95)` |
| `decay` · 25 | Daftar parameter minimal dua dimensi yang dikenai weight decay, termasuk embedding. | `elemen 0 dari ([], [])` |
| `no_decay` · 25 | Daftar parameter berdimensi lebih kecil dari dua yang tidak dikenai weight decay. | `elemen 1 dari ([], [])` |
| `name` · 26 | Nama parameter model, field konfigurasi, atau berkas yang sedang diperiksa. | `elemen 0 dari iterasi model.named_parameters()` |
| `p` · 26 | Satu tensor parameter model yang dihitung jumlah elemennya, dikelompokkan optimizer, atau dibekukan sesuai fungsi. | `elemen 1 dari iterasi model.named_parameters()` |
| `groups` · 33 | Dua dictionary kelompok parameter yang diberikan ke AdamW. | `[{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]` |

### Lingkup `cosine_lr`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `step` · 40 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `argumen int` |
| `warmup_steps` · 40 | Jumlah step awal untuk kenaikan learning rate linear. | `argumen int` |
| `max_steps` · 40 | Batas update; max_steps negatif pada konfigurasi SFT berarti gunakan batas epochs. | `argumen int` |
| `lr` · 40 | Learning rate, ukuran langkah pembaruan parameter. | `argumen float` |
| `min_lr` · 40 | Learning rate minimum/akhir atau nilai sesudah ambang decay trainer awal. | `argumen float` |
| `progress` · 47 | Fraksi langkah sesudah warmup terhadap panjang bagian cosine decay. | `(step - warmup_steps) / max(1, max_steps - warmup_steps)` |
| `coeff` · 48 | Koefisien 0.5*(1+cos(pi*progress)) untuk interpolasi learning rate. | `0.5 * (1.0 + math.cos(math.pi * progress))` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 9 |
| `math` | `math` | 11 |
| `torch` | `torch` | 13 |
| `nn` | `torch.nn` | 14 |
