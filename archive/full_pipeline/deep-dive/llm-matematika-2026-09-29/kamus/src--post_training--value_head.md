# src/post_training/value_head.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/value_head.py)

## Fungsi dan kelas

### `TransformerWithValueHead` (baris 19)

Actor-critic yang berbagi backbone untuk logits dan estimasi value.

### `TransformerWithValueHead.__init__` (baris 26)

Membuat komponen dan keadaan objek TransformerWithValueHead. Actor-critic yang berbagi backbone untuk logits dan estimasi value.

```python
def __init__(self, transformer: Transformer) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `TransformerWithValueHead.context_length` (baris 42)

Property yang meneruskan kapasitas konteks backbone ke wrapper.

```python
def context_length(self) -> int:
```

Keluaran eksplisit: `self.transformer.context_length`.

### `TransformerWithValueHead.forward` (baris 45)

Menghasilkan logits dan value tiap token dari hidden states backbone yang sama.

```python
def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(logits, values)`.

### `TransformerWithValueHead.value_only` (baris 52)

Menghasilkan values tanpa menghitung logits kosakata dan tanpa autograd.

```python
def value_only(self, idx: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `self.value_head(hidden).squeeze(-1)`.

## Variabel dan parameter menurut lingkup

### Lingkup `TransformerWithValueHead.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 26 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `transformer` · 26 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen Transformer` |
| `self.transformer` · 28 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `transformer` |
| `n_embed` · 29 | Banyak koordinat vektor representasi satu token. | `transformer.lm_head.in_features` |
| `self.value_head` · 30 | MLP hidden state ke prediksi return, lapisan terakhir berawal nol. | `nn.Sequential(nn.Linear(n_embed, n_embed), nn.ReLU(), nn.Linear(n_embed, 1))` |
| `last` · 36 | Lapisan terakhir value head atau indeks token respons terakhir, sesuai asal. | `self.value_head[-1]` |

### Lingkup `TransformerWithValueHead.context_length`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 42 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |

### Lingkup `TransformerWithValueHead.forward`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 45 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 45 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `hidden` · 46 | Hidden states backbone; sebagai atribut MLP berarti lapisan linear ekspansi C ke 4C. | `self.transformer.forward_hidden(idx)` |
| `logits` · 47 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `self.transformer.lm_head(hidden)` |
| `values` · 48 | Estimasi value pada state baru/lama/berikutnya di PPO; values pada utilitas adalah tensor angka umum. | `self.value_head(hidden).squeeze(-1)` |

### Lingkup `TransformerWithValueHead.value_only`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 52 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 52 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `hidden` · 55 | Hidden states backbone; sebagai atribut MLP berarti lapisan linear ekspansi C ke 4C. | `self.transformer.forward_hidden(idx)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 11 |
| `torch` | `torch` | 13 |
| `nn` | `torch.nn` | 14 |
| `Transformer` | `src.models.transformer.Transformer` | 16 |
