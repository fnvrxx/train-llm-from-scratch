# src/models/mlp.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/models/mlp.py)

## Fungsi dan kelas

### `MLP` (baris 5)

Feed-forward per posisi: ekspansi C ke 4C, ReLU, dan proyeksi kembali C.

### `MLP.__init__` (baris 16)

Membuat komponen dan keadaan objek MLP. Feed-forward per posisi: ekspansi C ke 4C, ReLU, dan proyeksi kembali C.

```python
def __init__(self, n_embed: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `MLP.forward` (baris 28)

Menjalankan forward_embedding lalu project_embedding agar output kembali selebar input.

```python
def forward(self, x: Tensor) -> Tensor:
```

Keluaran eksplisit: `x`.

### `MLP.forward_embedding` (baris 43)

Menghasilkan aktivasi setelah linear ekspansi dan ReLU, bentuk (B,T,4C).

```python
def forward_embedding(self, x: Tensor) -> Tensor:
```

Keluaran eksplisit: `x`.

### `MLP.project_embedding` (baris 56)

Memproyeksikan aktivasi 4C kembali menjadi C.

```python
def project_embedding(self, x: Tensor) -> Tensor:
```

Keluaran eksplisit: `x`.

## Variabel dan parameter menurut lingkup

### Lingkup `MLP.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 16 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `n_embed` · 16 | Banyak koordinat vektor representasi satu token. | `argumen int` |
| `self.hidden` · 24 | Hidden states backbone; sebagai atribut MLP berarti lapisan linear ekspansi C ke 4C. | `nn.Linear(n_embed, 4 * n_embed)` |
| `self.relu` · 25 | Aktivasi ReLU, mengubah nilai negatif menjadi nol. | `nn.ReLU()` |
| `self.proj` · 26 | Linear yang mengembalikan concatenated heads atau aktivasi MLP ke lebar n_embed. | `nn.Linear(4 * n_embed, n_embed)` |

### Lingkup `MLP.forward`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 28 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `x` · 28, 39, 40 | Representasi yang sedang ditransformasikan. Bentuknya mengikuti operasi pada fungsi; MLP memperlebar lalu mengembalikannya. | `argumen Tensor`; `self.forward_embedding(x)`; `self.project_embedding(x)` |

### Lingkup `MLP.forward_embedding`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 43 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `x` · 43, 53 | Representasi yang sedang ditransformasikan. Bentuknya mengikuti operasi pada fungsi; MLP memperlebar lalu mengembalikannya. | `argumen Tensor`; `self.relu(self.hidden(x))` |

### Lingkup `MLP.project_embedding`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 56 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `x` · 56, 66 | Representasi yang sedang ditransformasikan. Bentuknya mengikuti operasi pada fungsi; MLP memperlebar lalu mengembalikannya. | `argumen Tensor`; `self.proj(x)` |

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `batch_size` · 71 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `2` |
| `sequence_length` · 72 | Banyak posisi token pada urutan yang diproses. | `3` |
| `embedding_dim` · 73 | Banyak koordinat vektor representasi satu token. | `16` |
| `input_tensor` · 74 | Tensor acak masukan pada demo modul di __main__, bukan data bahasa terlatih. | `torch.randn(batch_size, sequence_length, embedding_dim)` |
| `mlp_module` · 76 | Objek jaringan yang dipakai dalam demo modul terkait. | `MLP(n_embed=embedding_dim)` |
| `output_tensor` · 77 | Tensor keluaran modul pada demo bentuk data. | `mlp_module(input_tensor)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `torch` | `torch` | 1 |
| `nn` | `torch.nn` | 2 |
| `Tensor` | `torch.Tensor` | 3 |
