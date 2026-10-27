# src/models/transformer_block.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/models/transformer_block.py)

## Fungsi dan kelas

### `Block` (baris 6)

Pre-LayerNorm attention dan MLP dengan dua jalur residual.

### `Block.__init__` (baris 18)

Membuat komponen dan keadaan objek Block. Pre-LayerNorm attention dan MLP dengan dua jalur residual.

```python
def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `Block.forward` (baris 33)

Menjalankan attention lalu MLP dengan normalisasi dan penjumlahan residual.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `x`.

### `Block.forward_embedding` (baris 49)

Mengembalikan aktivasi tengah MLP 4C dan residual C; tidak menjalankan proyeksi akhir MLP.

```python
def forward_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(x, res)`.

## Variabel dan parameter menurut lingkup

### Lingkup `Block.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 18 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `n_head` · 18 | Jumlah attention head dalam setiap blok. | `argumen int` |
| `n_embed` · 18 | Banyak koordinat vektor representasi satu token. | `argumen int` |
| `context_length` · 18 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `self.ln1` · 28 | LayerNorm sebelum submodul attention pada Block. | `nn.LayerNorm(n_embed)` |
| `self.attn` · 29 | Submodul MultiHeadAttention dalam Block. | `MultiHeadAttention(n_head, n_embed, context_length)` |
| `self.ln2` · 30 | LayerNorm sebelum MLP pada Block. | `nn.LayerNorm(n_embed)` |
| `self.mlp` · 31 | Submodul MLP feed-forward per posisi dalam Block. | `MLP(n_embed)` |

### Lingkup `Block.forward`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 33 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `x` · 33, 44, 46 | Representasi yang sedang ditransformasikan. Bentuknya mengikuti operasi pada fungsi; MLP memperlebar lalu mengembalikannya. | `argumen torch.Tensor`; `x + self.attn(self.ln1(x))`; `x + self.mlp(self.ln2(x))` |

### Lingkup `Block.forward_embedding`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 49 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `x` · 49, 60 | Representasi yang sedang ditransformasikan. Bentuknya mengikuti operasi pada fungsi; MLP memperlebar lalu mengembalikannya. | `argumen torch.Tensor`; `self.mlp.forward_embedding(self.ln2(res))` |
| `res` · 59 | Representasi residual yang disimpan untuk penjumlahan atau inspeksi; res pada evaluator juga dictionary hasil. | `x + self.attn(self.ln1(x))` |

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `batch_size` · 65 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `2` |
| `sequence_length` · 66 | Banyak posisi token pada urutan yang diproses. | `5` |
| `embedding_dim` · 67 | Banyak koordinat vektor representasi satu token. | `32` |
| `num_heads` · 68 | Jumlah attention head dalam setiap blok. | `4` |
| `context_len` · 69 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `5` |
| `input_tensor` · 70 | Tensor acak masukan pada demo modul di __main__, bukan data bahasa terlatih. | `torch.randn(batch_size, sequence_length, embedding_dim)` |
| `transformer_block` · 72 | Objek jaringan yang dipakai dalam demo modul terkait. | `Block(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)` |
| `output_tensor` · 73 | Tensor keluaran modul pada demo bentuk data. | `transformer_block(input_tensor)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `torch` | `torch` | 1 |
| `nn` | `torch.nn` | 2 |
| `MultiHeadAttention` | `src.models.attention.MultiHeadAttention` | 3 |
| `MLP` | `src.models.mlp.MLP` | 4 |
