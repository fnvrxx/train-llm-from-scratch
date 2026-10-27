# src/models/attention.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/models/attention.py)

## Fungsi dan kelas

### `Head` (baris 6)

Satu causal scaled dot-product attention head.

### `Head.__init__` (baris 19)

Membuat komponen dan keadaan objek Head. Satu causal scaled dot-product attention head.

```python
def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `Head.forward` (baris 35)

Membentuk Q/K/V, causal mask, softmax skor, lalu kombinasi values.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `out`.

### `MultiHeadAttention` (baris 60)

Kumpulan head paralel yang hasilnya disambung dan diproyeksikan.

### `MultiHeadAttention.__init__` (baris 72)

Membuat komponen dan keadaan objek MultiHeadAttention. Kumpulan head paralel yang hasilnya disambung dan diproyeksikan.

```python
def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `MultiHeadAttention.forward` (baris 85)

Menyambung output head pada dimensi fitur dan menjalankan proyeksi linear.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `x`.

## Variabel dan parameter menurut lingkup

### Lingkup `Head.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 19 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `head_size` · 19 | Dimensi proyeksi query, key, dan value satu head. | `argumen int` |
| `n_embed` · 19 | Banyak koordinat vektor representasi satu token. | `argumen int` |
| `context_length` · 19 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `self.key` · 29 | Linear dari n_embed ke head_size untuk membentuk keys; key pada utilitas cfg adalah nama field yang dicari. | `nn.Linear(n_embed, head_size, bias=False)` |
| `self.query` · 30 | Linear dari n_embed ke head_size untuk membentuk queries. | `nn.Linear(n_embed, head_size, bias=False)` |
| `self.value` · 31 | Linear pembentuk values jika atribut Head; pada utilitas konversi/reduksi berupa nilai skalar masukan. | `nn.Linear(n_embed, head_size, bias=False)` |
| `self.tril` · 33 | Buffer segitiga bawah yang menutup attention ke token masa depan. | `register_buffer: torch.tril(torch.ones(context_length, context_length))` |

### Lingkup `Head.forward`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 35 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `x` · 35 | Representasi input (B,T,n_embed) yang diproyeksikan menjadi Q/K/V. | `argumen torch.Tensor` |
| `B` · 45 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `elemen 0 dari x.shape` |
| `T` · 45 | Banyak posisi token pada urutan yang diproses. | `elemen 1 dari x.shape` |
| `C` · 45 | Dimensi embedding input, sama dengan n_embed. | `elemen 2 dari x.shape` |
| `head_size` · 46 | Dimensi proyeksi query, key, dan value satu head. | `self.key.out_features` |
| `k` · 47 | Key tiap posisi, bentuk (B,T,head_size), hasil self.key(x). | `self.key(x)` |
| `q` · 48 | Query tiap posisi, bentuk (B,T,head_size), hasil self.query(x). | `self.query(x)` |
| `scale_factor` · 49 | 1/sqrt(head_size), pengendali skala dot product query-key. | `1 / math.sqrt(head_size)` |
| `attn_weights` · 51, 53, 54 | Skor pasangan posisi sebelum softmax, lalu bobot attention setelah softmax, bentuk (B,T,T). | `q @ k.transpose(-2, -1) * scale_factor`; `attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))`; `F.softmax(attn_weights, dim=-1)` |
| `v` · 55 | Value tiap posisi, bentuk (B,T,head_size), informasi yang dicampur oleh bobot attention. | `self.value(x)` |
| `out` · 57 | Hasil kombinasi values berbobot attention, bentuk (B,T,head_size). | `attn_weights @ v` |

### Lingkup `MultiHeadAttention.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 72 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `n_head` · 72 | Jumlah attention head dalam setiap blok. | `argumen int` |
| `n_embed` · 72 | Banyak koordinat vektor representasi satu token. | `argumen int` |
| `context_length` · 72 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `self.heads` · 82 | ModuleList attention head dengan proyeksi masing-masing. | `nn.ModuleList([Head(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])` |
| `_` · 82 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `iterasi range(n_head)` |
| `self.proj` · 83 | Linear yang mengembalikan concatenated heads atau aktivasi MLP ke lebar n_embed. | `nn.Linear(n_embed, n_embed)` |

### Lingkup `MultiHeadAttention.forward`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 85 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `x` · 85, 96, 98 | Input (B,T,C), lalu concatenation seluruh head, lalu proyeksi kembali (B,T,C). | `argumen torch.Tensor`; `torch.cat([h(x) for h in self.heads], dim=-1)`; `self.proj(x)` |
| `h` · 96 | Satu instance Head yang dipanggil pada input x. | `iterasi self.heads` |

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `batch_size` · 103 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `2` |
| `sequence_length` · 104 | Banyak posisi token pada urutan yang diproses. | `5` |
| `embedding_dim` · 105 | Banyak koordinat vektor representasi satu token. | `32` |
| `num_heads` · 106 | Jumlah attention head dalam setiap blok. | `4` |
| `context_len` · 107 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `5` |
| `input_tensor` · 108 | Tensor acak masukan pada demo modul di __main__, bukan data bahasa terlatih. | `torch.randn(batch_size, sequence_length, embedding_dim)` |
| `multihead_attn` · 110 | Objek jaringan yang dipakai dalam demo modul terkait. | `MultiHeadAttention(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)` |
| `output_tensor` · 111 | Tensor keluaran modul pada demo bentuk data. | `multihead_attn(input_tensor)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `torch` | `torch` | 1 |
| `nn` | `torch.nn` | 2 |
| `F` | `torch.nn.functional` | 3 |
| `math` | `math` | 4 |
