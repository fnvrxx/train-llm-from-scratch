# src/models/transformer.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/models/transformer.py)

## Fungsi dan kelas

### `Transformer` (baris 9)

Embedding token/posisi, rangkaian blok causal, final norm, dan lm_head.

### `Transformer.__init__` (baris 23)

Membuat komponen dan keadaan objek Transformer. Embedding token/posisi, rangkaian blok causal, final norm, dan lm_head.

```python
def __init__(self, n_head: int, n_embed: int, context_length: int, vocab_size: int, N_BLOCKS: int) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `Transformer._pre_attn_pass` (baris 48)

Menjumlahkan vektor token dan posisi untuk setiap anggota batch.

```python
def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `tok_embedding + pos_embedding`.

### `Transformer.forward_hidden` (baris 63)

Menjalankan embedding, seluruh blok, dan final LayerNorm untuk menghasilkan hidden states (B,T,C).

```python
def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `self.layer_norm(x)`.

### `Transformer.forward` (baris 90)

Menghasilkan logits dan opsional cross-entropy terhadap target yang sudah sejajar.

```python
def forward(self, idx: torch.Tensor, targets: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor | None]:
```

Keluaran eksplisit: `(logits, loss)`.

### `Transformer.forward_embedding` (baris 114)

Jalur inspeksi aktivasi tengah; penerusan 4C ke blok berikutnya bermasalah untuk banyak blok. Lihat bab 3.

```python
def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(x, residual)`.

### `Transformer.generate` (baris 130)

Sampling token berikutnya berulang sampai batas token tambahan, tanpa stop EOT.

```python
def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
```

Keluaran eksplisit: `idx`.

## Variabel dan parameter menurut lingkup

### Lingkup `Transformer.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 23 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `n_head` · 23 | Jumlah attention head dalam setiap blok. | `argumen int` |
| `n_embed` · 23 | Banyak koordinat vektor representasi satu token. | `argumen int` |
| `context_length` · 23 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `argumen int` |
| `vocab_size` · 23 | Jumlah ID kandidat token, yaitu lebar keluaran lm_head. | `argumen int` |
| `N_BLOCKS` · 23 | Jumlah blok Transformer yang disusun berurutan. | `argumen int` |
| `self.context_length` · 35 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `context_length` |
| `self.N_BLOCKS` · 36 | Jumlah blok Transformer yang disusun berurutan. | `N_BLOCKS` |
| `self.gradient_checkpointing` · 40 | Sakelar menghitung ulang aktivasi blok saat backward untuk mengurangi memori. | `False` |
| `self.token_embed` · 41 | Tabel parameter embedding token (vocab_size,n_embed). | `nn.Embedding(vocab_size, n_embed)` |
| `self.position_embed` · 42 | Tabel parameter embedding posisi absolut (context_length,n_embed). | `nn.Embedding(context_length, n_embed)` |
| `self.attn_blocks` · 43 | ModuleList blok Transformer yang dilalui berurutan. | `nn.ModuleList([Block(n_head, n_embed, context_length) for _ in range(N_BLOCKS)])` |
| `_` · 43 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `iterasi range(N_BLOCKS)` |
| `self.layer_norm` · 44 | LayerNorm terakhir setelah seluruh blok Transformer. | `nn.LayerNorm(n_embed)` |
| `self.lm_head` · 45 | Linear dari hidden features ke skor seluruh kosakata. | `nn.Linear(n_embed, vocab_size)` |
| `self.pos_idxs` · 46 | Buffer indeks posisi, bukan parameter trainable. | `register_buffer: torch.arange(context_length)` |

### Lingkup `Transformer._pre_attn_pass`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 48 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 48 | ID token input (B,T) untuk lookup embedding. | `argumen torch.Tensor` |
| `B` · 58 | Jumlah baris input, dibaca tetapi tidak dipakai lagi di fungsi ini. | `elemen 0 dari idx.shape` |
| `T` · 58 | Banyak posisi token pada urutan yang diproses. | `elemen 1 dari idx.shape` |
| `tok_embedding` · 59 | Vektor embedding token, bentuk (B,T,C). | `self.token_embed(idx)` |
| `pos_embedding` · 60 | Vektor posisi (T,C), ditambahkan ke seluruh anggota batch dengan broadcasting. | `self.position_embed(self.pos_idxs[:T])` |

### Lingkup `Transformer.forward_hidden`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 63 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 63 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `x` · 79, 85, 87 | Representasi yang sedang ditransformasikan. Bentuknya mengikuti operasi pada fungsi; MLP memperlebar lalu mengembalikannya. | `self._pre_attn_pass(idx)`; `checkpoint.checkpoint(block, x, use_reentrant=False)`; `block(x)` |
| `block` · 80 | Satu head atau satu blok dalam iterasi modul model. | `iterasi self.attn_blocks` |

### Lingkup `Transformer.forward`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 90 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 90 | ID token input, bentuk (B,T). | `argumen torch.Tensor` |
| `targets` · 90, 110 | ID target sejajar logits, sudah digeser pemanggil; diratakan menjadi panjang B*T. | `argumen torch.Tensor; default None`; `targets.reshape(B * T).long()` |
| `x` · 101 | Hidden states setelah final LayerNorm, bentuk (B,T,n_embed). | `self.forward_hidden(idx)` |
| `logits` · 102 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `self.lm_head(x)` |
| `loss` · 103, 111 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `None`; `F.cross_entropy(flat_logits, targets)` |
| `B` · 105 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `elemen 0 dari logits.shape` |
| `T` · 105 | Banyak posisi token pada urutan yang diproses. | `elemen 1 dari logits.shape` |
| `C` · 105 | Ukuran kosakata V, karena diambil dari sumbu terakhir logits. | `elemen 2 dari logits.shape` |
| `flat_logits` · 109 | Logits dengan sumbu batch dan waktu digabung menjadi (B*T,V). | `logits.reshape(B * T, C)` |

### Lingkup `Transformer.forward_embedding`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 114 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 114 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `x` · 124, 127 | Representasi yang sedang ditransformasikan. Bentuknya mengikuti operasi pada fungsi; MLP memperlebar lalu mengembalikannya. | `self._pre_attn_pass(idx)`; `elemen 0 dari block.forward_embedding(x)` |
| `residual` · 125, 127 | Representasi residual yang disimpan untuk penjumlahan atau inspeksi; res pada evaluator juga dictionary hasil. | `x`; `elemen 1 dari block.forward_embedding(x)` |
| `block` · 126 | Satu head atau satu blok dalam iterasi modul model. | `iterasi self.attn_blocks` |

### Lingkup `Transformer.generate`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 130 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 130, 147 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor`; `torch.cat((idx, idx_next), dim=1)` |
| `max_new_tokens` · 130 | Batas jumlah token tambahan di luar prompt. | `argumen int` |
| `_` · 141, 143 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `iterasi range(max_new_tokens)`; `elemen 1 dari self(idx_cond)` |
| `idx_cond` · 142 | Potongan konteks terakhir yang masih muat untuk prediksi berikutnya. | `idx[:, -self.context_length:]` |
| `logits` · 143, 144 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `elemen 0 dari self(idx_cond)`; `logits[:, -1, :]` |
| `probs` · 145 | Peluang token setelah softmax, dipakai multinomial untuk sampling. | `F.softmax(logits, dim=-1)` |
| `idx_next` · 146 | Token berikutnya hasil sampling distribusi model. | `torch.multinomial(probs, num_samples=1)` |

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `batch_size` · 152 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `2` |
| `sequence_length` · 153 | Banyak posisi token pada urutan yang diproses. | `5` |
| `vocab_size` · 154 | Jumlah ID kandidat token, yaitu lebar keluaran lm_head. | `100` |
| `embedding_dim` · 155 | Banyak koordinat vektor representasi satu token. | `32` |
| `num_heads` · 156 | Jumlah attention head dalam setiap blok. | `4` |
| `num_blocks` · 157 | Jumlah blok Transformer yang disusun berurutan. | `2` |
| `context_len` · 158 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `5` |
| `input_indices` · 159 | Tensor acak masukan pada demo modul di __main__, bukan data bahasa terlatih. | `torch.randint(0, vocab_size, (batch_size, sequence_length))` |
| `transformer_model` · 161 | Objek jaringan yang dipakai dalam demo modul terkait. | `Transformer(n_head=num_heads, n_embed=embedding_dim, context_length=context_len, vocab_size=vocab_size, N_BLOCKS=num_blocks)` |
| `logits` · 162 | Skor sebelum softmax. Pada DPO, nama ini adalah selisih log-ratio preferensi per pasangan. | `elemen 0 dari transformer_model(input_indices, targets=input_indices)` |
| `loss` · 162 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `elemen 1 dari transformer_model(input_indices, targets=input_indices)` |
| `start_indices` · 168 | ID token awal sebagai prefix generation. | `input_indices[:, :1]` |
| `generated_tokens` · 169 | Token hasil generation; jalur dasar menyertakan prefix input. | `transformer_model.generate(start_indices, max_new_tokens=5)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 1 |
| `torch` | `torch` | 3 |
| `nn` | `torch.nn` | 4 |
| `F` | `torch.nn.functional` | 5 |
| `checkpoint` | `torch.utils.checkpoint` | 6 |
| `Block` | `src.models.transformer_block.Block` | 7 |
