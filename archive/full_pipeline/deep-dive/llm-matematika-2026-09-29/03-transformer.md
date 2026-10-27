# 3. Mengikuti satu forward pass

[Daftar bab](../llm-matematika-2026-09-29.md) · [Berikutnya: training](04-training.md)

Kode: [MLP](../../src/models/mlp.py), [attention](../../src/models/attention.py), [Block](../../src/models/transformer_block.py), [Transformer](../../src/models/transformer.py). Semua kelas dan deklarasinya juga terdaftar dalam [kamus](kamus/README.md).

## Embedding: memilih baris matriks

`Transformer.__init__` membuat `token_embed` berukuran $(V,C)$ dan `position_embed` berukuran $(T_{\max},C)$. Pada `_pre_attn_pass(idx)`, indeks token memilih baris matriks pertama; `pos_idxs[:T]` memilih baris matriks kedua.

$$
X_{b,t,:}=E_{x_{b,t},:}+P_{t,:}.
$$

`tok_embedding` berbentuk $(B,T,C)$, `pos_embedding` berbentuk $(T,C)$. Broadcasting menambahkan posisi yang sama untuk semua anggota batch. Tanpa informasi posisi, attention saja tidak membedakan urutan dengan cara yang dibutuhkan model ini.

Posisi di sini merupakan parameter terlatih. Alternatifnya antara lain posisi sinusoidal atau rotary position embedding. Kode lokal memakai learned absolute positions, sehingga panjang masukan tidak boleh melampaui tabel posisi.

## Satu attention head

Anggap satu anggota batch mempunyai matriks $X\in\mathbb R^{T\times C}$. Head membentuk tiga proyeksi:

$$
Q=XW_Q^\top,\qquad K=XW_K^\top,\qquad U=XW_V^\top,
\qquad Q,K,U\in\mathbb R^{T\times d}.
$$

Saya memakai $U$ untuk values agar tidak tertukar dengan $V$ ukuran kosakata. Di kode values bernama `v`.

Query posisi $t$ dibandingkan dengan key posisi $j$. Hasilnya mengatur seberapa banyak informasi value posisi $j$ disertakan dalam keluaran posisi $t$:

$$
S=\frac{QK^\top}{\sqrt d}+M,
\quad M_{tj}=\begin{cases}0&j\le t\\-\infty&j>t,\end{cases}
\quad A_{tj}=\frac{e^{S_{tj}}}{\sum_k e^{S_{tk}}},
\quad O=AU.
$$

Setiap baris $A$ adalah distribusi bobot atas posisi yang tersedia. Keluaran merupakan kombinasi berbobot vektor values. Skala $1/\sqrt d$ mengendalikan pertumbuhan dot product; jika koordinat query dan key independen dengan variansi satu, variansi jumlah $d$ produk bertumbuh sebanding dengan $d$.

| Baris kode | Apa yang dihitung |
|---|---|
| `B, T, C = x.shape` | Membaca ukuran input |
| `head_size = self.key.out_features` | Mengambil $d$ |
| `k = self.key(x)` | Matriks $K$ |
| `q = self.query(x)` | Matriks $Q$ |
| `scale_factor = 1 / math.sqrt(head_size)` | Faktor $1/\sqrt d$ |
| `attn_weights = q @ k.transpose(-2,-1) * scale_factor` | Skor pasangan posisi, $(B,T,T)$ |
| `masked_fill(self.tril[:T,:T] == 0, -inf)` | Menutup posisi masa depan |
| `F.softmax(attn_weights, dim=-1)` | Normalisasi per baris menjadi $A$ |
| `v = self.value(x)` | Matriks values $U$ |
| `out = attn_weights @ v` | Keluaran $(B,T,d)$ |

`self.key`, `self.query`, dan `self.value` adalah lapisan linear tanpa bias. `self.tril` merupakan buffer matriks segitiga bawah. Ia mengikuti perpindahan model ke device, tetapi optimizer tidak mempelajarinya.

Attention dalam proyek ini mengambil gagasan scaled dot-product attention. Arsitektur lokal merupakan model causal decoder dengan posisi terlatih, bukan salinan lengkap encoder-decoder pada makalah awal. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Ilustrasi attention yang bisa dihitung tangan

Ambil $T=2$, $d=1$, $Q=K=(1,2)^\top$, dan values $U=(10,20)^\top$. Ini angka buatan untuk latihan, bukan bobot model.

$$
S=\begin{pmatrix}1&-\infty\\2&4\end{pmatrix},
\quad A\approx\begin{pmatrix}1&0\\0.1192&0.8808\end{pmatrix},
\quad O\approx\begin{pmatrix}10\\18.808\end{pmatrix}.
$$

Posisi pertama hanya boleh membaca dirinya. Posisi kedua dapat mencampur kedua values. Angka attention tidak dengan sendirinya menjadi penjelasan lengkap mengapa model menjawab sesuatu; masih ada head lain, proyeksi, MLP, dan blok berikutnya.

## Banyak head dan satu proyeksi

`MultiHeadAttention.__init__` menyimpan `self.heads` sebagai `ModuleList`. Setiap `h` punya proyeksi sendiri. `forward(x)` menjalankan seluruh `h(x)`, menyambung hasil pada sumbu terakhir, lalu menjalankan `self.proj`.

$$
\operatorname{MHA}(X)=\operatorname{Concat}(O_1,\ldots,O_H)W_O^\top+b_O.
$$

Lebar gabungan harus sama dengan `n_embed`. Karena kode memakai pembagian bulat `n_embed // n_head`, pilih `n_embed` habis dibagi `n_head`. Tidak ada pemeriksaan eksplisit syarat ini dalam konstruktor.

## MLP memproses fitur per posisi

`MLP` memakai perluasan $C\to4C$, ReLU, lalu proyeksi $4C\to C$:

$$
\operatorname{MLP}(x)=\operatorname{ReLU}(xW_1^\top+b_1)W_2^\top+b_2,
\qquad \operatorname{ReLU}(z)=\max(0,z).
$$

`self.hidden` adalah linear pertama, `self.relu` fungsi nonlinier, dan `self.proj` linear kedua. `forward_embedding(x)` menghasilkan aktivasi lebar $4C$; `project_embedding(x)` mengembalikannya ke $C$; `forward(x)` menjalankan keduanya.

Tanpa aktivasi nonlinier, komposisi dua pemetaan affine tetap affine. ReLU memungkinkan fungsi yang lebih kaya. Attention mencampur informasi antar posisi, sementara MLP mentransformasikan fitur setiap posisi memakai bobot yang sama. Alternatif MLP antara lain GELU atau gated MLP; kode ini memilih ReLU yang sederhana.

## Block: normalisasi dan residual

`Block.__init__` menyimpan `ln1`, `attn`, `ln2`, dan `mlp`. `Block.forward` menjalankan:

$$
R=X+\operatorname{MHA}(\operatorname{LN}_1(X)),
\qquad Y=R+\operatorname{MLP}(\operatorname{LN}_2(R)).
$$

LayerNorm pada satu vektor fitur $x$ menghitung:

$$
\mu=\frac1C\sum_jx_j,\quad s^2=\frac1C\sum_j(x_j-\mu)^2,
\quad \operatorname{LN}(x)_j=\gamma_j\frac{x_j-\mu}{\sqrt{s^2+\epsilon}}+\beta_j.
$$

$\gamma_j,\beta_j$ adalah parameter skala dan pergeseran. Normalisasi dilakukan sepanjang fitur setiap token, bukan sepanjang batch. Residual menyediakan jalur penjumlahan identitas sehingga blok mempelajari perubahan atas representasi yang masuk.

`Block.forward_embedding` berbeda: ia mengembalikan aktivasi $4C$ sebelum proyeksi MLP beserta `res` berlebar $C$. Fungsi ini berguna untuk inspeksi aktivasi. Ia bukan pengganti `Block.forward`.

## Transformer lengkap

| Metode atau atribut | Perannya |
|---|---|
| `context_length`, `N_BLOCKS` | Batas panjang input dan jumlah blok |
| `attn_blocks` | Deretan `Block` yang dilalui representasi |
| `gradient_checkpointing` | Jika aktif saat training, hitung ulang aktivasi saat backward untuk menghemat memori |
| `_pre_attn_pass` | Gabungkan embedding token dan posisi |
| `forward_hidden` | Jalankan seluruh blok dan `layer_norm`; hasil $(B,T,C)$ |
| `lm_head` | Proyeksi fitur $C$ ke skor $V$ kandidat token |
| `forward` | Kembalikan `(logits, loss)`; loss hanya ada jika `targets` diberikan |
| `forward_embedding` | Jalur inspeksi aktivasi dan residual, dengan keterbatasan di bawah |
| `generate` | Tambahkan token sampel berulang kali |

Di `forward`, `flat_logits` berukuran $(BT,V)$ dan `targets` menjadi vektor panjang $BT$. `F.cross_entropy` menerima logits mentah. Jangan lakukan softmax lagi sebelum fungsi ini: normalisasi log-probabilitas sudah bagian dari perhitungannya.

Ada masalah bentuk pada jalur lokal `Transformer.forward_embedding`: ia meneruskan keluaran $4C$ dari `Block.forward_embedding` ke blok berikutnya yang mengharapkan $C$. Untuk lebih dari satu blok, jalur itu tidak cocok tanpa proyeksi tambahan. Jalur training biasa memakai `forward_hidden`, sehingga penjelasan forward pass utama di atas tidak melewati masalah ini.

Demo `targets=input_indices` di bagian `__main__` hanya memperlihatkan bentuk tensor. Untuk belajar next-token prediction yang benar, gunakan target bergeser seperti `yb` pada loader.

## Latihan

Ilustrasi: $B=2,T=5,C=32,H=4$. Berapa ukuran `attn_weights`, keluaran satu head, dan keluaran MLP?

Jawaban: `attn_weights` pada tiap head $(2,5,5)$, keluaran head $(2,5,8)$, aktivasi tengah MLP $(2,5,128)$, dan keluaran MLP $(2,5,32)$. Empat keluaran head disambung menjadi $(2,5,32)$.
