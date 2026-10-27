# 1. Dari matematika ke operasi tensor

[Daftar bab](../llm-matematika-2026-09-29.md) · [Berikutnya: data](02-data.md)

## Apa yang dipelajari model?

Misalkan ada urutan token $x_1,\ldots,x_T$. Aturan rantai probabilitas memberi:

$$
p_\theta(x_1,\ldots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

Transformer menghitung distribusi bersyarat tersebut. Bobot awalnya acak. Training menggeser bobot supaya token yang muncul dalam data mendapat peluang lebih besar pada konteksnya. Prediksi untuk token awal memerlukan konteks awal; dalam loader proyek, target pertama adalah token setelah input pertama.

Ini menghubungkan LLM dengan maximum likelihood estimation. Untuk pasangan konteks dan target, memaksimalkan likelihood sama dengan meminimalkan negative log-likelihood:

$$
\mathcal L(\theta)=-\frac1M\sum_{i=1}^{M}\log p_\theta(y_i\mid c_i).
$$

$M$ adalah banyak target yang dihitung, $c_i$ konteks, dan $y_i$ token target. Log dipakai karena produk peluang menjadi penjumlahan, sehingga perhitungannya lebih stabil.

## Tensor sebagai tabel berindeks

Anda sudah mengenal vektor dan matriks. Tensor pada PyTorch adalah array multidimensi beserta jenis angka dan lokasi komputasinya. Untuk memahami kode ini, cukup ikuti arti setiap sumbu.

| Nama | Arti | Contoh ilustratif |
|---|---|---|
| `B` | Batch size: jumlah urutan yang diproses bersama | 2 kalimat |
| `T` | Banyak posisi token pada setiap urutan | 5 token |
| `C`, `n_embed` | Banyak koordinat representasi satu token | 32 koordinat |
| `V`, `vocab_size` | Banyak kandidat ID token | 100 kandidat pada demo kecil |
| `n_head` | Banyak attention head | 4 |
| `head_size` | Lebar tiap head, biasanya `n_embed // n_head` | 8 |

`idx.shape == (B,T)` berarti setiap elemen adalah ID bilangan bulat. Setelah embedding, `x.shape == (B,T,C)` berarti setiap ID memiliki sebuah vektor. Sesudah `lm_head`, `logits.shape == (B,T,V)` berarti setiap posisi mempunyai satu skor untuk setiap kandidat token.

Jangan menentukan arti huruf hanya dari namanya. Dalam `Transformer.forward`, kode menulis `B,T,C = logits.shape`. Di sana `C` bernilai $V$, karena objek yang dibaca adalah logits.

## Membaca Python di proyek ini

| Sintaks | Cara membaca |
|---|---|
| `class MLP(nn.Module)` | Mendefinisikan jenis objek jaringan; mewarisi pengelolaan parameter dari PyTorch |
| `def __init__(self, n_embed)` | Mengatur lapisan saat objek dibuat |
| `self.hidden` | Lapisan yang disimpan dalam objek; `self` menunjuk objek yang sedang dipakai |
| `def forward(self, x)` | Mendefinisikan perhitungan maju |
| `model(x)` | PyTorch menjalankan mekanisme pemanggilan modul, termasuk `forward(x)` |
| `x: torch.Tensor` | Anotasi tipe untuk pembaca dan pemeriksa kode; tidak menciptakan tensor |
| `-> tuple[...]` | Anotasi tipe hasil fungsi |
| `None` | Tidak ada nilai; misalnya loss ketika tidak diberi target |
| `logits, _ = model(tokens)` | Ambil logits; hasil kedua tidak digunakan |
| `@torch.no_grad()` | Matikan pencatatan graph turunan pada pemanggilan fungsi |
| `@dataclass` | Python membuat konstruktor dan metode data sederhana dari deklarasi field |
| `yield xb, yb` | Keluarkan batch, lalu lanjutkan fungsi dari posisi ini ketika diminta lagi |
| `with ...` | Batasi masa berlaku sumber daya atau mode komputasi |
| `if __name__ == '__main__'` | Jalankan bagian demo/CLI hanya jika berkas dieksekusi langsung |

`torch`, `np`, `nn`, dan `F` adalah alias impor: PyTorch, NumPy, `torch.nn`, serta `torch.nn.functional`. `nn.Linear` menyimpan parameter, sementara `F.softmax` menjalankan operasi pada tensor yang diberikan. `h5py` membaca HDF5; `tiktoken` mengubah teks menjadi token; `argparse` membaca argumen terminal.

## Operasi yang sering muncul

| Kode | Arti matematis atau perubahan bentuk |
|---|---|
| `nn.Linear(C,D)(x)` | $xW^\top+b$, bobot PyTorch berbentuk $(D,C)$ |
| `a @ b` | Perkalian matriks pada dua sumbu terakhir, dengan batch bila ada |
| `k.transpose(-2,-1)` | Tukar sumbu panjang token dan dimensi head |
| `dim=-1` | Operasikan sepanjang sumbu terakhir |
| `x[:, :-1, :]` | Semua batch, semua posisi kecuali terakhir, semua fitur |
| `x[:, 1:]` | Semua posisi mulai indeks 1, sehingga target bergeser satu token |
| `reshape(-1,V)` | Gabungkan sumbu lain; PyTorch menghitung dimensi bertanda `-1` |
| `unsqueeze(-1)` | Tambah sumbu terakhir sepanjang 1 |
| `squeeze(-1)` | Hapus sumbu terakhir jika panjangnya 1 |
| `gather(-1, targets)` | Pilih komponen yang indeksnya adalah ID target |
| `detach()` | Ambil nilai tanpa meneruskan gradien melalui operasi pembentuknya |
| `item()` | Ubah tensor skalar menjadi angka Python, misalnya untuk log |
| `clamp(min=1)` | Batasi nilai bawah menjadi 1; sering untuk menjaga pembagian |
| `register_buffer(...)` | Simpan tensor bersama model, tetapi bukan parameter yang dilatih |

`model.eval()` mengubah mode modul seperti dropout. Ia tidak mematikan autograd. `no_grad()` mematikan autograd tetapi tidak otomatis mengganti mode modul. Keduanya menjawab kebutuhan berbeda. PyTorch menghitung turunan dengan menelusuri graph operasi, lalu `backward()` mengakumulasikan gradien pada parameter. [Dokumentasi autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).

## Latihan

Ilustrasi: logits berbentuk `(2,5,100)`. Setelah `logits[:, :-1, :]`, berapa bentuknya? Mengapa perlu `reshape(-1,100)`?

Jawaban: `(2,4,100)`, kemudian `(8,100)`. Delapan baris itu adalah delapan tugas klasifikasi token. Setiap baris mempunyai 100 kandidat kelas. Tidak ada informasi urutan yang hilang dari proses attention sebelumnya; penggabungan ini hanya memudahkan penghitungan loss.
