# Memahami Small LLM cerita pendek

**Tanggal:** 22 Oktober 2026  
**Fase:** implementasi model kecil, trainer bersama, dan perbandingan sparsity  
**Berkas:** `small_llm/*.py`, `configs/*.json`, `data/stories_demo/*.txt`, `tests/test_small_llm.py`

## Gambaran kerja

Program mempelajari kelanjutan cerita dengan satu tugas: untuk setiap token dalam suatu jendela, tebak token setelahnya. Satu ID mewakili satu byte UTF-8. Transformer menghasilkan 257 skor untuk tiap posisi, lalu cross-entropy menghukum skor yang rendah pada ID berikutnya. Perubahan bobot berasal dari gradien loss tersebut.

Eksperimen sparsity mempertahankan embedding, attention, data, dan trainer. Hanya MLP pada tiap blok yang diganti. Pembatasan ini membuat perubahan loss atau waktu lebih mudah dilacak ke metode sparsity, meski jumlah parameter dan operasi tidak sepenuhnya sama.

Korpus bawaan berisi 12 cerita contoh. Ia memungkinkan pemeriksaan alur kode, bukan pengukuran kemampuan menulis. Untuk penelitian kualitas, tambah cerita yang boleh digunakan dan ulangi pemisahan data serta seluruh varian.

## Baca kode dari awal sampai akhir

### 1. `small_llm/tokenizer.py`

`encode` mengubah string menjadi daftar byte. Bilangan `0..255` mencakup semua nilai byte; `256` menjadi `END_OF_STORY`. Contoh:

```python
ids = list(text.encode("utf-8"))
if end_of_story:
    ids.append(END_OF_STORY)
```

Baris pertama mengubah teks menjadi satuan yang dapat dimasukkan ke `nn.Embedding`. Baris kedua menambahkan batas cerita. `decode` berhenti pada batas itu dan memakai `errors="replace"` jika generasi berhenti di tengah huruf UTF-8. Karakter pengganti pada keluaran adalah tanda urutan byte belum lengkap, bukan kata yang dipelajari model.

**Alasan pilihan:** kosakata tetap membuat semua varian dapat memakai ID yang sama. Tokenizer BPE dapat memuat lebih banyak teks dalam konteks yang sama, tetapi memerlukan pelatihan tokenizer dan pemeriksaan kebocoran data. Byte berguna saat ingin mengisolasi cara kerja model bahasa terlebih dahulu.

### 2. `small_llm/data.py`

`load_stories` membaca berkas `.txt`, mengacak daftar berkas dengan seed, lalu menyisihkan sebagian berkas untuk validasi. Pengacakan dilakukan pada berkas, bukan pada potongan token. Jika satu cerita panjang dipotong menjadi 100 jendela, seluruh jendelanya tetap berada pada kelompok yang sama.

`sample_batch` mengambil potongan `context_length + 1` token dari aliran yang sudah digabung:

```python
x = tokens[i : i + context_length]
y = tokens[i + 1 : i + context_length + 1]
```

Setiap target bergeser satu posisi. Jika `x[0]` adalah byte `P`, model harus memberi skor tinggi pada byte sesudah `P`, bukan pada `P` itu sendiri. Fungsi mengirim tensor kecil ke CPU atau MPS setelah memilih titik awal di CPU. Generator acak khusus batch membuat urutan jendela dapat diulang untuk tiap varian.

**Batas:** jendela dapat melewati penanda akhir cerita. Model melihat penandanya, tetapi attention tidak direset pada batas tersebut. Ini cukup untuk demonstrasi prediksi aliran token. Isolasi penuh per cerita memerlukan sampler yang menahan jendela di satu berkas atau mask attention tambahan.

### 3. `small_llm/model.py`

`ModelConfig` memeriksa dimensi saat dibuat. `n_embed` harus habis dibagi `n_head` karena setiap head menerima bagian vektor dengan lebar sama.

`StoryTransformer.forward` menyusun:

```text
ID token + ID posisi → embedding → blok attention dan MLP → LayerNorm → logits
```

Jika bentuk token `(B, T)`, embedding menghasilkan `(B, T, C)`. `lm_head` mengubahnya menjadi `(B, T, 257)`. Baris `F.cross_entropy(logits.reshape(-1, 257), targets.reshape(-1))` menganggap setiap posisi sebagai satu prediksi klasifikasi. `reshape` mempertahankan urutan pasangan logit-target sambil menggabungkan dimensi batch dan waktu.

`CausalAttention` membagi proyeksi query, key, dan value menjadi `H` head. Skor setiap head berbentuk `(B, H, T, T)`. Mask segitiga menutup kolom yang berada di masa depan sebelum softmax:

```python
scores = scores.masked_fill(~causal_mask[:length, :length], float("-inf"))
weights = F.softmax(scores, dim=-1)
```

Nilai negatif tak hingga membuat probabilitas posisi tertutup menjadi nol. Tanpa mask, model bisa melihat target saat training dan loss tampak bagus tanpa belajar menghasilkan kelanjutan. Tes kausalitas mengubah token terakhir lalu memastikan logit posisi sebelumnya tetap sama.

`Block` melakukan dua penambahan residual: satu setelah attention, satu setelah MLP. MLP dense memakai `C → 4C → C` dan ReLU. `generate` memakai logit posisi terakhir untuk memilih token berikutnya, menambah token itu ke konteks, lalu mengulangi. Saat panjang konteks mencapai batas, hanya token terbaru yang dipakai pada langkah berikutnya. Tanpa KV cache, blok menghitung ulang konteks setiap langkah.

### 4. `small_llm/sparse.py`

`convert_model` mengganti `block.mlp` dengan salah satu kelas berikut. Semua varian dimulai dari `StoryTransformer` yang dibuat dengan seed sama.

| Kelas | Keputusan per token | Apa yang tersimpan |
|---|---|---|
| `PrunedMLP` | Mask magnitudo dipilih sekali saat awal | Bobot dense dan mask boolean |
| `TopKMLP` | Pilih aktivasi terbesar setelah proyeksi pertama | Seluruh bobot dense |
| `MoEMLP` | Router memilih beberapa expert | Bobot expert dan router |
| `SETMLP` | Mask diubah tiap beberapa langkah | Bobot dense, mask, dan state optimizer |

Pada pruning, `F.linear(x, weight * mask, bias)` membuat gradien untuk bobot yang dimask menjadi nol. Tensor bobot tetap berukuran penuh. Jadi persentase koneksi aktif mengukur struktur komputasi yang dimaksud, bukan penghematan memori fisik yang sudah terbukti.

Top-k tetap menghitung semua nilai pada lapisan `hidden`. Ia memilih `k` nilai dan mengambil kolom yang sesuai dari `proj.weight`. Metode ini terutama mengubah proyeksi kedua. Jumlah parameter tidak berubah.

MoE membagi lebar MLP dense menjadi beberapa expert. Dengan empat expert, masing-masing menerima seperempat lebar hidden. Router mengeluarkan skor untuk tiap token; hanya expert terpilih yang dijalankan. `aux_loss` menghitung hubungan antara probabilitas rata-rata router dan jumlah token yang diterima tiap expert. Trainer menambahkannya pada loss token dengan koefisien kecil. Loss validasi tetap hanya cross-entropy token agar metrik antarvarian sejenis.

SET mulai dengan mask acak. `rewire_set` mencari bobot aktif bermagnitudo terendah, mematikannya, lalu membuka jumlah koneksi tidur yang sama. Nilai bobot serta momen optimizer pada posisi yang berubah diatur nol. Tanpa pembersihan momen, koneksi baru dapat menerima riwayat gradien dari koneksi lama pada posisi tensor tersebut. Jumlah koneksi aktif diperiksa dalam tes.

**Alternatif:** pustaka sparse dan kernel khusus bisa menyimpan atau menghitung matriks sparse secara fisik. Kode ini memakai operasi tensor biasa supaya perubahan algoritme terlihat. Karena itu perbandingan harus melaporkan waktu dan memori yang benar-benar terukur, bukan menganggap sparsity selalu mempercepat training.

### 5. `small_llm/config.py`, `train.py`, dan `compare.py`

`config.py` memisahkan dimensi model, training, dan pengaturan sparse. Profil `smoke.json` dipakai untuk pemeriksaan singkat; `small_m1.json` adalah titik mulai pada M1 8 GB. `resolve_device("auto")` memilih MPS jika PyTorch menyatakan tersedia, lalu CPU. Konfigurasi tidak mengaktifkan mixed precision agar operasi awal mudah diperiksa dalam float32.

Satu langkah training di `train.py` menjalankan beberapa microbatch. Loss tiap microbatch dibagi `grad_accum` sebelum `backward`, kemudian satu `optimizer.step` memperbarui bobot:

```python
optimizer.zero_grad(set_to_none=True)
for _ in range(grad_accum):
    (loss / grad_accum).backward()
optimizer.step()
```

Kode lengkap menghitung `loss` baru untuk tiap microbatch dan memotong norm gradien sebelum langkah optimizer. Pembagian loss mencegah besar gradien bertambah hanya karena jumlah microbatch naik. Pada SET, mask ditegakkan kembali setelah AdamW memperbarui bobot, lalu rewiring terjadi pada interval yang dipilih.

`evaluate` memakai generator dengan seed tetap pada setiap pemanggilan, sehingga jendela validasi yang dibandingkan antarlangkah identik. Trainer menyimpan `metrics.jsonl`, ringkasan, dan checkpoint. Checkpoint memuat bobot untuk generasi, tetapi belum mendukung resume training persis karena optimizer dan posisi generator tidak disimpan.

`compare.py` menjalankan tiap varian sebagai proses Python tersendiri. Puncak RSS adalah puncak memori proses, sehingga satu proses yang sudah memuat varian sebelumnya tidak mencemari ukuran berikutnya. Tiap proses membaca config dan data yang sama. `--in-process` tersedia bila pembuatan proses anak dibatasi; dalam mode itu kolom RSS pada tabel dikosongkan. `train_tokens_per_second` membagi jumlah token training dengan waktu langkah optimizer, tanpa evaluasi, penyimpanan, atau generasi. Tabel tidak mengklaim operasi FLOP antarvarian sama.

### 6. `small_llm/generate.py` dan `tests/test_small_llm.py`

`generate.py` membaca metadata checkpoint, membangun arsitektur dengan varian yang sama, memuat bobot secara ketat, lalu meneruskan prompt. Jika Anda mengubah `n_embed` pada JSON setelah checkpoint dibuat, generasi tetap memakai dimensi yang tersimpan di checkpoint.

Tes memeriksa round-trip UTF-8, pemisahan cerita, pergeseran input-target, kausalitas, ekuivalensi top-k saat semua neuron aktif, rekonstruksi MoE ketika semua expert dipilih dengan bobot sama, konservasi koneksi SET, pembersihan state optimizer, dan pemuatan checkpoint sparse. Tes integrasi juga melatih setiap varian satu langkah lalu memuat checkpoint-nya. Tes tersebut memeriksa sifat yang bisa salah meski program selesai tanpa exception.

## Konsep yang perlu dikuasai

| Konsep | Arti dalam repo | Kapan berguna | Batas di sini |
|---|---|---|---|
| Prediksi token berikutnya | Logit posisi `t` menebak token `t+1` | Pretraining model autoregresif | Korpus contoh terlalu kecil untuk kualitas bahasa |
| Causal attention | Posisi `t` hanya membaca posisi `≤t` | Generasi tanpa melihat masa depan | Biaya attention naik kuadratis terhadap panjang konteks |
| Gradien dan AdamW | Loss menentukan arah perubahan bobot | Semua tahap training | Checkpoint belum menyimpan state untuk resume |
| Sparsity bobot | Mask menghilangkan koneksi dari perhitungan | Menguji struktur MLP | Mask dense belum menghemat penyimpanan bobot |
| Routing | Tiap token memilih expert | Mencoba komputasi bersyarat | Overhead router dapat mendominasi model kecil |

## Bacaan lanjut

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): asal arsitektur Transformer dan perhatian multi-head. Model repo ini memakai decoder kausal sederhana.
- [PyTorch: cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy): bentuk logits, target, dan rata-rata loss yang dipakai `StoryTransformer.forward`.
- [PyTorch: automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html): mengapa `backward` mengisi gradien parameter.
- [PyTorch: AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW): state optimizer dan pemisahan weight decay dari momen gradien.
- [PyTorch: MPS](https://docs.pytorch.org/docs/main/notes/mps.html): cara PyTorch menjalankan operasi pada GPU Apple dan memeriksa ketersediaannya.
- [Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity](https://arxiv.org/abs/1707.04780): gagasan Sparse Evolutionary Training. Implementasi repo ini memakai mask tensor dense agar langkah rewiring mudah diamati.
- [Switch Transformers](https://arxiv.org/abs/2101.03961): contoh penelitian routing expert. `MoEMLP` di repo ini adalah versi kecil untuk mempelajari mekanismenya.

## Latihan yang bisa Anda lakukan

1. Ambil satu cerita, tulis lima ID byte pertamanya, lalu tunjukkan pasangan input-target setelah digeser satu posisi.
2. Ubah `context_length` pada profil smoke dari 64 ke 128. Prediksi bagian kode mana yang memakai memori lebih besar sebelum melihat hasil.
3. Bandingkan `pruned` dengan `dense`: catat koneksi aktif, jumlah parameter, loss validasi, dan RSS. Jelaskan mengapa jumlah parameter dan RSS mungkin tetap mirip.
4. Set `top_k` MoE sama dengan `n_experts`, lalu periksa tes rekonstruksi dense. Ubah satu hal dan amati bagian asumsi yang rusak.
5. Tambahkan lebih banyak cerita, ulangi seluruh varian dengan seed yang sama, dan bandingkan lagi. Catat berkas train dan validasi yang tersimpan di `summary.json`.
