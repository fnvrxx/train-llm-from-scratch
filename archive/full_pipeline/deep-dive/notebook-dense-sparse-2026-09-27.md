# Memahami kode LLM: notebook dense, sparse, dan debugging

**Tanggal:** 2026-09-27  
**Fase:** notebook pembelajaran yang dapat dieksekusi  
**Cakupan:** alur data, Transformer, training, empat varian sparse, objective post-training,
generator notebook, dan pemeriksaan hasil. Ini peta belajar implementasi repo,
bukan audit setiap baris UI atau seluruh skrip pendukung.

## Mengapa sebelumnya hanya muncul file?

`%%writefile src/models/transformer.py` adalah perintah IPython untuk menyimpan isi
sel sebagai file. Definisi class di dalamnya belum dieksekusi di kernel. Output
`Writing ...` berarti file berhasil disiapkan, bukan model sudah belajar.

Setelah itu, `RUN_PRETRAIN = False` melewati blok training. Notebook lama juga
memanggil skrip melalui `subprocess.run`: training berlangsung dalam proses Python
lain. `model` dari proses itu tidak menjadi variabel notebook.

Sekarang kelima notebook mempunyai laboratorium otomatis dengan data karakter
lokal. Sel Python di laboratorium memanggil modul asli secara langsung, dan
menyimpan variabel antara di kernel. Training dataset penuh tetap merupakan bagian
opsional. Sel opsional mencetak status ketika dilewati.

## Peta kode dan urutan membaca

| File/folder | Tanggung jawab dan kaitannya |
|---|---|
| [learning_cells.py](../notebook/learning_cells.py) | Menyusun latihan executable dan penjelasan yang masuk ke kelima notebook |
| [build_notebooks.py](../notebook/build_notebooks.py) | Menyalin source ke `%%writefile`, menambahkan laboratorium dan pipeline, memvalidasi notebook |
| [data_loader.py](../data_loader/data_loader.py) | Membentuk input/target dari dataset HDF5 `tokens` |
| [transformer.py](../src/models/transformer.py) | Embedding, rangkaian blok, logits, cross entropy, generasi |
| [attention.py](../src/models/attention.py) | Proyeksi Q/K/V, causal mask, gabungan beberapa head |
| [transformer_block.py](../src/models/transformer_block.py) | LayerNorm dan residual di sekitar attention serta MLP |
| [mlp.py](../src/models/mlp.py) | Ekspansi C ke 4C, ReLU, proyeksi kembali ke C |
| [pretrain_base.py](../scripts/pretrain_base.py) | Orkestrasi pretraining, optimizer, evaluasi, checkpoint |
| [loader.py](../config/loader.py) | Memuat dan menggabungkan konfigurasi base/stage |
| [post_training](../src/post_training) | Objective dan komponen SFT, reward, DPO, PPO, GRPO, rollout, inference |
| [scripts](../scripts) | Entry point CLI untuk persiapan data, training tiap tahap, evaluasi, chat |
| [experiments](../experiments) | Penggantian MLP dan runner eksperimen sparse |
| [test_notebooks.py](../notebook/test_notebooks.py) | Eksekusi source notebook dalam proses IPython terisolasi |

Mulai dari laboratorium baseline, kemudian baca fungsi yang sedang dipanggil.
Setelah memahami satu forward dan satu update, bandingkan MLP sparse. Baru lanjut
ke pipeline dataset nyata dan post-training. File `ui/` menyediakan panel kontrol;
source UI yang disertakan notebook tidak berarti server UI sedang dijalankan.

## Alur satu batch, dari karakter sampai loss

### 1. Tokenisasi dan target

Komputer menerima ID integer, bukan makna kata. Demo memakai satu ID per karakter
agar pasangan teks/ID mudah diperiksa tanpa mengunduh tokenizer. Pipeline penuh
memakai tokenisasi berbeda; vocabulary dan embedding model harus cocok.

```python
samples = torch.stack([encoded[start:start + 13] for start in (0, 13)])
xb, yb = samples[:, :-1], samples[:, 1:]
```

Baris pertama mengambil dua potongan dengan 13 token. Baris kedua menghasilkan
12 input dan 12 target per contoh. Jika potongannya `saya `, input `saya` menargetkan
`aya `. Menggunakan target yang sama dengan input melatih tugas yang berbeda.

`get_batch_iterator` menerapkan ide ini pada HDF5, mengacak indeks potongan, lalu
mengembalikan batch melalui `yield`. Generator bersifat lazy: error file bisa baru
muncul saat `next(iterator)`. Jumlah contoh harus mencukupi batch; loader ini dapat
menghasilkan batch lebih kecil jika data terlalu pendek. Periksa shape dan rentang
ID sebelum mencoba training penuh.

### 2. Embedding dan posisi

`Transformer.__init__` membuat dua tabel embedding. `_pre_attn_pass` melakukan:

```python
tok_embedding = self.token_embed(idx)
pos_embedding = self.position_embed(self.pos_idxs[:T])
return tok_embedding + pos_embedding
```

`idx` berbentuk `[B,T]`. Tabel token mengubahnya menjadi `[B,T,C]`. Posisi berbentuk
`[T,C]` dan di-broadcast ke setiap contoh. Posisi membedakan urutan token yang sama.
`pos_idxs` adalah buffer: ikut pindah device dan masuk state dict, tetapi bukan
parameter yang dilatih. Alternatif posisi antara lain sinusoidal dan RoPE; model
repo ini memakai embedding posisi yang dipelajari.

| Simbol demo | Nilai | Makna |
|---|---|---|
| B | 2 | Contoh per batch |
| T | 12 | Posisi input |
| C | 16 | Lebar representasi |
| H | 2 | Jumlah attention head |
| D | 8 | Lebar satu head, C/H |
| V | `len(vocabulary)` | Jumlah karakter unik pada teks demo |

### 3. Causal attention

`Head.forward` memproyeksikan representasi menjadi Q, K, V. Q mewakili pencarian
informasi, K apa yang bisa dicocokkan, dan V informasi yang akan dicampurkan.

`q @ k.transpose(-2, -1)` menghasilkan skor `[B,T,T]`. Pembagian dengan akar lebar
head menjaga skala skor. Posisi masa depan diisi `-inf` sebelum softmax sehingga
probabilitasnya nol. `attention @ v` membentuk campuran informasi masa lalu.

Laboratorium menghitung matriks yang sama pada head pertama dan memeriksa bahwa
segitiga atas nol serta setiap baris berjumlah satu. Ini bukan attention seluruh
model yang dirata-ratakan. `MultiHeadAttention.forward` menggabungkan output head
melalui concatenation dan proyeksi linear.

### 4. Blok Transformer dan MLP

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

Baris pertama mencampurkan informasi antarposisi. Baris kedua mengolah fitur tiap
posisi. Keduanya menormalisasi input sebelum sublayer, lalu menambah residual.
Residual menyediakan jalur informasi dan gradien di luar transformasi sublayer.
Alternatif post-norm menempatkan normalisasi setelah penambahan; itu bukan susunan
yang digunakan di sini.

MLP dense menjalankan `Linear(C,4C)`, ReLU, lalu `Linear(4C,C)`. ReLU dapat membuat
aktivasi nol, tetapi tidak mengubah penyimpanan parameter menjadi sparse. Semua
varian eksperimen mengganti `block.mlp`; attention tetap sama.

### 5. Logits dan cross entropy

`forward_hidden` menjalankan blok dan LayerNorm akhir. `lm_head` memetakan C ke V.
Logits bukan probabilitas; softmax diperlukan ketika ingin menampilkan distribusi.
Untuk loss, `F.cross_entropy` sudah menangani normalisasi skor.

`Transformer.forward` me-reshape logits menjadi `[B*T,V]` dan target `[B*T]`.
Pemakaian `reshape` mengakomodasi slice target yang tidak contiguous. Loss rata-rata
mendorong skor token yang benar meningkat relatif terhadap kandidat lain.

## Mengapa backward dipisah dari optimizer?

`optimizer.zero_grad(set_to_none=True)` membuang akumulasi gradien sebelumnya.
`loss.backward()` mengisi `.grad` menggunakan autograd. `optimizer.step()` baru
memperbarui parameter menggunakan gradien dan state AdamW.

Laboratorium memisahkan pemeriksaan gradien dan update agar perubahan ini terlihat.
Sel backward menghitung forward baru sehingga dapat diulang tanpa memakai graph
yang sudah dibebaskan. Namun mengulang sel itu juga membuat optimizer baru dan
mereset momentum; loop sepuluh langkah berikutnya memakai optimizer yang sama.

`before_update` adalah clone embedding sebelum update. Selisih setelah `step`
membuktikan parameter berubah. Gradien tidak selalu nonzero untuk setiap parameter:
ReLU, mask, dan expert yang tidak terpilih dapat membuat jalur tertentu tidak aktif.

Loop demo menghafal satu batch. Loss menurun bukan bukti kemampuan bahasa atau
generalisasi. Training nyata membutuhkan split validasi terpisah dan evaluasi yang
konsisten. `model.eval()` mengatur mode modul, sedangkan `torch.no_grad()` mematikan
pencatatan graph; keduanya mempunyai fungsi berbeda.

## Empat makna sparse dalam repo

| Varian | Yang dipilih | Yang tetap dihitung/disimpan | Kegunaan dan batas |
|---|---|---|---|
| Pruned | Koneksi bobot berdasarkan magnitude | Tensor dense, operasi linear dengan bobot bermask | Menguji penghilangan bobot kecil; mask tidak berubah selama training |
| Top-k | Neuron hidden per token setelah ReLU | Seluruh skor hidden dan parameter dense | Menguji seleksi aktivasi; seleksi/gather bisa menambah overhead |
| MoE | Expert per token melalui router | Parameter semua expert dan router | Menguji routing bersyarat; penugasan bisa tidak seimbang |
| SET | Koneksi aktif yang dipangkas dan ditumbuhkan ulang | Tensor bobot serta state optimizer dense | Menguji perubahan topologi; bukan kernel sparse yang menghemat memori secara otomatis |

### Pruned: `experiments/pruned/model.py`

`_mask` meratakan nilai absolut bobot, mengurutkannya, lalu menonaktifkan proporsi
terkecil. `forward` memakai `weight * mask`, sehingga koneksi itu tidak menyumbang
ke output. Bias tidak dipangkas. Mask adalah buffer persisten dan ikut checkpoint.
Jumlah parameter tersimpan tetap sama walaupun koneksi efektif berkurang.

Magnitude pruning paling masuk akal dipelajari pada bobot yang sudah dilatih.
Demo memakai inisialisasi acak untuk memperlihatkan mekanismenya. Alternatif
structured pruning membuang neuron atau channel dan dapat mengubah dimensi matriks.

### Top-k: `experiments/topk/model.py`

`hidden = F.relu(self.hidden(x))` menghitung semua hidden. `topk` memberi nilai dan
indeks terpilih, kemudian `F.embedding` mengambil kolom proyeksi terkait. Kontribusi
ditimbang nilai aktivasi dan dijumlahkan. Keluaran tetap `[B,T,C]`.

Indeks dapat berbeda untuk setiap token. K slot terpilih tidak berarti semuanya
nonzero, karena hasil ReLU dapat nol. Tensor kolom terpilih mempunyai tambahan
dimensi k; ukuran intermediate dan overhead gather bisa besar. Bandingkan waktu
nyata pada hardware dan ukuran batch yang sama sebelum menyimpulkan percepatan.

### MoE: `experiments/moe/model.py`

Dense hidden dibagi menjadi beberapa expert, bukan diduplikasi utuh pada setiap
expert. Router linear menghasilkan skor expert; top-k memilih expert, softmax
menormalisasi bobot hanya pada pilihan itu. Kode mengalikan kontribusi dengan
`top_k` sebelum `index_add` mengumpulkan kembali output token.

Bobot expert diinisialisasi dari potongan dense, sementara router baru. Karena
routing dan bobot gate berubah, hasil umum tidak identik dengan dense. Auxiliary
loss memakai rata-rata probabilitas router dan frekuensi penugasan. Total objective
demo adalah `CE + 0.01 * aux_loss`; CE ditampilkan terpisah untuk perbandingan.

### SET: `experiments/evolutionary/model.py` dan `rewire.py`

Mask awal memilih jumlah koneksi acak sesuai density. `enforce_masks` membuat
bobot tidak aktif benar-benar nol. `rewire_weight` mencari bobot aktif terkecil,
membuangnya, lalu memilih koneksi dari himpunan yang sebelumnya tidak aktif.
Jumlah aktif dipertahankan, tetapi posisi berubah.

Koneksi baru dimulai dari nol dan memperoleh bobot pada update berikutnya.
`_clear_optimizer_state` menghapus state AdamW pada posisi yang berubah agar
momentum koneksi lama tidak terbawa. Laboratorium mencetak jumlah koneksi tumbuh
dan perbedaan mask; satu koneksi diganti berarti dua posisi mask berubah.

Implementasi ini merupakan eksperimen SET bermask dense. Jangan menyamakan angka
density dengan penghematan RAM/VRAM atau speedup. Paper SET memberi konteks algoritme,
tetapi klaim efisiensi implementasi lain tidak otomatis berlaku di sini.

## Dari pretraining ke post-training

| Tahap | Input dan objective | File inti |
|---|---|---|
| Pretraining | Token teks, prediksi token berikutnya | `scripts/pretrain_base.py`, `Transformer.forward` |
| SFT | Percakapan dengan mask jawaban, cross entropy pada jawaban | `chat_template.py`, `sft.py`, `data_loader/sft_dataset.py` |
| Reward | Pasangan chosen/rejected, `-log sigmoid(r_chosen-r_rejected)` | `reward_model.py`, `reward_train.py` |
| DPO | Pasangan preferensi, margin log-prob policy relatif reference | `dpo.py`, `rollout.sequence_logprobs` |
| PPO | Rollout, reward, value/advantage, objective policy dengan clipping | `ppo.py`, `value_head.py`, `scripts/train_ppo.py` |
| GRPO | Grup jawaban per prompt, advantage relatif reward grup | `grpo.py`, `scripts/train_grpo.py`, `rewards/` |
| Evaluasi/inference | Checkpoint, prompt, sampling, penilaian jawaban | `evaluation.py`, `inference.py`, `scripts/chat.py` |

SFT melakukan shift internal: logits posisi t memprediksi token t+1, dan mask juga
digeser. Jangan memberi target yang sudah bergeser seolah masih token asli pada
API tersebut. Demo SFT memberi `xb` sebagai rangkaian utuh beserta mask jawaban.

DPO membandingkan selisih log-prob chosen/rejected policy dengan reference beku.
Reward model tidak wajib dilalui untuk DPO. PPO memakai value model untuk membantu
estimasi advantage; `compute_gae` mengakumulasi temporal-difference error dari akhir
respons, dengan mask mencegah bootstrap melewati terminal.

GRPO memakai mean/std reward dalam grup sebagai baseline. Urutan completion harus
berkelompok menurut prompt. Demo memakai empat reward untuk satu prompt; ukuran
grup satu tidak cocok dengan standar deviasi sampel pada implementasi ini.
Contoh loss di notebook hanya memperlihatkan objective; rollout, scoring, KL,
checkpoint, dan orkestrasi training lengkap berada di skrip tiap tahap.

## Debugging yang dapat dicoba

1. Jalankan laboratorium sampai sel forward. Periksa `xb.shape`, `xb.dtype`,
   `xb.min()`, `xb.max()`, `logits.shape`, dan `loss.item()`.
2. Di sel baru, coba `model(torch.full_like(xb, len(vocabulary)))`. ID tersebut
   berada di luar vocabulary dan sengaja menyebabkan error embedding.
3. Jalankan `%debug`; gunakan `up`/`down` untuk memilih frame dan `p idx.shape`
   ketika berada di frame model. Ketik `q` untuk keluar, lalu hapus sel error.
4. Tambahkan `breakpoint()` sebelum `loss.backward()` jika ingin berhenti sebelum
   kegagalan. Gunakan `n`, `s`, `c`, dan `p nama_variabel`.
5. Untuk masalah NaN pada backward, ubah `DEBUG_ANOMALY=True`. Mode ini lebih lambat,
   sehingga sebaiknya digunakan pada reproduksi kecil.

Untuk pipeline baseline, setelah config dan data siap:

```python
%run -d scripts/pretrain_base.py --config configs/colab/pretrain.json
```

Untuk SET, salin argumen dari sel `command`, lalu ganti pemanggilan subprocess
dengan `%run -d -m experiments.evolutionary.train` diikuti argumen yang sama.
`%debug` terhadap `CalledProcessError` hanya memeriksa proses induk, bukan frame
model dalam subprocess. `%run -d` memberi debugger di proses kernel.

| Gejala | Periksa dahulu |
|---|---|
| Hanya `Writing ...` | Masih di bagian setup source; lanjut ke laboratorium |
| Tahap dilewati | Nilai `RUN_...`, kesiapan data/checkpoint |
| `NameError` | Sel sebelumnya belum dijalankan atau kernel baru saja di-reset |
| ID embedding di luar rentang | Tokenizer dan `vocab_size` harus cocok |
| Shape checkpoint berbeda | Dimensi config dan jenis dense/sparse harus sama |
| Backward graph dipakai dua kali | Hitung forward baru sebelum backward ulang |
| GPU kehabisan memori | Reproduksi di CPU dengan dimensi/batch kecil dahulu |
| Edit file tidak mengubah perilaku | Restart kernel karena modul lama masih ter-cache |
| Loss turun, jawaban tetap buruk | Demo menghafal batch kecil; belum melatih bahasa secara memadai |

## Pola desain dan alternatif

`nn.Module` menyusun model secara komposisional, sehingga MLP bisa diganti tanpa
mengubah forward Transformer. `convert_model` melakukan mutasi model yang diberikan;
itulah alasan demo memakai `copy.deepcopy` sebelum konversi. Optimizer dibuat
sesudah konversi agar memegang parameter model akhir.

Sel executable yang pendek mempermudah observasi dan breakpoint. Sebaliknya,
skrip CLI memberi konfigurasi dan eksekusi berulang yang lebih mudah diotomasi.
Notebook mempertahankan keduanya: laboratorium untuk belajar, pipeline opsional
untuk pekerjaan data nyata. Source model tidak diduplikasi menjadi implementasi
matematika baru; demonstrasi memakai class dan objective repo yang sama.

## Referensi belajar terpilih

- [PyTorch: Fundamentals of Autograd](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html):
  pelajari graph, gradien, dan hubungan backward dengan optimizer; tersedia materi
  video dalam seri tutorial resminya.
- [IPython: built-in magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html):
  cari `writefile`, `run`, `debug`, dan `pdb` untuk perilaku sel serta debugger.
- [Paper SET, Nature Communications](https://www.nature.com/articles/s41467-018-04316-3):
  konteks koneksi sparse adaptif dan evolusi topologi selama pembelajaran.
- [Panduan attention repo](../docs/foundations/attention.md),
  [optimisasi](../docs/foundations/optimization.md), dan
  [objective](../docs/foundations/objectives.md): lanjutkan konsep dasar setelah demo.
- [Panduan SFT](../docs/03_sft.md), [DPO](../docs/05_dpo.md),
  [PPO](../docs/06_ppo.md), [GRPO](../docs/07_grpo.md): detail tahap lanjutan.

## Latihan lanjutan

1. Ubah satu karakter input dan amati baris attention serta prediksi berikutnya.
2. Ubah active fraction top-k menjadi 1, reset model, dan periksa kedekatan logits
   terhadap dense sebelum training. Jelaskan mengapa semua neuron mengembalikan jalur dense.
3. Ubah sparsity pruning menjadi 0; hasil sebelum training seharusnya cocok dengan dense.
4. Pada SET, coba fraction rewiring 0 dan periksa bahwa mask tidak berubah.
5. Pada MoE, periksa distribusi penugasan expert, bukan hanya total loss.
6. Pisahkan teks latihan dan validasi sebelum menilai kualitas prediksi.

## Verifikasi dan batas pengujian

`python3 notebook/test_notebooks.py` mengeksekusi source tertanam dan semua sel
default selain `%pip` di lima proses IPython terpisah. Path runtime dialihkan ke
direktori sementara. Tes memeriksa keluaran tiap sel, pembaruan bobot, sepuluh langkah
training, panjang generasi, import dari source tertanam, dan traceback yang tersimpan
untuk debugger. Pengujian ini tidak menjalankan hosted Colab, GPU, pemasangan paket,
mount Drive, download dataset, atau training penuh.

Antislop-code PASS: kode laboratorium tidak memakai komentar dekoratif atau narasi
tiap baris; penjelasan pembelajaran ditempatkan di Markdown. Perubahan executable
dilakukan untuk memenuhi permintaan perbaikan notebook, bukan sebagai audit komentar.
Source model tertanam tetap diperiksa identik dengan file repo oleh generator.
Aturan desain UI pada antislop tidak berlaku pada perubahan notebook ini.
