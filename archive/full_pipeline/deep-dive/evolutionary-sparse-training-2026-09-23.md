# Deep Dive: Sparse Evolutionary Training pada MLP Transformer

**Dibuat**: 2026-09-23  
**Fase**: eksperimen arsitektur MLP  
**Kode**: `experiments/evolutionary/__init__.py`, `model.py`, `rewire.py`,
`train.py`, `test_evolutionary.py`, `README.md`, dan indeks `experiments/README.md`

## Gambaran

Eksperimen ini mengganti MLP pada setiap blok Transformer dengan dua matriks
bermasker. Hanya koneksi bertanda `True` yang ikut menghasilkan output.
Selama training, sejumlah koneksi aktif yang bobotnya paling dekat nol
dimatikan, lalu jumlah yang sama dihidupkan pada posisi acak yang sebelumnya
tidak aktif. Dengan demikian kepadatan tetap, tetapi topologi berubah.

Ini mengikuti gagasan dasar [Sparse Evolutionary Training (SET)](https://www.nature.com/articles/s41467-018-04316-3)
dari Mocanu dkk.: topologi awal acak, pemangkasan berdasarkan bobot kecil,
dan pertumbuhan kembali secara acak. Ada dua penyesuaian yang disengaja:
rewiring dilakukan tiap sejumlah langkah optimizer karena loader HDF5 tidak
menetapkan akhir epoch, dan bobot yang baru tumbuh dimulai dari nol agar tidak
langsung mengguncang output. Karena `F.linear` tetap menerima matriks padat,
eksperimen ini meneliti proses belajar dan topologi, belum menghemat memori
bobot/optimizer atau menjamin percepatan.

Gunakan eksperimen ini bila ingin membandingkan MLP padat, pruning statis,
dan konektivitas yang berevolusi pada data dan arsitektur Transformer yang sama.

## Alur satu langkah training

1. Loader membaca pasangan token input dan target dari HDF5.
2. MLP menghitung `weight * mask`, sehingga koneksi mati tidak berkontribusi.
3. Loss prediksi token menghasilkan gradien; gradien untuk bobot mati bernilai nol.
4. AdamW memperbarui bobot, lalu `enforce_model_masks` mengembalikan bobot mati ke nol.
5. Pada interval rewiring, bobot aktif terkecil dipangkas. Posisi lama yang mati
   dipilih acak untuk tumbuh, dengan jumlah sama persis.
6. Bobot dan momentum Adam pada posisi yang berubah diatur ke nol. Langkah
   training berikutnya mulai belajar nilai koneksi baru.
7. Model, mask, optimizer, konfigurasi, dan nomor langkah disimpan ke checkpoint.

## Penjelasan kode per file

### `experiments/evolutionary/__init__.py`

Satu docstring menandai folder ini sebagai paket Python. Berkat itu,
`python3 -m experiments.evolutionary.train` dapat mencari modul di dalamnya.
File tersebut tidak menjalankan training saat diimpor.

### `experiments/evolutionary/model.py`

- Baris 10-19, `EvolutionaryMLP.__init__`: menerima MLP padat asli dan memakai
  layer `hidden` serta `proj` yang sudah ada. Ini memungkinkan fine-tuning dari
  checkpoint padat. `density` harus berada di antara nol dan satu. Dua mask
  didaftarkan sebagai *buffer*, supaya ikut pindah device dan masuk ke
  `state_dict`, tetapi bukan parameter yang diperbarui optimizer. Koneksi yang
  tidak terpilih segera dinolkan.
- Baris 21-26, `_random_mask`: menghitung target koneksi aktif, mengambil indeks
  acak tanpa pengulangan, lalu membentuk mask dengan bentuk yang sama seperti
  matriks bobot. Pemilihan acak dengan jumlah tepat memberi kepadatan tetap.
  Ini pendekatan praktis terhadap topologi acak awal pada SET; kepadatan di sini
  diatur langsung, bukan melalui parameter epsilon pada paper.
- Baris 28-30, `forward`: urutannya masih sama seperti MLP lama,
  `linear → ReLU → linear`. Perkalian dengan mask adalah bagian dari grafik
  autograd; jika mask suatu bobot nol, kontribusi dan gradien bobot itu nol.
- Baris 32-39, `enforce_masks` dan `masked_layers`: operasi pertama menahan
  bobot nonaktif tetap nol setelah langkah optimizer; generator kedua memberi
  pasangan `(weight, mask)` kepada algoritme rewiring agar kedua matriks diproses.
- Baris 42-50, `convert_model` dan `evolutionary_mlps`: fungsi pertama mengganti
  `block.mlp` di seluruh blok Transformer; fungsi kedua mengunjungi MLP itu
  ketika training. Attention, embedding, dan head bahasa tidak diubah.
- Baris 53-63, `load_experiment_checkpoint`: membuat ulang Transformer dan
  MLP evolutionary dari metadata checkpoint, memuat bobot **dan mask** secara
  ketat, lalu menyiapkan mode evaluasi. Loader biasa pada pipeline produksi
  tidak mengetahui varian ini, jadi helper khusus ini diperlukan.

### `experiments/evolutionary/rewire.py`

- Baris 8-13, `_clear_optimizer_state`: AdamW menyimpan momentum per posisi
  bobot. Momentum lama pada koneksi yang dipangkas atau baru tumbuh tidak boleh
  terbawa ke topologi baru, maka tensor state yang bentuknya sama dengan bobot
  dinolkan pada posisi tersebut.
- Baris 16-29, awal `rewire_weight`: validasi fraksi dan identifikasi indeks
  `active` serta `dormant` dilakukan **sebelum** mengubah mask. Banyaknya
  koneksi yang ditukar dibatasi oleh jumlah posisi mati yang tersedia.
- Baris 31-42, inti rewiring: `topk(..., largest=False)` mencari bobot aktif
  dengan magnitudo terendah. Posisi pertumbuhan diundi dari himpunan yang
  sebelumnya mati, sehingga setiap peristiwa benar-benar mengubah topologi.
  Kedua kelompok berukuran sama, jadi jumlah koneksi aktif tidak berubah.
  Bobot baru nol dan state optimizer pada semua posisi yang berubah dibersihkan.
- Baris 45-68, fungsi tingkat model: `rewire_model` menerapkan operasi pada
  kedua matriks di setiap blok; `enforce_model_masks` menjaga bobot mati nol;
  `active_connections` menghitung kepadatan aktual untuk log dan pemeriksaan.
  `@torch.no_grad()` penting karena perubahan mask dan bobot ini adalah operasi
  pengelolaan topologi, bukan bagian dari diferensiasi loss.

### `experiments/evolutionary/train.py`

- Baris 18-58, konfigurasi CLI: file JSON memberi dimensi Transformer,
  sementara argumen mengatur data, checkpoint, device, learning rate,
  kepadatan, frekuensi rewiring, evaluasi, dan penyimpanan. Validasi menolak
  nilai yang tidak masuk akal serta mencegah checkpoint padat sumber tertimpa.
- Baris 61-74, `model_dimensions` dan `check_data`: mengambil lima dimensi
  arsitektur dari JSON dan memastikan HDF5 menyediakan setidaknya satu batch
  sekuens penuh. Dimensi harus cocok dengan checkpoint padat yang dimuat.
- Baris 77-93, `autocast` dan `evaluate`: bf16 hanya dipakai jika diminta di
  CUDA. Evaluasi menghitung rerata loss beberapa batch tanpa gradien dan
  mengembalikan model ke mode training.
- Baris 96-108, `save_checkpoint`: menyimpan bobot model **termasuk mask**,
  state AdamW, nomor langkah, dan pengaturan rewiring. Ini memungkinkan resume
  tanpa memilih mask baru secara permanen saat memuat ulang.
- Baris 111-141, awal `main`: seed disetel, data dan konfigurasi diperiksa,
  lalu Transformer dibangun. Checkpoint padat, bila diberikan, dimuat dengan
  `strict=True` sebelum MLP dikonversi. Pada `--resume`, checkpoint eksperimen
  harus cocok arsitektur dan setelan rewiringnya; model dan optimizer dipulihkan.
- Baris 143-162, training: iterator HDF5 memasok batch. Loss dibagi menurut
  `grad-accum`, lalu backward, clipping gradien, langkah AdamW, dan penegakan
  mask dijalankan berurutan. `grad-accum` membantu menambah batch efektif
  tanpa memegang seluruh batch sekaligus di memori.
- Baris 164-175, jadwal: rewiring dilakukan setelah langkah optimizer pada
  kelipatan `rewire-every`, kecuali pada langkah terakhir atau setelah
  `freeze-after` bila batas itu diatur. Log mencatat jumlah koneksi yang
  berubah dan loss; checkpoint disimpan menurut interval atau pada akhir run.
  Tidak ada rewiring sesudah langkah terakhir agar koneksi baru tidak dibiarkan
  tanpa kesempatan belajar.
- Baris 178-179, *entry point*: memanggil `main` hanya ketika modul dijalankan
  sebagai program, bukan ketika diimpor oleh kode lain.

### `experiments/evolutionary/test_evolutionary.py`

- Baris 14-18 membuat Transformer kecil dengan seed tetap untuk pemeriksaan
  yang dapat diulang pada mesin training.
- Baris 20-28 memastikan kepadatan awal tepat, bentuk logits tetap benar,
  backward bekerja, dan bobot nonaktif tidak menerima gradien.
- Baris 30-47 melakukan satu langkah AdamW lalu rewiring. Ia memeriksa jumlah
  koneksi tetap, mask benar-benar berubah, bobot posisi yang berubah nol, dan
  kedua momentum Adam pada posisi itu nol.
- Baris 49-57 menyimpan `state_dict` ke buffer memori dan memuatnya kembali ke
  model dengan mask awal berbeda. Mask hasil pemulihan harus sama dengan mask
  yang disimpan. Tes ini belum dijalankan di laptop lokal sesuai permintaan.

### `experiments/evolutionary/README.md` dan `experiments/README.md`

README folder eksperimen memberikan contoh perintah untuk mesin training,
menjelaskan mulai dari checkpoint padat, mulai dari nol, dan resume. Indeks
`experiments/README.md` menghubungkan pendekatan baru dengan tiga pendekatan
yang dibuat sebelumnya. Dokumentasi sengaja memisahkan hasil fungsional dari
klaim performa yang belum diukur.

## Konsep dan pilihan desain

| Konsep | Mengapa dipakai | Alternatif |
|---|---|---|
| Buffer untuk mask | Mask ikut checkpoint dan device, tanpa menjadi parameter gradien | Menyimpan mask di luar model, tetapi pemuatan lebih rawan tidak lengkap |
| Magnitudo kecil dipangkas | Mendekati aturan SET dan mudah ditelusuri | Pangkas acak atau berdasarkan statistik gradien |
| Pertumbuhan acak | Memberi kesempatan pada jalur baru tanpa menghitung gradien padat kandidat koneksi | RigL memilih pertumbuhan berdasar gradien, dengan biaya dan kompleksitas tambahan |
| Bobot tumbuh dari nol | Perubahan output saat rewiring lebih tenang | Inisialisasi acak seperti variasi SET lain, berpotensi mengubah output segera |
| Mask pada `F.linear` | Cocok dengan kode Transformer yang ada dan mudah diaudit | Kernel sparse khusus diperlukan untuk penghematan komputasi nyata |

Contoh: sebuah matriks dengan 1.000 posisi dan `density=0.2` memiliki 200
koneksi aktif. Jika `rewire-fraction=0.3`, pada suatu interval 60 koneksi aktif
terlemah dilepas dan 60 posisi lama yang mati diaktifkan. Hasilnya tetap 200
koneksi aktif, dengan jalur yang berbeda.

## Sumber belajar

- [Paper SET asli, Mocanu dkk.](https://www.nature.com/articles/s41467-018-04316-3): topologi awal acak, pemangkasan bobot kecil, pertumbuhan acak, dan jumlah koneksi tetap.
- [PyTorch `Module.register_buffer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): alasan mask dapat ikut `state_dict` tanpa menjadi parameter optimizer.
- [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html): bagaimana gradien mengalir melalui perkalian bobot dan mask, serta fungsi `no_grad`.
- [Panduan checkpoint PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): mengapa model dan optimizer sama-sama perlu disimpan untuk melanjutkan training.
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html): optimizer yang dipakai dan alasan state per bobot perlu dibersihkan saat topologi berubah.

## Kode terkait dan langkah belajar

`src/models/mlp.py` adalah MLP padat asal; `src/models/transformer_block.py`
menempatkannya di residual; `src/models/transformer.py` membangun seluruh
blok. `data_loader/data_loader.py` menyediakan pasangan token untuk prediksi
token berikutnya. Bandingkan `experiments/pruned/model.py` untuk melihat
perbedaan paling penting: mask pruning statis tidak pernah tumbuh lagi.

Setelah kode dijalankan pada mesin yang memadai, catat loss validasi, kualitas
generasi, kepadatan, penggunaan memori puncak, dan token per detik. Bandingkan
dengan model padat dan pruning statis menggunakan data, jumlah langkah, seed,
dan ukuran model yang sama. Waspadai resume: bobot dan optimizer dipulihkan,
tetapi posisi iterator data tidak, sehingga kelanjutan tidak identik bit demi
bit. Jangan menafsirkan banyaknya nol pada mask sebagai bukti percepatan
sebelum mengukur kernel dan perangkat yang sebenarnya.
