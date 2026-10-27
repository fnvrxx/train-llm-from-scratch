# Membaca eksperimen sparsity

Sparsity berarti sebagian bobot, neuron, atau expert tidak dipakai untuk suatu perhitungan. `small_llm/sparse.py` mengganti MLP pada setiap blok Transformer; embedding dan attention tetap sama.

`PrunedMLP` memilih bobot dengan magnitudo paling kecil saat model dibuat dan memasang mask tetap. `SETMLP` membuat mask acak lalu, setiap `rewire_every` langkah, membuang koneksi aktif terlemah dan membuka koneksi baru. Keduanya masih menyimpan tensor bobot padat. Karena operasi menggunakan perkalian bobot dengan mask, penghematan RAM dan percepatan tidak otomatis terjadi.

`TopKMLP` menghitung aktivasi seluruh neuron, memilih `k` nilai terbesar per token, lalu hanya mengambil kolom proyeksi kedua yang diperlukan. Ini mengurangi kerja pada proyeksi kedua, tetapi operasi pemilihan dan pengambilan kolom mempunyai biaya sendiri. `MoEMLP` membagi lebar MLP menjadi beberapa expert; router memilih `top_k` expert per token. Loss tambahan mendorong pemakaian expert lebih merata.

Perbandingan menyamakan data, seed, langkah, optimizer, dan batch. Jumlah parameter total, koneksi aktif, dan operasi aktual berbeda. Baca `mlp_usage` untuk melihat apa yang dipilih tiap metode. `active_mlp_connections` hanya berlaku untuk mask bobot `pruned` dan `set`; top-k memilih neuron, sedangkan MoE memilih expert. Bandingkan `valid_loss` bersama `train_tokens_per_second`, `parameters`, dan `peak_process_rss_mb`. Model dengan loss kecil setelah waktu training yang jauh lebih lama belum tentu pilihan yang tepat untuk batas perangkat Anda.

Urutan belajar yang disarankan: pahami dense terlebih dahulu, lalu pruning untuk melihat mask tetap, top-k untuk pilihan per token, SET untuk perubahan mask, dan MoE untuk routing. Jalankan tes kecil setelah mengubah satu mekanisme agar kesalahan bentuk tensor atau gradien terlihat sebelum training panjang.
