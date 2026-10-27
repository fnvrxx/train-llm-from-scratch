# Training dan evaluasi

`small_llm/train.py` menjalankan lima operasi berulang: ambil batch, hitung logits dan loss, jalankan `backward`, potong gradien bila terlalu besar, lalu panggil `optimizer.step`. Loss token memakai cross-entropy. Jika target sebenarnya mendapat probabilitas rendah, kontribusinya pada loss lebih besar.

`grad_accum` membagi satu langkah optimizer menjadi beberapa microbatch. Setiap loss dibagi dengan jumlah microbatch sebelum `backward`, sehingga gradien yang terkumpul berada pada skala rata-rata yang benar. Satu `optimizer.step` terjadi setelah semua microbatch selesai. Ini menjaga batch tensor kecil di memori tanpa mengubah jumlah token efektif per langkah.

Evaluasi memakai berkas validasi yang tidak dilatih. Titik awal jendela validasi dipilih dengan seed tetap setiap kali evaluasi, sehingga perubahan loss antarlangkah mengukur model pada jendela yang sama. `valid_loss` di `comparison.csv` adalah metrik utama, tetapi korpus bawaan terlalu kecil untuk menyimpulkan kualitas menulis cerita.

Baris langkah `0` menyimpan loss validasi sebelum bobot diperbarui. Bandingkan dengan loss setelah langkah terakhir untuk melihat apakah training mengubah prediksi pada jendela validasi tersebut.

Setiap varian menyimpan `model.pt`, `metrics.jsonl`, dan `summary.json`. Checkpoint memuat arsitektur, pilihan sparsity, dan bobot untuk inferensi. Ia belum menyimpan optimizer serta posisi generator acak, sehingga menjalankan ulang trainer akan memulai training baru. `small_llm/generate.py` membangun arsitektur dari metadata checkpoint sebelum memuat bobotnya.

`train_tokens_per_second` menghitung token training dibagi waktu langkah training; waktu evaluasi dan generasi tidak masuk. `peak_process_rss_mb` mencakup seluruh proses Python. Perbandingan menjalankan varian dalam proses terpisah agar puncak memori satu varian tidak terbawa ke varian berikutnya.
