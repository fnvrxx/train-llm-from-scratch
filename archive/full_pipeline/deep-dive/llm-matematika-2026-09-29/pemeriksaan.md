# Catatan pemeriksaan panduan

Tanggal pemeriksaan: 29 September 2026.

Panduan menggunakan kode lokal sebagai sumber utama dan situs penulis serta makalah asli sebagai referensi konsep. Kamus dibentuk dengan membaca struktur sintaks Python (AST), kemudian dilengkapi arti nama dan catatan per lingkup. Nomor baris adalah snapshot, bukan tautan yang otomatis mengikuti perubahan kode berikutnya.

## Cakupan

PASS: 51 berkas Python pipeline tercatat pada indeks kamus. Pencarian mencakup `src/models/` yang tidak muncul dalam daftar berkas default karena aturan ignore.

PASS: 208 deklarasi fungsi/kelas dalam kamus cocok dengan hitungan independen AST pada sumber.

PASS: 1.600 nama dalam lingkup mempunyai penjelasan dan asal nilai. Nama mencakup parameter, field dataclass, variabel, atribut yang ditulis, buffer terdaftar, serta pengikat loop/lambda. Impor dicatat terpisah. Pengikat comprehension dikelompokkan pada fungsi pembacanya; Python sebenarnya memberi comprehension lingkup lokal tersendiri.

PASS: Tautan relatif, pasangan code fence, pasangan display-math delimiter, dan ketiadaan placeholder penjelasan diperiksa secara programatis pada seluruh Markdown panduan.

## Contoh yang dihitung dengan kode CPU

PASS: Transformer ilustratif dengan `B=2`, `T=5`, `V=100`, `C=32`, empat head, dan dua blok menghasilkan logits `(2,5,100)`.

PASS: Rumus jumlah parameter pada bab 7 cocok dengan jumlah elemen parameter model ilustratif, yaitu 31.940. Ini ukuran contoh pemeriksaan, bukan klaim ukuran semua konfigurasi proyek.

PASS: Mengganti token posisi terakhir tidak mengubah logits pada posisi sebelumnya dalam contoh causal attention.

PASS: Contoh attention dua posisi menghasilkan keluaran sekitar 10 dan 18,80797.

PASS: Contoh loss SFT dengan tiga NLL `(2,1,3)` dan mask target `(0,1,1)` menghasilkan loss 2.

PASS: Contoh DPO pada bab 5 menghasilkan loss sekitar 0,64439666.

PASS: Contoh GAE menghasilkan advantage `(0.8,0.6)` dan returns `(1,1)`.

PASS: Reward kelompok `(0,0,1,1)` dengan epsilon nol menghasilkan advantage sekitar `(-0.866,-0.866,0.866,0.866)`; reward kelompok seragam menghasilkan advantage nol.

PASS: Keterbatasan bentuk `Transformer.forward_embedding` untuk banyak blok dikonfirmasi dengan pemanggilan model kecil yang menghasilkan ketidakcocokan dimensi. Catatan tersebut dipisahkan dari jalur normal `forward_hidden`.

Pemeriksaan tidak menjalankan training besar, tidak mengunduh dataset, dan tidak mengukur kualitas model terlatih. Pemeriksaan delimiter Markdown tidak sama dengan pemeriksaan visual pada semua aplikasi pembaca; pembaca perlu mendukung LaTeX Markdown dan Mermaid untuk menampilkan rumus/diagram.

## Pemeriksaan AntiVibe dan antislop-copywriting

PASS, AntiVibe: bab menghubungkan operasi kode dengan fungsi matematis, alasan desain, alternatif, batas penggunaan, sumber primer, serta latihan beserta jawaban.

PASS, R-02: panduan memakai bahasa Indonesia dengan istilah teknis yang dijelaskan, dan pemeriksaan karakter tidak menemukan em dash atau en dash.

PASS, R-16 dan R-36: pembahasan menyebut operasi konkret seperti menggeser target, mengambil log-prob, dan menghitung gradien. Tidak ada klaim kemampuan model tanpa pengukuran.

PASS, R-17 dan R-38: angka dibedakan menjadi nilai konfigurasi lokal, hasil perhitungan ilustratif, atau konstanta kode. Tidak ada benchmark training yang direkayasa.

PASS, C-1 dan C-3: bab disusun menurut aliran komputasi; tabel digunakan untuk memetakan simbol dan membandingkan objective. Kamus dipisahkan agar bacaan utama tetap dapat diikuti bertahap.

PASS, C-5: klaim perilaku implementasi dihubungkan ke berkas sumber; contoh hitung inti diperiksa dengan kode CPU.

PASS, pemeriksaan nada: tidak ada testimoni, promosi, emoji dekoratif, atau penutup chatbot dalam bab. Istilah yang sama dipertahankan agar tidak perlu mempelajari sinonim baru di setiap bagian.

Aturan Delivery Gate yang khusus antarmuka visual, aset, kontras, tombol, tema, breakpoint, dan animasi tidak berlaku pada tugas dokumentasi Markdown ini. Tidak ada antarmuka aplikasi yang dibangun.
