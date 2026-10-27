# Tokenisasi dan data

Model menerima bilangan bulat. `encode("A")` menghasilkan byte UTF-8 `65`; `encode("é")` menghasilkan dua bilangan karena huruf itu memakai dua byte dalam UTF-8. ID `256` disisihkan sebagai penanda akhir cerita. `decode` menyusun kembali byte menjadi teks dan berhenti ketika melihat penanda tersebut.

Kode ada di `small_llm/tokenizer.py`. Kosakata tetap 257 ID membuat ukuran embedding kecil dan menghindari tahap pelatihan tokenizer. Harganya adalah konteks 256 token yang sering hanya memuat sebagian paragraf. Jika nanti kita memakai BPE, tokenizer baru harus dilatih hanya pada data train dan dipakai identik oleh semua varian.

`small_llm/data.py` membaca satu cerita per berkas `.txt`. Nama berkas diacak memakai seed konfigurasi, lalu beberapa berkas ditahan untuk validasi. Pemisahan dilakukan sebelum tokenisasi digabung menjadi satu aliran. Potongan dari cerita yang sama tidak bisa muncul di kedua kelompok.

Untuk satu jendela panjang empat token, misalnya `[12, 13, 14, 15, 16]`, batch membuat:

```text
input  = [12, 13, 14, 15]
target = [13, 14, 15, 16]
```

Logit pada posisi pertama dipakai untuk menebak `13`, bukan `12`. `sample_batch` memilih titik awal dengan generator acak tersendiri. Seed generator itu sama pada tiap varian, sehingga urutan jendela training sama ketika eksperimen dimulai ulang.

Penanda akhir cerita berada di aliran token seperti token lain. Sebuah jendela dapat melintasi dua cerita, tetapi model melihat penanda pemisahnya. Untuk eksperimen dengan perhatian yang benar-benar terpisah per cerita, kita perlu sampler dan mask tambahan.
