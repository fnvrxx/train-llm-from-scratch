# Korpus cerita tambahan

Folder ini berisi 60 cerita pendek orisinal untuk percobaan Small LLM: 12 cerita dari `stories_demo` dan 48 cerita tambahan. Setiap berkas `.txt` berisi satu cerita. Kumpulan ini dibuat di repo; tidak diambil dari situs atau buku lain.

Gunakan `configs/stories_expanded_m1.json` untuk eksperimen terpisah. Konfigurasi tersebut menyisihkan 25% cerita, yaitu 15 berkas, sebagai validasi dengan seed tetap. Pemisahan dilakukan per cerita, bukan per potongan teks.

Enam puluh cerita masih merupakan korpus sangat kecil untuk model bahasa. Perubahan `valid_loss` dapat menunjukkan kemajuan prediksi byte pada cerita yang disisihkan, tetapi belum cukup untuk menyimpulkan model mampu menulis cerita baru yang runtut. Periksa juga contoh generasi secara manual. Nilai loss dari korpus ini tidak langsung sebanding dengan hasil `stories_demo` karena data validasinya berbeda.

