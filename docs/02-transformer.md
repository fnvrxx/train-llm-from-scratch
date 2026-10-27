# Transformer kecil

`StoryTransformer` di `small_llm/model.py` menerima tensor token berbentuk `(batch, panjang)`. Embedding mengubah tiap ID menjadi vektor `n_embed`; embedding posisi memberi informasi urutan. Empat blok pada profil M1 mengolah vektor itu sebelum `lm_head` menghasilkan 257 logit per posisi.

Di satu blok, attention menghitung query, key, dan value. Untuk tiap pasangan posisi, skor dihitung dari `query × key / sqrt(head_size)`. Mask segitiga mengubah skor posisi masa depan menjadi negatif tak hingga sebelum softmax. Akibatnya logit pada posisi ke-3 tidak dapat membaca token ke-4. Tes `test_attention_cannot_read_future_token` memeriksa sifat ini dengan mengganti token terakhir dan membandingkan logit sebelumnya.

MLP memproyeksikan vektor dari `C` ke `4C`, menerapkan ReLU, lalu mengembalikannya ke `C`. Residual menambahkan hasil attention dan MLP ke masukan masing-masing. LayerNorm sebelum kedua operasi membantu menjaga skala aktivasi selama training.

Model memakai attention yang ditulis dengan operasi tensor biasa agar bentuk `(batch, head, panjang, panjang)` tampak di kode. Cara ini mudah diperiksa, tetapi biaya memorinya naik kira-kira sebanding dengan kuadrat panjang konteks. Model juga menghitung ulang konteks saat menghasilkan setiap token. Optimisasi seperti attention terfusi dan KV cache dapat dipelajari setelah alur dasar benar.
