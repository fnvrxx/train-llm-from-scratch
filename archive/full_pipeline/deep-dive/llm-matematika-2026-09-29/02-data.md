# 2. Teks, token, dan target

[Daftar bab](../llm-matematika-2026-09-29.md) · [Berikutnya: Transformer](03-transformer.md)

Kode utama: [persiapan pretraining](../../scripts/prepare_pretrain_data.py), [loader](../../data_loader/data_loader.py), [format chat](../../src/post_training/chat_template.py), dan [persiapan SFT](../../scripts/prepare_sft_data.py).

## Token bukan selalu kata

Tokenizer memetakan potongan teks ke ID. Satu kata bisa menjadi beberapa token; spasi dan tanda baca juga memengaruhi pembagian. Proyek memakai `tiktoken.get_encoding("r50k_base")`, sebuah tokenizer BPE yang sudah tersedia. Model bahasa dilatih dari awal, tetapi tokenizernya tidak dilatih dari awal di sini. [Repositori tiktoken](https://github.com/openai/tiktoken).

ID adalah indeks, bukan besaran numerik yang menyatakan makna. ID 100 tidak berarti dua kali makna ID 50. Hubungan antar token dipelajari lewat embedding.

Dalam `prepare_pretrain_data.py`, `shard_urls` menyusun alamat pecahan dataset, `download` menyimpan berkas, `iter_texts` membaca teks dari JSONL terkompresi, dan `tokenize_to_h5` mengubahnya menjadi deretan ID dalam HDF5. Fungsi lokal `flush` memindahkan token dari buffer RAM ke disk.

| Variabel | Arti pada proses ini |
|---|---|
| `zst_paths`, `zp` | Daftar berkas terkompresi dan berkas yang sedang dibaca |
| `enc` | Objek tokenizer |
| `docs`, `txt` | Kumpulan teks sebelum tokenisasi dan satu teks |
| `ids` | ID token satu dokumen |
| `buf` | Token yang menunggu ditulis ke disk |
| `ENC_BATCH` | Batas banyak dokumen per pemanggilan batch tokenizer, 1024 dalam kode |
| `WRITE_CHUNK` | Ukuran chunk HDF5 dan ambang flush, 8.000.000 token |
| `dset` | Dataset `tokens` di dalam HDF5 |
| `total` | Banyak token yang sudah ditulis |
| `EOT_ID` | ID 50256 sebagai penanda akhir teks |

Membaca secara streaming menekan kebutuhan RAM dibanding memuat seluruh corpus sekaligus. Alternatifnya adalah menyimpan semua token di RAM, yang lebih sederhana untuk data latihan kecil. `max_tokens` diperiksa setelah flush, sehingga batas ini bisa terlampaui; ia bukan pemotongan presisi pada jumlah token yang diminta.

## Satu sampel mengandung input dan jawaban sekaligus

`get_batch_iterator` membaca `context_length + 1` token. Ilustrasi dengan ID buatan:

```python
random_samples = [[10, 20, 30, 40, 50]]
xb = [[10, 20, 30, 40]]
yb = [[20, 30, 40, 50]]
```

Pada posisi pertama, model melihat 10 dan menargetkan 20. Pada posisi kedua, model melihat konteks 10,20 dan menargetkan 30. Model menghitung semua posisi dalam satu forward pass; causal mask mencegah posisi awal membaca masa depan.

| Variabel loader | Arti |
|---|---|
| `data_path` | Lokasi HDF5 |
| `hdf5_file`, `dataset` | Berkas terbuka dan array token di dalamnya |
| `dataset_size` | Jumlah token dalam array |
| `n_examples` | Banyak jendela penuh: `(dataset_size - 1) // context_length` |
| `example_idxs` | Nomor jendela yang diacak |
| `counter` | Posisi pada daftar jendela untuk batch berikutnya |
| `epochs` | Hitungan putaran loader |
| `random_indices` | Offset awal setiap jendela, yaitu nomor jendela dikali panjang konteks |
| `random_samples` | Batch jendela, masing-masing sepanjang $T+1$ |
| `xb`, `yb` | Input dan target bergeser, masing-masing $(B,T)$ |

Perhatikan batas lingkup: `Transformer.forward` tidak menggeser target lagi. Ia menganggap pemanggil sudah menyediakan target sejajar seperti `yb`.

## Chat dan loss mask

Chat memakai header teks `<|user|>`, `<|assistant|>`, dan `<|system|>`. Header ini terpecah menjadi token biasa. Hanya EOT yang diperlakukan sebagai ID khusus dalam template ini.

`encode_chat(messages)` mengembalikan `ids` dan `mask` dengan panjang sama. `header_ids` adalah token header, `content_ids` token isi pesan, `role` jenis pembicara, dan `is_completion` bernilai benar untuk pesan assistant.

| Bagian percakapan | Nilai mask |
|---|---|
| Header role | 0 |
| Isi user/system dan EOT-nya | 0 |
| Isi assistant dan EOT-nya | 1 |

Mask nol membuat target tidak menyumbang langsung pada loss. Prompt tetap menjadi input attention dan dapat memengaruhi gradien prediksi jawaban. Jadi mask nol tidak berarti prompt dihapus dari model.

`render_chat` membuat teks untuk tampilan. `encode_prompt` membuat ID yang berakhir pada header assistant agar model mulai menjawab. `decode` mengubah ID kembali ke teks sambil membuang EOT dan ID di luar rentang token biasa. `get_tokenizer` memakai cache agar tokenizer tidak dibuat berulang kali.

`add_generation_prompt=True` menambahkan header dengan mask nol. Bila riwayat `messages` sudah berisi jawaban assistant, mask jawaban lama tetap satu; klaim bahwa seluruh mask selalu nol hanya berlaku jika pesan masuk belum memuat assistant.

## Tiga bentuk dataset post-training

| Tahap | Isi record | Keluaran loader |
|---|---|---|
| SFT | Contoh chat yang telah dipadatkan | `tokens`, `loss_mask`, `epoch` |
| RM/DPO | `prompt`, `chosen`, `rejected` | ID, mask jawaban, dan panjang nyata untuk kedua jawaban |
| PPO/GRPO | `prompt`, `gold` | Daftar prompt dan kunci numerik untuk verifier |

`pack_examples` menyambung daftar `ids` dan `mask` menjadi `flat_ids` dan `flat_mask`, menghitung `n_rows`, lalu membentuk `ids_arr` dan `mask_arr` berukuran `(n_rows, context_length)`. Sisa yang tidak memenuhi satu baris dibuang. EOT menjadi pemisah teks, tetapi bukan attention mask pemisah contoh: token setelah EOT masih dapat memperhatikan contoh sebelumnya dalam baris yang sama.

Loader preferensi memakai right padding agar `chosen` dan `rejected` punya lebar yang sama. `ch_ids/ch_mask/ch_len` milik jawaban chosen; `rj_ids/rj_mask/rj_len` milik rejected. Mask padding nol; reward model mengambil token terakhir berdasarkan panjang nyata, bukan ujung padding.

Pemotongan `ids[:max_len]` bisa menghabiskan seluruh ruang pada prompt. Jika tidak ada token jawaban tersisa, pasangan itu tidak memberi sinyal DPO yang dimaksud. Saat memeriksa dataset, lihat juga jumlah token jawaban setelah pemotongan.

## Latihan

Ilustrasi: ada 21 token dan `context_length=4`. Berapa `n_examples`? Mengapa bukan `21 // 5`?

Jawaban: $(21-1)//4=5$. Jendela dimulai pada offset 0,4,8,12,16. Jendela bersebelahan berbagi satu token batas, karena token akhir jendela sebelumnya menjadi awal jendela berikutnya.
