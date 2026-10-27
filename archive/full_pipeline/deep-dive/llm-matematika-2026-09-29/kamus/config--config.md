# config/config.py

[Indeks kamus](README.md) · [Kode sumber](../../../config/config.py)

## Fungsi dan kelas

Berkas ini berisi konfigurasi, ekspor, atau penanda paket; tidak mendefinisikan fungsi/kelas.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `VOCAB_SIZE` · 6 | Jumlah ID kandidat token, yaitu lebar keluaran lm_head. | `50304` |
| `CONTEXT_LENGTH` · 7 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `512` |
| `N_EMBED` · 8 | Banyak koordinat vektor representasi satu token. | `2048` |
| `N_HEAD` · 9 | Jumlah attention head dalam setiap blok. | `16` |
| `N_BLOCKS` · 10 | Jumlah blok Transformer yang disusun berurutan. | `64` |
| `TRAIN_PATH` · 13 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `'data/train/pile_train.h5'` |
| `DEV_PATH` · 14 | Path dataset tahap terkait; beberapa path evaluasi ditulis tetap dalam skrip. | `'data/val/pile_dev.h5'` |
| `T_BATCH_SIZE` · 17 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `32` |
| `T_CONTEXT_LENGTH` · 18 | Kapasitas/panjang konteks; konteks model dan panjang batch training dapat berbeda. | `16` |
| `T_TRAIN_STEPS` · 19 | Indeks/jumlah step training atau evaluasi, sesuai argumen dan range pada fungsi. | `200000` |
| `T_EVAL_STEPS` · 20 | Interval step/iterasi antar evaluasi. | `1000` |
| `T_EVAL_ITERS` · 21 | Jumlah atau batas batch untuk estimasi metrik evaluasi. | `250` |
| `T_LR_DECAY_STEP` · 22 | Ambang step sesudahnya trainer awal memakai learning rate decayed. | `50000` |
| `T_LR` · 23 | Learning rate, ukuran langkah pembaruan parameter. | `0.0005` |
| `T_LR_DECAYED` · 24 | Learning rate minimum/akhir atau nilai sesudah ambang decay trainer awal. | `5e-05` |
| `T_OUT_PATH` · 25 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `'models/transformer_B.pt'` |
| `T_CHECKPOINT_STEPS` · 26 | Interval save; nol menonaktifkan save periodik pada trainer awal. | `0` |
| `T_KEEP_LAST_CHECKPOINTS` · 27 | Banyak checkpoint terbaru yang dipertahankan; nol menyimpan semua. | `3` |
| `T_CHECKPOINT_DIR` · 28 | Direktori penyimpanan checkpoint. | `None` |
| `USE_AMP` · 32 | Sakelar mixed precision pada trainer awal. | `False` |
| `AMP_DTYPE` · 33 | Tipe atau nama presisi autocast, biasanya bf16/fp16; None berarti tanpa autocast. | `'bf16'` |
| `USE_GRADIENT_CHECKPOINTING` · 34 | Sakelar menghitung ulang aktivasi blok saat backward untuk mengurangi memori. | `False` |
| `GRAD_ACCUM_STEPS` · 35 | Jumlah micro-batch per optimizer step jika loop trainer mengimplementasikannya. | `1` |
| `REPORT_MEMORY_BUDGET` · 36 | Sakelar pelaporan perkiraan kebutuhan memori. | `False` |
| `DEVICE` · 39 | Lokasi komputasi, misalnya cpu atau cuda. | `'cuda' if torch.cuda.is_available() else 'cpu'` |
| `default_config` · 42 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `{'vocab_size': VOCAB_SIZE, 'context_length': CONTEXT_LENGTH, 'n_embed': N_EMBED, 'n_head': N_HEAD, 'n_blocks': N_BLOCKS, 'train_path': TRAIN_PATH, 'dev_path': DEV_PATH, 't_batch_size': T_BATCH_SIZE, 't_context_length': T_CONTEXT_LENGTH, 't_train_steps': T_TRAIN_STEPS, 't_eval_steps': T_EVAL_STEPS, 't_eval_iters': T_EVAL_ITERS, 't_lr_decay_step': T_ ...` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `torch` | `torch` | 3 |
