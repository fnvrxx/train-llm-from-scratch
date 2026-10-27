# Cara menjalankan seluruh program di Google Colab

Panduan ini memakai **hosted runtime Google Colab**, bukan *local runtime* di laptop. Lima notebook di folder [`notebook/`](notebook/) sudah memuat kode proyek dalam sel terpisah, satu file sumber per sel. Anda tidak perlu mengunduh repo dari GitHub saat menjalankannya. Sel `%%writefile` hanya menulis file kode ke mesin Colab; training baru dimulai setelah Anda mengubah sakelar tahap terkait menjadi `True`.

| Notebook | Kegunaan |
|---|---|
| [`01_baseline.ipynb`](notebook/01_baseline.ipynb) | Persiapan data, pretraining, seluruh post-training, evaluasi, chat |
| [`02_pruned.ipynb`](notebook/02_pruned.ipynb) | Pruning MLP statis, lalu fine-tuning opsional |
| [`03_topk.ipynb`](notebook/03_topk.ipynb) | Aktivasi neuron top-k per token, lalu fine-tuning opsional |
| [`04_moe.ipynb`](notebook/04_moe.ipynb) | Mixture of Experts, lalu fine-tuning opsional |
| [`05_evolutionary_set.ipynb`](notebook/05_evolutionary_set.ipynb) | Sparse Evolutionary Training (SET), termasuk rewiring dan resume |

Notebook sparse dan SET berdiri sendiri dari sisi kode. Namun, untuk membandingkan model pada data yang sama, gunakan HDF5 dan checkpoint baseline yang dibuat oleh notebook pertama. Semua sakelar `RUN_...` awalnya `False` agar unduhan dan training tidak berjalan tanpa sengaja.

Notebook baseline juga menulis file UI dan skrip pendukung ke runtime Colab agar kode proyek tersedia lengkap. Panduan ini menjalankan pipeline training, evaluasi, dan chat; ia tidak menyalakan server Streamlit, karena Colab tidak otomatis menyediakan alamat publik untuk UI tersebut.

## 1. Siapkan Colab

1. Buka [Google Colab](https://colab.research.google.com/), pilih **File → Upload notebook**, lalu unggah `notebook/01_baseline.ipynb`.
2. Pilih **Runtime → Change runtime type → GPU**, lalu pastikan Anda memakai runtime yang di-host Colab. Ketersediaan GPU dan batas sesi bergantung pada akun serta kapasitas Colab.
3. Jalankan sel dari atas sampai ke sel **Profil Colab dan data**: instalasi paket, pembuatan direktori, dan seluruh sel `%%writefile`. Setiap sel `%%writefile` mewakili satu file; jalankan semuanya berurutan agar import antarmodul tersedia.
4. Saat pertama kali, biarkan `RESTORE_FROM_DRIVE = False`. Jika data atau checkpoint sudah disimpan di Drive, ubah ke `True` lalu jalankan sel pemulihan itu. Colab akan meminta izin memasang Google Drive.
5. Pada sel profil, biarkan `PROFILE = 'smoke'` untuk percobaan awal. Pilih `'full'` hanya jika GPU, memori, waktu, dan penyimpanan mencukupi. Profil `smoke` memakai model kecil untuk memeriksa alur, bukan hasil penelitian model penuh.

Lokasi kerja kode adalah `/content/train-llm-from-scratch`. Data, checkpoint, dan log berada di `/ephemeral/data`, `/ephemeral/ckpts`, dan `/ephemeral/logs`. Jangan menganggap file di runtime Colab tersimpan permanen: [FAQ Google Colab](https://research.google.com/colaboratory/intl/id/faq.html) menjelaskan bahwa mesin virtual dapat berakhir dan file lokalnya hilang.

## 2. Siapkan data baseline

Anda dapat memakai sel persiapan data di notebook atau memulihkan file yang sudah ada dari Google Drive. Jika memakai Drive, letakkan file pada `MyDrive/train-llm-artifacts/data/`; sel `RESTORE_FROM_DRIVE` menyalinnya ke `/ephemeral/data/`.

Jalankan satu sel persiapan pada satu waktu. Ubah sakelar di sel itu menjadi `True`, jalankan, periksa hasilnya, lalu lanjut ke data berikutnya.

| Sakelar | Dipakai untuk | File hasil di `/ephemeral/data/` |
|---|---|---|
| `RUN_PREP_PRETRAIN` | Pretraining | `pile_train.h5`, `pile_dev.h5` |
| `RUN_PREP_SFT` | Supervised fine-tuning (SFT) | `sft_packed.h5`, `sft_dev_packed.h5` |
| `RUN_PREP_PREFERENCES` | Reward model dan DPO | `preferences.jsonl`, `preferences_test.jsonl` |
| `RUN_PREP_RL` | PPO dan GRPO | `rl_prompts_train.jsonl`, `rl_prompts_test.jsonl`, `arithmetic_prompts.jsonl` |

Sel persiapan pretraining menggunakan batas `200000` token untuk file HDF5 contoh, **tetapi shard Pile diunduh lebih dahulu sebelum batas itu berlaku**. Pastikan ruang disk, kuota jaringan, dan waktu Colab cukup. Jika tidak, siapkan HDF5 dengan format yang dipakai repo di tempat lain, lalu unggah ke Drive. Sel setelah profil dapat menunjukkan apakah dua HDF5 pretraining sudah tersedia.

Dataset SFT, preferensi, dan RL juga dapat mengunduh data dari Hugging Face. Jika ada dataset yang memerlukan persetujuan akses atau autentikasi, sel persiapannya akan berhenti sampai akses tersebut Anda penuhi.

## 3. Jalankan baseline, satu tahap per sel

Setelah data tahap tersebut tersedia, cari selnya di bagian **Jalankan tahap yang dipilih**, ubah hanya sakelar tahap itu menjadi `True`, lalu jalankan sel tersebut. Tidak perlu menjalankan ulang semua sel dari awal setiap kali mengubah satu sakelar. Simpan checkpoint ke Drive setelah tahap yang memakan waktu lama.

| Urutan | Sakelar | Masukan utama | Checkpoint hasil |
|---|---|---|---|
| 1. Pretraining | `RUN_PRETRAIN` | `pile_train.h5`, `pile_dev.h5` | `/ephemeral/ckpts/base_pretrained.pt` |
| 2. SFT | `RUN_SFT` | `base_pretrained.pt`, `sft_packed.h5`, `sft_dev_packed.h5` | `/ephemeral/ckpts/sft.pt` |
| 3. Reward model | `RUN_REWARD` | `sft.pt`, data preferensi | `/ephemeral/ckpts/reward.pt` |
| 4. DPO | `RUN_DPO` | `sft.pt`, data preferensi | `/ephemeral/ckpts/dpo.pt` |
| 5. PPO | `RUN_PPO` | `sft.pt`, prompt RL; `reward.pt` bila memakai reward model | `/ephemeral/ckpts/ppo.pt` |
| 6. GRPO | `RUN_GRPO` | `sft.pt`, prompt RL dan kurikulum aritmetika | `/ephemeral/ckpts/grpo.pt` |

Reward model dan DPO adalah cabang yang sama-sama dimulai dari SFT, bukan tahapan yang saling meneruskan bobot. Konfigurasi PPO bawaan memakai `reward_source: "verifier"`; jika Anda mengubahnya untuk memakai reward model, pastikan `reward.pt` sudah dibuat. Semua tahap mengambil pengaturan dari `configs/colab/<tahap>.json` yang dihasilkan sel profil. Baca pengaturan jumlah langkah, batch, serta path di sana sebelum memulai sesi yang panjang.

Setelah checkpoint tersedia:

1. Di sel evaluasi, pilih `EVAL_STAGE = 'sft'`, `'dpo'`, `'ppo'`, atau `'grpo'`, lalu set `RUN_EVALUATION = True`. Untuk checkpoint pretraining, nama filenya `base_pretrained.pt`, jadi gunakan `EVAL_STAGE = 'base_pretrained'`. Evaluasi GSM8K dapat mengunduh dataset dan membutuhkan waktu.
2. Di sel chat, pilih `CHAT_STAGE` dengan nama checkpoint yang sama, ubah `PROMPT` bila perlu, lalu set `RUN_CHAT = True`. Untuk model pretraining mentah, antarmuka chat belum tentu memberi respons percakapan yang bermakna.
3. Untuk menyimpan hasil, pada sel terakhir ubah `MOUNT_DRIVE = True`, jalankan, lalu tunggu pesan `Saved to Drive`. Sel ini menyalin file langsung di `/ephemeral/data` dan `/ephemeral/ckpts` ke `MyDrive/train-llm-artifacts/`. Jalankan lagi setelah menghasilkan checkpoint baru.

Jika runtime terputus, buka notebook baru, jalankan kembali sel setup dan semua `%%writefile`, lalu gunakan `RESTORE_FROM_DRIVE = True` sebelum melanjutkan tahap berikutnya. File yang belum disalin ke Drive tidak dapat dipulihkan hanya dari notebook.

## 4. Jalankan tiga eksperimen sparse secara terpisah

Ulangi langkah berikut untuk `02_pruned.ipynb`, `03_topk.ipynb`, dan `04_moe.ipynb`. Ketiganya tidak otomatis menjalankan seluruh pipeline post-training baseline; bagian trainingnya mengukur *loss* token pada HDF5.

1. Buka satu notebook di Colab dengan GPU. Jalankan sel instalasi, setup, dan seluruh `%%writefile` dari atas ke bawah.
2. Jika HDF5 dan checkpoint ada di Drive, ubah `RESTORE_FROM_DRIVE = True`, lalu jalankan sel pemulihan. Pastikan `/ephemeral/data/pile_train.h5` dan `/ephemeral/data/pile_dev.h5` tersedia.
3. Sel `RUN_SANITY` bersifat opsional. Bila `True`, ia melakukan pemeriksaan sintetis kecil atas eksperimen yang dipilih, bukan training pada data asli.
4. Pada sel pengaturan fine-tuning, pilih `MODEL_CONFIG_PATH = Path('configs/smoke/base.json')` untuk checkpoint profil `smoke`, atau `Path('configs/base.json')` untuk checkpoint profil `full`. Dimensi model **harus sama** dengan checkpoint yang akan dimuat.
5. Jika ingin memulai dari bobot baseline, isi `DENSE_CHECKPOINT = ARTIFACT_ROOT / 'ckpts' / 'base_pretrained.pt'`. Biarkan `None` jika ingin inisialisasi acak. Isi `TRAIN_PATH`, `VAL_PATH`, `TRAIN_STEPS`, `BATCH_SIZE`, dan `LEARNING_RATE` sesuai eksperimen Anda. Default notebook adalah 20 langkah, sehingga hanya cocok sebagai pemeriksaan alur awal.
6. Set `RUN_TRAIN = True` pada sel pengaturan, lalu jalankan sel training tepat di bawahnya. Perhatikan `train_loss` dan `val_loss`, lalu simpan checkpoint hasilnya dari `/ephemeral/ckpts/pruned.pt`, `topk.pt`, atau `moe.pt` ke Drive. Notebook sparse belum menyediakan sel backup akhir; salin file melalui panel Files Colab atau gunakan sel penyalinan Drive dari notebook baseline setelah menyesuaikan runtime yang aktif.

Pruning statis memakai sparsity awal `0.5`; top-k mengaktifkan fraksi neuron `0.25` per token; MoE memakai 4 expert dengan top-2 routing. Parameter itu ada di sel training dan dapat diubah sebelum eksperimen. Ketiga checkpoint menyimpan bobot model dan konfigurasi eksperimen, **bukan state optimizer**, sehingga tidak mendukung resume training secara persis. Perubahan sparsity tidak otomatis menjamin percepatan inferensi atau penghematan memori pada implementasi ini.

## 5. Jalankan Sparse Evolutionary Training (SET)

1. Buka `05_evolutionary_set.ipynb` di Colab dengan GPU. Jalankan instalasi, setup, seluruh `%%writefile`, dan pemulihan Drive bila diperlukan.
2. Opsional: ubah `RUN_CHECKS = True` untuk menjalankan tes kepadatan mask, perubahan koneksi, state optimizer, dan pemuatan checkpoint **di Colab**. Tes ini tidak dijalankan di laptop.
3. Pada sel pengaturan, pilih `MODEL_CONFIG_PATH` yang cocok dengan ukuran baseline. Siapkan `TRAIN_PATH`; `VAL_PATH` opsional, tetapi disarankan agar *validation loss* dapat dibandingkan.
4. Untuk memulai SET dari model padat, isi `DENSE_CHECKPOINT` dengan `base_pretrained.pt` yang cocok dan biarkan `RESUME_CHECKPOINT = None`. Untuk memulai dari bobot acak, biarkan keduanya `None`.
5. Tentukan `TRAIN_STEPS`, lalu ubah `RUN_TRAIN = True` dan jalankan sel training. Defaultnya density `0.2`, rewiring fraction `0.3`, rewiring setiap 100 langkah, dan batch size 2. Hasilnya ditulis ke `/ephemeral/ckpts/evolutionary.pt`.
6. Simpan `evolutionary.pt` ke Drive sebelum sesi Colab berakhir. Untuk melanjutkan pada sesi lain, pulihkan file tersebut, isi `RESUME_CHECKPOINT = ARTIFACT_ROOT / 'ckpts' / 'evolutionary.pt'`, set `DENSE_CHECKPOINT = None`, lalu jalankan training lagi dengan target `TRAIN_STEPS` yang lebih tinggi dari langkah yang sudah tercapai. Jangan isi `DENSE_CHECKPOINT` dan `RESUME_CHECKPOINT` sekaligus.

Berbeda dari tiga eksperimen sparse sebelumnya, checkpoint SET mencakup bobot, mask koneksi, optimizer, dan langkah training untuk resume. Urutan batch setelah resume dapat berubah karena iterator data dibuat ulang, jadi hasilnya tidak dijamin identik bit demi bit dengan satu sesi tanpa putus. Notebook SET juga belum menyediakan sel backup akhir; unduh checkpoint atau salin ke Drive dari Colab.

## 6. Bandingkan hasil secara adil

Gunakan profil/dimensi model yang sama, HDF5 train dan validasi yang sama, checkpoint awal yang sama, seed yang sama, dan jumlah langkah atau anggaran komputasi yang sebanding. Default tiga notebook sparse adalah 20 langkah, sedangkan SET 1000 langkah, sehingga **jangan membandingkan loss default keduanya sebagai hasil akhir**. Catat juga kepadatan, parameter routing, batch size, GPU, waktu training, serta *validation loss*. Notebook ini belum mengklaim bahwa satu pendekatan lebih akurat atau lebih cepat.

## Bila terjadi masalah

- **File HDF5 atau checkpoint tidak ditemukan:** periksa path di sel pengaturan, jalankan persiapan data atau `RESTORE_FROM_DRIVE`, lalu pastikan nama file tepat.
- **Ukuran tensor checkpoint tidak cocok:** jangan campur checkpoint `smoke` dengan `configs/base.json`, atau checkpoint `full` dengan `configs/smoke/base.json`.
- **GPU kehabisan memori:** mulai dari profil `smoke`, kurangi batch size atau panjang konteks sebelum memulai training baru. Mengubah dimensi model setelah checkpoint dibuat akan membuat checkpoint tersebut tidak kompatibel.
- **Runtime Colab terputus:** pulihkan artefak dari Drive. Menjalankan ulang `%%writefile` mengembalikan kode, tetapi tidak mengembalikan bobot yang belum dicadangkan.
- **Checkpoint sparse tidak terbaca oleh `scripts/chat.py`:** skrip chat baseline membuat arsitektur Transformer padat. Model pruned/top-k/MoE harus dibangun dengan converter eksperimen yang sesuai sebelum bobotnya dimuat. Untuk SET, gunakan pemuat checkpoint di `experiments/evolutionary/model.py`.
- **Unduhan data gagal:** periksa akses dataset, jaringan, kuota disk, dan autentikasi di Colab. Persiapan Pile khususnya dapat jauh lebih besar daripada HDF5 hasilnya.

Panduan ini menjelaskan alur berdasarkan kode dan konfigurasi yang tersedia. Training, unduhan dataset, dan tes model tidak dijalankan di laptop saat panduan dibuat.
