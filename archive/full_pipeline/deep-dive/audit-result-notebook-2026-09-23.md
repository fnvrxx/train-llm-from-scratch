# AntiVibe: analisis hasil Colab dan penyebab kegagalan notebook

Tanggal: 2026-09-23. Cakupan: satu notebook hasil di `result/`, kelima notebook di `notebook/`, generator notebook, dan skrip yang terkait. Pemeriksaan dilakukan dengan membaca JSON, keluaran yang tersimpan, dan kode sumber. Tidak ada sel notebook, training, instalasi paket, atau unduhan dataset yang dijalankan. Perubahan hanya berupa dokumen ini.

## Kesimpulan berdasarkan bukti

[`result/01_baseline_llm_scratch.ipynb`](../result/01_baseline_llm_scratch.ipynb) menyimpan dua kegagalan: persiapan SFT dan pretraining. Keduanya hanya merekam `CalledProcessError`, tanpa stderr atau traceback dari skrip anak. Karena itu, pesan exception yang pertama kali menghentikan kedua skrip belum tersedia.

Analisis kode menemukan kandidat kuat yang sebelumnya belum dibahas: **notebook mengatur jalur import hanya pada kernel, tetapi menjalankan skrip baseline melalui interpreter Python baru tanpa menyiapkan jalur import proyek untuk proses tersebut**. Ini dapat menggagalkan import `src` pada persiapan SFT dan `config` pada pretraining. Temuan kode ini nyata; keterkaitannya dengan dua kegagalan yang tersimpan masih perlu dikonfirmasi dengan stderr proses anak.

Ada pula bukti bahwa kedua HDF5 pretraining belum tersedia saat sel pengecekan data dijalankan. Namun, pengecekan itu mendahului sel persiapan pretraining. Hasil persiapan tersebut tidak tercatat dalam output, sehingga dokumen ini tidak menyimpulkan bahwa HDF5 pasti masih tidak ada ketika training dicoba.

## 1. Apa yang benar-benar tersimpan di `result/`

Folder `result/` berisi satu file, yaitu `01_baseline_llm_scratch.ipynb`. Tidak ada hasil eksekusi pruned, top-k, MoE, atau SET di folder tersebut.

File baseline memiliki 127 sel, terdiri dari 118 sel kode dan 9 sel Markdown. Sebanyak 108 sel kode mempunyai `execution_count`; 10 lainnya bernilai `null`. Ada 100 sel `%%writefile` dengan pesan `Writing ...`.

Nomor sel dalam laporan ini dihitung mulai 1, termasuk Markdown. Nomor ini berbeda dengan nomor eksekusi `In[...]`. ID sel disertakan agar mudah dicocokkan.

| Sel / ID | Bukti pada hasil tersimpan | Maknanya |
|---|---|---|
| 4 / `cell-003` | `Project: /content/train-llm-from-scratch | Artifacts: /ephemeral` | Setup dijalankan |
| 6 sampai 105 | Pesan `Writing ...` | Seratus file sumber/config ditulis ke runtime |
| 107 / `cell-106` | `RESTORE_FROM_DRIVE = False`, tanpa output | Blok pemulihan Drive pada sumber tersimpan dinonaktifkan |
| 109 / `cell-108` | `Config: configs/colab | Device: cuda` | Sel profil menghasilkan konfigurasi dan mendeteksi CUDA |
| 110 / `cell-109` | `Belum ada: /ephemeral/data/pile_train.h5` dan `pile_dev.h5` | Kedua file belum ada pada saat pengecekan ini |
| 112 / `cell-111` | `RUN_PREP_PRETRAIN = True`, execution count 107, `outputs: []` | Sel mempunyai catatan eksekusi, tetapi hasil proses anak tidak tersimpan |
| 113 / `cell-112` | Execution count 108, `CalledProcessError`, exit status 1 | Pemanggilan persiapan SFT gagal |
| 117 / `cell-116` | Execution count 109, `CalledProcessError`, exit status 1 | Pemanggilan pretraining gagal |
| 114, 115, 118 sampai 122, 124, 125, 127 | Execution count `null`, output kosong | Tidak ada catatan eksekusi untuk persiapan preference/RL, post-training, evaluasi, chat, dan backup |

Notebook menyimpan sumber sel saat file disimpan, bukan riwayat setiap pengeditan. Nilai `True` yang terlihat sekarang tidak membuktikan nilai itu sudah `True` ketika execution count terakhir diperoleh. Sebaliknya, `outputs: []` tidak membuktikan proses gagal atau tidak bekerja. Kita juga tidak boleh menyebut sel dengan `execution_count=null` pasti tidak pernah dijalankan di sesi lain. [Format notebook resmi Jupyter](https://nbformat.readthedocs.io/en/latest/format_description.html) menjelaskan struktur sumber, execution count, dan output yang terpisah.

Perbandingan sumber dengan `notebook/01_baseline.ipynb` menunjukkan perubahan tujuh sakelar menjadi `True`: persiapan pretraining, SFT, preference, RL, serta training pretrain, SFT, dan reward. Ada juga perubahan whitespace pada sel yang menulis `scripts/data_download.py`. Setup, skrip SFT, skrip pretrain, dan cara memanggil subprocess tetap sama. Dengan demikian, persoalan integrasi yang dibahas di bawah sudah terdapat pada notebook asal.

## 2. Prioritas tinggi: jalur import kernel tidak diwariskan ke Python baru

Lokasi: [`notebook/build_notebooks.py`](../notebook/build_notebooks.py), setup baris 45 dan pemanggilan skrip baseline baris 167 serta 198. Kode yang sama tertanam pada notebook asal dan hasil Colab.

Setup menjalankan:

```python
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
```

Langkah pertama mengganti direktori kerja. Langkah kedua membuat kernel bisa menemukan paket proyek. Setelah itu, sel SFT memulai interpreter lain:

```python
subprocess.run(
    [sys.executable, 'scripts/prepare_sft_data.py', ...],
    check=True,
    cwd=PROJECT_ROOT,
)
```

Interpreter baru menyusun `sys.path` sendiri. Untuk eksekusi berdasarkan nama file, direktori awal pencarian adalah direktori skrip, yaitu `/content/train-llm-from-scratch/scripts`. Perubahan `sys.path` dalam memori kernel tidak ikut dipindahkan. `cwd` menentukan direktori kerja, tetapi tidak dengan sendirinya menjamin akar proyek tersedia untuk import dalam mode pemanggilan ini. [Dokumentasi inisialisasi `sys.path`](https://docs.python.org/3/library/sys_path_init.html).

Sementara itu:

- [`scripts/prepare_sft_data.py`](../scripts/prepare_sft_data.py), baris 26, mengimpor `src.post_training.chat_template`.
- [`scripts/pretrain_base.py`](../scripts/pretrain_base.py), baris 26, mengimpor `config.post_training_config`.
- Paket `src`, `config`, dan `data_loader` berada di akar proyek, di luar direktori `scripts`.
- Sel instalasi notebook hanya memasang dependency; tidak memasang proyek dengan `pip install -e .` dan setup tidak menetapkan `PYTHONPATH`.

Pada lingkungan yang belum menyediakan akar proyek melalui mekanisme lain, gejala yang diperkirakan adalah `ModuleNotFoundError: No module named 'src'` atau `'config'`. Itu perkiraan dari kode, **bukan kutipan error yang ditemukan pada file hasil**. Lingkungan Colab yang tersimpan tidak menyertakan nilai `PYTHONPATH` atau daftar paket terpasang, sehingga kemungkinan akar proyek sudah disediakan oleh lingkungan tetap terbuka.

Petunjuk tambahan ada dalam contoh penggunaan kedua skrip: keduanya menuliskan `PYTHONPATH=.`. [`scripts/run_posttraining.sh`](../scripts/run_posttraining.sh) juga mengekspor variabel tersebut. Notebook belum membawa pengaturan ini ke proses anak.

Opsi perbaikan yang dapat dipilih kemudian:

1. Jalankan modul dari akar proyek, misalnya `python -m scripts.pretrain_base`, dengan `cwd` yang benar. Python menambahkan direktori kerja pada pencarian modul dalam mode ini.
2. Teruskan akar proyek lewat `PYTHONPATH` pada environment subprocess, sambil mempertahankan environment yang sudah ada.
3. Pasang proyek sebagai paket setelah seluruh file sumber ditulis. Pilihan ini perlu memperhatikan kelengkapan berkas packaging.

Tidak ada opsi tersebut yang diterapkan dalam audit ini. Untuk memastikan kandidat import tanpa training, pemeriksaan selanjutnya dapat memakai pemanggilan skrip asli dengan `--help` di Colab dan menangkap stderr. Import tingkat modul terjadi sebelum parser menampilkan bantuan.

## 3. Prioritas tinggi: notebook tidak meneruskan output subprocess secara eksplisit

Sel 113 dan 117 hanya memiliki output bertipe `error`. Tidak ada output bertipe `stream` yang membawa error dari `prepare_sft_data.py` atau `pretrain_base.py`. Sel 112 juga tidak menyimpan progres unduhan/tokenisasi.

Semua pemanggilan terkait memakai `subprocess.run(..., check=True)` tanpa pipe, capture, atau pencetakan ulang stdout/stderr. Secara default, proses anak mewarisi stream sistem operasi. Itu tidak sama dengan jaminan output akan diteruskan ke tampilan kernel dan disimpan dalam `.ipynb`. Bukti di file ini menunjukkan pesan proses anak memang tidak ikut tersimpan. Kita tidak dapat menentukan apakah pesan sempat muncul di konsol lain atau hilang saat ekspor.

Ketika proses anak gagal, `check=True` membuat kernel melempar `CalledProcessError`. Tanpa penangkapan stream, exception itu tidak membawa rincian stderr yang dibutuhkan untuk diagnosis. [Dokumentasi Python `subprocess`](https://docs.python.org/3/library/subprocess.html).

Konsekuensinya, saran sebelumnya untuk mencari 10 sampai 30 baris sebelum traceback tidak cukup untuk file hasil ini: baris tersebut memang tidak ada di JSON yang disimpan.

Perbaikan yang disarankan untuk tahap berikutnya adalah satu fungsi pemanggil bersama yang membaca stdout dan stderr subprocess, mencetaknya kembali ke output sel, menyimpan log, lalu memeriksa exit code. Untuk training panjang, `Popen` dengan pipe yang dibaca bertahap lebih sesuai daripada menampung seluruh output dalam memori sampai selesai. Untuk diagnosis singkat, `run(capture_output=True, text=True)` dapat dipakai bila hasilnya dicetak sebelum error diteruskan. Opsi `-u` membantu mengurangi buffering Python anak, tetapi tetap membutuhkan mekanisme penerusan output.

## 4. Prasyarat data belum diperiksa ulang sebelum training

Sel 110 membuktikan data belum ada **sebelum** sel 112. Karena sel 112 memiliki catatan eksekusi dan tidak menyimpan exception, belum ada dasar untuk mengatakan persiapan pretraining pasti gagal. Sumber sel juga dapat diedit setelah dijalankan. Status final HDF5 tidak tersedia dalam ekspor ini.

Notebook tidak menjalankan pemeriksaan pascapersiapan yang melaporkan keberadaan file, jumlah token, dan kesiapan batch sebelum model dibuat. Skrip pretraining membuat model serta optimizer terlebih dahulu, kemudian loader membuka HDF5 ketika batch pertama diminta. Kekurangan input bisa baru muncul setelah alokasi model.

Pemeriksaan yang disarankan sebelum training:

- Pastikan file train ada, dapat dibuka, memiliki `tokens` satu dimensi, dan berisi cukup token untuk batch yang dipilih.
- Periksa data dev secara terpisah; evaluasi tidak boleh diam-diam hilang karena file tidak tersedia.
- Cetak konfigurasi efektif, bukan hanya nama profil dan status CUDA.
- Laporkan kembali hasil persiapan setelah selesai, termasuk lokasi serta jumlah token.

Jika import gagal lebih dahulu, HDF5 yang sebenarnya sudah lengkap pun tidak akan menyelesaikan kegagalan tersebut. Urutan diagnosis harus mengikuti traceback pertama.

## 5. Bug tambahan: batas 200.000 token dapat terlewati jauh

Lokasi: [`scripts/prepare_pretrain_data.py`](../scripts/prepare_pretrain_data.py), baris 34, 91, dan 100 sampai 115; salinannya berada pada sel 43 baseline. Notebook mengirim `--max_tokens 200000`.

Variabel `total` hanya menghitung token yang sudah ditulis saat `flush()`. Token baru ditampung di `buf`. Flush normal baru terjadi ketika buffer mencapai `WRITE_CHUNK = 8_000_000`, sedangkan kondisi berhenti memeriksa `total >= max_tokens`.

Contoh konseptual: sesudah 200.000 token terkumpul di buffer, `total` bisa masih 0. Kondisi berhenti belum terpenuhi dan pembacaan berlanjut. Untuk korpus yang cukup besar, proses dapat baru berhenti sesudah flush pertama yang mencapai atau melewati 8 juta token. Tidak ada pemotongan buffer ke sisa anggaran token.

Ini temuan pasti dari alur kode, tetapi tidak terbukti sebagai penyebab dua error yang tersimpan. Ia menjelaskan mengapa percobaan yang terlihat kecil dapat memproses jauh lebih banyak data. Selain itu, seluruh shard terpilih diunduh sebelum tokenisasi dimulai; membetulkan batas token saja tidak membatasi ukuran unduhan.

Perbaikan nantinya perlu menghitung token tertulis dan token tertunda, membatasi penambahan sesuai sisa anggaran, serta menentukan aturan saat pemotongan bertepatan dengan batas dokumen.

## 6. Mengapa tidak ada hasil untuk setiap file kode?

Seratus sel `%%writefile` sudah memiliki output `Writing ...`. Magic ini menyimpan isi sel ke file, bukan menjalankan fungsi atau blok `main()` di dalamnya. Jadi `Writing scripts/pretrain_base.py` membuktikan file skrip dibuat, bukan bahwa model selesai dilatih. [Dokumentasi IPython `%%writefile`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-writefile).

Sepuluh sel kode lain tidak mempunyai catatan eksekusi dalam file hasil, termasuk SFT training dan reward meski sumber tersimpan bernilai `True`. Tidak ada hasil training sukses yang dapat disimpulkan untuk tahap-tahap itu.

Checkpoint dan metrik juga merupakan artefak terpisah dari notebook:

| Artefak | Lokasi default pada runtime Colab | Bukti ketersediaan dalam `result/` |
|---|---|---|
| Kode proyek | `/content/train-llm-from-scratch/` | Ada pesan penulisan file |
| HDF5 pretraining | `/ephemeral/data/pile_train.h5`, `pile_dev.h5` | Tidak ada pada pengecekan awal; status setelah persiapan belum diketahui |
| HDF5 SFT | `/ephemeral/data/sft_packed.h5`, `sft_dev_packed.h5` | Persiapan SFT gagal; tidak ada inventaris file hasil |
| Metrik baseline | `/ephemeral/logs/<stage>_<timestamp>.jsonl` | File JSONL tidak disertakan dalam folder hasil |
| Checkpoint baseline | `/ephemeral/ckpts/base_pretrained.pt`, lalu checkpoint tahap lain | Tidak ada bukti penyimpanan berhasil dalam ekspor |
| Backup Drive | `MyDrive/train-llm-artifacts/data/` dan `ckpts/` | Sel backup bernilai `False` dan execution count `null` |

Ketiadaan artefak di folder `result/` lokal tidak membuktikan ketiadaannya pada mesin Colab. Ekspor `.ipynb` tidak menyertakan otomatis semua file runtime.

## 7. Dampak terhadap kelima notebook

Semua notebook di `notebook/` adalah template dengan execution count kosong. Hanya baseline memiliki pasangan hasil eksekusi dalam `result/`, sehingga kesimpulan untuk empat eksperimen lain berasal dari pemeriksaan kode.

| Notebook | Temuan yang relevan |
|---|---|
| `01_baseline.ipynb` | Pemanggilan `python scripts/...` berisiko kehilangan import proyek; keluaran proses anak tidak diteruskan eksplisit; pemeriksaan data dilakukan sebelum persiapan; batas token bermasalah berasal dari skrip tertanam |
| `02_pruned.ipynb` | Fine-tuning berjalan langsung dalam kernel, sehingga memakai `sys.path` kernel. Sanity check menggunakan `-m`, tetapi output subprocess tetap belum diteruskan eksplisit. Membutuhkan dua HDF5 yang sudah tersedia |
| `03_topk.ipynb` | Pola eksekusi dan kebutuhan data sama dengan pruned; tidak ada hasil eksekusi untuk membuktikan berhasil/gagal |
| `04_moe.ipynb` | Pola eksekusi sama; `train_loss` mencakup loss tambahan routing, sedangkan `val_loss` hanya loss token. Keduanya tidak identik secara definisi |
| `05_evolutionary_set.ipynb` | Training memakai `python -m experiments.evolutionary.train` dari akar proyek, sehingga jalur pemanggilannya berbeda dari baseline. Risiko tidak tersimpannya stdout/stderr tetap ada; notebook memeriksa keberadaan train, sedangkan validasi opsional |

Keterbatasan penyimpanan yang terpisah dari error saat ini:

- Backup baseline hanya menyalin file langsung di `data` dan `ckpts`, tidak menyertakan `logs` atau direktori data mentah.
- Empat notebook eksperimen mempunyai pemulihan Drive, tetapi belum mempunyai sel backup akhir.
- Training HDF5 pruned/top-k/MoE mencetak metrik di sel tanpa menyimpan JSONL; checkpoint menyimpan model, bukan optimizer. Parameter seperti fraksi aktif top-k juga tidak dicatat sebagai konfigurasi eksperimen tersendiri.

Ini perlu diperhatikan ketika membandingkan atau melanjutkan eksperimen, tetapi bukan bukti bahwa sparse atau SET mengalami error yang sama dengan baseline.

## 8. Mengapa validasi pembuat notebook tidak menangkap masalah ini?

Fungsi `validate()` pada [`notebook/build_notebooks.py`](../notebook/build_notebooks.py), mulai baris 415, membandingkan isi sumber tertanam dan memakai `ast.parse` untuk memeriksa sintaks sel Python. Pemeriksaan ini tidak memulai subprocess, menguji pencarian paket, mengecek stream notebook, atau membaca HDF5.

Karena itu, laporan sebelumnya bahwa notebook lolos pemeriksaan sintaks memang terbatas pada struktur dan sintaks. Itu belum merupakan bukti pipeline berhasil di Colab. Kesalahan integrasi proses dapat muncul meskipun setiap file Python valid secara sintaks.

## 9. Urutan verifikasi dan perbaikan berikutnya

Semua langkah berikut adalah rekomendasi, belum dieksekusi atau diterapkan.

1. Sediakan penangkapan stderr/stdout agar kegagalan berikutnya menyimpan traceback asli. Pertahankan file hasil saat ini sebagai bukti.
2. Di Colab, periksa pemanggilan skrip asli dengan `--help`. Jika muncul `ModuleNotFoundError` untuk paket proyek, benahi jalur import subprocess dan ulangi pemeriksaan ringan tersebut.
3. Gunakan `--print-config` pada skrip pretrain untuk memeriksa konfigurasi efektif setelah import berhasil; opsi ini keluar sebelum pembuatan model.
4. Inventarisasi HDF5 yang sudah ada sebelum mengulang unduhan. Periksa struktur dan jumlah token di file itu.
5. Benahi batas token sebelum menjalankan persiapan data baru yang dimaksudkan kecil.
6. Jalankan pretraining profil `smoke` di Colab, lalu pastikan pesan akhir, file checkpoint, dan metrik tersimpan. Persiapan SFT perlu diperiksa terpisah karena memakai dataset lain.
7. Baru lanjutkan tahap yang memerlukan checkpoint sebelumnya, lalu simpan checkpoint serta log eksperimen.

## Referensi dan kode terkait

- [Python: inisialisasi jalur import](https://docs.python.org/3/library/sys_path_init.html), untuk memahami perbedaan skrip dan modul.
- [Python: subprocess](https://docs.python.org/3/library/subprocess.html), untuk environment, capture output, pipe, dan exit code.
- [IPython: writefile](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-writefile), untuk memahami penulisan file dari sel.
- [Jupyter: format notebook](https://nbformat.readthedocs.io/en/latest/format_description.html), untuk menafsirkan execution count serta output tersimpan.
- [`scripts/prepare_sft_data.py`](../scripts/prepare_sft_data.py), import proyek terjadi sebelum pemuatan dataset SFT.
- [`scripts/pretrain_base.py`](../scripts/pretrain_base.py), import, konfigurasi, alokasi model, pembacaan batch, lalu checkpoint.
- [`data_loader/data_loader.py`](../data_loader/data_loader.py), format HDF5 pretraining dan pengambilan batch.
- [`src/post_training/logging_utils.py`](../src/post_training/logging_utils.py), lokasi dan format metrik baseline.

Laporan ini memperbarui diagnosis sebelumnya dengan bukti dari notebook hasil. Penyebab langsung dua exception tetap belum dapat dinyatakan pasti karena stderr proses anak tidak tersedia; kandidat import, kekurangan pencatatan output, dan bug batas token kini mempunyai dasar kode yang spesifik.
