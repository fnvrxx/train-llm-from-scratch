# Deep Dive: lokasi output notebook dan error pretraining

**Tanggal:** 2026-09-23  
**Fase:** diagnosis tanpa perubahan kode  
**File yang dibaca:** `notebook/build_notebooks.py`, `scripts/pretrain_base.py`, `data_loader/data_loader.py`, `src/post_training/logging_utils.py`, `config/loader.py`, `configs/smoke/pretrain.json`, `configs/pretrain.json`

## Gambaran umum

Menjalankan semua sel tidak berarti setiap sel menghasilkan metrik. Notebook berisi sel dokumentasi, instalasi, penulisan file sumber, pengaturan, dan tahap training yang awalnya dinonaktifkan. Hasil training tidak disimpan di bawah setiap sel sebagai file notebook: kode menggunakan direktori `/ephemeral` pada mesin Colab.

Traceback `CalledProcessError ... exit status 1` pada `RUN_PRETRAIN` belum menunjukkan akar masalah. Ia hanya menyatakan bahwa `scripts/pretrain_base.py` gagal. Keluaran proses anak sebelum traceback itu diperlukan untuk mengetahui penyebabnya. `check=True` sengaja membuat notebook berhenti ketika proses training gagal, sesuai [dokumentasi Python](https://docs.python.org/3/library/subprocess.html).

## Di mana output tiap jenis sel?

| Jenis sel | Yang terjadi | Lokasi output |
|---|---|---|
| Markdown | Menampilkan penjelasan | Isi sel notebook |
| Instalasi `%pip` | Memasang paket | Pesan singkat di bawah sel; tidak menghasilkan model |
| Setup | Membuat `/content/train-llm-from-scratch` dan `/ephemeral/{data,ckpts,logs}` | Pesan `Project: ... | Artifacts: ...` di bawah sel |
| `%%writefile` | Menulis satu file sumber/config ke runtime | File di `/content/train-llm-from-scratch/...`; biasanya hanya pesan `Writing ...` atau `Overwriting ...` |
| Sel dengan `RUN_... = False` | Melewati isi blok `if` | Biasanya kosong, karena tidak menjalankan tahap itu |
| Persiapan data yang diaktifkan | Mengunduh dan mengonversi data | Pesan progres di bawah sel; file di `/ephemeral/data/` |
| Pretraining yang berhasil | Mencetak langkah/loss, mencatat metrik, menyimpan checkpoint | Pesan di bawah sel; metrik `/ephemeral/logs/pretrain_*.jsonl`; checkpoint `/ephemeral/ckpts/base_pretrained.pt` |
| Post-training yang berhasil | Mencatat metrik dan checkpoint per tahap | `/ephemeral/logs/` dan `/ephemeral/ckpts/` |
| Backup Drive | Menyalin file data dan checkpoint bila `MOUNT_DRIVE=True` | `MyDrive/train-llm-artifacts/data/` dan `ckpts/` |

Sel backup Drive di notebook baseline menyalin `data` dan `ckpts`, tetapi **tidak menyalin `logs`**. Bila ingin menyimpan metrik JSONL, file log perlu disalin secara terpisah. File di `/ephemeral` juga dapat hilang saat runtime Colab berakhir; [FAQ Colab](https://research.google.com/colaboratory/intl/en-GB/faq.html) menerangkan bahwa mesin virtual memiliki masa hidup terbatas.

## Walkthrough `RUN_PRETRAIN`

### Sel notebook

Generator `notebook/build_notebooks.py` membuat sel pretrain dengan `RUN_PRETRAIN = False` sebagai awal. Jika diubah menjadi `True`, sel menjalankan:

```python
subprocess.run(
    [sys.executable, "scripts/pretrain_base.py",
     "--config", str(config_dir / "pretrain.json")],
    check=True,
    cwd=PROJECT_ROOT,
)
```

Notebook Anda mengirim `configs/colab/pretrain.json` melalui `/usr/bin/python3`. Baris `subprocess.run` adalah pemanggil. Ia tidak menjelaskan mengapa proses training keluar dengan status 1.

### `config/loader.py`

Skrip training membaca `configs/colab/pretrain.json` dan base config di direktori yang sama. Dengan profil `smoke`, base config itu memakai dimensi model kecil dan stage config meminta 20 langkah, batch size 4, evaluasi pada interval yang lebih besar daripada 20, serta `save_every` yang juga lebih besar daripada 20. Checkpoint akhir tetap disimpan setelah seluruh loop selesai. Bila proses berhenti sebelumnya, file checkpoint mungkin belum ada.

### `scripts/pretrain_base.py`

1. Baris 57 sampai 63 memuat konfigurasi, memilih perangkat, dan memasang seed.
2. Baris 65 sampai 90 membuat model serta optimizer, mencetak jumlah parameter dan effective batch, lalu membuka logger. Logger membuat file bernama `pretrain_<timestamp>.jsonl` di `/ephemeral/logs/`.
3. Baris 93 menyiapkan iterator HDF5 train. Iterator Python ini bersifat *lazy*: file HDF5 baru dibuka saat batch pertama diminta pada baris 107.
4. Baris 98 sampai 125 menjalankan training. Pada langkah 0 dan setiap 20 langkah, proses mencetak `step ... | loss ...` dan menulis metrik JSONL.
5. Baris 128 sampai 139 mengevaluasi/menyimpan secara berkala jika interval tercapai. Baris 141 sampai 144 menyimpan checkpoint akhir hanya bila loop selesai.

Karena itu, bila error terjadi sebelum baris logger, tidak ada log; bila error terjadi sesudah logger tetapi sebelum langkah pertama selesai, file log bisa ada namun kosong; bila error terjadi sebelum akhir 20 langkah pada profil `smoke`, checkpoint akhir belum tentu tersedia.

### `data_loader/data_loader.py`

Baris 20 membuka `cfg.train_path`, yaitu `/ephemeral/data/pile_train.h5` pada konfigurasi repo. Baris 23 mengharapkan dataset HDF5 bernama `tokens`. Baris 29 menghitung jumlah urutan yang bisa dibuat dari panjang token dan `context_length`. HDF5 yang belum ada, rusak, atau tidak memiliki `tokens` adalah kemungkinan kegagalan, tetapi **belum dapat dinyatakan sebagai penyebab error Anda** tanpa traceback dari proses anak.

## Pemeriksaan aman di Colab

Jalankan sel baru berikut **di hosted runtime Colab yang sama**, bila runtime masih aktif. Sel ini hanya membaca nama, ukuran file, dan konfigurasi; ia tidak melatih model atau mengunduh data.

```python
from pathlib import Path
import json

for name in (
    "/ephemeral/data/pile_train.h5",
    "/ephemeral/data/pile_dev.h5",
    "/ephemeral/data/sft_packed.h5",
    "/ephemeral/ckpts/base_pretrained.pt",
):
    path = Path(name)
    print(name, "ada" if path.is_file() else "tidak ada",
          f"({path.stat().st_size:,} byte)" if path.is_file() else "")

for directory in (Path("/ephemeral/logs"), Path("/ephemeral/ckpts")):
    print(directory, sorted(p.name for p in directory.iterdir()) if directory.exists() else "tidak ada")

stage = Path("/content/train-llm-from-scratch/configs/colab/pretrain.json")
print("pretrain config:", json.loads(stage.read_text()) if stage.exists() else "tidak ada")
```

Pemeriksaan ini menunjukkan keberadaan file, bukan memastikan isinya valid. Jika `pile_train.h5` tidak ada, pretraining tidak bisa membaca input train. Jika ada, pesan error dari proses anak masih diperlukan untuk membedakan masalah format HDF5, GPU/memori, dependency, atau bagian training lain.

## Apa yang perlu dikirim untuk diagnosis pasti

Pada output sel pretrain yang gagal, cari baris **di atas** traceback `CalledProcessError`. Kirim pesan terakhir yang tercetak dan traceback yang menyebut `pretrain_base.py`, `data_loader.py`, `torch`, atau `h5py`. Jika output skrip tidak tampak, kirim hasil sel pemeriksaan aman di atas. Jangan kirim kredensial atau token akses.

| Bukti yang terlihat | Kesimpulan yang boleh diambil |
|---|---|
| `pile_train.h5` tidak ada | Input train belum tersedia di runtime tersebut; persiapan/pemulihan data perlu diperiksa |
| File data ada tetapi tidak ada `Model parameters:` | Kegagalan terjadi sebelum logger/training dimulai; lihat traceback proses anak |
| `Model parameters:` muncul lalu gagal | Model berhasil dibuat; cek pesan sesudahnya untuk menemukan langkah yang gagal |
| `step 0 | loss ...` muncul | Sedikitnya satu langkah training selesai; error terjadi kemudian |
| `Done. Final checkpoint -> ...` muncul | Training mencapai akhir dan checkpoint akhir ditulis |

## Konsep dan alternatif

`subprocess.run` menjaga skrip CLI tetap sama antara notebook dan penggunaan terminal. Konsekuensinya, traceback proses anak bisa terpisah dari traceback notebook. Memanggil fungsi `main()` langsung dapat membuat debugging satu proses lebih mudah, tetapi mengubah cara kerja notebook dan belum diperlukan sebelum error asli dibaca. `check=True` tepat untuk pipeline ini karena mencegah tahap berikutnya dianggap berhasil setelah pretrain gagal.

Notebook memakai file HDF5 dan JSONL agar data, checkpoint, serta metrik dapat diperiksa terpisah dari tampilan sel. Alternatifnya adalah menyimpan semuanya sebagai output notebook, tetapi output panjang dapat menjadi berat, dan checkpoint model tetap harus disimpan sebagai file.

## Referensi belajar

- [Python `subprocess`](https://docs.python.org/3/library/subprocess.html): status proses, `check=True`, dan `CalledProcessError`.
- [Google Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html): umur runtime dan penyimpanan sementara.
- [Dokumentasi h5py](https://docs.h5py.org/en/stable/high/dataset.html): struktur dataset di dalam HDF5.

Tidak ada training, unduhan, atau pengujian model yang dijalankan untuk membuat dokumen ini. Kode program tidak diubah.
