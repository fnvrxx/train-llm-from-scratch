# Laboratorium LLM dalam notebook

Kelima notebook berisi kode model langsung, penjelasan berbahasa Indonesia, rumus LaTeX, dan eksperimen kecil di CPU. Semua hasil eksperimen tampil di output sel setelah dijalankan. Notebook yang dibagikan memiliki output kosong dan execution count `null`.

| Notebook | Yang dipelajari |
| :--- | :--- |
| [01_baseline.ipynb](01_baseline.ipynb) | Transformer dense, next-token prediction, gradien, SFT, reward, DPO, PPO, GRPO |
| [02_pruned.ipynb](02_pruned.ipynb) | Mask bobot tetap berdasarkan magnitudo |
| [03_topk.ipynb](03_topk.ipynb) | Pemilihan neuron aktif untuk setiap token |
| [04_moe.ipynb](04_moe.ipynb) | Routing expert dan loss penyeimbang |
| [05_evolutionary_set.ipynb](05_evolutionary_set.ipynb) | Mask acak, rewiring, dan pembersihan state optimizer |

Buka notebook di Jupyter atau Colab dengan Python 3.10+ dan PyTorch terpasang, lalu pilih **Run all**. Notebook tidak memerlukan checkout repositori, dataset unduhan, atau GPU. Jika PyTorch belum tersedia, pasang di lingkungan Python sebelum menjalankan notebook. Instalasi dependensi dikelola oleh lingkungan, bukan sel eksperimen.

Baca 01 lebih dulu. Setiap notebook berikutnya mengulang definisi dasar agar dapat dijalankan sendiri. Tabel AntiVibe di dekat sel menjelaskan nama, scope, tipe/bentuk, serta fungsi. Paragraf sebelumnya menghubungkan kode dengan rumus dan alasan pemilihan metode. Latihan kecil memiliki jawaban yang dapat dibuka.

Data karakter, parameter, history sepuluh langkah, dan teks generasi berada di RAM. Eksekusi notebook tidak membuat folder proyek, berkas Python, checkpoint, log, JSON hasil, atau salinan Drive. Restart kernel menghapus state tersebut. Penyimpanan notebook oleh Jupyter sendiri tetap mengikuti pengaturan aplikasi.

Sepuluh langkah mengulang satu batch untuk memperlihatkan proses optimisasi. Hasilnya tidak mengukur kemampuan bahasa atau kecepatan sparse. Contoh post-training pada baseline menghitung objective pada tensor kecil, bukan menjalankan pipeline RLHF lengkap. Untuk training data besar, gunakan skrip repositori dan panduan utama; notebook ini tidak lagi memuat pipeline opsional tersebut.

## Mengubah materi

`build_notebooks.py` mengambil definisi terpilih dari source model dan objective, kemudian menaruhnya langsung di sel. `learning_cells.py` memuat eksperimen dan penjelasan AntiVibe. Generator menolak nama yang belum memiliki penjelasan. Source asal dicatat dalam metadata setiap sel definisi dan diperiksa kesamaannya saat pengujian.

Dari root repositori:

```bash
python3 notebook/build_notebooks.py
python3 notebook/test_notebooks.py
```

Pengujian memerlukan PyTorch, IPython, dan nbformat. Test memeriksa struktur notebook, output tersimpan yang kosong, penjelasan nama, kesamaan source, dan eksekusi setiap notebook dalam proses terpisah dengan impor proyek serta penulisan berkas diblokir. Pemeriksaan numerik mencakup loss/gradien finite, perubahan bobot, bentuk generasi, mask pruning, proyeksi top-k, ekuivalensi MoE saat semua expert dipilih, serta konservasi koneksi dan reset momen Adam pada SET.

Perintah generator memang menulis lima berkas `.ipynb`; sel di dalam notebook tidak menulis artifact eksperimen. Penjelasan lama di `deep-dive/` tetap tersedia, tetapi petunjuk eksekusi notebook mengikuti README ini.
