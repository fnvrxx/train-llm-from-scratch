# 4. Bagaimana bobot belajar, lalu menghasilkan teks

[Daftar bab](../llm-matematika-2026-09-29.md) · [Berikutnya: SFT dan preferensi](05-sft-preferensi.md)

Kode: [pretraining](../../scripts/pretrain_base.py), [trainer awal](../../scripts/train_transformer.py), [optimizer](../../src/post_training/optim.py), [rollout](../../src/post_training/rollout.py).

## Dari skor menjadi loss

Untuk logits $z\in\mathbb R^V$ dan token target $y$, loss satu posisi adalah:

$$
\ell(z,y)=-\log\frac{e^{z_y}}{\sum_{v=1}^V e^{z_v}}
=-z_y+\log\sum_{v=1}^V e^{z_v}.
$$

Turunannya terhadap logit kandidat $j$:

$$
\frac{\partial\ell}{\partial z_j}=p_j-\mathbf1[j=y].
$$

Jika peluang target masih kecil, gradien untuk logit target negatif. Gradient descent cenderung menaikkan logit itu. Aturan rantai meneruskan pengaruhnya ke `lm_head`, seluruh blok, dan embedding.

Ilustrasi: peluang token benar 0,25 memberi loss $-\log(0.25)\approx1.3863$. Jika naik menjadi 0,5, loss turun menjadi sekitar 0,6931. Ini ukuran kecocokan distribusi terhadap target, bukan persentase jawaban benar.

## Empat baris yang mengubah model

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
optimizer.step()
```

`zero_grad` membersihkan gradien sebelumnya. `backward` menghitung gradien loss saat ini. `clip_grad_norm_` membatasi norma gabungan gradien. `step` memperbarui parameter berdasarkan aturan optimizer.

Untuk memahami arah pembaruan, gradient descent biasa adalah $\theta_{k+1}=\theta_k-\eta g_k$, dengan $g_k=\nabla_\theta\mathcal L_k$. Kode memakai AdamW, yang mengadaptasi skala berdasarkan riwayat gradien:

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
$$
$$
\hat m_k=\frac{m_k}{1-\beta_1^k},\quad
\hat v_k=\frac{v_k}{1-\beta_2^k},\quad
\theta_{k+1}=(1-\eta\lambda)\theta_k-\eta\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
$$

Kuadrat, akar, dan pembagian terakhir bekerja per koordinat. $\lambda$ adalah weight decay pada kelompok parameter yang menerapkannya. `configure_optimizer` mengelompokkan parameter berdimensi minimal dua ke `decay`, sisanya ke `no_decay`. Karena embedding berbentuk matriks, ia termasuk `decay` pada implementasi ini, walaupun komentar awal berkas menyatakan sebaliknya.

## Ikuti `pretrain_base.main`

| Variabel | Arti dan pemakaiannya |
|---|---|
| `cfg`, `extras` | Konfigurasi final dan argumen tambahan seperti resume |
| `ctx` | Konteks proses/device, termasuk `rank` dan `world_size` |
| `model`, `optimizer` | Jaringan yang dilatih dan pengubah bobotnya |
| `batch_iter` | Sumber pasangan `xb,yb` berikutnya |
| `step`, `start_step` | Indeks update sekarang dan awal loop |
| `lr` | Learning rate pada step ini |
| `micro` | Indeks micro-batch dalam gradient accumulation |
| `sync`, `cm` | Apakah gradien DDP disinkronkan pada micro-step ini dan context manager-nya |
| `accum_loss` | Jumlah loss yang sudah dibagi faktor accumulation untuk pelaporan |
| `tokens_per_step` | Banyak token nominal yang diproses seluruh proses per update |
| `ev` | Hasil evaluasi train/dev |
| `logger`, `t0`, `dt`, `tok_s` | Pencatat metrik, awal interval, durasi, dan throughput token |

Gradient accumulation menjalankan beberapa backward sebelum satu optimizer step. Setiap loss dibagi `grad_accum`, sehingga untuk micro-batch sama besar gradiennya adalah rata-rata:

$$
g=\frac1K\sum_{k=1}^K g^{(k)},\qquad
B_{\mathrm{efektif}}=B_{\mathrm{micro}}K\,W.
$$

$K$ adalah jumlah micro-batch dan $W$ banyak proses DDP. Ini dipakai pada `pretrain_base.py`. Jangan menerapkan rumus itu hanya karena nama konfigurasi ada; lihat apakah loop benar-benar menggunakannya.

`cosine_lr` memulai dengan warmup linear. Setelah warmup, ia menurunkan learning rate mengikuti kosinus sampai `min_lr`:

$$
\eta_s=\eta_{\min}+\tfrac12\left(1+\cos\left(\pi\frac{s-w}{S-w}\right)\right)(\eta_{\max}-\eta_{\min}).
$$

$s$ adalah step, $w$ jumlah warmup, dan $S$ batas step. Pada bagian warmup, kode memakai $\eta_{\max}(s+1)/\max(1,w)$. Trainer awal `train_transformer.py` memakai jadwal berbeda: learning rate berpindah setelah ambang `t_lr_decay_step`.

## Checkpoint dan memori

Checkpoint menyimpan bobot agar bisa dimuat kembali. Untuk melanjutkan optimisasi, state optimizer juga berguna karena AdamW menyimpan $m$ dan $v$. `save_stage_ckpt` menyimpan bobot, konfigurasi, step, optimizer, dan metrik.

Pada trainer awal, `save_training_checkpoint` menulis file sementara lalu menggantinya secara atomik. `restore_training_checkpoint` membaca step selanjutnya; `list_checkpoints` dan `resolve_resume_path` mencari checkpoint; `prune_old_checkpoints` membatasi jumlah berkas; `unique_output_path` menghindari penimpaan hasil final. Detail setiap parameter dan variabelnya ada di kamus.

AMP menjalankan sebagian operasi dengan presisi lebih rendah. Activation checkpointing menyimpan lebih sedikit aktivasi dan menghitungnya ulang saat backward. Gradient accumulation mengurangi jumlah sampel yang harus aktif sekaligus. Ketiganya menukar biaya yang berbeda; tidak ada yang otomatis menambah kemampuan model.

## Generation mengambil sampel dari distribusi

`Transformer.generate(idx,max_new_tokens)` mengulang:

1. Ambil `idx_cond`, yaitu bagian akhir konteks yang masih muat.
2. Hitung logits, kemudian ambil posisi terakhir.
3. Bentuk `probs` dengan softmax.
4. Sampel `idx_next` memakai `torch.multinomial`.
5. Sambungkan ID baru ke `idx`.

Bobot tetap selama inference biasa. Urutan inputnya yang bertambah. Fungsi dasar ini tidak menghentikan generasi pada EOT. Jalur post-training `generate_with_logprobs` menambahkan penghentian per baris dan mask padding.

Untuk suhu $\tau>0$:

$$
p_j=\frac{\exp(z_j/\tau)}{\sum_k\exp(z_k/\tau)}.
$$

Suhu kecil membuat distribusi lebih terkonsentrasi. `top_k` membatasi kandidat pada skor tertinggi; `top_p` mempertahankan kumpulan kandidat berpeluang kumulatif mencapai ambang. Kode memindahkan tanda penghapusan satu posisi agar token yang melampaui ambang tetap disertakan.

| Variabel rollout | Arti |
|---|---|
| `B`, `P`, `cap` | Jumlah prompt, panjang prompt bersama, dan kapasitas konteks |
| `sequences` | Prompt diikuti token hasil generasi |
| `finished`, `active` | Baris yang sudah selesai dan yang masih boleh menghasilkan token |
| `next_tok`, `token_lp` | Token hasil sampel dan log-probabilitasnya |
| `gen_mask`, `response_mask` | Posisi generated yang nyata; versi kedua juga memuat bagian prompt bernilai false |
| `stop_set`, `hit_stop` | ID penghentian dan apakah token terbaru cocok |
| `gen_logprobs` | Log-probabilitas token generated, nol pada padding |

`rollout_prompts` mengelompokkan prompt dengan panjang sama karena model tidak mempunyai attention mask untuk left padding. `pieces` menyimpan hasil kelompok sementara; `prompt_lens` menyimpan panjang prompt tiap baris. Ia memulihkan urutan input setelah generasi.

## Latihan

Ilustrasi: `batch_size=4`, `grad_accum=3`, dan `world_size=2`. Berapa batch efektif pada pretraining?

Jawaban: 24 urutan per optimizer step. Jika panjang urutan 16, jumlah target nominal adalah 384 token per step. Jumlah parameter model tidak berubah akibat pengaturan ini.
