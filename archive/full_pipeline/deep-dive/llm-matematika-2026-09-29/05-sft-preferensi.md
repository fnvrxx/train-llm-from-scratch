# 5. SFT, reward model, DPO, ORPO, dan KTO

[Daftar bab](../llm-matematika-2026-09-29.md) · [Berikutnya: PPO dan GRPO](06-rl.md)

Kode: [SFT](../../src/post_training/sft.py), [reward model](../../src/post_training/reward_model.py), [loss reward](../../src/post_training/reward_train.py), [DPO/ORPO/KTO](../../src/post_training/dpo.py).

## SFT: meniru jawaban contoh

Supervised Fine-Tuning melanjutkan training model dengan pasangan instruksi dan jawaban yang diharapkan. Pada proyek ini, format chat dan mask menentukan bagian jawaban yang dihitung. Penggunaan demonstrasi sebagai tahap instruction tuning juga muncul pada [makalah InstructGPT](https://arxiv.org/abs/2203.02155).

Untuk indeks Python $t=0,\ldots,T-2$:

$$
\mathcal L_{\mathrm{SFT}}=-\frac{\sum_{b,t}m_{b,t+1}\log p_\theta(x_{b,t+1}\mid x_{b,\le t})}{\max(1,\sum_{b,t}m_{b,t+1})}.
$$

`logits[:, :-1, :]` memprediksi `tokens[:, 1:]`. `mask = loss_mask[:, 1:]` ikut bergeser karena yang dinilai adalah token target. `V` mengambil ukuran kosakata, `ce` menyimpan cross-entropy per posisi lalu dikalikan mask. Fungsi `sft_loss` mengembalikan satu skalar.

Ilustrasi: tiga target memiliki NLL 2, 1, 3 dan mask 0,1,1. Loss adalah $(1+3)/2=2$, bukan $(2+1+3)/3$. Prompt tetap membantu prediksi walaupun targetnya tidak dihitung.

`scripts/train_sft.py` memuat `pretrained_ckpt`, menghitung loss tersebut, menjalankan backward, dan menyimpan model baru. Ia melatih seluruh parameter backbone, bukan adapter LoRA. Alternatif loss semua token juga mungkin, tetapi akan mengalokasikan sebagian tujuan training untuk meniru prompt.

## Reward model: belajar membandingkan dua jawaban

Misalkan prompt $x$ memiliki jawaban yang dipilih $y^+$ dan ditolak $y^-$. Reward model menghasilkan skor real $r_\phi(x,y)$. Skor itu tidak harus berada dalam interval nol sampai satu.

`RewardModel` membungkus `Transformer`. `token_rewards(idx)` mengambil `hidden` melalui `forward_hidden`, kemudian `reward_head` memetakan tiap vektor menjadi skalar. `forward(idx,seq_lengths)` mengambil skor pada token nyata terakhir. Tanpa `seq_lengths`, ia memakai kolom terakhir, sehingga pemanggil harus memastikan kolom itu bukan padding.

$$
r_\phi(x,y)=w_r^\top h_{\mathrm{last}},\qquad
P(y^+\succ y^-\mid x)=\sigma(r^+-r^-),\quad
\sigma(u)=\frac1{1+e^{-u}}.
$$
$$
\mathcal L_{\mathrm{RM}}=-\frac1B\sum_b\log\sigma(r_b^+-r_b^-).
$$

`bradley_terry_loss(chosen_rewards,rejected_rewards)` menghitung rumus terakhir. `preference_accuracy` menghitung proporsi pasangan dengan skor chosen lebih besar. `reward_margin` merata-ratakan selisih skor. `load_reward_model` memuat model terlatih dan membekukan parameternya untuk dipakai sebagai penilai.

Ilustrasi: skor 2 dan 1 menghasilkan probabilitas preferensi $\sigma(1)\approx0.7311$. Jika kedua skor ditambah 100, probabilitas tidak berubah. Yang dilatih oleh loss pasangan adalah perbedaannya, sehingga skor absolut tidak otomatis bermakna probabilitas kebenaran.

Verifier numerik adalah alternatif reward model untuk tugas yang mempunyai kunci jawaban. Ia lebih mudah diperiksa, tetapi hanya menilai aturan yang diprogram.

## Log-probabilitas seluruh jawaban

`compute_logprobs` menghitung log-softmax lalu `gather` untuk token yang benar-benar ada dalam urutan. `sequence_logprobs` menjumlahkan hanya token jawaban:

$$
\ell_\theta(x,y)=\log\pi_\theta(y\mid x)
=\sum_{t\in\mathrm{jawaban}}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

`lp` berisi log-probabilitas per token; `mask` memilih jawaban; `m` adalah mask dalam tipe angka yang sesuai. Hasil fungsi adalah `(sum_logprob,n_tokens)`, masing-masing satu angka per urutan.

Pada `train_dpo._compute_losses`, chosen dan rejected digabung dalam satu batch. `psum` adalah jumlah log-prob policy dan `pn` jumlah token. `pc/pr` membaginya menjadi chosen/rejected; `ncn/nrn` adalah jumlah token masing-masing. `rsum`, `rc`, dan `rr` adalah pasangan hasil yang sama untuk reference model.

## DPO: preferensi relatif terhadap reference

Direct Preference Optimization memuat policy yang dilatih dan reference model yang dibekukan. Ia memakai pasangan jawaban yang tersedia, tanpa harus menghasilkan rollout baru dalam setiap update. [Makalah DPO](https://arxiv.org/abs/2305.18290).

$$
\Delta_\theta=\ell_\theta(x,y^+)-\ell_\theta(x,y^-),\qquad
\Delta_{\mathrm{ref}}=\ell_{\mathrm{ref}}(x,y^+)-\ell_{\mathrm{ref}}(x,y^-),
$$
$$
\mathcal L_{\mathrm{DPO}}=-\frac1B\sum_b\log\sigma\!\left(\beta(\Delta_{\theta,b}-\Delta_{\mathrm{ref},b})\right).
$$

| Variabel dalam `dpo_loss` | Arti |
|---|---|
| `policy_chosen_logps`, `policy_rejected_logps` | $\ell_\theta$ untuk jawaban chosen/rejected |
| `ref_chosen_logps`, `ref_rejected_logps` | Log-probabilitas kedua jawaban pada reference |
| `pi_logratios` | $\Delta_\theta$ |
| `ref_logratios` | $\Delta_{\mathrm{ref}}$ |
| `logits` | Selisih kedua delta; ini logit preferensi per pasangan, bukan logits kosakata |
| `beta` | Koefisien skala perbandingan policy-reference |
| `chosen_reward`, `rejected_reward` | Reward implisit $\beta(\ell_\theta-\ell_{\mathrm{ref}})$, di-detach untuk diagnostik |

Reference memberi titik pembanding yang tetap. DPO tidak memerlukan model reward terpisah dalam loop ini. `beta` bukan sampling temperature dan bukan learning rate; pengaruhnya harus dibaca dari objective.

Ilustrasi: $\ell_\theta^+=-2$, $\ell_\theta^-=-4$, $\ell_{\mathrm{ref}}^+=-3$, dan $\ell_{\mathrm{ref}}^-=-4$. Maka delta policy 2, delta reference 1. Dengan $\beta=0.1$, loss $-\log\sigma(0.1)\approx0.6444$. Saat policy sama persis dengan reference, loss awal setiap pasangan adalah $\log2$.

## ORPO: NLL dan odds ratio dalam satu loss

Odds Ratio Preference Optimization menggabungkan peniruan jawaban chosen dan preferensi tanpa reference model. [Makalah ORPO](https://arxiv.org/abs/2403.07691).

Kode merata-ratakan log-probabilitas per token:

$$
\bar\ell^+=\frac{\ell_\theta(x,y^+)}{\max(1,n^+)},\quad
\bar\ell^-=\frac{\ell_\theta(x,y^-)}{\max(1,n^-)},\quad
o(a)=a-\log(1-e^a).
$$
$$
\mathcal L_{\mathrm{ORPO}}=-\operatorname{mean}(\bar\ell^+)
-\lambda\operatorname{mean}\bigl[\log\sigma(o(\bar\ell^+)-o(\bar\ell^-))\bigr].
$$

`chosen_mean/rejected_mean` adalah $\bar\ell^+/\bar\ell^-$. `log_odds` adalah selisih $o$; `or_loss` loss preferensi; `nll` loss chosen; `orpo_lambda` bobot kedua bagian. `_log1mexp(x)` menghitung $\log(1-e^x)$ untuk $x<0$ memakai dua bentuk numerik agar pengurangan dekat satu lebih stabil. $e^{\bar\ell}$ merupakan rata-rata geometrik peluang token, bukan peluang gabungan seluruh jawaban.

## KTO: sinyal disukai atau tidak disukai

Kahneman-Tversky Optimization dirancang untuk sinyal desirable/undesirable, yang tidak harus berbentuk pasangan. Implementasi lokal mengambil sinyal itu dari `chosen` dan `rejected`, lalu menggunakan baseline batch sederhana. [Makalah KTO](https://arxiv.org/abs/2402.01306).

Untuk $u^+=\ell_\theta^+-\ell_{\mathrm{ref}}^+$ dan $u^-=\ell_\theta^--\ell_{\mathrm{ref}}^-$, kode menghitung:

$$
z=\operatorname{stopgrad}\left(\max(0,\operatorname{mean}(u^+\cup u^-))\right),
$$
$$
\mathcal L=\operatorname{mean}\left[w_+(1-\sigma(\beta(u^+-z)))\right]
+\operatorname{mean}\left[w_-(1-\sigma(\beta(z-u^-)))\right].
$$

`chosen_logratio/rejected_logratio` adalah $u^+/u^-$, `kl` adalah $z$, `chosen_losses/rejected_losses` kedua komponen loss, dan `desirable_weight/undesirable_weight` adalah $w_+/w_-$. Nama `kl` di sini tidak menjamin nilainya estimator KL yang tepat: batch berisi jawaban dataset, bukan sampel policy yang memenuhi asumsi ekspektasi KL. Baca ini sebagai varian sederhana yang memang ditulis di repo.

`implicit_accuracy` menghitung seberapa sering reward implisit chosen melebihi rejected. Angka itu bukan akurasi menjawab soal matematika.

## Latihan

Apakah loss DPO di bawah $\log2$ membuktikan chosen lebih mungkin daripada rejected pada policy?

Jawaban: tidak. Ia menunjukkan perubahan log-ratio yang menguntungkan chosen relatif terhadap reference, dalam rata-rata objective. Policy bisa tetap memberi peluang absolut lebih kecil pada chosen, terutama jika reference jauh lebih menyukai rejected.
