# 8. Glosarium dan sumber belajar

[Daftar bab](../llm-matematika-2026-09-29.md) · [Kamus kode](kamus/README.md)

## Istilah yang muncul sepanjang pipeline

| Istilah | Arti dalam konteks kode ini |
|---|---|
| LLM, Large Language Model | Model bahasa berparameter yang memprediksi distribusi token; arsitektur yang sama dapat dibuat kecil untuk belajar |
| Corpus | Kumpulan teks sumber pretraining |
| Token / token ID | Potongan teks hasil tokenizer dan nomor indeksnya |
| BPE, Byte Pair Encoding | Keluarga metode tokenisasi melalui penggabungan unit yang sering muncul; proyek memakai encoding yang sudah tersedia |
| Vocabulary | Himpunan ID kandidat token |
| Embedding | Tabel vektor terlatih yang dicari berdasarkan ID |
| Hidden state | Vektor representasi internal setiap posisi setelah sebagian/seluruh jaringan |
| Logit | Skor real sebelum normalisasi menjadi probabilitas |
| Softmax | Pemetaan vektor skor menjadi distribusi peluang |
| Log-prob, logp | Logaritma probabilitas suatu token atau urutan |
| Autoregressive | Menghasilkan token berikutnya dengan mengondisikan pada token sebelumnya |
| Causal mask | Membatasi attention agar posisi sekarang tidak membaca posisi masa depan |
| Loss mask / response mask | Memilih posisi mana yang masuk loss atau statistik; tidak sama dengan causal mask |
| Attention head | Satu set proyeksi query-key-value yang mencampur informasi posisi |
| MLP / FFN | Multi-Layer Perceptron / Feed-Forward Network, transformasi fitur per token |
| Residual connection | Menambahkan masukan ke hasil submodul |
| LayerNorm | Normalisasi fitur di dalam vektor satu token, dengan skala dan bias terlatih |
| Parameter / weight | Bilangan dalam model yang diperbarui optimizer |
| Hyperparameter | Pengaturan seperti learning rate, jumlah layer, atau koefisien KL |
| Forward / backward | Menghitung keluaran; menghitung turunan loss terhadap parameter |
| Autograd | Mekanisme diferensiasi otomatis PyTorch |
| Cross-entropy / NLL | Loss prediksi token; untuk target kelas tunggal sama dengan negative log-prob target |
| Batch / micro-batch | Kumpulan contoh yang diproses; micro-batch merupakan bagian dari update dengan accumulation |
| Epoch / step | Satu putaran dataset; satu indeks update optimizer sesuai loop |
| Pretraining | Belajar prediksi token dari corpus umum, biasanya mulai dari bobot acak |
| Fine-tuning | Melanjutkan training bobot yang sudah dipelajari dengan tujuan/data tertentu |
| SFT, Supervised Fine-Tuning | Fine-tuning menggunakan demonstrasi jawaban yang diharapkan |
| Post-training | Tahap lanjutan sesudah pretraining, termasuk SFT dan optimisasi preferensi/reward |
| Alignment | Penyesuaian perilaku model ke kriteria data atau reward yang dipakai; bukan jaminan semua jawaban benar |
| Chosen / rejected | Dua jawaban untuk prompt yang sama, dengan label lebih disukai dan kurang disukai |
| RM, Reward Model | Jaringan yang belajar memberi skor dari data preferensi |
| Bradley-Terry | Model probabilitas preferensi yang memakai sigmoid selisih skor |
| DPO, Direct Preference Optimization | Objective preferensi berdasarkan log-ratio policy dan reference pada pasangan jawaban |
| ORPO, Odds Ratio Preference Optimization | NLL chosen ditambah loss odds-ratio preferensi tanpa reference terpisah |
| KTO, Kahneman-Tversky Optimization | Optimisasi memakai label desirable/undesirable; kode lokal membentuk label dari pasangan |
| RL, Reinforcement Learning | Optimisasi perilaku berdasarkan reward dari action yang diambil |
| RLHF | RL dengan sinyal preferensi manusia, sering diringkas melalui reward model |
| RLVR | RL dengan reward yang dapat diperiksa program |
| PPO, Proximal Policy Optimization | Update policy melalui rasio peluang action dan surrogate clipping |
| GRPO, Group Relative Policy Optimization | Update policy memakai advantage relatif terhadap kelompok jawaban untuk prompt yang sama |
| Policy / actor | Distribusi action; pada LLM berarti distribusi token berikutnya |
| Critic / value head | Prediksi return yang diharapkan dari state/prefix |
| Rollout | Sampel jawaban hasil menjalankan policy pada prompt |
| Reward / return / advantage | Skor langsung; akumulasi skor mendatang; perbandingan hasil terhadap baseline |
| GAE, Generalized Advantage Estimation | Estimasi advantage melalui akumulasi galat temporal difference |
| Reference model | Salinan model awal yang dibekukan sebagai pembanding |
| Old policy | Policy saat rollout dikumpulkan, sebelum update berulang pada sampel itu |
| KL, Kullback-Leibler divergence | Ukuran perbedaan distribusi; bukan jarak simetris |
| Entropy | Ukuran ketidakpastian distribusi kandidat token |
| Teacher forcing | Menghitung prediksi dengan prefix token yang sudah disediakan, bukan sampling prefix baru |
| Packing / padding | Memadatkan contoh dalam baris konteks / menambah token pengisi agar bentuk seragam |
| EOT / EOS | End of text / end of sequence; proyek memakai EOT sebagai penghentian turn/generation |
| Top-k / top-p | Penyaringan kandidat token berdasarkan jumlah kandidat atau massa probabilitas |
| Temperature | Pembagi logits untuk mengubah konsentrasi distribusi sampling |
| Checkpoint | Berkas bobot dan metadata suatu keadaan training |
| AMP / bf16 / fp16 / fp32 | Mixed precision serta format bilangan floating point dengan rentang dan ketelitian berbeda |
| DDP, DistributedDataParallel | Beberapa proses memiliki model masing-masing dan menyinkronkan gradien |
| Rank / world size | Identitas satu proses / jumlah proses DDP |
| HDF5 / JSONL | Format penyimpanan array / satu objek JSON per baris |
| GSM8K | Dataset soal cerita aritmetika yang dipakai sebagai tugas dan evaluasi numerik di repo |
| MoE / pruning | Mixture of Experts memilih jalur expert; pruning menghapus/memask sebagian koneksi pada eksperimen alternatif |

KL dan entropy, untuk distribusi diskret $p,q$ dengan dukungan yang sesuai:

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_a p(a)\log\frac{p(a)}{q(a)},\qquad
H(p)=-\sum_a p(a)\log p(a).
$$

KL nol jika kedua distribusi sama. Nilai positif tidak menunjukkan arah mana yang lebih baik menjawab soal. Distribusi sangat yakin pada jawaban salah bisa mempunyai entropy rendah.

## Referensi yang terhubung ke kode

| Sumber | Gunakan setelah membaca |
|---|---|
| [Dokumentasi penulis](https://fareedkhan-dev.github.io/train-llm-from-scratch/) | Peta tahap proyek; bandingkan dengan implementasi lokal |
| [tiktoken, sumber resmi](https://github.com/openai/tiktoken) | Bab 2: tokenisasi BPE dan encoding |
| [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | Bab 1 dan 4: computational graph dan backward |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Bab 3: scaled dot-product dan multi-head attention |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | Bab 5 dan 6: demonstrasi, model reward, lalu RLHF |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Bab 5: objective policy/reference dari preferensi |
| [ORPO](https://arxiv.org/abs/2403.07691) | Bab 5: odds ratio tanpa reference model |
| [KTO](https://arxiv.org/abs/2402.01306) | Bab 5: label desirable/undesirable dan perbedaan dengan varian lokal |
| [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | Bab 6: rasio policy dan clipped surrogate |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Bab 6: asal GRPO dan baseline kelompok |

Rumus rinci pada bab utama mengikuti operasi dalam kode lokal. Makalah menjadi rujukan konsep dan pembanding, bukan klaim bahwa repo mengimplementasikan seluruh detail eksperimen makalah.

## Cek pemahaman akhir

Coba jelaskan `model(tokens)` tanpa memakai kata cerdas atau memahami. Jawaban yang cukup tepat: model mengambil vektor embedding token dan posisi, mengolahnya dengan attention dan MLP, lalu menghasilkan skor kandidat token untuk setiap posisi. Sesudah normalisasi, kita dapat menghitung loss terhadap target atau mengambil sampel token baru.

Kemudian jelaskan apa yang berubah pada SFT dan DPO. Pada SFT, contoh dan mask menentukan target yang ditiru. Pada DPO, objective membandingkan peluang dua jawaban relatif terhadap reference. Struktur dasar Transformer tetap dipakai.
