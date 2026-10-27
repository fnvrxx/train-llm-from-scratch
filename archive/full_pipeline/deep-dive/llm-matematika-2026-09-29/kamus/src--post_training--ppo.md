# src/post_training/ppo.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/ppo.py)

## Fungsi dan kelas

### `compute_gae` (baris 24)

Menghitung rekursi advantage mundur dan target returns, nol di luar action jawaban.

```python
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, values_next: torch.Tensor, resp_mask: torch.Tensor, gamma: float=1.0, lam: float=0.95) -> tuple[torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(adv * m, returns * m)`.

### `whiten` (baris 60)

Menormalkan advantage memakai mean/variansi hanya pada posisi mask aktif.

```python
def whiten(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `(advantages - mean) / (var.sqrt() + 1e-08) * m`.

### `ppo_policy_loss` (baris 68)

Menghitung clipped surrogate policy loss dan fraksi rasio di luar batas clip.

```python
def ppo_policy_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor, clip: float=0.2) -> tuple[torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(loss, masked_mean(clipped, mask))`.

### `ppo_value_loss` (baris 84)

Menghitung setengah maksimum galat kuadrat value biasa dan clipped value.

```python
def ppo_value_loss(new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor, vf_clip: float=0.2) -> torch.Tensor:
```

Keluaran eksplisit: `0.5 * masked_mean(torch.max(loss_unclipped, loss_clipped), mask)`.

### `approx_kl` (baris 98)

Merata-ratakan old_logp-new_logp sebagai diagnostik aproksimasi KL old terhadap new.

```python
def approx_kl(new_logp: torch.Tensor, old_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `masked_mean(old_logp - new_logp, mask)`.

## Variabel dan parameter menurut lingkup

### Lingkup `compute_gae`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rewards` · 25 | Reward per action termasuk penalti KL dan task reward akhir, bentuk (B,L). | `argumen torch.Tensor` |
| `values` · 26 | Prediksi V(s_t), satu per action, bentuk (B,L). | `argumen torch.Tensor` |
| `values_next` · 27 | Prediksi value state setelah action sekarang, sejajar dengan rewards (B,T-1). | `argumen torch.Tensor` |
| `resp_mask` · 28 | Mask posisi jawaban nyata, bukan prompt/padding. | `argumen torch.Tensor` |
| `gamma` · 29 | Discount factor untuk reward/value mendatang. | `argumen float; default 1.0` |
| `lam` · 30 | Koefisien jejak temporal difference pada GAE. | `argumen float; default 0.95` |
| `B` · 46 | Jumlah urutan atau pasangan dalam batch, biasanya per proses DDP. | `elemen 0 dari rewards.shape` |
| `L` · 46 | Jumlah action, sama dengan panjang urutan token dikurangi satu. | `elemen 1 dari rewards.shape` |
| `adv` · 47 | Array advantage (B,L), diisi lewat rekursi mundur. | `torch.zeros_like(rewards)` |
| `lastgae` · 48, 54 | Akumulator rekursi GAE dari posisi kanan ke kiri, satu angka per baris. | `torch.zeros(B, device=rewards.device)`; `delta + gamma * lam * nonterminal * lastgae` |
| `m` · 49 | resp_mask yang dikonversi float untuk mengalikan advantage/return. | `resp_mask.float()` |
| `t` · 50 | Indeks action yang ditelusuri dari L-1 ke nol. | `iterasi reversed(range(L))` |
| `nonterminal` · 52 | Indikator action berikutnya masih bagian jawaban, untuk menghentikan bootstrap. | `m[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)` |
| `delta` · 53 | Galat temporal difference: reward + discounted next value - current value. | `rewards[:, t] + gamma * values_next[:, t] * nonterminal - values[:, t]` |
| `returns` · 56 | Target return untuk value head, dihitung sebagai advantage GAE ditambah value lama. | `adv + values` |

### Lingkup `whiten`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `advantages` · 60 | Advantage per token pada PPO atau per jawaban pada GRPO. | `argumen torch.Tensor` |
| `mask` · 60 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen torch.Tensor` |
| `m` · 62 | Mask posisi advantage yang dihitung statistiknya, dikonversi float. | `mask.float()` |
| `mean` · 63 | Rata-rata angka pada posisi bermask atau pada satu kelompok, sesuai fungsi. | `masked_mean(advantages, m)` |
| `var` · 64 | Variansi, yaitu rerata kuadrat simpangan dari mean pada posisi mask aktif. | `masked_mean((advantages - mean) ** 2, m)` |

### Lingkup `ppo_policy_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `new_logp` · 69 | Log-prob action pada policy yang sedang di-update, bentuk (B,T-1). | `argumen torch.Tensor` |
| `old_logp` · 70 | Log-prob action saat rollout dikumpulkan, disimpan tetap selama update atas rollout itu. | `argumen torch.Tensor` |
| `advantages` · 71 | Advantage per token pada PPO atau per jawaban pada GRPO. | `argumen torch.Tensor` |
| `mask` · 72 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen torch.Tensor` |
| `clip` · 73 | Epsilon lebar interval rasio [1-clip,1+clip] pada PPO/GRPO. | `argumen float; default 0.2` |
| `ratio` · 76 | exp(new_logp-old_logp), rasio peluang action yang sama. | `torch.exp(new_logp - old_logp)` |
| `surr1` · 77 | Rasio peluang baru/lama dikali advantage tanpa pembatasan rasio. | `ratio * advantages` |
| `surr2` · 78 | Rasio peluang yang di-clip ke [1-clip,1+clip] lalu dikali advantage. | `torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages` |
| `loss` · 79 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `-masked_mean(torch.min(surr1, surr2), mask)` |
| `clipped` · 80 | Indikator rasio token berada di luar interval clipping, dikonversi menjadi float. | `((ratio - 1.0).abs() > clip).float()` |

### Lingkup `ppo_value_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `new_values` · 85 | Prediksi value state dari actor saat update sekarang. | `argumen torch.Tensor` |
| `old_values` · 86 | Prediksi value state yang disimpan sebelum update pada rollout. | `argumen torch.Tensor` |
| `returns` · 87 | Target return untuk value head, dihitung sebagai advantage GAE ditambah value lama. | `argumen torch.Tensor` |
| `mask` · 88 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen torch.Tensor` |
| `vf_clip` · 89 | Batas perubahan prediksi value pada cabang clipped value loss. | `argumen float; default 0.2` |
| `v_clipped` · 92 | Value baru setelah perubahannya terhadap old_values dibatasi. | `old_values + torch.clamp(new_values - old_values, -vf_clip, vf_clip)` |
| `loss_unclipped` · 93 | Galat kuadrat new_values terhadap target returns tanpa clipping perubahan value. | `(new_values - returns) ** 2` |
| `loss_clipped` · 94 | Galat kuadrat v_clipped terhadap target returns. | `(v_clipped - returns) ** 2` |

### Lingkup `approx_kl`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `new_logp` · 98 | Log-prob action pada policy yang sedang di-update, bentuk (B,T-1). | `argumen torch.Tensor` |
| `old_logp` · 98 | Log-prob action saat rollout dikumpulkan, disimpan tetap selama update atas rollout itu. | `argumen torch.Tensor` |
| `mask` · 98 | Penanda posisi yang dihitung oleh loss/reduksi; biasanya target assistant bernilai satu. | `argumen torch.Tensor` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 17 |
| `torch` | `torch` | 19 |
| `masked_mean` | `src.post_training.utils.masked_mean` | 21 |
