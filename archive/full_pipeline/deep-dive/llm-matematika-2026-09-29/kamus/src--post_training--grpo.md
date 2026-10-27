# src/post_training/grpo.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/grpo.py)

## Fungsi dan kelas

### `group_advantages` (baris 17)

Menormalkan reward per kelompok memakai mean dan simpangan baku sampel; perlu G lebih dari satu.

```python
def group_advantages(rewards: torch.Tensor, group_size: int, eps: float=0.0001) -> torch.Tensor:
```

Keluaran eksplisit: `adv.reshape(-1)`.

### `k3_kl` (baris 31)

Menghitung exp(diff)-diff-1, diff=ref_logp-new_logp; interpretasi estimator KL bergantung distribusi sampling.

```python
def k3_kl(new_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `torch.exp(diff) - diff - 1.0`.

### `grpo_loss` (baris 37)

Menghitung surrogate clipped dengan advantage kelompok dan penalti reference KL, lalu statistiknya.

```python
def grpo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, ref_logp: torch.Tensor, advantages: torch.Tensor, resp_mask: torch.Tensor, clip: float=0.2, kl_coef: float=0.04) -> tuple[torch.Tensor, dict]:
```

Keluaran eksplisit: `(loss, stats)`.

## Variabel dan parameter menurut lingkup

### Lingkup `group_advantages`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `rewards` · 17 | Reward per urutan/action/kelompok sesuai fungsi; berbeda dari peluang token. | `argumen torch.Tensor` |
| `group_size` · 17 | Jumlah jawaban per prompt pada GRPO. | `argumen int` |
| `eps` · 17 | Konstanta kecil untuk penyebut normalisasi; tidak memperbaiki NaN std pada G=1. | `argumen float; default 0.0001` |
| `r` · 24 | Reward di-reshape menjadi (jumlah_prompt,group_size), satu kelompok per baris. | `rewards.view(-1, group_size)` |
| `mean` · 25 | Mean reward tiap kelompok dengan dimensi singleton agar broadcasting tepat. | `r.mean(dim=1, keepdim=True)` |
| `std` · 26 | Simpangan baku sampel tiap kelompok; default koreksi satu sehingga G=1 tidak valid. | `r.std(dim=1, keepdim=True)` |
| `adv` · 27 | Reward dikurangi mean kelompok lalu dibagi std+eps, masih berbentuk kelompok. | `(r - mean) / (std + eps)` |

### Lingkup `k3_kl`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `new_logp` · 31 | Log-prob action pada policy yang sedang di-update, bentuk (B,T-1). | `argumen torch.Tensor` |
| `ref_logp` · 31 | Log-prob action pada reference model yang dibekukan di awal tahap. | `argumen torch.Tensor` |
| `diff` · 33 | ref_logp-new_logp pada estimator k3. | `ref_logp - new_logp` |

### Lingkup `grpo_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `new_logp` · 38 | Log-prob action pada policy yang sedang di-update, bentuk (B,T-1). | `argumen torch.Tensor` |
| `old_logp` · 39 | Log-prob action saat rollout dikumpulkan, disimpan tetap selama update atas rollout itu. | `argumen torch.Tensor` |
| `ref_logp` · 40 | Log-prob action pada reference model yang dibekukan di awal tahap. | `argumen torch.Tensor` |
| `advantages` · 41 | Advantage per token pada PPO atau per jawaban pada GRPO. | `argumen torch.Tensor` |
| `resp_mask` · 42 | Mask posisi jawaban nyata, bukan prompt/padding. | `argumen torch.Tensor` |
| `clip` · 43 | Epsilon lebar interval rasio [1-clip,1+clip] pada PPO/GRPO. | `argumen float; default 0.2` |
| `kl_coef` · 44 | Bobot penalti penyimpangan terhadap reference. | `argumen float; default 0.04` |
| `adv` · 57 | Advantage per jawaban diberi sumbu tambahan (B,1), lalu disiarkan ke semua token jawaban. | `advantages[:, None]` |
| `ratio` · 58 | exp(new_logp-old_logp), rasio peluang action yang sama. | `torch.exp(new_logp - old_logp)` |
| `surr1` · 59 | Rasio peluang baru/lama dikali advantage tanpa pembatasan rasio. | `ratio * adv` |
| `surr2` · 60 | Rasio peluang yang di-clip ke [1-clip,1+clip] lalu dikali advantage. | `torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv` |
| `surrogate` · 61 | Minimum surr1 dan surr2, yaitu objective policy clipped per token. | `torch.min(surr1, surr2)` |
| `kl` · 62 | Penalti/estimasi KL pada RL; pada KTO baseline mean log-ratio batch yang di-clamp dan detach. | `k3_kl(new_logp, ref_logp)` |
| `per_token` · 64 | Surrogate GRPO sesudah dikurangi penalti KL per token. | `surrogate - kl_coef * kl` |
| `loss` · 65 | Negative masked mean dari surrogate dikurangi penalti KL; merata-ratakan seluruh token aktif. | `-masked_mean(per_token, resp_mask)` |
| `stats` · 66 | Metrik loss, KL, clip fraction, dan hitungan untuk agregasi. | `{'kl': masked_mean(kl, resp_mask).item(), 'clipfrac': masked_mean(((ratio - 1.0).abs() > clip).float(), resp_mask).item()}` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 10 |
| `torch` | `torch` | 12 |
| `masked_mean` | `src.post_training.utils.masked_mean` | 14 |
