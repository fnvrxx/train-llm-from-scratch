# src/post_training/dpo.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/dpo.py)

## Fungsi dan kelas

### `dpo_loss` (baris 21)

Menghitung negative log-sigmoid selisih log-ratio preferensi policy/reference serta reward implisit diagnostik.

```python
def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor, beta: float=0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(loss, chosen_reward, rejected_reward)`.

### `_log1mexp` (baris 43)

Menghitung log(1-exp(x)) secara numerik stabil untuk x negatif.

```python
def _log1mexp(x: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))`.

### `orpo_loss` (baris 48)

Menggabungkan NLL chosen dan loss log-odds mean log-prob, tanpa reference model.

```python
def orpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, chosen_n_tokens: torch.Tensor, rejected_n_tokens: torch.Tensor, orpo_lambda: float=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(loss, chosen_mean.detach(), rejected_mean.detach())`.

### `kto_loss` (baris 71)

Menghitung loss desirable/undesirable dari pasangan dengan baseline mean log-ratio batch yang di-detach.

```python
def kto_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor, beta: float=0.1, desirable_weight: float=1.0, undesirable_weight: float=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

Keluaran eksplisit: `(loss, (beta * chosen_logratio).detach(), (beta * rejected_logratio).detach())`.

### `implicit_accuracy` (baris 93)

Menghitung proporsi pasangan dengan skor chosen lebih tinggi; ties tidak dianggap benar.

```python
def implicit_accuracy(chosen_reward: torch.Tensor, rejected_reward: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `(chosen_reward > rejected_reward).float().mean()`.

## Variabel dan parameter menurut lingkup

### Lingkup `dpo_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `policy_chosen_logps` · 22 | Jumlah log-prob token jawaban chosen pada policy trainable, bentuk (B,). | `argumen torch.Tensor` |
| `policy_rejected_logps` · 23 | Jumlah log-prob token jawaban rejected pada policy trainable, bentuk (B,). | `argumen torch.Tensor` |
| `ref_chosen_logps` · 24 | Jumlah log-prob token jawaban chosen pada reference beku, bentuk (B,). | `argumen torch.Tensor` |
| `ref_rejected_logps` · 25 | Jumlah log-prob token jawaban rejected pada reference beku, bentuk (B,). | `argumen torch.Tensor` |
| `beta` · 26 | Skala objective preferensi DPO/KTO, bukan learning rate atau sampling temperature. | `argumen float; default 0.1` |
| `pi_logratios` · 34 | Log-prob chosen minus rejected menurut policy trainable. | `policy_chosen_logps - policy_rejected_logps` |
| `ref_logratios` · 35 | Log-prob chosen minus rejected menurut reference beku. | `ref_chosen_logps - ref_rejected_logps` |
| `logits` · 36 | Delta log-ratio chosen/rejected policy dikurangi delta reference, satu per pasangan. | `pi_logratios - ref_logratios` |
| `loss` · 37 | Rerata -logsigmoid(beta*logits) pada batch preferensi. | `-F.logsigmoid(beta * logits).mean()` |
| `chosen_reward` · 38 | Skor sisi chosen, dari reward model atau reward implisit menurut fungsi. | `beta * (policy_chosen_logps - ref_chosen_logps).detach()` |
| `rejected_reward` · 39 | Skor sisi rejected, dari reward model atau reward implisit menurut fungsi. | `beta * (policy_rejected_logps - ref_rejected_logps).detach()` |

### Lingkup `_log1mexp`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `x` · 43 | Log-probabilitas negatif; fungsi mengembalikan log(1-exp(x)). | `argumen torch.Tensor` |

### Lingkup `orpo_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `policy_chosen_logps` · 49 | Jumlah log-prob token jawaban chosen pada policy trainable, bentuk (B,). | `argumen torch.Tensor` |
| `policy_rejected_logps` · 50 | Jumlah log-prob token jawaban rejected pada policy trainable, bentuk (B,). | `argumen torch.Tensor` |
| `chosen_n_tokens` · 51 | Jumlah token jawaban aktif pada sisi chosen. | `argumen torch.Tensor` |
| `rejected_n_tokens` · 52 | Jumlah token jawaban aktif pada sisi rejected. | `argumen torch.Tensor` |
| `orpo_lambda` · 53 | Bobot penalti preferensi ORPO yang ditambahkan ke NLL. | `argumen float; default 1.0` |
| `chosen_mean` · 61 | Jumlah log-prob chosen dibagi jumlah token jawaban aktif, untuk ORPO. | `policy_chosen_logps / chosen_n_tokens.clamp(min=1)` |
| `rejected_mean` · 62 | Jumlah log-prob rejected dibagi jumlah token jawaban aktif, untuk ORPO. | `policy_rejected_logps / rejected_n_tokens.clamp(min=1)` |
| `log_odds` · 63 | Selisih log-odds chosen/rejected berdasarkan exp(mean log-prob). | `chosen_mean - _log1mexp(chosen_mean) - (rejected_mean - _log1mexp(rejected_mean))` |
| `or_loss` · 64 | Negative log-sigmoid selisih log-odds chosen/rejected pada ORPO. | `-F.logsigmoid(log_odds).mean()` |
| `nll` · 65 | Rerata negative mean log-prob jawaban chosen pada ORPO. | `-chosen_mean.mean()` |
| `loss` · 66 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `nll + orpo_lambda * or_loss` |

### Lingkup `kto_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `policy_chosen_logps` · 72 | Jumlah log-prob token jawaban chosen pada policy trainable, bentuk (B,). | `argumen torch.Tensor` |
| `policy_rejected_logps` · 73 | Jumlah log-prob token jawaban rejected pada policy trainable, bentuk (B,). | `argumen torch.Tensor` |
| `ref_chosen_logps` · 74 | Jumlah log-prob token jawaban chosen pada reference beku, bentuk (B,). | `argumen torch.Tensor` |
| `ref_rejected_logps` · 75 | Jumlah log-prob token jawaban rejected pada reference beku, bentuk (B,). | `argumen torch.Tensor` |
| `beta` · 76 | Skala objective preferensi DPO/KTO, bukan learning rate atau sampling temperature. | `argumen float; default 0.1` |
| `desirable_weight` · 77 | Bobot kontribusi loss contoh desirable/chosen dalam KTO. | `argumen float; default 1.0` |
| `undesirable_weight` · 78 | Bobot kontribusi loss contoh undesirable/rejected dalam KTO. | `argumen float; default 1.0` |
| `chosen_logratio` · 84 | Log-prob jawaban chosen pada policy dikurangi pada reference. | `policy_chosen_logps - ref_chosen_logps` |
| `rejected_logratio` · 85 | Log-prob jawaban rejected pada policy dikurangi pada reference. | `policy_rejected_logps - ref_rejected_logps` |
| `kl` · 86 | Baseline nonnegatif dari mean log-ratio batch yang di-detach; bukan KL persis tanpa asumsi sampling tambahan. | `torch.cat([chosen_logratio, rejected_logratio]).mean().clamp(min=0).detach()` |
| `chosen_losses` · 87 | Penalti KTO untuk sinyal desirable: 1-sigmoid(beta*(chosen_logratio-kl)). | `1.0 - torch.sigmoid(beta * (chosen_logratio - kl))` |
| `rejected_losses` · 88 | Penalti KTO untuk sinyal undesirable: 1-sigmoid(beta*(kl-rejected_logratio)). | `1.0 - torch.sigmoid(beta * (kl - rejected_logratio))` |
| `loss` · 89 | Skalar objective yang diminimalkan; rumusnya bergantung tahap training. | `(desirable_weight * chosen_losses).mean() + (undesirable_weight * rejected_losses).mean()` |

### Lingkup `implicit_accuracy`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `chosen_reward` · 93 | Skor sisi chosen, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |
| `rejected_reward` · 93 | Skor sisi rejected, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 15 |
| `torch` | `torch` | 17 |
| `F` | `torch.nn.functional` | 18 |
