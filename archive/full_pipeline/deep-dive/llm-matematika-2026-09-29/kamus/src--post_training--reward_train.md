# src/post_training/reward_train.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/reward_train.py)

## Fungsi dan kelas

### `bradley_terry_loss` (baris 18)

Menghitung mean -log sigmoid(reward chosen - reward rejected).

```python
def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `-F.logsigmoid(chosen_rewards - rejected_rewards).mean()`.

### `preference_accuracy` (baris 23)

Menghitung proporsi pasangan dengan skor chosen lebih tinggi; ties tidak dianggap benar.

```python
def preference_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `(chosen_rewards > rejected_rewards).float().mean()`.

### `reward_margin` (baris 28)

Mengembalikan rerata selisih reward chosen minus rejected.

```python
def reward_margin(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `(chosen_rewards - rejected_rewards).mean()`.

## Variabel dan parameter menurut lingkup

### Lingkup `bradley_terry_loss`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `chosen_rewards` · 18 | Skor sisi chosen, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |
| `rejected_rewards` · 18 | Skor sisi rejected, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |

### Lingkup `preference_accuracy`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `chosen_rewards` · 23 | Skor sisi chosen, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |
| `rejected_rewards` · 23 | Skor sisi rejected, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |

### Lingkup `reward_margin`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `chosen_rewards` · 28 | Skor sisi chosen, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |
| `rejected_rewards` · 28 | Skor sisi rejected, dari reward model atau reward implisit menurut fungsi. | `argumen torch.Tensor` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 12 |
| `torch` | `torch` | 14 |
| `F` | `torch.nn.functional` | 15 |
