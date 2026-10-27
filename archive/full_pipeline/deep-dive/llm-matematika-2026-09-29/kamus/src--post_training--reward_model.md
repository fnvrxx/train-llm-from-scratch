# src/post_training/reward_model.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/reward_model.py)

## Fungsi dan kelas

### `load_reward_model` (baris 21)

Membuat wrapper reward, memuat bobot, memasang eval, dan membekukan parameter untuk menilai rollout.

```python
def load_reward_model(cfg, ckpt_path: str, device: str) -> 'RewardModel':
```

Keluaran eksplisit: `rm`.

### `RewardModel` (baris 37)

Wrapper backbone dengan kepala skor reward per token/urutan.

### `RewardModel.__init__` (baris 40)

Membuat komponen dan keadaan objek RewardModel. Wrapper backbone dengan kepala skor reward per token/urutan.

```python
def __init__(self, transformer: Transformer) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `RewardModel.context_length` (baris 48)

Property yang meneruskan kapasitas konteks backbone ke wrapper.

```python
def context_length(self) -> int:
```

Keluaran eksplisit: `self.transformer.context_length`.

### `RewardModel.token_rewards` (baris 51)

Menghasilkan skor reward tiap posisi dari hidden states, berbentuk (B,T).

```python
def token_rewards(self, idx: torch.Tensor) -> torch.Tensor:
```

Keluaran eksplisit: `self.reward_head(hidden).squeeze(-1)`.

### `RewardModel.forward` (baris 56)

Memilih reward pada token nyata terakhir berdasarkan seq_lengths, atau kolom terakhir tanpa panjang.

```python
def forward(self, idx: torch.Tensor, seq_lengths: torch.Tensor | None=None) -> torch.Tensor:
```

Keluaran eksplisit: `rewards[:, -1]`; `gather_last(rewards, seq_lengths)`.

## Variabel dan parameter menurut lingkup

### Lingkup `load_reward_model`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg` · 21 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `argumen tanpa anotasi tipe` |
| `ckpt_path` · 21 | Path checkpoint masukan; awalan menunjukkan asal tahap bobot. | `argumen str` |
| `device` · 21 | Lokasi komputasi, misalnya cpu atau cuda. | `argumen str` |
| `backbone` · 24 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `build_model_from_config(cfg)` |
| `rm` · 25 | Model reward yang menghasilkan skor jawaban. | `RewardModel(backbone)` |
| `ck` · 26 | Isi berkas checkpoint, termasuk bobot dan metadata bila tersedia. | `_torch.load(ckpt_path, map_location='cpu', weights_only=False)` |
| `state` · 27, 29 | Pemetaan nama parameter/buffer ke tensor bobot. | `ck['model_state_dict'] if 'model_state_dict' in ck else ck`; `{k.removeprefix('module.'): v for k, v in state.items()}` |
| `k` · 28, 29 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `iterasi state`; `elemen 0 dari iterasi state.items()` |
| `v` · 29 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi state.items()` |
| `p` · 32 | Satu tensor parameter model yang dihitung jumlah elemennya, dikelompokkan optimizer, atau dibekukan sesuai fungsi. | `iterasi rm.parameters()` |

### Lingkup `RewardModel.__init__`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 40 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `transformer` · 40 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `argumen Transformer` |
| `self.transformer` · 42 | Jaringan Transformer utama; backbone mengacu bagian representasi yang dipakai ulang oleh wrapper. | `transformer` |
| `n_embed` · 43 | Banyak koordinat vektor representasi satu token. | `transformer.lm_head.in_features` |
| `self.reward_head` · 44 | Linear dari hidden state ke satu reward, dengan bobot awal nol. | `nn.Linear(n_embed, 1, bias=False)` |

### Lingkup `RewardModel.context_length`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 48 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |

### Lingkup `RewardModel.token_rewards`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 51 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 51 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `hidden` · 53 | Hidden states backbone; sebagai atribut MLP berarti lapisan linear ekspansi C ke 4C. | `self.transformer.forward_hidden(idx)` |

### Lingkup `RewardModel.forward`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `self` · 56 | Objek yang sedang menjalankan metode; self.* menyimpan lapisan atau keadaan objek. | `argumen tanpa anotasi tipe` |
| `idx` · 56 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen torch.Tensor` |
| `seq_lengths` · 56 | Banyak token nyata setiap urutan, sebelum right padding. | `argumen torch.Tensor &#124; None; default None` |
| `rewards` · 65 | Reward per urutan/action/kelompok sesuai fungsi; berbeda dari peluang token. | `self.token_rewards(idx)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 10 |
| `torch` | `torch` | 12 |
| `nn` | `torch.nn` | 13 |
| `_torch` | `torch` | 15 |
| `Transformer` | `src.models.transformer.Transformer` | 17 |
| `build_model_from_config` | `src.post_training.utils.build_model_from_config` | 18 |
| `gather_last` | `src.post_training.utils.gather_last` | 18 |
| `unwrap` | `src.post_training.utils.unwrap` | 18 |
