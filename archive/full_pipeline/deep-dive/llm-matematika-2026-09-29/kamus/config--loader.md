# config/loader.py

[Indeks kamus](README.md) · [Kode sumber](../../../config/loader.py)

## Fungsi dan kelas

### `_deep_merge` (baris 28)

Menggabungkan dictionary rekursif dengan nilai sumber menimpa tujuan.

```python
def _deep_merge(dst: dict, src: dict) -> dict:
```

Keluaran eksplisit: `dst`.

### `_resolve_base` (baris 38)

Memilih base.json eksplisit, sibling JSON tahap, atau default configs/base.json.

```python
def _resolve_base(json_path: str | None, base_path: str | None) -> str:
```

Keluaran eksplisit: `base_path`; `sibling`; `'configs/base.json'`.

### `load_config` (baris 48)

Memuat JSON dan override, membuang field yang tidak dikenal, lalu membuat dataclass tahap.

```python
def load_config(cfg_cls, json_path: str | None=None, overrides: dict[str, Any] | None=None, *, base_path: str | None=None):
```

Keluaran eksplisit: `cfg_cls(**merged)`.

## Variabel dan parameter menurut lingkup

### Lingkup `_deep_merge`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `dst` · 28 | Dictionary tujuan/sumber pada merge rekursif konfigurasi. | `argumen dict` |
| `src` · 28 | Dictionary tujuan/sumber pada merge rekursif konfigurasi. | `argumen dict` |
| `k` · 30 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `elemen 0 dari iterasi src.items()` |
| `v` · 30 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi src.items()` |

### Lingkup `_resolve_base`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `json_path` · 38 | Path konfigurasi tahap/base atau hasil resolusi base.json terdekat. | `argumen str &#124; None` |
| `base_path` · 38 | Path konfigurasi tahap/base atau hasil resolusi base.json terdekat. | `argumen str &#124; None` |
| `sibling` · 42 | Path konfigurasi tahap/base atau hasil resolusi base.json terdekat. | `os.path.join(os.path.dirname(json_path), 'base.json')` |

### Lingkup `load_config`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg_cls` · 49 | Kelas dataclass konfigurasi yang akan dibuat instance-nya. | `argumen tanpa anotasi tipe` |
| `json_path` · 50 | Path konfigurasi tahap/base atau hasil resolusi base.json terdekat. | `argumen str &#124; None; default None` |
| `overrides` · 51 | Nilai pengganti konfigurasi dengan prioritas di atas default/JSON. | `argumen dict[str, Any] &#124; None; default None` |
| `base_path` · 53 | Path konfigurasi tahap/base atau hasil resolusi base.json terdekat. | `argumen str &#124; None; default None` |
| `base` · 69 | Path konfigurasi tahap/base atau hasil resolusi base.json terdekat. | `_resolve_base(json_path, base_path)` |
| `merged` · 70 | Dictionary hasil penggabungan konfigurasi sebelum konstruksi dataclass. | `{}` |
| `path` · 71 | Path/nama artefak yang dibaca atau ditulis; lihat jenis fungsi dan ekspresi asal. | `iterasi (base, json_path)` |
| `fh` · 73 | Handle berkas yang dibuka oleh with; mode r membaca, w menulis ulang, a menambahkan, dan b berarti byte. | `open(path)` |
| `field_names` · 76 | Nama field yang sah pada kelas konfigurasi. | `{f.name for f in fields(cfg_cls)}` |
| `f` · 76 | Satu field dataclass konfigurasi, dengan atribut name dan type. | `iterasi fields(cfg_cls)` |
| `key` · 77 | Linear dari n_embed ke head_size untuk membentuk keys; key pada utilitas cfg adalah nama field yang dicari. | `iterasi list(merged)` |
| `k` · 83 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `elemen 0 dari iterasi overrides.items()` |
| `v` · 83 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi overrides.items()` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 20 |
| `json` | `json` | 22 |
| `os` | `os` | 23 |
| `fields` | `dataclasses.fields` | 24 |
| `Any` | `typing.Any` | 25 |
