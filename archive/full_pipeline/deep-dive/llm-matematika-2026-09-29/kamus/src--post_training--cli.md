# src/post_training/cli.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/cli.py)

## Fungsi dan kelas

### `_add_typed` (baris 16)

Membuat opsi argparse berdasarkan nama dan anotasi tipe field.

```python
def _add_typed(p: argparse.ArgumentParser, name: str, type_str) -> None:
```

Tidak mengembalikan nilai secara eksplisit. Fungsi mengatur objek, mengubah state, menulis berkas, atau menjalankan loop seperti dijelaskan di atas.

### `parse_config` (baris 29)

Mengubah CLI menjadi instance dataclass serta namespace argumen tambahan.

```python
def parse_config(cfg_cls, extra: dict | None=None):
```

Keluaran eksplisit: `(cfg_cls(**overrides), SimpleNamespace(**extras))`.

### `parse_config_with_json` (baris 50)

Menggabungkan default, base JSON, stage JSON, dan CLI; print-config menampilkan hasil lalu keluar.

```python
def parse_config_with_json(cfg_cls, default_json: str, extra: dict | None=None):
```

Keluaran eksplisit: `(cfg, SimpleNamespace(**extras))`.

## Variabel dan parameter menurut lingkup

### Lingkup `_add_typed`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `p` · 16 | ArgumentParser yang ditambahi opsi field. | `argumen argparse.ArgumentParser` |
| `name` · 16 | Nama parameter model, field konfigurasi, atau berkas yang sedang diperiksa. | `argumen str` |
| `type_str` · 16 | Nama flag, tipe, opsi argparse, atau nama CLI yang dikecualikan dari field konfigurasi. | `argumen tanpa anotasi tipe` |
| `t` · 17 | Representasi string anotasi tipe field untuk memilih parser int/float/bool/string. | `str(type_str)` |

### Lingkup `_add_typed.lambda@23`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `s` · 23 | Teks argumen CLI yang dianggap true jika lower-case-nya 1, true, atau yes. | `argumen lambda: s.lower() in ('1', 'true', 'yes')` |

### Lingkup `parse_config`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg_cls` · 29 | Kelas dataclass konfigurasi yang akan dibuat instance-nya. | `argumen tanpa anotasi tipe` |
| `extra` · 29 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `argumen dict &#124; None; default None` |
| `p` · 37 | Parser opsi terminal yang sedang ditambahkan field atau dibaca hasilnya. | `argparse.ArgumentParser()` |
| `field_names` · 38 | Nama field yang sah pada kelas konfigurasi. | `{f.name for f in fields(cfg_cls)}` |
| `f` · 38, 39 | Satu field dataclass konfigurasi, dengan atribut name dan type. | `iterasi fields(cfg_cls)` |
| `flag` · 41 | Nama flag, tipe, opsi argparse, atau nama CLI yang dikecualikan dari field konfigurasi. | `elemen 0 dari iterasi (extra or {}).items()` |
| `kwargs` · 41 | Nama flag, tipe, opsi argparse, atau nama CLI yang dikecualikan dari field konfigurasi. | `elemen 1 dari iterasi (extra or {}).items()` |
| `args` · 43 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `vars(p.parse_args())` |
| `overrides` · 45 | Nilai pengganti konfigurasi dengan prioritas di atas default/JSON. | `{k: v for k, v in args.items() if k in field_names and v is not None}` |
| `k` · 45, 46 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `elemen 0 dari iterasi args.items()` |
| `v` · 45, 46 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi args.items()` |
| `extras` · 46 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `{k: v for k, v in args.items() if k not in field_names}` |

### Lingkup `parse_config_with_json`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `cfg_cls` · 50 | Kelas dataclass konfigurasi yang akan dibuat instance-nya. | `argumen tanpa anotasi tipe` |
| `default_json` · 50 | Path konfigurasi tahap/base atau hasil resolusi base.json terdekat. | `argumen str` |
| `extra` · 50 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `argumen dict &#124; None; default None` |
| `p` · 63 | Parser opsi terminal yang sedang ditambahkan field atau dibaca hasilnya. | `argparse.ArgumentParser()` |
| `field_names` · 64 | Nama field yang sah pada kelas konfigurasi. | `{f.name for f in fields(cfg_cls)}` |
| `f` · 64, 65 | Satu field dataclass konfigurasi, dengan atribut name dan type. | `iterasi fields(cfg_cls)` |
| `flag` · 69 | Nama flag, tipe, opsi argparse, atau nama CLI yang dikecualikan dari field konfigurasi. | `elemen 0 dari iterasi (extra or {}).items()` |
| `kwargs` · 69 | Nama flag, tipe, opsi argparse, atau nama CLI yang dikecualikan dari field konfigurasi. | `elemen 1 dari iterasi (extra or {}).items()` |
| `args` · 71 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `vars(p.parse_args())` |
| `reserved` · 73 | Nama flag, tipe, opsi argparse, atau nama CLI yang dikecualikan dari field konfigurasi. | `{'config', 'print_config'}` |
| `overrides` · 74 | Nilai pengganti konfigurasi dengan prioritas di atas default/JSON. | `{k: v for k, v in args.items() if k in field_names and v is not None}` |
| `k` · 74, 75 | Kunci dictionary yang sedang diiterasi, seperti nama bobot atau field konfigurasi. | `elemen 0 dari iterasi args.items()` |
| `v` · 74, 75 | Nilai dictionary untuk kunci terkait, seperti tensor bobot atau nilai konfigurasi. | `elemen 1 dari iterasi args.items()` |
| `extras` · 75 | Opsi terminal atau definisi argumen tambahan; extra pada save adalah metadata tambahan checkpoint. | `{k: v for k, v in args.items() if k not in field_names and k not in reserved}` |
| `cfg` · 77 | Pengaturan arsitektur, runtime, data, dan training; dapat berupa dataclass atau dictionary. | `load_config(cfg_cls, args['config'], overrides)` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 6 |
| `argparse` | `argparse` | 8 |
| `json` | `json` | 9 |
| `asdict` | `dataclasses.asdict` | 10 |
| `fields` | `dataclasses.fields` | 10 |
| `SimpleNamespace` | `types.SimpleNamespace` | 11 |
| `load_config` | `config.loader.load_config` | 13 |
