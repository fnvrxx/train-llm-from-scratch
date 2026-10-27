# src/post_training/chat_template.py

[Indeks kamus](README.md) · [Kode sumber](../../../src/post_training/chat_template.py)

## Fungsi dan kelas

### `get_tokenizer` (baris 53)

Mengembalikan tokenizer r50k_base yang di-cache agar cukup dibuat sekali.

```python
def get_tokenizer() -> 'tiktoken.Encoding':
```

Keluaran eksplisit: `tiktoken.get_encoding('r50k_base')`.

### `_encode_ordinary` (baris 58)

Mengubah teks biasa menjadi ID tanpa memproses literal special token.

```python
def _encode_ordinary(text: str) -> list[int]:
```

Keluaran eksplisit: `get_tokenizer().encode_ordinary(text)`.

### `_header_for` (baris 63)

Memilih header user/assistant/system, atau menolak role tidak dikenal.

```python
def _header_for(role: str) -> str:
```

Keluaran eksplisit: `USER_HEADER`; `ASSISTANT_HEADER`; `SYSTEM_HEADER`.

### `render_chat` (baris 73)

Menggabungkan role, isi, dan EOT menjadi teks tampilan/debugging.

```python
def render_chat(messages: Iterable[dict], add_generation_prompt: bool=False) -> str:
```

Keluaran eksplisit: `''.join(parts)`.

### `encode_chat` (baris 95)

Menghasilkan ID dan mask sejajar; isi assistant dan EOT-nya mask satu, bagian lain nol.

```python
def encode_chat(messages: Iterable[dict], add_generation_prompt: bool=False) -> tuple[list[int], list[int]]:
```

Keluaran eksplisit: `(ids, mask)`.

### `encode_prompt` (baris 139)

Menghasilkan ID chat dengan header assistant di ujung untuk memulai generation.

```python
def encode_prompt(messages: Iterable[dict]) -> list[int]:
```

Keluaran eksplisit: `ids`.

### `decode` (baris 145)

Mengubah ID token biasa menjadi teks setelah menyaring EOT dan ID di luar rentang.

```python
def decode(ids: Iterable[int]) -> str:
```

Keluaran eksplisit: `get_tokenizer().decode(clean)`.

## Variabel dan parameter menurut lingkup

### Lingkup `modul`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `EOT_ID` · 39 | Token akhir teks 50256; juga dipakai sebagai stop dan padding post-training. | `50256` |
| `USER_HEADER` · 42 | Header role berbentuk teks biasa, bukan special token baru. | `'<&#124;user&#124;>\n'` |
| `ASSISTANT_HEADER` · 43 | Header role berbentuk teks biasa, bukan special token baru. | `'<&#124;assistant&#124;>\n'` |
| `SYSTEM_HEADER` · 44 | Header role berbentuk teks biasa, bukan special token baru. | `'<&#124;system&#124;>\n'` |
| `THINK_OPEN` · 48 | String tag pembuka/penutup reasoning dan jawaban, ditokenisasi sebagai teks biasa. | `elemen 0 dari ('<think>', '</think>')` |
| `THINK_CLOSE` · 48 | String tag pembuka/penutup reasoning dan jawaban, ditokenisasi sebagai teks biasa. | `elemen 1 dari ('<think>', '</think>')` |
| `ANSWER_OPEN` · 49 | String tag pembuka/penutup reasoning dan jawaban, ditokenisasi sebagai teks biasa. | `elemen 0 dari ('<answer>', '</answer>')` |
| `ANSWER_CLOSE` · 49 | String tag pembuka/penutup reasoning dan jawaban, ditokenisasi sebagai teks biasa. | `elemen 1 dari ('<answer>', '</answer>')` |

### Lingkup `_encode_ordinary`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `text` · 58 | Teks system/pengguna/dokumen yang diproses sesuai fungsi. | `argumen str` |

### Lingkup `_header_for`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `role` · 63 | Identitas pembicara user, assistant, atau system. | `argumen str` |

### Lingkup `render_chat`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `messages` · 73 | Daftar pesan percakapan berupa dictionary role/content. | `argumen Iterable[dict]` |
| `add_generation_prompt` · 73 | Tambahkan header assistant di ujung prompt untuk memulai generation. | `argumen bool; default False` |
| `parts` · 85 | Potongan teks sebelum digabung menjadi tampilan chat. | `[]` |
| `m` · 86 | Satu pesan role/content yang akan dirender menjadi teks. | `iterasi messages` |

### Lingkup `encode_chat`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `messages` · 96 | Daftar pesan percakapan berupa dictionary role/content. | `argumen Iterable[dict]` |
| `add_generation_prompt` · 97 | Tambahkan header assistant di ujung prompt untuk memulai generation. | `argumen bool; default False` |
| `ids` · 110 | Daftar ID chat lengkap yang dibangun header, isi, dan EOT setiap pesan. | `[]` |
| `mask` · 111 | Daftar angka 0/1 sejajar ID: target assistant dan EOT-nya bernilai satu. | `[]` |
| `m` · 113 | Satu dictionary pesan role/content dalam messages. | `iterasi messages` |
| `role` · 114 | Identitas pembicara user, assistant, atau system. | `m['role']` |
| `header_ids` · 117, 132 | ID token teks header role; target header diberi mask nol. | `_encode_ordinary(_header_for(role))`; `_encode_ordinary(ASSISTANT_HEADER)` |
| `content_ids` · 121 | ID token isi pesan; mask bergantung apakah role assistant. | `_encode_ordinary(m['content'])` |
| `is_completion` · 122 | Apakah pesan berasal dari assistant sehingga isi dan EOT-nya diberi mask satu. | `role == 'assistant'` |

### Lingkup `encode_prompt`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `messages` · 139 | Daftar pesan percakapan berupa dictionary role/content. | `argumen Iterable[dict]` |
| `ids` · 141 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `elemen 0 dari encode_chat(messages, add_generation_prompt=True)` |
| `_` · 141 | Hasil/indeks yang sengaja tidak dipakai pada bagian kode ini. | `elemen 1 dari encode_chat(messages, add_generation_prompt=True)` |

### Lingkup `decode`

| Nama dan baris | Arti | Asal nilai / deklarasi |
|---|---|---|
| `ids` · 145 | Token atau indeks masukan; kolom asal membedakan ID kosakata dari indeks pemilihan record. | `argumen Iterable[int]` |
| `clean` · 153 | ID yang telah difilter agar hanya token biasa valid yang didekode. | `[t for t in ids if 0 <= t < EOT_ID]` |
| `t` · 153 | Satu ID token; hanya ID 0 sampai 50255 dimasukkan ke clean. | `iterasi ids` |

## Alias impor

Alias mengacu pustaka atau simbol dari berkas lain. Impor tidak otomatis menjadi bobot yang dilatih.

| Nama lokal | Sumber | Baris |
|---|---|---:|
| `annotations` | `__future__.annotations` | 31 |
| `lru_cache` | `functools.lru_cache` | 33 |
| `Iterable` | `typing.Iterable` | 34 |
| `tiktoken` | `tiktoken` | 36 |
