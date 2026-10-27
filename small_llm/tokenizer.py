"""Tokenizer byte: setiap nilai UTF-8 adalah satu token."""

VOCAB_SIZE = 257
END_OF_STORY = 256


def encode(text: str, *, end_of_story: bool = False) -> list[int]:
    ids = list(text.encode("utf-8"))
    if end_of_story:
        ids.append(END_OF_STORY)
    return ids


def decode(ids: list[int]) -> str:
    """Berhenti di penanda cerita; ganti byte UTF-8 yang belum lengkap."""
    content = []
    for token in ids:
        if token == END_OF_STORY:
            break
        if not 0 <= token < 256:
            raise ValueError(f"Token tidak dikenal: {token}")
        content.append(token)
    return bytes(content).decode("utf-8", errors="replace")

