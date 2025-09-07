import numpy as np
import torch
from torch import Tensor, tensor


class ByteTokenizer:
    def __init__(self, special_tokens: dict[str, int] | None = None):
        super().__init__()
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.pad_idx = 0
        self.special_tokens = special_tokens or {}
        self.dtype = torch.uint8

        for idx in [self.bos_idx, self.eos_idx] + list(self.special_tokens.values()):
            assert (
                0xF8 <= idx <= 0xFF
            ), f"Special token index {idx} is not in the safe range (0xF8~0xFF) for UTF-8."

    def encode(self, seqs: list[str], add_bos=False, add_eos=False) -> list[Tensor]:
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")
            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = tensor(text_byte, dtype=self.dtype)

            total_outputs.append(text_byte_ids)

        return total_outputs

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()
        if "errors" not in kwargs:
            kwargs["errors"] = "replace"
        return bytearray(tokens).decode("utf-8", **kwargs)

    def add_special_tokens(
        self, tokens: torch.Tensor, bos=True, eos=True
    ) -> torch.Tensor:
        if bos:
            tokens = torch.cat([torch.tensor([self.bos_idx]), tokens])
        if eos:
            tokens = torch.cat([tokens, torch.tensor([self.eos_idx])])
        return tokens
