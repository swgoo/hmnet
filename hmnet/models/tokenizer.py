import numpy as np
import torch


class ByteTokenizer:
    def __init__(self, special_tokens: dict[str, int] | None = None):
        super().__init__()
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.pad_idx = 0
        self.special_tokens = special_tokens or {}
        self.dtype = np.uint8

        for idx in [self.bos_idx, self.eos_idx] + list(self.special_tokens.values()):
            assert (
                0xF8 <= idx <= 0xFF
            ), f"Special token index {idx} is not in the safe range (0xF8~0xFF) for UTF-8."

    def encode(self, seqs: list[str], add_bos=False, add_eos=False, **kwargs):
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")
            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)

    def add_special_tokens(self, tokens: np.ndarray | torch.Tensor, bos=True, eos=True):
        if bos:
            if isinstance(tokens, np.ndarray):
                tokens = np.insert(tokens, 0, self.bos_idx)
            else:
                tokens = torch.cat([torch.tensor([self.bos_idx]), tokens])
        if eos:
            if isinstance(tokens, np.ndarray):
                tokens = np.append(tokens, self.eos_idx)
            else:
                tokens = torch.cat([tokens, torch.tensor([self.eos_idx])])
        return tokens
