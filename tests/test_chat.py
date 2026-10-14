"""Host chat REPL continuation logic (`src/chat.py:reply`).

The REPL must show only what the model generated *after* the prompt, not the
prompt echoed back. That trimming is the only non-trivial logic in the module,
and it is easy to get wrong when the tokenizer round-trip is not byte-exact.
"""

import torch

from chat import is_exit, reply


class FakeTok:
    """ASCII stand-in: encode -> ord ids, decode -> chr join (exact inverse)."""

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return type("E", (), {"ids": [ord(c) for c in text]})()

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class EchoModel:
    """Generate that appends a fixed continuation to the prompt it was given."""

    def __init__(self, continuation, echo_prompt=True):
        self.continuation = continuation
        self.echo_prompt = echo_prompt
        self.cfg = type("C", (), {"seq_len": 64})()

    def generate(self, ids, max_new_tokens, temperature, top_k):
        prompt = "".join(chr(i) for i in ids[0].tolist())
        tail = (prompt if self.echo_prompt else "") + self.continuation
        return torch.tensor([[ord(c) for c in tail]])


def test_reply_strips_the_prompt_echo():
    m = EchoModel(" dunia!")
    out = reply(m, FakeTok(), "halo", 5, 0.8, 40, "cpu")
    assert out == "dunia!"


def test_reply_passes_the_chat_prompt_format():
    tok = FakeTok()
    reply(EchoModel("x"), tok, "apa kabar", 5, 0.8, 40, "cpu")
    assert tok.calls == ["user: apa kabar\nJawaban:"]


def test_reply_without_echo_returns_full_text():
    # Some checkpoints may not reproduce the prompt exactly; the fallback must
    # return the model's text rather than an empty string.
    out = reply(EchoModel("jawaban", echo_prompt=False), FakeTok(), "halo", 5, 0.8, 40, "cpu")
    assert out == "jawaban"


def test_is_exit_is_case_insensitive():
    # The firmware matches its command words case-insensitively; the REPL should
    # not make the user guess the exact case to quit.
    assert is_exit("keluar") and is_exit("Keluar") and is_exit("QUIT")
    assert is_exit("exit") and is_exit("Exit")


def test_is_exit_ignores_surrounding_space():
    assert is_exit("  keluar  ")
    assert not is_exit("keluar nanti")


def test_is_exit_treats_blank_line_as_stop():
    # main() breaks on an empty line so a stray Enter ends the session.
    assert is_exit("") and is_exit("   ")

