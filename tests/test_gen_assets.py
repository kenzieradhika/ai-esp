"""The GPT-2 byte<->unicode alphabet in gen_assets must be a bijection over all
256 byte values. The firmware's `bpe.h` byte map is built from it, so a missing
or duplicated entry would silently mis-tokenize on the device.
"""

from gen_assets import bytes_to_unicode


def test_covers_every_byte_exactly_once():
    b2u = bytes_to_unicode()
    assert len(b2u) == 256
    assert sorted(b2u.keys()) == list(range(256))
    assert len(set(b2u.values())) == 256  # injective


def test_identity_on_printable_ascii():
    b2u = bytes_to_unicode()
    for b in range(ord("!"), ord("~") + 1):
        assert b2u[b] == chr(b)


def test_control_bytes_shift_above_256():
    b2u = bytes_to_unicode()
    # Bytes not in the printable/Latin-1 ranges get code points >= 256, in
    # first-seen order, so the alphabet is contiguous and reversible.
    assert ord(b2u[0]) >= 256
    assert ord(b2u[10]) >= 256
    assert ord(b2u[32]) >= 256


def test_latin1_extras_are_identity():
    b2u = bytes_to_unicode()
    # 0xA1..0xAC and 0xAE..0xFF map to themselves (in the Latin-1 plane).
    assert b2u[0xA1] == chr(0xA1)
    assert b2u[0xFF] == chr(0xFF)
