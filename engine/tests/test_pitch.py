"""Pitch mapping and geometry unit tests (§7, §57)."""
from synthesia.model import midi_to_name, name_to_midi, is_black_key


def test_midi_name_roundtrip():
    for m in range(21, 109):
        assert name_to_midi(midi_to_name(m)) == m


def test_known_pitches():
    assert midi_to_name(60) == "C4"
    assert midi_to_name(69) == "A4"
    assert name_to_midi("C4") == 60
    assert name_to_midi("A0") == 21
    assert name_to_midi("C8") == 108


def test_black_keys():
    assert is_black_key(name_to_midi("C#4"))
    assert is_black_key(name_to_midi("A#3"))
    assert not is_black_key(name_to_midi("C4"))
    assert not is_black_key(name_to_midi("E4"))
    assert not is_black_key(name_to_midi("B4"))
