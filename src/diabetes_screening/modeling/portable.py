"""Load fitted pipelines on any platform, including XGBoost models pickled on another operating system.

XGBoost pickles a booster as a "memory snapshot" with two parts: "Model" (every tree, the base score
and the objective, in XGBoost's portable model format) and "Config" (training settings). The config
includes the random-number generator's state as text, and that text layout depends on the C++
standard library: Linux builds write plain decimal numbers, Windows builds expect a different layout.
So Windows XGBoost refuses a snapshot pickled on Linux with "input stream corrupted", although its
trees are fine. When that happens we load only the "Model" part. Predictions are unchanged, because
they depend only on the model part, and refitting uses the scikit-learn settings pickled alongside it.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/evaluation.md.
"""
import struct

import joblib
from xgboost import Booster
from xgboost.core import XGBoostError

_SIZES = {"i": 1, "U": 1, "I": 2, "l": 4, "L": 8, "d": 4, "D": 8, "C": 1}
_INTEGERS = {"i": ">b", "U": ">B", "I": ">h", "l": ">i", "L": ">q"}


def _integer(raw, pos):
    """A UBJSON length: type marker, then a big-endian integer. Returns (value, next position)."""
    marker = chr(raw[pos])
    if marker not in _INTEGERS:
        raise ValueError(f"Expected a UBJSON length at byte {pos}")
    end = pos + 1 + _SIZES[marker]
    return struct.unpack(_INTEGERS[marker], raw[pos + 1:end])[0], end


def _skip(raw, pos, marker=None):
    """Position just after the UBJSON value at `pos`; pass `marker` when the container already gave the type."""
    if marker is None:
        marker, pos = chr(raw[pos]), pos + 1
    if marker in "ZTFN":
        return pos
    if marker in _SIZES:
        return pos + _SIZES[marker]
    if marker in "SH":
        length, pos = _integer(raw, pos)
        return pos + length
    if marker not in "{[":
        raise ValueError(f"Unknown UBJSON marker {marker!r} at byte {pos - 1}")
    item, count = None, None
    if chr(raw[pos]) == "$":  # every item has this type, so items carry no marker of their own
        item, pos = chr(raw[pos + 1]), pos + 2
    if chr(raw[pos]) == "#":  # item count up front, so there is no closing bracket
        count, pos = _integer(raw, pos + 1)
    if marker == "[" and count is not None and item in _SIZES:
        return pos + count * _SIZES[item]  # a packed array of numbers
    close, seen = "}" if marker == "{" else "]", 0
    while (seen < count) if count is not None else (chr(raw[pos]) != close):
        if marker == "{":
            length, pos = _integer(raw, pos)
            pos += length  # skip the key
        pos = _skip(raw, pos, item)
        seen += 1
    return pos if count is not None else pos + 1


def snapshot_model(raw):
    """The "Model" member of an XGBoost memory snapshot, as standalone UBJSON that `load_model` accepts."""
    raw = bytes(raw)
    if raw[:1] != b"{":
        raise ValueError("Not a UBJSON XGBoost snapshot")
    pos = 1
    while chr(raw[pos]) != "}":
        length, pos = _integer(raw, pos)
        key, pos = raw[pos:pos + length], pos + length
        end = _skip(raw, pos)
        if key == b"Model":
            return raw[pos:end]
        pos = end
    raise ValueError("The snapshot has no Model member")


def _restore_model_only(booster, state):
    state = dict(state)
    handle = state.pop("handle")
    Booster.__init__(booster)
    if handle is not None:
        booster.load_model(bytearray(snapshot_model(handle)))
    booster.__dict__.update(state)


def load_pipeline(path):
    """`joblib.load`, falling back to each XGBoost booster's portable model part if its snapshot won't load."""
    try:
        return joblib.load(path)
    except XGBoostError:
        original = Booster.__setstate__
        Booster.__setstate__ = _restore_model_only
        try:
            return joblib.load(path)
        finally:
            Booster.__setstate__ = original
