"""Piano-music backend for the bitstring program-synthesis harness.

Encodes few-shot bitstring tasks as solo-piano "PNO" text — the native training
format of the music50m spark-gpt byte model (vocab 259 = 256 raw bytes +
BOS/EOS/PAD, so token id == byte value and no tokenizer is needed).

PNO grammar (matches music-gpt/midi_codec.py exactly):
  wait char  = ALPH[t-1] for t in 1..31 ten-ms ticks (ALPH[31] = 2-char escape)
  note       = ALPH[32+vel_bucket] + ALPH[pitch-21] + ALPH[dur_bucket]
where ALPH = printable ASCII minus '"' and '\\' (92 chars).

Trial layout (one PNO document per trial):
  <bos> in_1 SEPIO out_1 SEPEX in_2 SEPIO out_2 SEPEX ... query SEPIO [generate]
Bits ride on per-trial randomized note attributes (pitch pair by default);
separators are long bass notes (SEPEX one octave below SEPIO), which read as
phrase/cadence gestures in the training distribution.

Decode modes:
  constrained: teacher-force the canonical note bytes, read logits only at the
    discriminating byte, argmax over the candidate byte ids. 100% valid.
  greedy: free-run byte argmax, tolerant parse, nearest-candidate assignment.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from dsl import few_shot

ALPH = [chr(c) for c in range(33, 127) if chr(c) not in ('"', "\\")]
assert len(ALPH) == 92
VEL_BASE, VEL_N = 32, 16
PITCH_MIN = 21
MUSIC_BOS_ID = 256
MUSIC_EOS_ID = 257
MUSIC_CONTEXT_LIMIT = 4090  # model max_position_embeddings is 4096
MUSIC_ENCODINGS = ("pitch", "octave", "rhythm", "2bit-pitch", "slice-pitch")

# slice-pitch (for ROLL/time-slice models, e.g. Musicroll-50M): one bitstring =
# one 8-slice line "|C4|A4|...|"; bits ride on two natural letters in a fixed
# octave, so candidates differ only at the letter byte. Separator lines: bass
# note held 4 slices + 4 silent (io), or two held bass notes (ex).
SLICE_NATURALS = "CDEFGAB"
_SLICE_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# melodic register for bit notes; separators live an octave-plus below
_BIT_PITCH_LO, _BIT_PITCH_HI = 55, 84  # G3..C6
_SEP_PITCH_LO, _SEP_PITCH_HI = 36, 48  # C2..C3


def wait_str(ticks: int) -> str:
    out = []
    while ticks > 0:
        if ticks <= 31:
            out.append(ALPH[ticks - 1])
            ticks = 0
        else:
            v = min(ticks, 92 * 92 - 1)
            out.append(ALPH[31] + ALPH[v // 92] + ALPH[v % 92])
            ticks -= v
    return "".join(out)


def note_str(vel_bucket: int, pitch: int, dur_bucket: int) -> str:
    return ALPH[VEL_BASE + vel_bucket] + ALPH[pitch - PITCH_MIN] + ALPH[dur_bucket]


def sample_music_params(music_encoding: str, rng) -> Dict[str, Any]:
    """Per-trial randomized symbol assignment (the music analogue of ImageGPT's
    random color tokens)."""
    if music_encoding not in MUSIC_ENCODINGS:
        raise ValueError(f"Unsupported music encoding: {music_encoding}")
    vel = rng.randint(7, 11)
    dur = rng.randint(38, 48)
    wait = rng.randint(12, 25)
    sep_pitch = rng.randint(_SEP_PITCH_LO, _SEP_PITCH_HI)
    params: Dict[str, Any] = {
        "encoding": music_encoding,
        "vel": vel,
        "dur": dur,
        "wait": wait,
        "sep_io_pitch": sep_pitch,
        "sep_ex_pitch": sep_pitch - 12,
        "sep_dur": rng.randint(55, 65),
        "bits_per_note": 2 if music_encoding == "2bit-pitch" else 1,
    }
    if music_encoding == "pitch":
        p0 = rng.randint(_BIT_PITCH_LO, _BIT_PITCH_HI - 9)
        p1 = p0 + rng.choice([2, 3, 4, 5, 7, 9])
        pitches = [p0, p1]
        rng.shuffle(pitches)
        params["symbol_pitches"] = {"0": pitches[0], "1": pitches[1]}
    elif music_encoding == "octave":
        p0 = rng.randint(_BIT_PITCH_LO, _BIT_PITCH_HI - 12)
        pitches = [p0, p0 + 12]
        rng.shuffle(pitches)
        params["symbol_pitches"] = {"0": pitches[0], "1": pitches[1]}
    elif music_encoding == "2bit-pitch":
        while True:
            pitches = sorted(rng.sample(range(_BIT_PITCH_LO, _BIT_PITCH_HI + 1), 4))
            if all(b - a >= 2 for a, b in zip(pitches, pitches[1:])):
                break
        rng.shuffle(pitches)
        params["symbol_pitches"] = {
            "00": pitches[0], "01": pitches[1], "10": pitches[2], "11": pitches[3],
        }
    elif music_encoding == "rhythm":
        params["pitch"] = rng.randint(_BIT_PITCH_LO, _BIT_PITCH_HI)
        d_short = rng.randint(25, 32)
        durs = [d_short, d_short + rng.randint(18, 25)]
        rng.shuffle(durs)
        params["symbol_durs"] = {"0": durs[0], "1": durs[1]}
    elif music_encoding == "slice-pitch":
        letters = rng.sample(SLICE_NATURALS, 2)
        params["slice_letters"] = {"0": letters[0], "1": letters[1]}
        params["slice_octave"] = rng.randint(3, 5)
        params["slice_sep_letter"] = rng.choice(SLICE_NATURALS)
        params["slice_sep_octave"] = rng.randint(0, 1)
    return params


def _slice_bits_line(bits: str, params: Dict[str, Any]) -> str:
    toks = [params["slice_letters"][b] + str(params["slice_octave"]) for b in bits]
    return "|" + "|".join(toks) + "|"


def _slice_sep_line(params: Dict[str, Any], kind: str) -> str:
    on = params["slice_sep_letter"].upper() + str(params["slice_sep_octave"])
    held = params["slice_sep_letter"].lower() + str(params["slice_sep_octave"])
    if kind == "io":
        slices = [on, held, held, held, "", "", "", ""]
    else:
        slices = [on, held, held, held, on, held, held, held]
    return "|" + "|".join(slices) + "|"


def _bit_symbols(bits: str, params: Dict[str, Any]) -> List[str]:
    """Split a bitstring into per-note symbols ('0'/'1', or 2-bit chunks)."""
    step = params["bits_per_note"]
    if len(bits) % step != 0:
        raise ValueError(f"bit length {len(bits)} not divisible by {step}")
    return [bits[i:i + step] for i in range(0, len(bits), step)]


def bits_to_pno(bits: str, params: Dict[str, Any], *, lead_wait: bool) -> str:
    """Encode a bitstring as PNO notes. lead_wait=False only for the very first
    note of the document (training chunks start with a note at t=0)."""
    w = wait_str(params["wait"])
    parts = []
    for i, sym in enumerate(_bit_symbols(bits, params)):
        parts.append(w if (i > 0 or lead_wait) else "")
        if params["encoding"] == "rhythm":
            parts.append(note_str(params["vel"], params["pitch"], params["symbol_durs"][sym]))
        else:
            parts.append(note_str(params["vel"], params["symbol_pitches"][sym], params["dur"]))
    return "".join(parts)


def sep_pno(params: Dict[str, Any], kind: str) -> str:
    pitch = params["sep_io_pitch"] if kind == "io" else params["sep_ex_pitch"]
    return wait_str(params["wait"]) + note_str(params["vel"], pitch, params["sep_dur"])


def make_music_trial_spec(
    program: Sequence[str],
    *,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    music_encoding: str,
    ablate_labels: bool,
    verbose: bool,
    permute_fn=None,
) -> Dict[str, Any]:
    if max_new_tokens != bit_length:
        raise ValueError("Music backend expects max_new_tokens == bit_length")
    params = sample_music_params(music_encoding, random)
    if bit_length % params["bits_per_note"] != 0:
        raise ValueError("2bit-pitch music encoding requires an even bit_length")

    bit_prompt, query_input, expected_output, _, query_input_raw = few_shot(
        program,
        in_context_examples,
        "\n",
        "->",
        bit_length=bit_length,
    )
    lines = bit_prompt.split("\n") if bit_prompt else []
    context_lines = lines[:-1] if lines else []
    context_examples: List[Tuple[str, str]] = []
    for line in context_lines:
        if "->" not in line:
            raise ValueError(f"Malformed few-shot line for music backend: {line!r}")
        input_bits, output_bits = line.split("->", 1)
        if len(input_bits) != bit_length or len(output_bits) != bit_length:
            raise ValueError("Music backend expected fixed-length context bitstrings")
        context_examples.append((input_bits, output_bits))

    if ablate_labels and len(context_examples) > 1:
        if permute_fn is None:
            raise ValueError("ablate_labels requires a permute_fn")
        inputs = [inp for inp, _ in context_examples]
        outputs = permute_fn([out for _, out in context_examples], random)
        context_examples = list(zip(inputs, outputs))

    if music_encoding == "slice-pitch":
        if bit_length != 8:
            raise ValueError("slice-pitch encoding requires bit_length == 8 "
                             "(one 8-slice line per bitstring)")
        lines: List[str] = []
        for input_bits, output_bits in context_examples:
            lines.append(_slice_bits_line(input_bits, params))
            lines.append(_slice_sep_line(params, "io"))
            lines.append(_slice_bits_line(output_bits, params))
            lines.append(_slice_sep_line(params, "ex"))
        lines.append(_slice_bits_line(query_input, params))
        lines.append(_slice_sep_line(params, "io"))
        prompt_text = "\n".join(lines)
        gen_budget = 2 + bit_length * 3
    else:
        parts: List[str] = []
        for idx, (input_bits, output_bits) in enumerate(context_examples):
            parts.append(bits_to_pno(input_bits, params, lead_wait=idx > 0))
            parts.append(sep_pno(params, "io"))
            parts.append(bits_to_pno(output_bits, params, lead_wait=True))
            parts.append(sep_pno(params, "ex"))
        parts.append(bits_to_pno(query_input, params, lead_wait=bool(context_examples)))
        parts.append(sep_pno(params, "io"))
        prompt_text = "".join(parts)
        gen_budget = (bit_length // params["bits_per_note"]) * (len(wait_str(params["wait"])) + 3)
    if 1 + len(prompt_text) + gen_budget > MUSIC_CONTEXT_LIMIT:
        raise ValueError(
            f"Music prompt exceeds model context: prompt={len(prompt_text)} bytes "
            f"+ generation={gen_budget} > {MUSIC_CONTEXT_LIMIT}"
        )

    spec: Dict[str, Any] = {
        "prompt": prompt_text,
        "query": query_input,
        "expected": expected_output,
        "query_unaltered": query_input_raw,
        "music_params": params,
    }
    if verbose:
        spec["few_shot_examples"] = [
            {"input": inp, "output": out} for inp, out in context_examples
        ]
    return spec


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def music_prompt_ids(text: str) -> List[int]:
    return [MUSIC_BOS_ID] + list(text.encode("ascii"))


def _feed(model, cache, token_ids: Sequence[int]) -> mx.array:
    tokens = mx.array([list(token_ids)], dtype=mx.uint32)
    logits = model(tokens, cache=cache)
    mx.eval(logits, [c.state for c in cache])
    return logits[0, -1]


def _candidate_map(params: Dict[str, Any]) -> Dict[int, str]:
    """Discriminating-byte token id -> bit symbol."""
    if params["encoding"] == "rhythm":
        return {ord(ALPH[db]): sym for sym, db in params["symbol_durs"].items()}
    return {ord(ALPH[p - PITCH_MIN]): sym for sym, p in params["symbol_pitches"].items()}


def _note_pre_post(params: Dict[str, Any]) -> Tuple[str, str]:
    """Bytes fed before the discriminating byte, and after it, per note."""
    if params["encoding"] == "rhythm":
        pre = wait_str(params["wait"]) + ALPH[VEL_BASE + params["vel"]] \
            + ALPH[params["pitch"] - PITCH_MIN]
        post = ""
    else:
        pre = wait_str(params["wait"]) + ALPH[VEL_BASE + params["vel"]]
        post = ALPH[params["dur"]]
    return pre, post


def generate_music_completions(
    model,
    trial_specs: List[Dict[str, Any]],
    *,
    max_bits: int,
    music_decode: str,
) -> List[str]:
    from mlx_lm.models import cache as kv_cache

    outputs: List[str] = []
    for spec in trial_specs:
        params = spec["music_params"]
        if params["encoding"] == "slice-pitch":
            outputs.append(_generate_slice_completion(
                model, kv_cache, spec, max_bits=max_bits, music_decode=music_decode))
            continue
        n_notes = max_bits // params["bits_per_note"]
        cache = kv_cache.make_prompt_cache(model)
        last_logits = _feed(model, cache, music_prompt_ids(spec["prompt"]))

        if music_decode == "constrained":
            cand = _candidate_map(params)
            cand_ids = list(cand.keys())
            pre, post = _note_pre_post(params)
            bits_parts: List[str] = []
            gen_parts: List[str] = []
            for _ in range(n_notes):
                logits = _feed(model, cache, [ord(c) for c in pre])
                scores = logits[mx.array(cand_ids)]
                best_id = cand_ids[int(mx.argmax(scores).item())]
                bits_parts.append(cand[best_id])
                gen_parts.append(pre + chr(best_id) + post)
                fed = [best_id] + [ord(c) for c in post]
                last_logits = _feed(model, cache, fed)
            spec["music_generated_pno"] = "".join(gen_parts)
            outputs.append("".join(bits_parts))
        elif music_decode == "greedy":
            budget = n_notes * (len(wait_str(params["wait"])) + 3) + 16
            gen_ids: List[int] = []
            logits = last_logits
            for _ in range(budget):
                next_id = int(mx.argmax(logits).item())
                if next_id >= 256:  # EOS/PAD/BOS -> stop
                    break
                gen_ids.append(next_id)
                logits = _feed(model, cache, [next_id])
            gen_text = bytes(gen_ids).decode("latin-1")
            spec["music_generated_pno"] = gen_text
            outputs.append(_parse_bits_from_pno(gen_text, params, n_notes))
        else:
            raise ValueError(f"Unsupported music decode mode: {music_decode}")
    return outputs


def _generate_slice_completion(model, kv_cache, spec: Dict[str, Any], *,
                               max_bits: int, music_decode: str) -> str:
    """slice-pitch generation. The prompt ends with an io-separator line; the
    answer is the next 8-slice line. Constrained mode feeds '\\n|' then reads
    logits only at each slice's letter byte."""
    params = spec["music_params"]
    octave = str(params["slice_octave"])
    cache = kv_cache.make_prompt_cache(model)
    logits = _feed(model, cache, music_prompt_ids(spec["prompt"]))

    if music_decode == "constrained":
        cand = {ord(v): k for k, v in params["slice_letters"].items()}
        cand_ids = list(cand.keys())
        logits = _feed(model, cache, [ord("\n"), ord("|")])
        bits_parts: List[str] = []
        gen_parts: List[str] = ["\n|"]
        for _ in range(max_bits):
            scores = logits[mx.array(cand_ids)]
            best_id = cand_ids[int(mx.argmax(scores).item())]
            bits_parts.append(cand[best_id])
            tail = octave + "|"
            gen_parts.append(chr(best_id) + tail)
            logits = _feed(model, cache, [best_id] + [ord(c) for c in tail])
        spec["music_generated_pno"] = "".join(gen_parts)
        return "".join(bits_parts)

    if music_decode == "greedy":
        budget = 2 + max_bits * 4 + 16
        gen_ids: List[int] = []
        logits = _feed(model, cache, [ord("\n")])
        for _ in range(budget):
            next_id = int(mx.argmax(logits).item())
            if next_id >= 256:
                break
            gen_ids.append(next_id)
            logits = _feed(model, cache, [next_id])
        gen_text = bytes(gen_ids).decode("latin-1")
        spec["music_generated_pno"] = gen_text
        # tolerant parse: melodic-register tokens -> nearest candidate letter
        targets = {bit: (params["slice_octave"] + 1) * 12 + _SLICE_LETTER_PC[letter]
                   for bit, letter in params["slice_letters"].items()}
        token_re = re.compile(r"([A-Ga-g]#?)([0-8])")
        bits_parts = []
        for m in token_re.finditer(gen_text):
            if len(bits_parts) >= max_bits:
                break
            letters, oct_digit = m.group(1), int(m.group(2))
            if oct_digit <= 2 or not letters[0].isupper():
                continue  # separator register or held tail
            pc = _SLICE_LETTER_PC.get(letters[0].upper())
            if pc is None:
                continue
            pitch = (oct_digit + 1) * 12 + pc + (1 if "#" in letters else 0)
            best = min(targets, key=lambda b: abs(targets[b] - pitch))
            bits_parts.append(best)
        return "".join(bits_parts)

    raise ValueError(f"Unsupported music decode mode: {music_decode}")


def _parse_bits_from_pno(text: str, params: Dict[str, Any], n_notes: int) -> str:
    """Tolerant parse of free-run PNO: extract notes, skip bass separators, map
    each note's discriminating attribute to the nearest candidate symbol."""
    char_idx = {c: i for i, c in enumerate(ALPH)}
    if params["encoding"] == "rhythm":
        targets = {sym: db for sym, db in params["symbol_durs"].items()}
    else:
        targets = {sym: p for sym, p in params["symbol_pitches"].items()}

    bits_parts: List[str] = []
    i, n = 0, len(text)
    while i < n and len(bits_parts) < n_notes:
        idx = char_idx.get(text[i])
        if idx is None or idx < VEL_BASE or idx >= VEL_BASE + VEL_N:
            i += 1  # wait chars / junk
            continue
        if i + 2 >= n:
            break
        pitch_idx = char_idx.get(text[i + 1])
        dur_idx = char_idx.get(text[i + 2])
        i += 3
        if pitch_idx is None or dur_idx is None:
            continue
        pitch = pitch_idx + PITCH_MIN
        if pitch <= _SEP_PITCH_HI + 2:  # bass register -> separator, not a bit
            continue
        value = dur_idx if params["encoding"] == "rhythm" else pitch
        best_sym = min(targets, key=lambda s: abs(targets[s] - value))
        bits_parts.append(best_sym)
    return "".join(bits_parts)
