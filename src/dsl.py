import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def reverse_bits(s: str) -> str:
    return s[::-1]

def flip_bits(s: str) -> str:
    return ''.join('1' if b == '0' else '0' for b in s)

def swap_halves(s: str) -> str:
    mid = len(s) // 2
    return s[mid:] + s[:mid]

def rotl1(s: str) -> str:
    # circular rotate left by 1
    return s[1:] + s[:1]

def identity(s: str) -> str:
    return s

# Runtime only (not in enumeration)
def majority(s):
    ones = s.count('1')
    zeros = s.count('0')
    if ones >= zeros:
        return '1' * len(s)
    else:
        return '0' * len(s)


def minority(s):
    return flip_bits(majority(s))


def parity_fill(s):
    ones = s.count('1')
    fill_bit = '1' if ones % 2 else '0'
    return fill_bit * len(s)


def alternating_start_one(s):
    pattern = ['1' if idx % 2 == 0 else '0' for idx in range(len(s))]
    return ''.join('1' if bit != pattern[idx] else '0' for idx, bit in enumerate(s))


def alternating_start_zero(s):
    pattern = ['0' if idx % 2 == 0 else '1' for idx in range(len(s))]
    return ''.join('1' if bit != pattern[idx] else '0' for idx, bit in enumerate(s))


def left_half(s):
    mid = len(s) // 2
    return s[:mid] + '0' * (len(s) - mid)


def right_half(s):
    mid = len(s) // 2
    return '0' * mid + s[mid:]


def double_rotl(s):
    return rotl1(rotl1(s)) if s else s


def rotr1(s):
    return s[-1:] + s[:-1]


def double_rotr(s):
    return rotr1(rotr1(s)) if s else s


def ones_if_palindrome(s):
    return '1' * len(s) if s == s[::-1] else '0' * len(s)


def mirror_half(s):
    mid = len(s) // 2
    left = s[:mid]
    if len(s) % 2:
        center = s[mid]
        return left + center + left[::-1]
    return left + left[::-1]


def spread_first_bit(s):
    if not s:
        return s
    return s[0] * len(s)


def spread_last_bit(s):
    if not s:
        return s
    return s[-1] * len(s)


def invert_prefix(s):
    mid = len(s) // 2
    prefix = flip_bits(s[:mid])
    return prefix + s[mid:]


def invert_suffix(s):
    mid = len(s) // 2
    suffix = flip_bits(s[mid:])
    return s[:mid] + suffix


def meta_constant(s, s_m1=None):
    if s_m1 is None:
        raise ValueError("meta_constant requires an external bitstring s_m1")
    return s_m1


def shift_left_zero(s):
    if not s:
        return s
    return s[1:] + '0'


def shift_right_zero(s):
    if not s:
        return s
    return '0' + s[:-1]


def swap_pairs(s):
    chars = list(s)
    for i in range(0, len(chars) - 1, 2):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return ''.join(chars)


def reverse_each_half(s):
    mid = len(s) // 2
    return s[:mid][::-1] + s[mid:][::-1]


def keep_even_positions(s):
    return ''.join(bit if idx % 2 == 0 else '0' for idx, bit in enumerate(s))


def keep_odd_positions(s):
    return ''.join('0' if idx % 2 == 0 else bit for idx, bit in enumerate(s))


def flip_even_positions(s):
    return ''.join(
        ('1' if bit == '0' else '0') if idx % 2 == 0 else bit
        for idx, bit in enumerate(s)
    )


def flip_odd_positions(s):
    return ''.join(
        bit if idx % 2 == 0 else ('1' if bit == '0' else '0')
        for idx, bit in enumerate(s)
    )


def edge_mask(s):
    if not s:
        return s
    if len(s) == 1:
        return s
    middle = '0' * (len(s) - 2)
    return s[0] + middle + s[-1]


def center_mask(s):
    if len(s) <= 2:
        return '0' * len(s)
    middle_len = len(s) - 2
    return '0' + s[1:-1] + '0'


def _wrap_unary(func):
    def wrapped(s, s_m1=None):
        return func(s)

    return wrapped


def _random_bitstring(length: int, rng: Optional[random.Random] = None) -> str:
    if length < 0:
        raise ValueError("Bitstring length must be non-negative")
    if rng is None:
        rng = random
    return ''.join(rng.choice('01') for _ in range(length))


def _unique_bitstrings(count: int, length: int, rng: Optional[random.Random] = None) -> List[str]:
    if count < 0:
        raise ValueError("Count must be non-negative")
    max_unique = 1 << length if length <= 16 else None
    if max_unique is not None and count > max_unique:
        raise ValueError("Requested more unique bitstrings than possible for given length")

    rng = rng or random
    seen = set()
    values: List[str] = []
    while len(values) < count:
        candidate = _random_bitstring(length, rng)
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return values

# Binary (combine current s with s0)
def xor_with_s0(s: str, s0: str) -> str:
    return ''.join('0' if a == b else '1' for a, b in zip(s, s0))

def or_with_s0(s: str, s0: str) -> str:
    return ''.join('1' if (a == '1' or b == '1') else '0' for a, b in zip(s, s0))

def and_with_s0(s: str, s0: str) -> str:
    return ''.join('1' if (a == '1' and b == '1') else '0' for a, b in zip(s, s0))

# Catalog for program synthesis (enumeration subset)
ENUM_UNARY = {
    'rotl1': _wrap_unary(rotl1),
    'reverse_bits': _wrap_unary(reverse_bits),
    'flip_bits': _wrap_unary(flip_bits),
    'swap_halves': _wrap_unary(swap_halves),
    'identity': _wrap_unary(identity)
}
ENUM_BINARY = {
    'xor_with_s0': xor_with_s0,
    'or_with_s0': or_with_s0,
    'and_with_s0': and_with_s0,
}

# Additional primitives available at runtime but excluded from enumeration by default
RUNTIME_ONLY_UNARY = {
    'majority': _wrap_unary(majority),
    'minority': _wrap_unary(minority),
    'parity_fill': _wrap_unary(parity_fill),
    'alternating_start_one': _wrap_unary(alternating_start_one),
    'alternating_start_zero': _wrap_unary(alternating_start_zero),
    'left_half': _wrap_unary(left_half),
    'right_half': _wrap_unary(right_half),
    'double_rotl': _wrap_unary(double_rotl),
    'rotr1': _wrap_unary(rotr1),
    'double_rotr': _wrap_unary(double_rotr),
    'ones_if_palindrome': _wrap_unary(ones_if_palindrome),
    'mirror_half': _wrap_unary(mirror_half),
    'spread_first_bit': _wrap_unary(spread_first_bit),
    'spread_last_bit': _wrap_unary(spread_last_bit),
    'invert_prefix': _wrap_unary(invert_prefix),
    'invert_suffix': _wrap_unary(invert_suffix),
    'shift_left_zero': _wrap_unary(shift_left_zero),
    'shift_right_zero': _wrap_unary(shift_right_zero),
    'swap_pairs': _wrap_unary(swap_pairs),
    'reverse_each_half': _wrap_unary(reverse_each_half),
    'keep_even_positions': _wrap_unary(keep_even_positions),
    'keep_odd_positions': _wrap_unary(keep_odd_positions),
    'flip_even_positions': _wrap_unary(flip_even_positions),
    'flip_odd_positions': _wrap_unary(flip_odd_positions),
    'edge_mask': _wrap_unary(edge_mask),
    'center_mask': _wrap_unary(center_mask),
    'meta_constant': meta_constant,
}
RUNTIME_ONLY_BINARY = {}

UNARY = {**ENUM_UNARY, **RUNTIME_ONLY_UNARY}
BINARY = {**ENUM_BINARY, **RUNTIME_ONLY_BINARY}

# Helper to run a program (list of op names) on s0
def run_program(ops, s0, s_m1=None):
    s = s0
    for name in ops:
        if name in BINARY:
            s = BINARY[name](s, s0)
        elif name in UNARY:
            if name == 'meta_constant':
                if s_m1 is None:
                    raise ValueError("meta_constant requires an external bitstring s_m1")
                if len(s_m1) != len(s):
                    raise ValueError(
                        "meta_constant expects s_m1 to match the current bitstring length"
                    )
            s = UNARY[name](s, s_m1)
        else:
            raise KeyError(f"Unknown primitive: {name}")
    return s


def few_shot(
    program: Sequence[str],
    shots: int,
    newline_sep: str,
    apply_sep: str,
    *,
    bit_length: int = 8,
    rng: Optional[random.Random] = None,
    map_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[str, str, str, str, str]:
    """Construct a few-shot prompt for the given program.

    Returns a tuple of (prompt, query_input, expected_output, s_m1, query_input_raw).
    If map_fn is provided, it is applied to every sampled input before use.
    """

    if shots < 0:
        raise ValueError("shots must be non-negative")
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")

    if isinstance(program, str):
        program = [program]
    program = list(program)

    rng = rng or random

    sample_count = shots + 1  # include one extra for the query

    def _identity(bits: str) -> str:
        return bits

    if map_fn is None:
        transforms = [_identity] * sample_count
    elif callable(map_fn):
        transforms = [map_fn] * sample_count
    else:
        if not isinstance(map_fn, Sequence) or isinstance(map_fn, (str, bytes)):
            raise TypeError("map_fn must be callable or a sequence of callables")
        transforms = list(map_fn)
        if len(transforms) != sample_count:
            raise ValueError(
                "map_fn sequence must have the same length as the number of sampled inputs"
            )
        if not all(callable(fn) for fn in transforms):
            raise TypeError("Every entry in map_fn sequence must be callable")

    inputs = _unique_bitstrings(sample_count, bit_length, rng)
    mapped_inputs: List[str] = []
    for bits, transform_fn in zip(inputs, transforms):
        mapped = transform_fn(bits)
        if not isinstance(mapped, str):
            raise TypeError("Mapping function must return strings")
        mapped_inputs.append(mapped)

    if not mapped_inputs:
        raise ValueError("No inputs generated for few-shot prompt")

    mapped_length = len(mapped_inputs[0])
    if mapped_length <= 0:
        raise ValueError("Mapped bitstrings must be non-empty")
    if not all(len(mi) == mapped_length for mi in mapped_inputs):
        raise ValueError("Mapping function must produce bitstrings of a consistent length")

    context_inputs_raw = inputs[:-1]
    query_input_raw = inputs[-1]

    context_inputs = mapped_inputs[:-1]
    query_input = mapped_inputs[-1]

    s_m1 = _random_bitstring(len(inputs[0]), rng)

    context_lines: List[str] = []
    context_transforms = transforms[:-1]
    for raw_bits, mapped_bits, transform_fn in zip(
        context_inputs_raw, context_inputs, context_transforms
    ):
        output = run_program(program, raw_bits, s_m1=s_m1)
        mapped_output = transform_fn(output)
        if not isinstance(mapped_output, str):
            raise TypeError("Mapping function must return strings")
        if len(mapped_output) != len(mapped_bits):
            raise ValueError(
                "Mapping function must produce outputs of a consistent length"
            )
        context_lines.append(f"{mapped_bits}{apply_sep}{mapped_output}")

    final_line = f"{query_input}{apply_sep}"

    if context_lines:
        prompt = newline_sep.join(context_lines) + newline_sep + final_line
    else:
        prompt = final_line

    expected_output = run_program(program, query_input_raw, s_m1=s_m1)

    return prompt, query_input, expected_output, s_m1, query_input_raw


def program_signature(
    program: Sequence[str],
    *,
    bit_length: int = 8,
    s_m1: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return the output signature of ``program`` for all inputs of ``bit_length``."""

    if bit_length <= 0:
        raise ValueError("bit_length must be positive")

    program = list(program)
    outputs: List[str] = []
    for value in range(1 << bit_length):
        bits = format(value, f"0{bit_length}b")
        outputs.append(run_program(program, bits, s_m1=s_m1))
    return tuple(outputs)


def verify_unique_programs(
    programs: Iterable[Sequence[str]],
    *,
    bit_length: int = 8,
    s_m1: Optional[str] = None,
    s_m1_provider: Optional[Callable[[Sequence[str]], str]] = None,
) -> Tuple[bool, List[Tuple[int, int, Sequence[str], Sequence[str]]]]:
    """Check that every program in ``programs`` has a unique behavior.

    Returns (is_unique, duplicates) where ``duplicates`` is a list of
    (idx_a, idx_b, program_a, program_b) entries describing conflicts.
    """

    signatures: Dict[Tuple[str, ...], Tuple[int, Sequence[str]]] = {}
    duplicates: List[Tuple[int, int, Sequence[str], Sequence[str]]] = []

    for idx, program in enumerate(programs):
        program = list(program)
        needs_meta = 'meta_constant' in program
        resolved_s_m1: Optional[str]
        if needs_meta:
            if s_m1_provider is not None:
                resolved_s_m1 = s_m1_provider(program)
            else:
                resolved_s_m1 = s_m1
            if resolved_s_m1 is None:
                raise ValueError(
                    "Program uses meta_constant but no s_m1 or s_m1_provider was supplied"
                )
            if len(resolved_s_m1) != bit_length:
                raise ValueError(
                    "Provided s_m1 must have the same length as the evaluation bit length"
                )
        else:
            resolved_s_m1 = None

        signature = program_signature(
            program, bit_length=bit_length, s_m1=resolved_s_m1
        )
        if signature in signatures:
            prev_idx, prev_program = signatures[signature]
            duplicates.append((prev_idx, idx, prev_program, program))
        else:
            signatures[signature] = (idx, program)

    return (len(duplicates) == 0, duplicates)
