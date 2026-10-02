"""Reproduce the committed v1 fit table using only Python's standard library.

Run from any directory. This writes src/fit_table.rs beside this script.
Changing the curve or numerical constants changes the law; version and retest it.
Do not run this at game startup.
"""
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path


with localcontext() as context:
    context.prec = 90
    pi = Decimal(
        "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342"
    )

    def cosine(x):
        x = (x + pi) % (2 * pi) - pi
        term = total = Decimal(1)
        for n in range(1, 240):
            term *= -(x * x) / Decimal((2 * n - 1) * (2 * n))
            total += term
            if abs(term) < Decimal("1e-88"):
                return total
        raise RuntimeError("Taylor series did not converge")

    quantum = 1 << 20
    values = [
        int((quantum * (cosine(2 * pi * d / 256) - cosine(4 * pi * d / 256)))
            .to_integral_value(rounding=ROUND_HALF_UP))
        for d in range(256)
    ]

assert values[0] == 0 and values[64] == quantum and values[128] == -2 * quantum
assert all(values[d] == values[-d % 256] for d in range(256))
output = (
    "// Authoritative table for watt-selective-transfer-v1. Do not regenerate at startup.\n"
    "// Q = 2^20; round half away from zero of Q*(cos(2*pi*d/256)-cos(4*pi*d/256)).\n"
    "// Generated once with Decimal at 90 digits. Symmetric under d -> -d mod 256.\n"
    "pub const FIT: [i32; 256] = [\n"
    + "".join("    " + ", ".join(map(str, values[i:i+8])) + ",\n" for i in range(0, 256, 8))
    + "];\n"
)
(Path(__file__).resolve().parent / "src" / "fit_table.rs").write_text(output)
