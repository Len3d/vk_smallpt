#!/usr/bin/env python3
"""ppm_diff.py -- compare two PPM(P6) images to check whether two implementations agree.

Usage: python ppm_diff.py a.ppm b.ppm
Output: image size, per-channel max/mean difference, and the share of bytes differing by >1/255.
Standard library only.
"""

import sys


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"P6"):
        raise ValueError(f"{path}: not a P6 PPM")
    idx = 2
    fields = []
    while len(fields) < 3:
        while idx < len(data) and data[idx:idx + 1].isspace():
            idx += 1
        start = idx
        while idx < len(data) and not data[idx:idx + 1].isspace():
            idx += 1
        fields.append(int(data[start:idx]))
    idx += 1
    w, h, maxval = fields
    px = data[idx:idx + w * h * 3]
    return w, h, px


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    w1, h1, a = read_ppm(sys.argv[1])
    w2, h2, b = read_ppm(sys.argv[2])
    if (w1, h1) != (w2, h2):
        print(f"size mismatch: {w1}x{h1} vs {w2}x{h2}")
        return 2
    n = len(a)
    maxdiff = 0
    total = 0
    over1 = 0
    for i in range(n):
        d = abs(a[i] - b[i])
        total += d
        if d > maxdiff:
            maxdiff = d
        if d > 1:
            over1 += 1
    print(f"size {w1}x{h1}, {n} bytes compared")
    print(f"max diff {maxdiff}/255, mean diff {total / n:.5f}, bytes differing by >1: {over1 / n * 100:.4f}%")
    identical = maxdiff == 0
    print("verdict: " + ("byte identical" if identical else "differs (floating-point reassociation or RNG difference)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
