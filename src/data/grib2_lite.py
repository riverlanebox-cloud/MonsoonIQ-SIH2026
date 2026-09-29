"""
Minimal GRIB2 decoder for the fields MonsoonIQ reads from NOAA GFS.

Supports regular lat-lon grids (grid template 3.0) with data templates 5.0 (simple
packing), 5.2 (complex packing) and 5.3 (complex packing + spatial differencing,
what GFS pgrb2 files use), without bitmaps. Fully vectorised with NumPy, so no
eccodes/cfgrib installation is needed.

Checked value-for-value against an independent JavaScript implementation used by
the browser fetcher (scripts/fetch/), and against physical ranges (MSLP 990-1025 hPa
over India, rainfall >= 0).
"""

import struct
from typing import Dict, Tuple

import numpy as np


def _sm(v: int, nbits: int) -> int:
    """Sign-magnitude integer (GRIB2 convention for signed octets)."""
    sign = 1 << (nbits - 1)
    return -(v & (sign - 1)) if v & sign else v


def _read_bits(buf: np.ndarray, bitpos: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Read unsigned ints of `width` bits starting at `bitpos` (vectorised, width <= 32)."""
    out = np.zeros(bitpos.shape, dtype=np.uint64)
    nz = width > 0
    if not nz.any():
        return out
    bp = bitpos[nz].astype(np.int64)
    w = width[nz].astype(np.int64)
    byte = bp >> 3
    pad = np.concatenate([buf, np.zeros(8, dtype=np.uint8)])
    win = np.zeros(len(bp), dtype=np.uint64)
    for k in range(8):  # big-endian 64-bit window
        win = (win << np.uint64(8)) | pad[byte + k].astype(np.uint64)
    shift = (np.int64(64) - (bp & 7) - w).astype(np.uint64)
    mask = (np.uint64(1) << w.astype(np.uint64)) - np.uint64(1)
    out[nz] = (win >> shift) & mask
    return out


def _unpack_seq(buf: np.ndarray, start_bit: int, n: int, width: int) -> Tuple[np.ndarray, int]:
    if width == 0 or n == 0:
        return np.zeros(n, dtype=np.int64), start_bit
    pos = start_bit + width * np.arange(n, dtype=np.int64)
    vals = _read_bits(buf, pos, np.full(n, width, dtype=np.int64)).astype(np.int64)
    return vals, start_bit + width * n


def decode(msg: bytes) -> Dict[str, object]:
    """Decode one GRIB2 message. Returns {'ni','nj','la1','lo1','values' (nj, ni) float32}."""
    b = np.frombuffer(msg, dtype=np.uint8)
    if bytes(b[:4]) != b"GRIB":
        raise ValueError("not a GRIB message")
    o, secs = 16, {}
    while o < len(b) - 4:
        if bytes(b[o:o + 4]) == b"7777":
            break
        ln = struct.unpack(">I", msg[o:o + 4])[0]
        secs[int(b[o + 4])] = (o, ln)
        o += ln
    s3, s5, s6, s7 = secs[3][0], secs[5][0], secs[6][0], secs[7][0]
    if struct.unpack(">H", msg[s3 + 12:s3 + 14])[0] != 0:
        raise ValueError("only regular lat-lon grids are supported")
    ni, nj = struct.unpack(">II", msg[s3 + 30:s3 + 38])
    la1 = _sm(struct.unpack(">I", msg[s3 + 46:s3 + 50])[0], 32) / 1e6
    lo1 = _sm(struct.unpack(">I", msg[s3 + 50:s3 + 54])[0], 32) / 1e6
    if b[s6 + 5] != 255:
        raise ValueError("bitmaps are not supported")
    npts = struct.unpack(">I", msg[s5 + 5:s5 + 9])[0]
    tmpl = struct.unpack(">H", msg[s5 + 9:s5 + 11])[0]
    R = struct.unpack(">f", msg[s5 + 11:s5 + 15])[0]
    E = _sm(struct.unpack(">H", msg[s5 + 15:s5 + 17])[0], 16)
    D = _sm(struct.unpack(">H", msg[s5 + 17:s5 + 19])[0], 16)
    nbits = int(b[s5 + 19])
    data = b[s7 + 5: s7 + secs[7][1]]
    bitpos = 0

    if tmpl == 0:
        X, _ = _unpack_seq(data, 0, npts, nbits)
        X = X.astype(np.float64)
    elif tmpl in (2, 3):
        if b[s5 + 22] != 0:
            raise ValueError("missing-value management is not supported")
        NG = struct.unpack(">I", msg[s5 + 31:s5 + 35])[0]
        refW, bitsW = int(b[s5 + 35]), int(b[s5 + 36])
        refL = struct.unpack(">I", msg[s5 + 37:s5 + 41])[0]
        incL = int(b[s5 + 41])
        lastL = struct.unpack(">I", msg[s5 + 42:s5 + 46])[0]
        bitsL = int(b[s5 + 46])
        order = int(b[s5 + 47]) if tmpl == 3 else 0
        noct = int(b[s5 + 48]) if tmpl == 3 else 0
        ival1 = ival2 = minsd = 0
        if order:
            def rdsm(pos):
                v = int.from_bytes(bytes(data[pos // 8: pos // 8 + noct]), "big")
                return _sm(v, 8 * noct), pos + 8 * noct
            ival1, bitpos = rdsm(bitpos)
            if order == 2:
                ival2, bitpos = rdsm(bitpos)
            minsd, bitpos = rdsm(bitpos)
        gref, bitpos = _unpack_seq(data, bitpos, NG, nbits)
        bitpos = (bitpos + 7) // 8 * 8
        gw, bitpos = _unpack_seq(data, bitpos, NG, bitsW)
        gw = gw + refW
        bitpos = (bitpos + 7) // 8 * 8
        gl, bitpos = _unpack_seq(data, bitpos, NG, bitsL)
        gl = refL + incL * gl
        gl[-1] = lastL
        bitpos = (bitpos + 7) // 8 * 8
        if int(gl.sum()) != npts:
            raise ValueError("group lengths do not add up")
        widths = np.repeat(gw, gl)
        starts = bitpos + np.concatenate([[0], np.cumsum(gw * gl)[:-1]])
        within = np.arange(npts) - np.repeat(np.concatenate([[0], np.cumsum(gl)[:-1]]), gl)
        pos = np.repeat(starts, gl) + within * widths
        X = np.repeat(gref, gl).astype(np.float64) + _read_bits(data, pos, widths).astype(np.float64)
        if order == 1:
            X[0] = ival1
            X[1:] += minsd
            X = np.cumsum(X)
        elif order == 2:
            X[0], X[1] = ival1, ival2
            d = X[2:] + minsd
            # x[i] = d[i] + 2x[i-1] - x[i-2]  <=>  second-order cumulative sum
            first = np.empty(npts)
            first[0], first[1] = ival1, ival2 - ival1
            first[2:] = d
            first[1:] = np.cumsum(first[1:])
            X = np.cumsum(first)
    else:
        raise ValueError(f"data template 5.{tmpl} not supported")

    vals = ((R + X * 2.0 ** E) / 10.0 ** D).astype(np.float32)
    return {"ni": ni, "nj": nj, "la1": la1, "lo1": lo1, "values": vals.reshape(nj, ni)}
