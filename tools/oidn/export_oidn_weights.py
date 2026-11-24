# export_oidn_weights.py -- convert an OpenImageDenoise .tza weight blob into a flat
# float32 file that the Vulkan denoiser in this repository can upload to an SSBO.
#
# The .tza container (OIDN 1.4.x, version 2.0) is a small directory followed by
# plain float32 payloads; see core/tza.cpp in the OIDN sources. Nothing is reordered here,
# but each 3x3 kernel row (the 9 weights of one output/input channel pair) is padded to
# 12 floats so the shader can fetch it with three vec4 loads instead of nine scalar ones.
# Conv weight rows are thus:  w[((o * cin + i) * 12) + ky * 3 + kx],  padded to 12.
# Biases follow the padded weights.
#
# Usage:
#   python export_oidn_weights.py <path-to-rt_hdr.tza> <out.bin>
#
# The layer order below matches core/unet.cpp (buildNet): encoder layers, then
# the decoder chain, weight and bias adjacent.

import struct
import sys

import numpy as np

LAYERS = [
    'enc_conv0', 'enc_conv1', 'enc_conv2', 'enc_conv3', 'enc_conv4',
    'enc_conv5a', 'enc_conv5b',
    'dec_conv4a', 'dec_conv4b', 'dec_conv3a', 'dec_conv3b',
    'dec_conv2a', 'dec_conv2b', 'dec_conv1a', 'dec_conv1b', 'dec_conv0',
]


def parse_tza(path):
    data = open(path, 'rb').read()
    magic, = struct.unpack_from('<H', data, 0)
    if magic != 0x41D7:
        raise SystemExit('not an OIDN weights blob (magic %04X)' % magic)
    if data[2] != 2:
        raise SystemExit('unsupported weights blob version %d.%d' % (data[2], data[3]))
    table_ofs, = struct.unpack_from('<Q', data, 4)
    count, = struct.unpack_from('<I', data, table_ofs)
    pos = table_ofs + 4
    tensors = {}
    for _ in range(count):
        name_len, = struct.unpack_from('<H', data, pos)
        pos += 2
        name = data[pos:pos + name_len].decode('ascii')
        pos += name_len
        ndims = data[pos]
        pos += 1
        dims = list(struct.unpack_from('<%dI' % ndims, data, pos))
        pos += 4 * ndims
        layout = data[pos:pos + ndims].decode('ascii')
        pos += ndims
        dtype = chr(data[pos])
        pos += 1
        ofs, = struct.unpack_from('<Q', data, pos)
        pos += 8
        if dtype != 'f':
            raise SystemExit('unsupported tensor data type %r' % dtype)
        n = 1
        for d in dims:
            n *= d
        tensors[name] = (dims, layout, data[ofs:ofs + 4 * n])
    return tensors


def main():
    src, dst = sys.argv[1], sys.argv[2]
    tensors = parse_tza(src)
    out = bytearray()
    print('layer        in  out  weight floats   offset   offset %% 4')
    for name in LAYERS:
        w_dims, w_layout, w_data = tensors[name + '.weight']
        _, _, b_data = tensors[name + '.bias']
        if w_layout != 'oihw' or len(w_dims) != 4:
            raise SystemExit('%s.weight is not a 4D oihw tensor' % name)
        cout, cin = w_dims[0], w_dims[1]
        if w_dims[2:] != [3, 3]:
            raise SystemExit('%s.weight is not 3x3' % name)
        ofs = len(out) // 4
        print('%-12s %3d  %3d  %8d  %9d  %8d' % (name, cin, cout, cin * cout * 12, ofs, ofs % 4))
        rows = np.frombuffer(w_data, dtype=np.float32).reshape(cout, cin, 9)
        padded = np.zeros((cout, cin, 12), dtype=np.float32)
        padded[:, :, :9] = rows
        out += padded.tobytes()
        # biases, padded to a multiple of 4 floats to keep the next layer 16-byte aligned
        bias = np.frombuffer(b_data, dtype=np.float32)
        biasPad = np.zeros(((cout + 3) // 4) * 4, dtype=np.float32)
        biasPad[:cout] = bias
        out += biasPad.tobytes()
    with open(dst, 'wb') as f:
        f.write(out)
    print('total %d floats (%.2f MB) -> %s' % (len(out) // 4, len(out) / 1024 / 1024, dst))


main()
