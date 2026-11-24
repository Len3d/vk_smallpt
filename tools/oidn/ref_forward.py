# ref_forward.py -- numpy reference implementation of the OIDN RT U-Net (rt_hdr,
# rt_hdr_alb or rt_hdr_alb_nrm weights), used to validate the Vulkan compute
# implementation in terms of float tolerance.
#
# Layout convention used everywhere here and in the shaders: NHWC (one pixel's
# channels are contiguous), float32. The .tza weights keep their oihw layout.
#
# Usage:
#   python ref_forward.py <weights.bin> <input.f32> <width> <height> <out_prefix> \
#                         [exposure_scale] [albedo.f32] [normal.f32]
#
# <input.f32> is raw float32 linear RGB, HWC, width*height*3 values. The albedo and
# normal images use the same layout; giving the albedo selects the 6 channel tensor
# (rt_hdr_alb weights), giving both selects the 9 channel tensor (rt_hdr_alb_nrm).

import struct
import sys
import numpy as np

LAYERS_FIXED = [
    ('enc_conv0', 0, 32, True),   # cin = input channels, set by the feature set below

    ('enc_conv1', 32, 32, True),
    ('enc_conv2', 32, 48, True),
    ('enc_conv3', 48, 64, True),
    ('enc_conv4', 64, 80, True),
    ('enc_conv5a', 80, 96, True),
    ('enc_conv5b', 96, 96, True),
    ('dec_conv4a', 160, 112, True),
    ('dec_conv4b', 112, 112, True),
    ('dec_conv3a', 160, 96, True),
    ('dec_conv3b', 96, 96, True),
    ('dec_conv2a', 128, 64, True),
    ('dec_conv2b', 64, 64, True),
    ('dec_conv1a', None, 64, True),  # cin = 64 + input channels (concat1 with the input)
    ('dec_conv1b', 64, 32, True),
    ('dec_conv0', 32, 3, False),
]

# PU transfer function of core/color.ispc
PU_A = 1.41283765e+03
PU_B = 1.64593172e+00
PU_C = 4.31384981e-01
PU_D = -2.94139609e-03
PU_E = 1.92653254e-01
PU_F = 6.26026094e-03
PU_G = 9.98620152e-01
PU_Y0 = 1.57945760e-06
PU_Y1 = 3.22087631e-02
PU_X0 = 2.23151711e-03
PU_X1 = 3.70974749e-01
HDR_Y_MAX = 65504.0


def pu_forward(y):
    if y <= PU_Y0:
        return PU_A * y
    if y <= PU_Y1:
        return PU_B * np.power(y, PU_C) + PU_D
    return PU_E * np.log(y + PU_F) + PU_G


def pu_inverse(x):
    if x <= PU_X0:
        return x / PU_A
    if x <= PU_X1:
        return np.power((x - PU_D) / PU_B, 1.0 / PU_C)
    return np.exp((x - PU_G) / PU_E) - PU_F


def layers_for(input_channels):
    return [(name, (64 + input_channels) if cin is None else (input_channels if cin == 0 else cin),
            cout, relu) for name, cin, cout, relu in LAYERS_FIXED]


def load_weights(path, layers):
    raw = np.fromfile(path, dtype=np.float32)
    out = {}
    ofs = 0
    for name, cin, cout, _ in layers:
        # each 3x3 kernel row is padded to 12 floats, biases to a multiple of 4
        rows = raw[ofs:ofs + cout * cin * 12].reshape(cout, cin, 12)
        ofs += cout * cin * 12
        w = np.ascontiguousarray(rows[:, :, :9]).reshape(cout, cin, 3, 3)
        b = raw[ofs:ofs + cout]
        ofs += ((cout + 3) // 4) * 4
        out[name] = (w, b)
    assert ofs == raw.size, (ofs, raw.size)
    return out


def conv3x3(x, w, b, relu):
    h, wd, cin = x.shape
    cout = w.shape[0]
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    acc = np.zeros((h, wd, cout), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            patch = p[ky:ky + h, kx:kx + wd, :]              # H,W,Cin
            acc += patch @ np.ascontiguousarray(w[:, :, ky, kx].T)  # Cin,Cout
    acc += b
    if relu:
        np.maximum(acc, 0.0, out=acc)
    return acc


def pool2(x):
    h, w, c = x.shape
    return x[:h & ~1, :w & ~1].reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


def upsample2(x):
    return x.repeat(2, axis=0).repeat(2, axis=1)


def autoexposure(color):
    h, w, _ = color.shape
    k = 16
    hk, wk = (h + k // 2) // k, (w + k // 2) // k
    logs, count = 0.0, 0
    for i in range(hk):
        for j in range(wk):
            b0, b1 = i * h // hk, (i + 1) * h // hk
            c0, c1 = j * w // wk, (j + 1) * w // wk
            block = color[b0:b1, c0:c1, :]
            lum = float(np.sum(0.212671 * block[:, :, 0] + 0.715160 * block[:, :, 1] +
                              0.072169 * block[:, :, 2]) / ((b1 - b0) * (c1 - c0)))
            if lum > 1e-8:
                logs += np.log2(lum)
                count += 1
    return 0.18 / np.exp2(logs / count) if count else 1.0


def main():
    weights_path, in_path, width, height, out_prefix = sys.argv[1:6]
    width, height = int(width), int(height)
    # Optional 7th argument overrides the automatic exposure scale, so a padded reference
    # image can be compared against the Vulkan path, which computes the scale from the
    # unpadded image. The albedo and normal images come after it.
    forced_scale = float(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != '-' else None
    alb_path = sys.argv[7] if len(sys.argv) > 7 else None
    nrm_path = sys.argv[8] if len(sys.argv) > 8 else None
    color = np.fromfile(in_path, dtype=np.float32).reshape(height, width, 3)
    albedo = (np.fromfile(alb_path, dtype=np.float32).reshape(height, width, 3)
              if alb_path else None)
    normal = (np.fromfile(nrm_path, dtype=np.float32).reshape(height, width, 3)
              if nrm_path else None)
    aux = (1 if albedo is not None else 0) + (1 if normal is not None else 0)
    if normal is not None and albedo is None:
        sys.exit('the normal image needs the albedo image (OIDN rejects color + normal only)')
    input_channels = 3 + 3 * aux
    layers = layers_for(input_channels)
    weights = load_weights(weights_path, layers)

    # Preprocess (core/input_reorder.ispc): the colour channels take the exposure scale and the
    # PU transfer function, the albedo is only clamped to [0, 1] and the normal is clamped to
    # [-1, 1] and remapped to [0, 1].
    input_scale = autoexposure(color) if forced_scale is None else forced_scale
    norm_scale = 1.0 / pu_forward(HDR_Y_MAX)
    y = np.clip(np.nan_to_num(color) * input_scale, 0.0, HDR_Y_MAX)
    x = np.array([[pu_forward(v) for v in row.reshape(-1)] for row in y],
                 dtype=np.float32).reshape(y.shape) * norm_scale
    extra = [x]
    if albedo is not None:
        extra.append(np.clip(albedo, 0.0, 1.0).astype(np.float32))
    if normal is not None:
        extra.append((np.clip(normal, -1.0, 1.0) * 0.5 + 0.5).astype(np.float32))
    xin = np.concatenate(extra, axis=2) if aux else x

    # Encoder
    c0 = conv3x3(xin, *weights['enc_conv0'], True)
    c1 = conv3x3(c0, *weights['enc_conv1'], True)
    p1 = pool2(c1)
    c2 = conv3x3(p1, *weights['enc_conv2'], True)
    p2 = pool2(c2)
    c3 = conv3x3(p2, *weights['enc_conv3'], True)
    p3 = pool2(c3)
    c4 = conv3x3(p3, *weights['enc_conv4'], True)
    p4 = pool2(c4)
    c5a = conv3x3(p4, *weights['enc_conv5a'], True)
    c5b = conv3x3(c5a, *weights['enc_conv5b'], True)

    # Decoder
    u4 = upsample2(c5b)
    d4 = conv3x3(np.concatenate([u4, p3], axis=2), *weights['dec_conv4a'], True)
    d4 = conv3x3(d4, *weights['dec_conv4b'], True)
    u3 = upsample2(d4)
    d3 = conv3x3(np.concatenate([u3, p2], axis=2), *weights['dec_conv3a'], True)
    d3 = conv3x3(d3, *weights['dec_conv3b'], True)
    u2 = upsample2(d3)
    d2 = conv3x3(np.concatenate([u2, p1], axis=2), *weights['dec_conv2a'], True)
    d2 = conv3x3(d2, *weights['dec_conv2b'], True)
    u1 = upsample2(d2)
    d1 = conv3x3(np.concatenate([u1, xin], axis=2), *weights['dec_conv1a'], True)
    d1 = conv3x3(d1, *weights['dec_conv1b'], True)
    out = conv3x3(d1, *weights['dec_conv0'], False)

    # Postprocess (core/output_reorder.ispc). outputScale is 1/inputScale, which
    # is how OIDN undoes the autoexposure applied before the network.
    out = np.clip(np.nan_to_num(out), 0.0, np.finfo(np.float32).max)
    rcp_norm_scale = pu_forward(HDR_Y_MAX)
    z = out * rcp_norm_scale
    res = np.array([[pu_inverse(v) for v in row.reshape(-1)] for row in z],
                   dtype=np.float32).reshape(z.shape)
    res *= np.float32(1.0 / input_scale)

    res.astype(np.float32).tofile(out_prefix + '.f32')

    # Gamma-encoded PPM for a quick visual check
    def enc(v):
        v = np.clip(v, 0.0, 1.0) ** (1.0 / 2.2) * 255.0 + 0.5
        return v.astype(np.uint8)
    with open(out_prefix + '.ppm', 'wb') as f:
        f.write(b'P6\n%d %d\n255\n' % (width, height))
        f.write(enc(res).tobytes())

    print('feature set: %d input channels (albedo %s, normal %s)'
          % (input_channels, 'yes' if albedo is not None else 'no',
             'yes' if normal is not None else 'no'))
    print('input scale %.6f' % input_scale)
    print('output  min %.4f  max %.4f  mean %.4f' %
          (res.min(), res.max(), res.mean()))
    print('wrote %s.f32 and %s.ppm' % (out_prefix, out_prefix))


if __name__ == '__main__':
    main()
