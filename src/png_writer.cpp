// png_writer.cpp -- a small, self-contained PNG (8-bit RGB) writer.
//
// PNG needs four things that a plain C++ program does not have, all implemented
// here: CRC-32 (chunk checksums), Adler-32 (zlib checksum), the per-row filters
// and a deflate stream. The deflate encoder is a single dynamic-Huffman block
// (BTYPE=10) driven by a greedy LZ77 matcher over a 32 KiB window, which puts
// the output within a few percent of zlib level 6 on real renders while keeping
// the whole thing to one source file with no external dependency.
//
// Layout produced:
//   IHDR  width, height, bit depth 8, colour type 2 (RGB), no interlace
//   IDAT  zlib stream (0x78 0x01 header, deflate data, Adler-32 trailer)
//   IEND

#include "png_writer.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <queue>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

std::uint32_t g_crc_table[256];
bool g_crc_ready = false;

void crc32_init() {
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    g_crc_table[i] = c;
  }
  g_crc_ready = true;
}

std::uint32_t crc32(const std::uint8_t *data, std::size_t n) {
  if (!g_crc_ready) crc32_init();
  std::uint32_t c = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) c = g_crc_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

std::uint32_t adler32(const std::uint8_t *data, std::size_t n) {
  std::uint32_t a = 1, b = 0;
  std::size_t i = 0;
  while (i < n) {
    // 5552 additions stay below 2^32 before a modulo is needed.
    const std::size_t block = (n - i < 5552) ? (n - i) : 5552;
    for (std::size_t k = 0; k < block; ++k) {
      a += data[i + k];
      b += a;
    }
    a %= 65521u;
    b %= 65521u;
    i += block;
  }
  return (b << 16) | a;
}

// ---------------------------------------------------------------------------
// Bit writer: deflate fills bytes least-significant bit first
// ---------------------------------------------------------------------------

struct BitWriter {
  std::vector<std::uint8_t> &out;
  std::uint32_t buf = 0;
  int count = 0;

  explicit BitWriter(std::vector<std::uint8_t> &dst) : out(dst) {}

  void put(std::uint32_t bits, int n) {
    buf |= bits << count;
    count += n;
    while (count >= 8) {
      out.push_back(static_cast<std::uint8_t>(buf & 0xFFu));
      buf >>= 8;
      count -= 8;
    }
  }

  void flush() {
    if (count > 0) out.push_back(static_cast<std::uint8_t>(buf & 0xFFu));
    buf = 0;
    count = 0;
  }
};

std::uint32_t reverse_bits(std::uint32_t v, int n) {
  std::uint32_t r = 0;
  for (int i = 0; i < n; ++i) {
    r = (r << 1) | (v & 1u);
    v >>= 1;
  }
  return r;
}

struct HuffCode {
  std::uint16_t code = 0;
  std::uint8_t len = 0;
};

// ---------------------------------------------------------------------------
// Huffman code length construction
//
// Deflate caps the code length at 15 bits (7 for the code-length alphabet).
// A plain Huffman tree over wildly skewed frequencies can be deeper than that,
// and repairing an over-deep tree by hand is fiddly, so the depth is instead
// bounded by flattening the histogram: halve every frequency that is still
// above 1 and rebuild. The ratios converge quickly, and once all frequencies
// are equal the tree is balanced (depth ceil(log2(n)) <= 9 for the largest
// alphabet here), so the loop always terminates with a complete code.
// ---------------------------------------------------------------------------

struct HuffTree {
  std::vector<std::uint8_t> len;
  int max_len = 0;
};

HuffTree huffman_tree(const std::vector<std::uint32_t> &freq) {
  const int n = static_cast<int>(freq.size());
  std::vector<int> parent(static_cast<std::size_t>(2 * n), -1);
  int next_node = n;

  typedef std::pair<std::uint32_t, int> HeapItem;
  auto greater = [](const HeapItem &a, const HeapItem &b) { return a.first > b.first; };
  std::vector<HeapItem> heap;
  heap.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    if (freq[i] != 0) heap.push_back(HeapItem(freq[i], i));
  std::make_heap(heap.begin(), heap.end(), greater);

  while (heap.size() > 1) {
    std::pop_heap(heap.begin(), heap.end(), greater);
    HeapItem a = heap.back();
    heap.pop_back();
    std::pop_heap(heap.begin(), heap.end(), greater);
    HeapItem b = heap.back();
    heap.pop_back();
    parent[static_cast<std::size_t>(a.second)] = next_node;
    parent[static_cast<std::size_t>(b.second)] = next_node;
    heap.push_back(HeapItem(a.first + b.first, next_node));
    std::push_heap(heap.begin(), heap.end(), greater);
    ++next_node;
  }

  HuffTree t;
  t.len.assign(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    if (freq[i] == 0) continue;
    int depth = 0;
    int p = i;
    while (parent[static_cast<std::size_t>(p)] >= 0) {
      p = parent[static_cast<std::size_t>(p)];
      ++depth;
    }
    t.len[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(depth);
    if (depth > t.max_len) t.max_len = depth;
  }
  return t;
}

std::vector<std::uint8_t> build_lengths(const std::vector<std::uint32_t> &freq_in, int max_bits) {
  std::vector<std::uint32_t> freq(freq_in);

  // A deflate code needs at least two symbols to be complete.
  std::uint32_t used = 0;
  for (std::size_t i = 0; i < freq.size(); ++i)
    if (freq[i] != 0) ++used;
  for (std::size_t i = 0; i < freq.size() && used < 2; ++i) {
    if (freq[i] == 0) {
      freq[i] = 1;
      ++used;
    }
  }

  for (int attempt = 0; attempt < 64; ++attempt) {
    HuffTree t = huffman_tree(freq);
    if (t.max_len <= max_bits) return t.len;
    for (std::size_t i = 0; i < freq.size(); ++i)
      if (freq[i] > 1) freq[i] = (freq[i] + 1) / 2;
  }

  // Unreachable in practice: equal frequencies give a balanced tree.
  std::vector<std::uint32_t> flat(freq.size(), 1);
  return huffman_tree(flat).len;
}

std::vector<HuffCode> build_codes(const std::vector<std::uint8_t> &len) {
  const int n = static_cast<int>(len.size());
  int max_bits = 0;
  for (int i = 0; i < n; ++i)
    if (len[static_cast<std::size_t>(i)] > max_bits) max_bits = len[static_cast<std::size_t>(i)];

  std::vector<int> bl_count(static_cast<std::size_t>(max_bits) + 1, 0);
  for (int i = 0; i < n; ++i) bl_count[len[static_cast<std::size_t>(i)]]++;

  std::vector<std::uint32_t> next_code(static_cast<std::size_t>(max_bits) + 1, 0);
  std::uint32_t code = 0;
  bl_count[0] = 0;
  for (int bits = 1; bits <= max_bits; ++bits) {
    code = (code + static_cast<std::uint32_t>(bl_count[bits - 1])) << 1;
    next_code[bits] = code;
  }

  std::vector<HuffCode> out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int l = len[static_cast<std::size_t>(i)];
    if (l == 0) continue;
    out[static_cast<std::size_t>(i)].len = static_cast<std::uint8_t>(l);
    out[static_cast<std::size_t>(i)].code = static_cast<std::uint16_t>(next_code[l]++);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Length / distance alphabets
// ---------------------------------------------------------------------------

const int kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10,  11,  13,  15,  17,  19,  23, 27,
                          31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const int kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                           2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const int kDistBase[30] = {1,    2,    3,    4,    5,    7,     9,     13,    17,    25,
                           33,   49,   65,   97,   129,  193,   257,   385,   513,   769,
                           1025, 1537, 2049, 3073, 4097, 6145,  8193,  12289, 16385, 24577};
const int kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                            6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

int len_symbol(int len, int &extra) {
  int j = 0;
  for (int k = 0; k < 29; ++k)
    if (kLenBase[k] <= len) j = k;
  extra = len - kLenBase[j];
  return 257 + j;
}

int dist_symbol(int dist, int &extra) {
  int j = 0;
  for (int k = 0; k < 30; ++k)
    if (kDistBase[k] <= dist) j = k;
  extra = dist - kDistBase[j];
  return j;
}

// ---------------------------------------------------------------------------
// LZ77 tokens: `dist == 0` means a literal (value in `len`), otherwise a match
// ---------------------------------------------------------------------------

struct Token {
  std::uint16_t len;  // literal byte value, or match length 3..258
  std::uint16_t dist; // 0 for a literal, 1..32768 for a match
};

const std::size_t kWindowSize = 32768;
const int kMinMatch = 3;
const int kMaxMatch = 258;
const int kHashBits = 15;
const std::uint32_t kHashMask = (1u << kHashBits) - 1u;
const int kMaxChain = 256;

void tokenize(const std::uint8_t *data, std::size_t size, std::vector<Token> &tokens) {
  std::vector<int> head(static_cast<std::size_t>(kHashMask) + 1, -1);
  std::vector<int> prev(size ? size : 1, -1);
  tokens.clear();

  auto hash_at = [&](std::size_t p) -> std::uint32_t {
    return ((static_cast<std::uint32_t>(data[p]) << 10) ^
            (static_cast<std::uint32_t>(data[p + 1]) << 5) ^
            static_cast<std::uint32_t>(data[p + 2])) & kHashMask;
  };

  std::size_t i = 0;
  while (i < size) {
    int best_len = 0;
    std::size_t best_pos = 0;
    if (i + static_cast<std::size_t>(kMinMatch) <= size) {
      const std::uint32_t h = hash_at(i);
      int cand = head[h];
      int chain = 0;
      while (cand >= 0 && i - static_cast<std::size_t>(cand) <= kWindowSize && chain < kMaxChain) {
        std::size_t maxlen = size - i;
        if (maxlen > static_cast<std::size_t>(kMaxMatch)) maxlen = static_cast<std::size_t>(kMaxMatch);
        std::size_t l = 0;
        while (l < maxlen && data[static_cast<std::size_t>(cand) + l] == data[i + l]) ++l;
        if (static_cast<int>(l) > best_len) {
          best_len = static_cast<int>(l);
          best_pos = static_cast<std::size_t>(cand);
          if (best_len >= kMaxMatch) break;
        }
        cand = prev[static_cast<std::size_t>(cand)];
        ++chain;
      }
      prev[i] = head[h];
      head[h] = static_cast<int>(i);
    }

    if (best_len >= kMinMatch) {
      tokens.push_back(Token{static_cast<std::uint16_t>(best_len),
                             static_cast<std::uint16_t>(i - best_pos)});
      for (int k = 1; k < best_len; ++k) {
        const std::size_t p = i + static_cast<std::size_t>(k);
        if (p + static_cast<std::size_t>(kMinMatch) <= size) {
          const std::uint32_t h = hash_at(p);
          prev[p] = head[h];
          head[h] = static_cast<int>(p);
        }
      }
      i += static_cast<std::size_t>(best_len);
    } else {
      tokens.push_back(Token{static_cast<std::uint16_t>(data[i]), 0});
      ++i;
    }
  }
}

// ---------------------------------------------------------------------------
// Dynamic-Huffman deflate block
// ---------------------------------------------------------------------------

const int kLitAlphabet = 286; // 0..255 literals, 256 end-of-block, 257..285 lengths
const int kDistAlphabet = 30;
const int kClAlphabet = 19;

const int kClOrder[kClAlphabet] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

struct ClItem {
  std::uint8_t sym;
  std::uint16_t extra;
  std::uint8_t extra_bits;
};

void encode_run_lengths(const std::vector<std::uint8_t> &lens, std::vector<ClItem> &items) {
  items.clear();
  std::size_t i = 0;
  while (i < lens.size()) {
    const std::uint8_t v = lens[i];
    std::size_t run = 1;
    while (i + run < lens.size() && lens[i + run] == v) ++run;

    if (v == 0) {
      std::size_t left = run;
      while (left >= 11) {
        const std::size_t c = left > 138 ? 138 : left;
        items.push_back(ClItem{18, static_cast<std::uint16_t>(c - 11), 7});
        left -= c;
      }
      while (left >= 3) {
        const std::size_t c = left > 10 ? 10 : left;
        items.push_back(ClItem{17, static_cast<std::uint16_t>(c - 3), 3});
        left -= c;
      }
      while (left > 0) {
        items.push_back(ClItem{0, 0, 0});
        --left;
      }
    } else {
      items.push_back(ClItem{v, 0, 0});
      std::size_t left = run - 1;
      while (left >= 3) {
        const std::size_t c = left > 6 ? 6 : left;
        items.push_back(ClItem{16, static_cast<std::uint16_t>(c - 3), 2});
        left -= c;
      }
      while (left > 0) {
        items.push_back(ClItem{v, 0, 0});
        --left;
      }
    }
    i += run;
  }
}

void put_huff(BitWriter &w, const HuffCode &c) { w.put(reverse_bits(c.code, c.len), c.len); }

void deflate_dynamic(const std::uint8_t *data, std::size_t size, std::vector<std::uint8_t> &out) {
  std::vector<Token> tokens;
  tokenize(data, size, tokens);

  std::vector<std::uint32_t> lit_freq(kLitAlphabet, 0);
  std::vector<std::uint32_t> dist_freq(kDistAlphabet, 0);
  for (const Token &t : tokens) {
    if (t.dist == 0) {
      lit_freq[t.len]++;
    } else {
      int lit_extra = 0, dist_extra = 0;
      lit_freq[static_cast<std::size_t>(len_symbol(t.len, lit_extra))]++;
      dist_freq[static_cast<std::size_t>(dist_symbol(t.dist, dist_extra))]++;
    }
  }
  lit_freq[256]++; // end-of-block

  const std::vector<std::uint8_t> lit_len = build_lengths(lit_freq, 15);
  const std::vector<std::uint8_t> dist_len = build_lengths(dist_freq, 15);

  int hlit = kLitAlphabet;
  while (hlit > 257 && lit_len[static_cast<std::size_t>(hlit - 1)] == 0) --hlit;
  int hdist = kDistAlphabet;
  while (hdist > 1 && dist_len[static_cast<std::size_t>(hdist - 1)] == 0) --hdist;

  std::vector<std::uint8_t> cl_lens;
  cl_lens.reserve(static_cast<std::size_t>(hlit + hdist));
  cl_lens.insert(cl_lens.end(), lit_len.begin(), lit_len.begin() + hlit);
  cl_lens.insert(cl_lens.end(), dist_len.begin(), dist_len.begin() + hdist);

  std::vector<ClItem> items;
  encode_run_lengths(cl_lens, items);

  std::vector<std::uint32_t> cl_freq(kClAlphabet, 0);
  for (const ClItem &it : items) cl_freq[it.sym]++;

  const std::vector<std::uint8_t> cl_len = build_lengths(cl_freq, 7);
  const std::vector<HuffCode> cl_codes = build_codes(cl_len);

  int hclen = kClAlphabet;
  while (hclen > 4 && cl_len[static_cast<std::size_t>(kClOrder[hclen - 1])] == 0) --hclen;

  BitWriter w(out);
  w.put(1, 1); // BFINAL
  w.put(2, 2); // BTYPE = 10, dynamic Huffman
  w.put(static_cast<std::uint32_t>(hlit - 257), 5);
  w.put(static_cast<std::uint32_t>(hdist - 1), 5);
  w.put(static_cast<std::uint32_t>(hclen - 4), 4);
  for (int i = 0; i < hclen; ++i)
    w.put(cl_len[static_cast<std::size_t>(kClOrder[i])], 3);
  for (const ClItem &it : items) {
    put_huff(w, cl_codes[it.sym]);
    if (it.extra_bits > 0) w.put(it.extra, it.extra_bits);
  }

  const std::vector<HuffCode> lit_codes = build_codes(lit_len);
  const std::vector<HuffCode> dist_codes = build_codes(dist_len);

  for (const Token &t : tokens) {
    if (t.dist == 0) {
      put_huff(w, lit_codes[t.len]);
    } else {
      int lit_extra = 0, dist_extra = 0;
      const int lsym = len_symbol(t.len, lit_extra);
      put_huff(w, lit_codes[static_cast<std::size_t>(lsym)]);
      const int lbits = kLenExtra[lsym - 257];
      if (lbits > 0) w.put(static_cast<std::uint32_t>(lit_extra), lbits);

      const int dsym = dist_symbol(t.dist, dist_extra);
      put_huff(w, dist_codes[static_cast<std::size_t>(dsym)]);
      const int dbits = kDistExtra[dsym];
      if (dbits > 0) w.put(static_cast<std::uint32_t>(dist_extra), dbits);
    }
  }
  put_huff(w, lit_codes[256]);
  w.flush();
}

// ---------------------------------------------------------------------------
// PNG chunks
// ---------------------------------------------------------------------------

void append_u32be(std::vector<std::uint8_t> &out, std::uint32_t v) {
  out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>(v & 0xFFu));
}

void append_chunk(std::vector<std::uint8_t> &out, const char type[4], const std::uint8_t *data,
                  std::size_t n) {
  append_u32be(out, static_cast<std::uint32_t>(n));
  const std::size_t crc_start = out.size();
  for (int k = 0; k < 4; ++k) out.push_back(static_cast<std::uint8_t>(type[k]));
  if (n > 0) out.insert(out.end(), data, data + n);
  append_u32be(out, crc32(&out[crc_start], 4 + n));
}

// Picks the cheapest of the three PNG filters for one row (sum of absolute
// differences, the heuristic libpng uses) and stores the filtered bytes.
void filter_row(const std::uint8_t *cur, const std::uint8_t *up, std::size_t stride,
                std::uint8_t *dst, std::vector<std::uint8_t> &scratch) {
  int best = 0;
  std::uint64_t cost[3] = {0, 0, 0};
  scratch.assign(static_cast<std::size_t>(3) * stride, 0);

  for (std::size_t x = 0; x < stride; ++x) {
    const std::uint8_t raw = cur[x];
    const std::uint8_t left = (x >= 3) ? cur[x - 3] : 0u;
    const std::uint8_t above = up ? up[x] : 0u;
    scratch[x] = raw;
    scratch[stride + x] = static_cast<std::uint8_t>(raw - left);
    scratch[2 * stride + x] = static_cast<std::uint8_t>(raw - above);
    for (int f = 0; f < 3; ++f) {
      const int s = static_cast<int>(static_cast<signed char>(scratch[static_cast<std::size_t>(f) * stride + x]));
      cost[f] += static_cast<std::uint64_t>(s < 0 ? -s : s);
    }
  }

  if (cost[1] < cost[best]) best = 1;
  if (cost[2] < cost[best]) best = 2;
  dst[0] = static_cast<std::uint8_t>(best);
  for (std::size_t x = 0; x < stride; ++x)
    dst[1 + x] = scratch[static_cast<std::size_t>(best) * stride + x];
}

} // namespace

bool write_png_rgb(const std::string &path, const std::uint8_t *rgb, std::uint32_t width,
                   std::uint32_t height) {
  if (rgb == nullptr || width == 0 || height == 0) return false;

  const std::size_t stride = static_cast<std::size_t>(width) * 3;
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(stride + 1) * height);
  std::vector<std::uint8_t> row(stride + 1);
  std::vector<std::uint8_t> scratch;

  for (std::uint32_t y = 0; y < height; ++y) {
    const std::uint8_t *cur = rgb + static_cast<std::size_t>(y) * stride;
    const std::uint8_t *up = (y > 0) ? rgb + static_cast<std::size_t>(y - 1) * stride : nullptr;
    filter_row(cur, up, stride, row.data(), scratch);
    std::copy(row.begin(), row.end(), raw.begin() + static_cast<std::size_t>(y) * (stride + 1));
  }

  std::vector<std::uint8_t> idat;
  idat.reserve(raw.size() / 2 + 1024);
  idat.push_back(0x78); // zlib: CM = 8 (deflate), CINFO = 7 (32 KiB window)
  idat.push_back(0x01); // no preset dictionary, default compression level, valid check bits
  deflate_dynamic(raw.data(), raw.size(), idat);
  append_u32be(idat, adler32(raw.data(), raw.size()));

  std::vector<std::uint8_t> png;
  png.reserve(idat.size() + 64);
  png.insert(png.end(), {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A});

  std::uint8_t ihdr[13];
  ihdr[0] = static_cast<std::uint8_t>((width >> 24) & 0xFFu);
  ihdr[1] = static_cast<std::uint8_t>((width >> 16) & 0xFFu);
  ihdr[2] = static_cast<std::uint8_t>((width >> 8) & 0xFFu);
  ihdr[3] = static_cast<std::uint8_t>(width & 0xFFu);
  ihdr[4] = static_cast<std::uint8_t>((height >> 24) & 0xFFu);
  ihdr[5] = static_cast<std::uint8_t>((height >> 16) & 0xFFu);
  ihdr[6] = static_cast<std::uint8_t>((height >> 8) & 0xFFu);
  ihdr[7] = static_cast<std::uint8_t>(height & 0xFFu);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // colour type: truecolour (RGB)
  ihdr[10] = 0; // compression method: deflate
  ihdr[11] = 0; // filter method 0
  ihdr[12] = 0; // no interlace
  append_chunk(png, "IHDR", ihdr, sizeof(ihdr));
  append_chunk(png, "IDAT", idat.data(), idat.size());
  append_chunk(png, "IEND", nullptr, 0);

  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char *>(png.data()), static_cast<std::streamsize>(png.size()));
  return out.good();
}
