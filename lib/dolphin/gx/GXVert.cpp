#include "gx.hpp"
#include "__gx.h"
#include "../../gx/fifo.hpp"

// Track vertex count between GXBegin/GXEnd for mismatch detection
static u16 sBeginNVerts = 0;
static u32 sBeginFifoSize = 0;
static bool sInBegin = false;

// GX_AUTO: offset of the u32 byte-length placeholder to patch in GXEnd
static u32 sBeginSizeOffset = 0;
static bool sBeginAuto = false;

namespace aurora::gx::fifo {
// The GX thread's kick (fifo.cpp) asks whether the FIFO can be handed over here. Only an open
// GX_AUTO draw can't: its GXEnd patches the size by offset into this buffer. Any other open
// GXBegin has all its vertices written by the time the game reaches a kick point (between GX
// calls) -- the game simply never called GXEnd, an empty macro in the GameCube's release SDK,
// which TTYD leaves out -- so it is closed here, and a late GXEnd won't measure across buffers.
bool __gx_in_begin() noexcept {
  if (sInBegin && !sBeginAuto) {
    sInBegin = false;
  }
  return sInBegin;
}
} // namespace aurora::gx::fifo

extern "C" {

void GXBegin(GXPrimitive primitive, GXVtxFmt vtxFmt, u16 nVerts) {
  CHECK(!sInBegin, "GXBegin: called without matching GXEnd");

  // Flush dirty state before starting a draw
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Flush pending primitives if needed
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }

  const u8 drawCmd = static_cast<u8>(vtxFmt) | static_cast<u8>(primitive);
  sBeginAuto = nVerts == GX_AUTO;
  if (sBeginAuto) {
    ASSERT(!aurora::gx::fifo::in_display_list(), "GXBegin: GX_AUTO not supported in display lists");
    GX_WRITE_AURORA(GX_AURORA_DRAW_SIZED);
    GX_WRITE_U8(drawCmd);
    sBeginSizeOffset = aurora::gx::fifo::get_buffer_size();
    GX_WRITE_U32(0);
  } else {
    GX_WRITE_U8(drawCmd);
    GX_WRITE_U16(nVerts);
  }

  // Record state for vertex count validation in GXEnd
  sBeginNVerts = nVerts;
  sBeginFifoSize = aurora::gx::fifo::get_buffer_size();
  sInBegin = true;
}

void GXEnd() {
  if (sInBegin) {
    u32 bytesWritten = aurora::gx::fifo::get_buffer_size() - sBeginFifoSize;
    if (sBeginAuto) {
      aurora::gx::fifo::patch_u32(sBeginSizeOffset, bytesWritten);
      sBeginAuto = false;
    } else if (sBeginNVerts > 0 && bytesWritten > 0) {
      u32 vtxSize = bytesWritten / sBeginNVerts;
      u32 remainder = bytesWritten % sBeginNVerts;
      if (remainder != 0) {
        Log.warn("GXEnd: vertex data not evenly divisible: {} bytes for {} vertices", bytesWritten, sBeginNVerts);
      }
      u32 actualVerts = (vtxSize > 0) ? bytesWritten / vtxSize : 0;
      CHECK(actualVerts == sBeginNVerts,
            "GXEnd: vertex count mismatch: GXBegin declared {} vertices ({}B each, {}B total) "
            "but {} bytes were written ({} vertices)",
            sBeginNVerts, vtxSize, sBeginNVerts * vtxSize, bytesWritten, actualVerts);
    }
    sInBegin = false;
  }
  // TEMP: debugging aid
  // if (!aurora::gx::fifo::in_display_list()) {
  //   aurora::gx::fifo::drain();
  // }
}

void GXPosition3f32(f32 x, f32 y, f32 z) {
  GX_WRITE_F32(x);
  GX_WRITE_F32(y);
  GX_WRITE_F32(z);
}

void GXPosition3u16(u16 x, u16 y, u16 z) {
  GX_WRITE_U16(x);
  GX_WRITE_U16(y);
  GX_WRITE_U16(z);
}

void GXPosition3s16(s16 x, s16 y, s16 z) {
  GX_WRITE_U16(static_cast<u16>(x));
  GX_WRITE_U16(static_cast<u16>(y));
  GX_WRITE_U16(static_cast<u16>(z));
}

void GXPosition3u8(u8 x, u8 y, u8 z) {
  GX_WRITE_U8(x);
  GX_WRITE_U8(y);
  GX_WRITE_U8(z);
}

void GXPosition3s8(s8 x, s8 y, s8 z) {
  GX_WRITE_U8(static_cast<u8>(x));
  GX_WRITE_U8(static_cast<u8>(y));
  GX_WRITE_U8(static_cast<u8>(z));
}

void GXPosition2f32(f32 x, f32 y) {
  GX_WRITE_F32(x);
  GX_WRITE_F32(y);
}

void GXPosition2u16(u16 x, u16 y) {
  GX_WRITE_U16(x);
  GX_WRITE_U16(y);
}

void GXPosition2s16(s16 x, s16 y) {
  GX_WRITE_U16(static_cast<u16>(x));
  GX_WRITE_U16(static_cast<u16>(y));
}

void GXPosition2u8(u8 x, u8 y) {
  GX_WRITE_U8(x);
  GX_WRITE_U8(y);
}

void GXPosition2s8(s8 x, s8 y) {
  GX_WRITE_U8(static_cast<u8>(x));
  GX_WRITE_U8(static_cast<u8>(y));
}

void GXPosition1x16(u16 idx) { GX_WRITE_U16(idx); }

void GXPosition1x8(u8 idx) { GX_WRITE_U8(idx); }

void GXNormal3f32(f32 x, f32 y, f32 z) {
  GX_WRITE_F32(x);
  GX_WRITE_F32(y);
  GX_WRITE_F32(z);
}

void GXNormal3s16(s16 x, s16 y, s16 z) {
  GX_WRITE_U16(static_cast<u16>(x));
  GX_WRITE_U16(static_cast<u16>(y));
  GX_WRITE_U16(static_cast<u16>(z));
}

void GXNormal3s8(s8 x, s8 y, s8 z) {
  GX_WRITE_U8(static_cast<u8>(x));
  GX_WRITE_U8(static_cast<u8>(y));
  GX_WRITE_U8(static_cast<u8>(z));
}

void GXNormal1x16(u16 index) { GX_WRITE_U16(index); }

void GXNormal1x8(u8 index) { GX_WRITE_U8(index); }

void GXColor4f32(f32 r, f32 g, f32 b, f32 a) {
  GX_WRITE_U8(static_cast<u8>(r * 255.0));
  GX_WRITE_U8(static_cast<u8>(g * 255.0));
  GX_WRITE_U8(static_cast<u8>(b * 255.0));
  GX_WRITE_U8(static_cast<u8>(a * 255.0));
}

void GXColor4u8(u8 r, u8 g, u8 b, u8 a) {
  GX_WRITE_U8(r);
  GX_WRITE_U8(g);
  GX_WRITE_U8(b);
  GX_WRITE_U8(a);
}

void GXColor3u8(u8 r, u8 g, u8 b) {
  GX_WRITE_U8(r);
  GX_WRITE_U8(g);
  GX_WRITE_U8(b);
}

void GXColor1u32(u32 clr) { GX_WRITE_U32(clr); }

void GXColor1u16(u16 clr) { GX_WRITE_U16(clr); }

void GXColor1x16(u16 index) { GX_WRITE_U16(index); }

void GXColor1x8(u8 index) { GX_WRITE_U8(index); }

void GXTexCoord2f32(f32 s, f32 t) {
  GX_WRITE_F32(s);
  GX_WRITE_F32(t);
}

void GXTexCoord2u16(u16 s, u16 t) {
  GX_WRITE_U16(s);
  GX_WRITE_U16(t);
}

void GXTexCoord2s16(s16 s, s16 t) {
  GX_WRITE_U16(static_cast<u16>(s));
  GX_WRITE_U16(static_cast<u16>(t));
}

void GXTexCoord2u8(u8 s, u8 t) {
  GX_WRITE_U8(s);
  GX_WRITE_U8(t);
}

void GXTexCoord2s8(s8 s, s8 t) {
  GX_WRITE_U8(static_cast<u8>(s));
  GX_WRITE_U8(static_cast<u8>(t));
}

void GXTexCoord1f32(f32 s) { GX_WRITE_F32(s); }

void GXTexCoord1u16(u16 s) { GX_WRITE_U16(s); }

void GXTexCoord1s16(s16 s) { GX_WRITE_U16(static_cast<u16>(s)); }

void GXTexCoord1u8(u8 s) { GX_WRITE_U8(s); }

void GXTexCoord1s8(s8 s) { GX_WRITE_U8(static_cast<u8>(s)); }

void GXTexCoord1x16(u16 index) { GX_WRITE_U16(index); }

void GXTexCoord1x8(u8 index) { GX_WRITE_U8(index); }

void GXCmd1u8(const u8 x) { aurora::gx::fifo::write_u8(x); }
void GXCmd1u16(const u16 x) { aurora::gx::fifo::write_u16(x); }
void GXCmd1u32(const u32 x) { aurora::gx::fifo::write_u32(x); }
void GXCmd1u64(const u64 x) { aurora::gx::fifo::write_u64(x); }

void GXParam1u8(const u8 x) { aurora::gx::fifo::write_u8(x); }
void GXParam1u16(const u16 x) { aurora::gx::fifo::write_u16(x); }
void GXParam1u32(const u32 x) { aurora::gx::fifo::write_u32(x); }
void GXParam1s8(const s8 x) { aurora::gx::fifo::write_u8(static_cast<uint8_t>(x)); }
void GXParam1s16(const s16 x) { aurora::gx::fifo::write_u16(static_cast<uint16_t>(x)); }
void GXParam1s32(const s32 x) { aurora::gx::fifo::write_u32(static_cast<uint32_t>(x)); }
void GXParam1f32(const f32 x) { aurora::gx::fifo::write_f32(x); }
void GXParam3f32(const f32 x, const f32 y, const f32 z) {
  aurora::gx::fifo::write_f32(x);
  aurora::gx::fifo::write_f32(y);
  aurora::gx::fifo::write_f32(z);
}
void GXParam4f32(const f32 x, const f32 y, const f32 z, const f32 w) {
  aurora::gx::fifo::write_f32(x);
  aurora::gx::fifo::write_f32(y);
  aurora::gx::fifo::write_f32(z);
  aurora::gx::fifo::write_f32(w);
}

static inline u8 prepare_anim_trifan_cmd(u8 vtxFmt) {
  CHECK(!sInBegin, "GXEmitAnimTriFan16: called without matching GXEnd");
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }
  return static_cast<u8>(vtxFmt & 7u) | static_cast<u8>(GX_TRIANGLEFAN);
}

static inline void write_be16_header(u8* dst, u8 drawCmd, u16 count) {
  dst[0] = drawCmd;
  const u16 countBe = bswap(count);
  std::memcpy(dst + 1, &countBe, 2);
}

void GXEmitAnimTriFan16_1Tex(u8 vtxFmt, u16 count, const void* posIdx, const void* nrmIdx, const void* clr0Idx,
                             const void* texIdx) {
  using namespace aurora::gx::fifo;
  const u8 drawCmd = prepare_anim_trifan_cmd(vtxFmt);
  const u32 totalBytes = 3u + static_cast<u32>(count) * 8u;
  const auto* p = static_cast<const u8*>(posIdx);
  const auto* n = static_cast<const u8*>(nrmIdx);
  const auto* c = static_cast<const u8*>(clr0Idx);
  const auto* t = static_cast<const u8*>(texIdx);

  if (!detail::sInDisplayList && detail::sBufferSize + totalBytes <= detail::sBufferCapacity)
    LIKELY {
      u8* dst = detail::sBufferData + detail::sBufferSize;
      detail::sBufferSize += totalBytes;
      write_be16_header(dst, drawCmd, count);
      dst += 3;
      for (u16 k = 0; k < count; ++k, dst += 8, p += 4, n += 4, c += 4, t += 4) {
        std::memcpy(dst + 0, p, 2);
        std::memcpy(dst + 2, n, 2);
        std::memcpy(dst + 4, c, 2);
        std::memcpy(dst + 6, t, 2);
      }
      return;
    }

  if (totalBytes <= 2048u) {
    u8 tmp[2048];
    write_be16_header(tmp, drawCmd, count);
    u8* dst = tmp + 3;
    for (u16 k = 0; k < count; ++k, dst += 8, p += 4, n += 4, c += 4, t += 4) {
      std::memcpy(dst + 0, p, 2);
      std::memcpy(dst + 2, n, 2);
      std::memcpy(dst + 4, c, 2);
      std::memcpy(dst + 6, t, 2);
    }
    write_data(tmp, totalBytes);
    return;
  }

  u8 hdr[3];
  write_be16_header(hdr, drawCmd, count);
  write_data(hdr, 3);
  for (u16 k = 0; k < count; ++k, p += 4, n += 4, c += 4, t += 4) {
    u8 vtx[8];
    std::memcpy(vtx + 0, p, 2);
    std::memcpy(vtx + 2, n, 2);
    std::memcpy(vtx + 4, c, 2);
    std::memcpy(vtx + 6, t, 2);
    write_data(vtx, 8);
  }
}

void GXEmitAnimTriFan16_NTex(u8 vtxFmt, u16 count, const void* posIdx, const void* nrmIdx, const void* clr0Idx,
                             const void* const* texIdxPtrs, u32 numTex) {
  using namespace aurora::gx::fifo;
  const u8 drawCmd = prepare_anim_trifan_cmd(vtxFmt);
  const u32 strideBytes = 6u + 2u * numTex;
  const u32 totalBytes = 3u + static_cast<u32>(count) * strideBytes;
  const auto* p = static_cast<const u8*>(posIdx);
  const auto* n = static_cast<const u8*>(nrmIdx);
  const auto* c = static_cast<const u8*>(clr0Idx);

  if (!detail::sInDisplayList && detail::sBufferSize + totalBytes <= detail::sBufferCapacity)
    LIKELY {
      u8* dst = detail::sBufferData + detail::sBufferSize;
      detail::sBufferSize += totalBytes;
      write_be16_header(dst, drawCmd, count);
      dst += 3;
      for (u16 k = 0; k < count; ++k, p += 4, n += 4, c += 4) {
        std::memcpy(dst + 0, p, 2);
        std::memcpy(dst + 2, n, 2);
        std::memcpy(dst + 4, c, 2);
        dst += 6;
        const size_t byteOff = static_cast<size_t>(k) * 4u;
        for (u32 ti = 0; ti < numTex; ++ti, dst += 2) {
          std::memcpy(dst, static_cast<const u8*>(texIdxPtrs[ti]) + byteOff, 2);
        }
      }
      return;
    }

  u8 hdr[3];
  write_be16_header(hdr, drawCmd, count);
  write_data(hdr, 3);
  for (u16 k = 0; k < count; ++k, p += 4, n += 4, c += 4) {
    u8 vtx[22];
    std::memcpy(vtx + 0, p, 2);
    std::memcpy(vtx + 2, n, 2);
    std::memcpy(vtx + 4, c, 2);
    const size_t byteOff = static_cast<size_t>(k) * 4u;
    for (u32 ti = 0; ti < numTex && ti < 8u; ++ti) {
      std::memcpy(vtx + 6 + ti * 2u, static_cast<const u8*>(texIdxPtrs[ti]) + byteOff, 2);
    }
    write_data(vtx, strideBytes <= sizeof(vtx) ? strideBytes : sizeof(vtx));
  }
}

} // extern "C"
