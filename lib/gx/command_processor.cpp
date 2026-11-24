#include "command_processor.hpp"

#include "../gfx/common.hpp"
#include "../gfx/depth_peek.hpp"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "gx_fmt.hpp"
#include "pipeline.hpp"
#include "shader_info.hpp"
#include "../internal.hpp"
#include <aurora/paper.h>

#include <absl/container/flat_hash_map.h>
#include <limits>
#include <memory>
#include <tracy/Tracy.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <vector>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

// Bumped by GX_CMD_INVL_VC (GXInvalidateVtxCache); invalidates the memoized attribute
// array content hashes the native geometry cache keys indexed draws by.
static u32 s_gxArrayInvalidateGen = 0;
// Bumped whenever an attribute array binding (data/size/le) changes or content is invalidated,
// refreshing per-format layoutMetaHash without blowing away unrelated arrays in s_arrayHashMemo.
static u32 s_gxArrayBindGen = 0;
// Bumped ONLY when disc-backed (!le) blobs/memory ranges are forgotten, moved, or reloaded,
// so read-only big-endian map/model arrays do not need to be re-hashed every frame or per NPC.
static u32 s_gxDiscArrayInvalidateGen = 0;

// Generation counters for O(1) vertex/matrix/cache-pointer validation in hot draw loops.
static std::array<u32, MaxPnMtx> s_pnMtxGen = []() constexpr {
  std::array<u32, MaxPnMtx> g{};
  for (u32 i = 0; i < MaxPnMtx; ++i) g[i] = i + 1;
  return g;
}();
static u32 s_pnMtxNextGen = MaxPnMtx + 1;
static std::array<u32, MaxVtxFmt> s_vtxFmtGen = {1, 1, 1, 1, 1, 1, 1, 1};
static std::array<u32, 12> s_arrayStateGen = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static u32 s_vtxStateDirtyGen = 1;
static u32 s_curVtxStateVerifiedGen = 0;
static u16 s_curVtxStateVerifiedId = UINT16_MAX;
static u64 s_vtxDescPacked = 0;
static bool s_vtxDescPackedDirty = true;

static inline u64 current_vtx_desc_bits() noexcept {
  if (s_vtxDescPackedDirty) UNLIKELY {
    static_assert(MaxVtxAttr * 2 <= 64);
    u64 bits = 0;
    for (u32 a = 0; a < MaxVtxAttr; ++a) {
      bits |= (static_cast<u64>(g_gxState.vtxDesc[a]) & 3ull) << (a * 2u);
    }
    s_vtxDescPacked = bits;
    s_vtxDescPackedDirty = false;
  }
  return s_vtxDescPacked;
}

static u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));

    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    buf.reserve_extra(vtxCount * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    buf.reserve_extra(((u32(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    buf.reserve_extra(((static_cast<u32>(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

// Emit index data that keeps a triangle strip AS a strip on the GPU (TriangleStrip
// topology), rather than unrolling it into an independent triangle list. `restart`
// prepends a 0xffff primitive-restart index so an earlier strip in the same index
// buffer ends before this one begins. Returns the index count written (incl. restart).
static u16 build_triangle_strip_topology_indices(ByteBuffer& buf, u16 vtxStart, u16 vtxCount, bool restart) {
  const u16 numIndices = static_cast<u16>(vtxCount + (restart ? 1 : 0));
  buf.reserve_extra(static_cast<size_t>(numIndices) * sizeof(u16));
  if (restart) {
    buf.append<u16>(0xffff);
  }
  for (u16 v = 0; v < vtxCount; ++v) {
    buf.append(static_cast<u16>(vtxStart + v));
  }
  return numIndices;
}

// Concatenate several strips into one index buffer with 0xffff restart markers between
// them, so a whole run of adjacent strip draws collapses into a single DrawIndexed.
static u32 build_triangle_strip_batch_topology_indices(ByteBuffer& buf, const std::vector<u16>& stripCounts) {
  u32 indexCount = 0;
  u16 vtxStart = 0;
  for (size_t i = 0; i < stripCounts.size(); ++i) {
    const bool restart = i != 0;
    indexCount += build_triangle_strip_topology_indices(buf, vtxStart, stripCounts[i], restart);
    vtxStart = static_cast<u16>(vtxStart + stripCounts[i]);
  }
  return indexCount;
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
static constexpr u8 CP_CMD_NOP = GX_NOP;
static constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
static constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
static constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
static constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
static constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
static constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
static constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
static constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
static constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
static constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
static constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

// Read helpers for big/little endian
#if _MSC_VER
template <typename T>
__forceinline // Yes, this was necessary.
    inline T unaligned_load(const T* ptr) {
  return *static_cast<const __unaligned T*>(ptr);
}
#else
template <typename T>
inline T unaligned_load(const T* ptr) {
  T copy;
  memcpy(&copy, ptr, sizeof(T));
  return copy;
}
#endif

static inline u16 read_u16(const u8* ptr, bool bigEndian) {
  const u16 val = unaligned_load(reinterpret_cast<const u16*>(ptr));
  if (bigEndian) {
    return bswap(val);
  }
  return val;
}

static inline u32 read_u32(const u8* ptr, bool bigEndian) {
  const u32 val = unaligned_load(reinterpret_cast<const u32*>(ptr));
  if (bigEndian) {
    return bswap(val);
  }
  return val;
}

static u32 bp_get(u32 reg, u32 size, u32 shift);

static GXPixelFmt decode_pixel_fmt(u32 peCtrl, u32 cmode1) {
  switch (bp_get(peCtrl, 3, 0)) {
  case 0:
    return GX_PF_RGB8_Z24;
  case 1:
    return GX_PF_RGBA6_Z24;
  case 2:
    return GX_PF_RGB565_Z16;
  case 3:
    return GX_PF_Z24;
  case 4:
    switch (bp_get(cmode1, 2, 9)) {
    case 0:
      return GX_PF_Y8;
    case 1:
      return GX_PF_U8;
    case 2:
      return GX_PF_V8;
    default:
      Log.warn("command_processor: unsupported cmode1 pixel subtype {}", bp_get(cmode1, 2, 9));
      return GX_PF_Y8;
    }
  case 5:
    return GX_PF_YUV420;
  default:
    Log.warn("command_processor: unsupported PE pixel format {}", bp_get(peCtrl, 3, 0));
    return GX_PF_RGB8_Z24;
  }
}

static inline u64 read_u64(const u8* ptr, bool bigEndian) {
  u64 loaded;
  // Unaligned-safe load
  memcpy(&loaded, ptr, sizeof(u64));

  if (bigEndian) {
    return bswap(loaded);
  }

  return loaded;
}

struct TexBpRegMapping {
  u8 texMapId;
  enum class Kind : uint8_t { Mode0, Mode1, Image0, Image1, Image2, Image3, Tlut } kind;
};

static inline std::optional<TexBpRegMapping> decode_tex_bp_reg(u32 regId) {
  if (regId >= 0x80u && regId <= 0x9Bu) {
    const u32 off = regId - 0x80u;
    return TexBpRegMapping{
        .texMapId = static_cast<u8>(off & 3u),
        .kind = static_cast<TexBpRegMapping::Kind>(off >> 2),
    };
  }
  if (regId >= 0xA0u && regId <= 0xBBu) {
    const u32 off = regId - 0xA0u;
    return TexBpRegMapping{
        .texMapId = static_cast<u8>(4u + (off & 3u)),
        .kind = static_cast<TexBpRegMapping::Kind>(off >> 2),
    };
  }
  return std::nullopt;
}

// Helper to convert packed RGBA8 to Vec4<float>
static Vec4<float> unpack_color(u32 packed) {
  return {
      static_cast<float>(packed >> 24 & 0xFF) / 255.f,
      static_cast<float>(packed >> 16 & 0xFF) / 255.f,
      static_cast<float>(packed >> 8 & 0xFF) / 255.f,
      static_cast<float>(packed & 0xFF) / 255.f,
  };
}

static inline f32 read_f32(const u8* ptr, bool bigEndian) {
  u32 bits = read_u32(ptr, bigEndian);
  f32 val;
  std::memcpy(&val, &bits, sizeof(val));
  return val;
}

static inline bool copy_xf_flat_mtx(void* dst, const u8* data, u32 len, bool bigEndian) {
  u32* flatU = reinterpret_cast<u32*>(dst);
  f32* flat = reinterpret_cast<f32*>(dst);
  bool changed = false;
  for (u32 i = 0; i < len; i++) {
    const u32 bits = read_u32(data + i * 4, bigEndian);
    if (flatU[i] != bits) {
      f32 v;
      std::memcpy(&v, &bits, sizeof(v));
      if (flat[i] != v) {
        changed = true;
      }
      flatU[i] = bits;
    }
  }
  return changed;
}

static u32 s_splitTexMtxGen = 0;
static u32 s_splitLightsGen = 0;

static bool copy_xf_data(u32 addr, const u8* data, u32 len, bool bigEndian) {
  if (addr < 0x78) {
    // Position matrices (0x0000 - 0x0077)
    u32 mtxIdx = addr / 12;
    u32 startOffset = addr % 12;
    // We only support full writes to matrices
    CHECK(mtxIdx < MaxPnMtx, "XF: PosMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 12, "XF: PosMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    if (copy_xf_flat_mtx(&g_gxState.pnMtx[mtxIdx].pos, data, len, bigEndian)) {
      g_gxState.stateDirty = true;
      s_pnMtxGen[mtxIdx] = ++s_pnMtxNextGen;
    }
    return true;
  } else if (addr < 0x0F0) {
    // Texture matrices (0x078-0x0EF)
    u32 texBase = addr - 0x078;
    u32 mtxIdx = texBase / 12;
    u32 startOffset = texBase % 12;
    CHECK(mtxIdx < MaxTexMtx, "XF TexMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && (len == 8 || len == 12), "XF TexMtx sub-copy unsupported: offs={}, len={}", startOffset,
          len);

    // Determine if 2x4 or 3x4 from count
    if (copy_xf_flat_mtx(&g_gxState.texMtxs[mtxIdx], data, len, bigEndian)) {
      g_gxState.stateDirty = true;
      ++s_splitTexMtxGen;
    }
    return true;
  } else if (addr >= 0x400 && addr < 0x45A) {
    // Normal matrices (0x400-0x459)
    u32 nrmBase = addr - 0x400;
    u32 mtxIdx = nrmBase / 9;
    u32 startOffset = nrmBase % 9;
    // We only support full writes to matrices
    CHECK(mtxIdx < MaxPnMtx, "XF: NrmMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 9, "XF: NrmMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    auto& mtx = g_gxState.pnMtx[mtxIdx].nrm;
    u32* flatU = reinterpret_cast<u32*>(&mtx);
    f32* flat = reinterpret_cast<f32*>(&mtx);
    bool changed = false;
    static constexpr uint8_t s_nrmIdx[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};
    for (u32 i = 0; i < len; i++) {
      const u32 bits = read_u32(data + i * 4, bigEndian);
      const u32 dstIdx = s_nrmIdx[i];
      if (flatU[dstIdx] != bits) {
        f32 v;
        std::memcpy(&v, &bits, sizeof(v));
        if (flat[dstIdx] != v) {
          changed = true;
        }
        flatU[dstIdx] = bits;
      }
    }
    if (changed) {
      g_gxState.stateDirty = true;
      s_pnMtxGen[mtxIdx] = ++s_pnMtxNextGen;
    }
    return true;
  } else if (addr >= 0x500 && addr < 0x5F0) {
    // Post-transform texture matrices (0x500-0x5EF)
    u32 ptBase = addr - 0x500;
    u32 mtxIdx = ptBase / 12;
    u32 startOffset = ptBase % 12;
    CHECK(mtxIdx < MaxPTTexMtx, "XF: PTTexMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 12, "XF: PTTexMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    if (copy_xf_flat_mtx(&g_gxState.ptTexMtxs[mtxIdx], data, len, bigEndian)) {
      g_gxState.stateDirty = true;
      ++s_splitTexMtxGen;
    }
    return true;
  } else if (addr >= 0x600 && addr < 0x680) {
    // Lights (0x600-0x67F) - 8 lights, 16 values each
    u32 lightBase = addr - 0x600;
    u32 lightIdx = lightBase / 0x10;
    u32 startOffset = lightBase % 0x10;
    CHECK(lightIdx < 8, "XF: Light copy oob? Should never happen; lightIdx={}", lightIdx);
    CHECK(startOffset + len <= 0x10, "XF: Light copy that crosses across light boundaries unsupported: offs={}, len={}",
          startOffset, len);
    auto& light = g_gxState.lights[lightIdx];
    bool changed = false;
    const auto setFloat = [&](f32& dst, f32 src) {
      if (dst != src) {
        dst = src;
        changed = true;
      }
    };
    for (u32 i = 0; i < len; i++) {
      u32 field = startOffset + i;
      f32 val = read_f32(data + i * 4, bigEndian);
      u32 ival = read_u32(data + i * 4, bigEndian);
      switch (field) {
      case 3: { // Color (packed u32)
        const auto c = unpack_color(ival);
        if (light.color.x() != c.x() || light.color.y() != c.y() || light.color.z() != c.z() || light.color.w() != c.w()) {
          light.color = c;
          changed = true;
        }
        break;
      }
      case 4:
        setFloat(light.cosAtt[0], val);
        break; // a0
      case 5:
        setFloat(light.cosAtt[1], val);
        break; // a1
      case 6:
        setFloat(light.cosAtt[2], val);
        break; // a2
      case 7:
        setFloat(light.distAtt[0], val);
        break; // k0
      case 8:
        setFloat(light.distAtt[1], val);
        break; // k1
      case 9:
        setFloat(light.distAtt[2], val);
        break; // k2
      case 10:
        setFloat(light.pos[0], val);
        break; // px
      case 11:
        setFloat(light.pos[1], val);
        break; // py
      case 12:
        setFloat(light.pos[2], val);
        break; // pz
      case 13:
        setFloat(light.dir[0], val);
        break; // nx
      case 14:
        setFloat(light.dir[1], val);
        break; // ny
      case 15:
        setFloat(light.dir[2], val);
        break; // nz
      default:
        break; // padding (0-2)
      }
    }
    if (changed) {
      g_gxState.stateDirty = true;
      ++s_splitLightsGen;
    }
    return true;
  }
  return false;
}

// Forward declarations for register handlers
static inline u32& bp_cache_slot(u32 regId, u32 value) noexcept {
  if (regId >= 0xE0 && regId <= 0xE7 && (value & (1u << 23)) != 0) {
    return g_gxState.tevKColorBpCache[regId - 0xE0];
  }
  return g_gxState.bpRegCache[regId];
}
static bool handle_bp(u32 value, bool bigEndian);
static void handle_cp(u8 addr, u32 value, bool bigEndian);
static void handle_xf(const u8* data, u32& pos, u32 size, bool bigEndian, bool& pipelineChanged);
static void handle_draw(u8 cmd, const u8* data, u32& pos, u32 size, bool bigEndian);
static void handle_aurora(const u8* data, u32& pos, u32 size, bool bigEndian);
static void flush_pending_pnmtx_batch();

namespace {
static bool s_hasLastDrawState = false;
static bool s_hasLastUniformRange = false;
static PipelineConfig s_lastConfig{};
static ShaderConfig s_lastShaderConfig{};
static ShaderInfo s_lastInfo{};
static gfx::PipelineRef s_lastPipeline{};
static size_t s_lastPipeMemoIdx = SIZE_MAX;
static uint64_t s_lastCfgHash = 0;
static GXBindGroups s_lastBindGroups{};
static GXBindGroups s_candidateBg{};
static uint32_t s_lastTextureDirty = 0;
static uint32_t s_lastUniformDirty = 0;
static gfx::Range s_lastUniformRange{};

static u64 s_uniformCacheHits = 0;
static u64 s_uniformCacheMisses = 0;

struct DrainTiming {
  double ms_bp = 0.0;
  double ms_cp = 0.0;
  double ms_xf = 0.0;
  double ms_indx = 0.0;
  double ms_draw_total = 0.0;
  double ms_geom_hash = 0.0;
  double ms_geom_expand = 0.0;
  double ms_pipe_cfg = 0.0;
  double ms_pipe_ref = 0.0;
  double ms_uni_build = 0.0;
  double ms_push_cmd = 0.0;
  double ms_scan = 0.0;
  double ms_tex_resolve = 0.0;
  double ms_bg_build = 0.0;
  double ms_geom_miss = 0.0;
  double ms_static_dl = 0.0;
  double ms_submit = 0.0;
  double ms_flush = 0.0;
  double ms_total = 0.0;
  u32 frames = 0;
  u32 draw_count = 0;
  u32 bg_calls = 0;
  u32 arr_hash_calls = 0;
  u64 arr_hash_bytes = 0;
  u64 fifo_hash_bytes = 0;
  u32 xf_calls = 0;
  // Draw-split diagnostics (only updated when TTYD_TIMING_DETAIL is set)
  u32 split_pairs = 0;
  u32 split_same_pipe = 0;
  u32 split_same_tex = 0;
  u32 split_same_both = 0;
  u32 split_only_pnmtx = 0;
  u32 split_only_tex = 0;
  u32 split_only_pipe = 0;
  u32 split_chg_texmtx = 0;
  u32 split_chg_lights = 0;
  u32 split_chg_bp_uni = 0;
  u32 split_chg_oth_uni = 0;
  u32 split_same_all = 0;
  // Detailed split/flush breakdown
  u32 fl_nonbat = 0;
  u32 fl_topo = 0;
  u32 fl_da = 0;
  u32 fl_tm = 0;
  u32 fl_li = 0;
  u32 fl_bp = 0;
  u32 fl_oth = 0;
  u32 fl_vtx = 0;
  u32 fl_pipe = 0;
  u32 fl_tex = 0;
  u32 fl_mtx = 0;
  u32 skip_fixtg = 0;
  u32 bat_ok = 0;
  u32 bat_single = 0;
  u32 bat_merged = 0;
  u32 oth_amb = 0;
  u32 oth_mat = 0;
  u32 oth_vp = 0;
  u32 oth_proj = 0;
  u32 oth_tdim = 0;
  u32 oth_off = 0;
  u32 bp_fog = 0;
  u32 bp_tcol = 0;
  u32 bp_su = 0;
  u32 bp_ind = 0;
  u32 bp_line = 0;
  u32 op_cull = 0;
  u32 op_z = 0;
  u32 op_blend = 0;
  u32 op_topo = 0;
  u32 op_fog = 0;
  u32 op_attrs = 0;
  u32 op_tev = 0;
  u32 op_chan = 0;
  u32 op_tcg = 0;
  u32 op_acmp = 0;
  u32 op_paper = 0;
  u32 pd_cull = 0;
  u32 pd_z = 0;
  u32 pd_blend = 0;
  u32 pd_topo = 0;
  u32 pd_fog = 0;
  u32 pd_attrs = 0;
  u32 pd_tev = 0;
  u32 pd_chan = 0;
  u32 pd_tcg = 0;
  u32 pd_acmp = 0;
  u32 pd_paper = 0;
  std::array<u32, 256> bp_uni_reg{};
};
static DrainTiming s_drainTiming;

struct BatchUniState {
  u32 vpProjGen = 0;
  u32 offGen = 0;
  u32 bpFogGen = 0;
  u32 bpSuGen = 0;
  u32 bpIndGen = 0;
  u32 bpLineGen = 0;
  std::array<u32, 4> bpTevColGen{};
  std::array<u32, 4> bpKColGen{};
  std::array<u32, 2> othAmbGen{};
  std::array<u32, 2> othMatGen{};
  std::array<u32, MaxTextures> texDimGen{};
};
static BatchUniState s_curUniState{};

enum class UniMismatch : u8 { None = 0, Lights, Bp, Oth };

static inline UniMismatch batch_uniforms_mismatch(const BatchUniState& batchUni, u32 batchLightsGen,
                                                  const ShaderInfo& info, bool ignoreTex0Dim = false) noexcept {
  if (info.lightingEnabled && batchLightsGen != s_splitLightsGen) {
    return UniMismatch::Lights;
  }
  if (batchUni.vpProjGen != s_curUniState.vpProjGen || batchUni.offGen != s_curUniState.offGen) {
    return UniMismatch::Oth;
  }
  if (info.usesFog && batchUni.bpFogGen != s_curUniState.bpFogGen) {
    return UniMismatch::Bp;
  }
  if ((info.usesTexCoordScales || info.lineMode != 0) && batchUni.bpSuGen != s_curUniState.bpSuGen) {
    return UniMismatch::Bp;
  }
  if (info.usedIndTexMtxs.any() && batchUni.bpIndGen != s_curUniState.bpIndGen) {
    return UniMismatch::Bp;
  }
  if (info.lineMode != 0 && batchUni.bpLineGen != s_curUniState.bpLineGen) {
    return UniMismatch::Bp;
  }
  if (((info.usesAmbColorMask & 1u) != 0 && batchUni.othAmbGen[0] != s_curUniState.othAmbGen[0]) ||
      ((info.usesAmbColorMask & 2u) != 0 && batchUni.othAmbGen[1] != s_curUniState.othAmbGen[1]) ||
      ((info.usesMatColorMask & 1u) != 0 && batchUni.othMatGen[0] != s_curUniState.othMatGen[0]) ||
      ((info.usesMatColorMask & 2u) != 0 && batchUni.othMatGen[1] != s_curUniState.othMatGen[1])) {
    return UniMismatch::Oth;
  }
  if (info.loadsTevReg.any()) {
    u32 m = static_cast<u32>(info.loadsTevReg.to_ulong());
    while (m != 0) {
      const u32 idx = static_cast<u32>(std::countr_zero(m));
      m &= m - 1;
      if (batchUni.bpTevColGen[idx] != s_curUniState.bpTevColGen[idx]) {
        return UniMismatch::Bp;
      }
    }
  }
  if (info.sampledKColors.any()) {
    u32 m = static_cast<u32>(info.sampledKColors.to_ulong());
    while (m != 0) {
      const u32 idx = static_cast<u32>(std::countr_zero(m));
      m &= m - 1;
      if (batchUni.bpKColGen[idx] != s_curUniState.bpKColGen[idx]) {
        return UniMismatch::Bp;
      }
    }
  }
  if (info.sampledTextures.any()) {
    u32 m = static_cast<u32>(info.sampledTextures.to_ulong());
    while (m != 0) {
      const u32 idx = static_cast<u32>(std::countr_zero(m));
      m &= m - 1;
      if (ignoreTex0Dim && idx == 0) {
        continue;
      }
      if (batchUni.texDimGen[idx] != s_curUniState.texDimGen[idx]) {
        return UniMismatch::Oth;
      }
    }
  }
  return UniMismatch::None;
}

static u32 s_splitBpUniGen = 0;
static u32 s_splitOthUniGen = 0;
static u32 s_splitOthAmbGen = 0;
static u32 s_splitOthMatGen = 0;
static u32 s_splitOthVpGen = 0;
static u32 s_splitOthProjGen = 0;
static u32 s_splitOthTexDimGen = 0;
static u32 s_splitOthOffGen = 0;
static u32 s_splitBpFogGen = 0;
static u32 s_splitBpTevColGen = 0;
static u32 s_splitBpSuGen = 0;
static u32 s_splitBpIndGen = 0;
static u32 s_splitBpLineGen = 0;
static bool s_splitHasPrev = false;
static gfx::PipelineRef s_splitPrevPipe = 0;
static gfx::BindGroupRef s_splitPrevTex = 0;
static u32 s_splitPrevDstAlpha = 0;
static u32 s_splitPrevUniformDirty = 0;
static u32 s_splitPrevTexMtxGen = 0;
static u32 s_splitPrevLightsGen = 0;
static u32 s_splitPrevBpUniGen = 0;
static u32 s_splitPrevOthUniGen = 0;
static u32 s_splitPrevOthAmbGen = 0;
static u32 s_splitPrevOthMatGen = 0;
static u32 s_splitPrevOthVpGen = 0;
static u32 s_splitPrevOthProjGen = 0;
static u32 s_splitPrevOthTexDimGen = 0;
static u32 s_splitPrevOthOffGen = 0;
static u32 s_splitPrevBpFogGen = 0;
static u32 s_splitPrevBpTevColGen = 0;
static u32 s_splitPrevBpSuGen = 0;
static u32 s_splitPrevBpIndGen = 0;
static u32 s_splitPrevBpLineGen = 0;
static PipelineConfig s_splitPrevConfig{};

struct CachedShaderDraw {
  uint64_t shaderHash = 0;
  ShaderInfo info{};
  GXBindGroups bindGroups{};
  gfx::Range uniformRange{};
  uint32_t uniformDirty = UINT32_MAX;
  uint32_t textureDirty = UINT32_MAX;
  uint32_t frame = UINT32_MAX;
};

static constexpr size_t kShaderDrawCacheSize = 512;
static std::array<CachedShaderDraw, kShaderDrawCacheSize> s_shaderDrawCache{};
static u32 s_vatGen = 1;
} // namespace

static std::array<u32, 256> s_cpRegCache{};
static std::array<bool, 256> s_cpRegValid{};

static void process_stream(const u8* data, u32 size, bool bigEndian, bool logTiming);
static void native_geom_cache_maybe_log_stats();
static bool s_staticDlCaptureActive = false;
static bool s_staticDlCaptureAborted = false;

static void reset_frame_draw_cache() {
  flush_pending_pnmtx_batch();
  s_hasLastDrawState = false;
  s_hasLastUniformRange = false;
  s_lastUniformRange = {};
  s_lastPipeMemoIdx = SIZE_MAX;
  s_lastCfgHash = 0;
  s_splitHasPrev = false;
  s_cpRegValid.fill(false);
  s_vtxDescPackedDirty = true;
  s_curVtxStateVerifiedId = UINT16_MAX;
  g_gxState.uniformDirty++;
  g_gxState.pipelineDirty++;
}

static inline bool bp_affects_pipeline(u32 regId) noexcept {
  if (regId == 0x00) return true; // genMode (cullMode, numTexGens, numChans, numTevStages, numIndStages)
  if (regId >= 0x10 && regId <= 0x1F) return true; // TEV indirect stages
  if (regId >= 0x25 && regId <= 0x2F) return true; // Ind tex scale (0x25..0x26), Ind tex order (0x27), TEV stage orders (0x28..0x2F)
  if (regId >= 0x40 && regId <= 0x43) return true; // Z mode, blend mode, clamp
  if (regId >= 0xC0 && regId <= 0xDF) return true; // TEV stages color/alpha combiners
  if (regId == 0xF1) return true; // Fog type (GXSetFog)
  if (regId == 0xF3) return true; // Alpha compare (GXSetAlphaCompare)
  if (regId >= 0xF6 && regId <= 0xFD) return true; // TEV swap table
  return false;
}

static inline bool bp_affects_uniforms(u32 regId) noexcept {
  // Indirect texture matrices (0x06..0x0E)
  if (regId >= 0x06 && regId <= 0x0E) return true;
  // Line / point size (0x22)
  if (regId == 0x22) return true;
  // SU texture coordinate scale (0x30..0x3F)
  if (regId >= 0x30 && regId <= 0x3F) return true;
  // TEV color / K color registers (0xE0..0xE7)
  if (regId >= 0xE0 && regId <= 0xE7) return true;
  // Fog parameters (0xEE..0xF2)
  if (regId >= 0xEE && regId <= 0xF2) return true;
  return false;
}

static inline bool is_timing_enabled() noexcept {
  static const bool s_enabled = std::getenv("TTYD_TIMING_DETAIL") != nullptr;
  return s_enabled;
}

static inline std::chrono::steady_clock::time_point timing_now() noexcept {
  return is_timing_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}

struct DrainScopeTimer {
  double& acc;
  bool active;
  std::chrono::steady_clock::time_point t0;
  explicit DrainScopeTimer(double& a) noexcept : acc(a), active(is_timing_enabled()) {
    if (active) UNLIKELY t0 = std::chrono::steady_clock::now();
  }
  ~DrainScopeTimer() {
    if (active) UNLIKELY {
      acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
  }
};

void process_chunk(const u8* data, u32 size, bool fresh) {
  ZoneScoped;
  if (fresh) {
    native_geom_cache_maybe_log_stats();
    reset_frame_draw_cache();
  }
  process_stream(data, size, true, is_timing_enabled());
}

void process_flush() { flush_pending_pnmtx_batch(); }

void process(const u8* data, u32 size, bool bigEndian) {
  ZoneScoped;
  native_geom_cache_maybe_log_stats();
  const bool logTiming = is_timing_enabled();
  const auto t_proc_start = logTiming ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  reset_frame_draw_cache();
  process_stream(data, size, bigEndian, logTiming);

  flush_pending_pnmtx_batch();

  if (logTiming) UNLIKELY {
    const auto t_proc_end = std::chrono::steady_clock::now();
    s_drainTiming.ms_total += std::chrono::duration<double, std::milli>(t_proc_end - t_proc_start).count();
    if (++s_drainTiming.frames >= 30) {
      const double f = 1.0 / s_drainTiming.frames;
      std::printf("[DRAIN-TIMING-30f] avg: total %.1fms | draw_tot %.1fms (draws:%.0f hash:%.1f exp:%.1f g_miss:%.1f scan:%.1f pipe_cfg:%.1f pipe_ref:%.1f res:%.1f bg:%.1f[calls:%.0f] uni:%.1f push:%.1f) | xf %.1fms[n:%.0f] | indx %.1fms | bp %.1fms | cp %.1fms | hbytes(fifo:%.1fKB arr:%.1fKB[%u])\n",
                  s_drainTiming.ms_total * f,
                  s_drainTiming.ms_draw_total * f,
                  s_drainTiming.draw_count * f,
                  s_drainTiming.ms_geom_hash * f,
                  s_drainTiming.ms_geom_expand * f,
                  s_drainTiming.ms_geom_miss * f,
                  s_drainTiming.ms_scan * f,
                  s_drainTiming.ms_pipe_cfg * f,
                  s_drainTiming.ms_pipe_ref * f,
                  s_drainTiming.ms_tex_resolve * f,
                  s_drainTiming.ms_bg_build * f,
                  s_drainTiming.bg_calls * f,
                  s_drainTiming.ms_uni_build * f,
                  s_drainTiming.ms_push_cmd * f,
                  s_drainTiming.ms_xf * f,
                  s_drainTiming.xf_calls * f,
                  s_drainTiming.ms_indx * f,
                  s_drainTiming.ms_bp * f,
                  s_drainTiming.ms_cp * f,
                  (s_drainTiming.fifo_hash_bytes * f) / 1024.0,
                  (s_drainTiming.arr_hash_bytes * f) / 1024.0,
                  static_cast<unsigned>(s_drainTiming.arr_hash_calls / s_drainTiming.frames));
      std::printf("[SPLIT-30f] avg pairs:%.0f | samePipe:%.0f sameTex:%.0f sameBoth:%.0f | onlyPnMtx:%.0f onlyTex:%.0f onlyPipe:%.0f | chgTexMtx:%.0f chgLights:%.0f chgBpUni:%.0f chgOthUni:%.0f sameAll:%.0f\n",
                  s_drainTiming.split_pairs * f,
                  s_drainTiming.split_same_pipe * f,
                  s_drainTiming.split_same_tex * f,
                  s_drainTiming.split_same_both * f,
                  s_drainTiming.split_only_pnmtx * f,
                  s_drainTiming.split_only_tex * f,
                  s_drainTiming.split_only_pipe * f,
                  s_drainTiming.split_chg_texmtx * f,
                  s_drainTiming.split_chg_lights * f,
                  s_drainTiming.split_chg_bp_uni * f,
                  s_drainTiming.split_chg_oth_uni * f,
                  s_drainTiming.split_same_all * f);
      u32 topReg[4] = {0, 0, 0, 0};
      u32 topCnt[4] = {0, 0, 0, 0};
      for (u32 r = 0; r < 256; ++r) {
        const u32 c = s_drainTiming.bp_uni_reg[r];
        if (c == 0) continue;
        for (int k = 0; k < 4; ++k) {
          if (c > topCnt[k]) {
            for (int m = 3; m > k; --m) {
              topCnt[m] = topCnt[m - 1];
              topReg[m] = topReg[m - 1];
            }
            topCnt[k] = c;
            topReg[k] = r;
            break;
          }
        }
      }
      std::printf("[SPLIT-WHY-30f] flush(nonbat:%.0f topo:%.0f da:%.0f tm:%.0f li:%.0f bp:%.0f oth:%.0f vtx:%.0f pipe:%.0f tex:%.0f mtx:%.0f fixtg:%.0f | batOk:%.0f 1x:%.0f mx:%.0f) | oth(amb:%.0f mat:%.0f vp:%.0f proj:%.0f tdim:%.0f off:%.0f) | bp(fog:%.0f tcol:%.0f su:%.0f ind:%.0f line:%.0f top:%02X=%.0f,%02X=%.0f,%02X=%.0f,%02X=%.0f)\n",
                  s_drainTiming.fl_nonbat * f, s_drainTiming.fl_topo * f, s_drainTiming.fl_da * f,
                  s_drainTiming.fl_tm * f, s_drainTiming.fl_li * f, s_drainTiming.fl_bp * f,
                  s_drainTiming.fl_oth * f, s_drainTiming.fl_vtx * f, s_drainTiming.fl_pipe * f,
                  s_drainTiming.fl_tex * f, s_drainTiming.fl_mtx * f, s_drainTiming.skip_fixtg * f,
                  s_drainTiming.bat_ok * f, s_drainTiming.bat_single * f, s_drainTiming.bat_merged * f,
                  s_drainTiming.oth_amb * f, s_drainTiming.oth_mat * f, s_drainTiming.oth_vp * f,
                  s_drainTiming.oth_proj * f, s_drainTiming.oth_tdim * f, s_drainTiming.oth_off * f,
                  s_drainTiming.bp_fog * f, s_drainTiming.bp_tcol * f, s_drainTiming.bp_su * f,
                  s_drainTiming.bp_ind * f, s_drainTiming.bp_line * f,
                  topReg[0], topCnt[0] * f, topReg[1], topCnt[1] * f,
                  topReg[2], topCnt[2] * f, topReg[3], topCnt[3] * f);
      std::printf("[PIPE-WHY-30f] onlyPipe(cull:%.0f z:%.0f blend:%.0f topo:%.0f fog:%.0f attrs:%.0f tev:%.0f chan:%.0f tcg:%.0f acmp:%.0f paper:%.0f) | allDiffPipe(cull:%.0f z:%.0f blend:%.0f topo:%.0f fog:%.0f attrs:%.0f tev:%.0f chan:%.0f tcg:%.0f acmp:%.0f paper:%.0f)\n",
                  s_drainTiming.op_cull * f, s_drainTiming.op_z * f, s_drainTiming.op_blend * f,
                  s_drainTiming.op_topo * f, s_drainTiming.op_fog * f, s_drainTiming.op_attrs * f,
                  s_drainTiming.op_tev * f, s_drainTiming.op_chan * f, s_drainTiming.op_tcg * f,
                  s_drainTiming.op_acmp * f, s_drainTiming.op_paper * f,
                  s_drainTiming.pd_cull * f, s_drainTiming.pd_z * f, s_drainTiming.pd_blend * f,
                  s_drainTiming.pd_topo * f, s_drainTiming.pd_fog * f, s_drainTiming.pd_attrs * f,
                  s_drainTiming.pd_tev * f, s_drainTiming.pd_chan * f, s_drainTiming.pd_tcg * f,
                  s_drainTiming.pd_acmp * f, s_drainTiming.pd_paper * f);
      std::fflush(stdout);
      s_drainTiming = DrainTiming{};
    }
  }
}

static void process_stream(const u8* data, u32 size, bool bigEndian, bool logTiming) {
  u32 pos = 0;

  while (pos < size) {
    u8 cmd = data[pos++];
    u8 opcode = cmd & CP_OPCODE_MASK;
    // Log.warn("Processing opcode {:02x} at pos {} (size {})", opcode, pos - 1, size);

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      CHECK(pos + 4 <= size, "BP reg read overrun");
      u32 value = read_u32(data + pos, bigEndian);
      pos += 4;
      const u32 regId = (value >> 24) & 0xFF;
      if (regId != 0xFE && regId != 0x52 && regId != 0x55 && regId != 0x56 && g_gxState.bpRegCache[0xFE] == 0x00FFFFFF &&
          bp_cache_slot(regId, value) == value) {
        continue;
      }
      bool bpChanged;
      if (logTiming) UNLIKELY {
        const auto t0 = std::chrono::steady_clock::now();
        bpChanged = handle_bp(value, bigEndian);
        const auto t1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_bp += std::chrono::duration<double, std::milli>(t1 - t0).count();
      } else {
        bpChanged = handle_bp(value, bigEndian);
      }
      if (bpChanged) {
        if (bp_affects_uniforms(regId)) {
          g_gxState.uniformDirty++;
          ++s_splitBpUniGen;
          if (regId >= 0xEE && regId <= 0xF2) {
            ++s_curUniState.bpFogGen;
            if (logTiming) UNLIKELY ++s_splitBpFogGen;
          } else if (regId >= 0xE0 && regId <= 0xE7) {
            const u32 idx = (regId - 0xE0) / 2;
            if ((value & (1u << 23)) != 0) {
              if (idx < GX_MAX_KCOLOR) ++s_curUniState.bpKColGen[idx];
            } else {
              if (idx < MaxTevRegs) ++s_curUniState.bpTevColGen[idx];
            }
            if (logTiming) UNLIKELY ++s_splitBpTevColGen;
          } else if (regId >= 0x30 && regId <= 0x3F) {
            ++s_curUniState.bpSuGen;
            if (logTiming) UNLIKELY ++s_splitBpSuGen;
          } else if (regId >= 0x06 && regId <= 0x0E) {
            ++s_curUniState.bpIndGen;
            if (logTiming) UNLIKELY ++s_splitBpIndGen;
          } else if (regId == 0x22) {
            ++s_curUniState.bpLineGen;
            if (logTiming) UNLIKELY ++s_splitBpLineGen;
          }
          if (logTiming) UNLIKELY {
            ++s_drainTiming.bp_uni_reg[regId];
          }
        }
        if (bp_affects_pipeline(regId)) {
          g_gxState.pipelineDirty++;
        }
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      CHECK(pos + 5 <= size, "CP reg read overrun");
      u8 addr = data[pos++];
      u32 value = read_u32(data + pos, bigEndian);
      pos += 4;
      if (s_cpRegValid[addr] && s_cpRegCache[addr] == value) {
        break;
      }
      s_cpRegCache[addr] = value;
      s_cpRegValid[addr] = true;
      if (logTiming) UNLIKELY {
        const auto t0 = std::chrono::steady_clock::now();
        handle_cp(addr, value, bigEndian);
        const auto t1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_cp += std::chrono::duration<double, std::milli>(t1 - t0).count();
      } else {
        handle_cp(addr, value, bigEndian);
      }
      if (addr != 0x30 && addr != 0x40) {
        g_gxState.pipelineDirty++;
        s_vatGen++;
        g_gxState.lastVtxFmt = GX_MAX_VTXFMT;
      }
      // CP registers set vertex attribute tables (VCD/VAT) and array bases/strides.
      // These affect PipelineConfig vertex attributes, NOT uniform buffer contents.
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const bool prevDirty = g_gxState.stateDirty;
      g_gxState.stateDirty = false;
      bool pipelineChanged = false;
      if (logTiming) UNLIKELY {
        const auto t0 = std::chrono::steady_clock::now();
        ++s_drainTiming.xf_calls;
        handle_xf(data, pos, size, bigEndian, pipelineChanged);
        const auto t1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_xf += std::chrono::duration<double, std::milli>(t1 - t0).count();
      } else {
        handle_xf(data, pos, size, bigEndian, pipelineChanged);
      }
      if (g_gxState.stateDirty) {
        g_gxState.uniformDirty++;
      }
      if (pipelineChanged) {
        g_gxState.pipelineDirty++;
      }
      g_gxState.stateDirty |= prevDirty;
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const auto t0 = logTiming ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      // Indexed XF load: 4 bytes of data
      CHECK(pos + 4 <= size, "indexed XF read overrun");
      u32 arrayType = GX_POS_MTX_ARRAY + (opcode - (CP_CMD_LOAD_INDX_A / 0x08));
      u8 srcArrayIdx = data[pos++];
      auto const& array = g_gxState.arrays[arrayType];
      u8* srcData = ((u8*)array.data) + srcArrayIdx * array.stride;
      u16 addrLen = read_u16(data + pos, bigEndian);
      u16 len = (addrLen >> 12) + 1;
      u16 dstAddr = addrLen & 0x0FFF;
      const bool prevDirty = g_gxState.stateDirty;
      g_gxState.stateDirty = false;
      if (!copy_xf_data(dstAddr, srcData, len, bigEndian)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      if (g_gxState.stateDirty) {
        g_gxState.uniformDirty++;
      }
      g_gxState.stateDirty |= prevDirty;
      pos += 4;
      if (logTiming) UNLIKELY {
        const auto t1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_indx += std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      CHECK(pos + 8 <= size, "call DL read overrun");
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      pos += 8;
      break;
    }

    case CP_CMD_INVAL_VTX: {
      // Invalidate vertex cache: the game may have rewritten attribute array contents in
      // place (e.g. particle systems), so drop memoized array content hashes.
      ++s_gxArrayInvalidateGen;
      ++s_gxArrayBindGen;
      break;
    }

    case GX_AURORA: {
      handle_aurora(data, pos, size, bigEndian);
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      if (logTiming) UNLIKELY {
        const auto td0 = std::chrono::steady_clock::now();
        ++s_drainTiming.draw_count;
        handle_draw(cmd, data, pos, size, bigEndian);
        const auto td1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_draw_total += std::chrono::duration<double, std::milli>(td1 - td0).count();
      } else {
        handle_draw(cmd, data, pos, size, bigEndian);
      }
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        if (logTiming) UNLIKELY {
          const auto td0 = std::chrono::steady_clock::now();
          ++s_drainTiming.draw_count;
          handle_draw(cmd, data, pos, size, bigEndian);
          const auto td1 = std::chrono::steady_clock::now();
          s_drainTiming.ms_draw_total += std::chrono::duration<double, std::milli>(td1 - td0).count();
        } else {
          handle_draw(cmd, data, pos, size, bigEndian);
        }
      } else {
        // Hex dump surrounding bytes for debugging
        {
          u32 dumpStart = (pos > 17) ? pos - 17 : 0;
          u32 dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (u32 i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
          for (int a = GX_VA_PNMTXIDX; a <= GX_VA_TEX7; ++a) {
            if (g_gxState.vtxDesc[a] != GX_NONE) {
              const auto& af = g_gxState.vtxFmts[0].attrs[a];
              Log.error("  vtxDesc[{}]={}: cnt={} type={} frac={}", a, static_cast<int>(g_gxState.vtxDesc[a]),
                        static_cast<int>(af.cnt), static_cast<int>(af.type), af.frac);
            }
          }
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, pos - 1);
      }
      break;
    }
  }
}

// Helper to extract bit fields from a 32-bit register
inline static u32 bp_get(u32 reg, u32 size, u32 shift) { return reg >> shift & (1u << size) - 1; }

// BP register handler - decodes BP (RAS/pixel engine) register writes and updates g_gxState
static bool handle_bp(u32 value, bool bigEndian) {
  u32 regId = (value >> 24) & 0xFF;
  // Mask off the register ID from the value for field extraction
  // (the regId is stored in bits 24-31, data is in bits 0-23)

  if (regId == 0xFE) {
    g_gxState.bpRegCache[regId] = value & 0x00FFFFFF;
    return false;
  } else {
    u32 ssMask = g_gxState.bpRegCache[0xFE];
    g_gxState.bpRegCache[0xFE] = 0x00FFFFFF;
    u32& cacheSlot = bp_cache_slot(regId, value);
    const u32 merged = (cacheSlot & ~ssMask) | (value & ssMask);
    value = (regId << 24) | (merged & 0x00FFFFFF);
    if (cacheSlot == value && regId != 0x52 && regId != 0x55 && regId != 0x56)
      return false;
    cacheSlot = value;
  }

  // TEV color combiner stages (0xC0, 0xC2, 0xC4, ... 0xDE)
  if (regId >= 0xC0 && regId <= 0xDE && (regId & 1) == 0) {
    u32 stage = (regId - 0xC0) / 2;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.colorPass.d = static_cast<GXTevColorArg>(bp_get(value, 4, 0));
      s.colorPass.c = static_cast<GXTevColorArg>(bp_get(value, 4, 4));
      s.colorPass.b = static_cast<GXTevColorArg>(bp_get(value, 4, 8));
      s.colorPass.a = static_cast<GXTevColorArg>(bp_get(value, 4, 12));
      s.colorOp.clamp = bp_get(value, 1, 19) != 0;
      s.colorOp.outReg = static_cast<GXTevRegID>(bp_get(value, 2, 22));
      if (bp_get(value, 2, 16) == 3) {
        // Bias==3 means compare mode: reconstruct GXTevOp enum (8 + 3-bit hw value)
        u32 hwOp = bp_get(value, 1, 18) | (bp_get(value, 2, 20) << 1);
        s.colorOp.op = static_cast<GXTevOp>(hwOp + 8);
        s.colorOp.bias = GX_TB_ZERO;
        s.colorOp.scale = GX_CS_SCALE_1;
      } else {
        // Normal mode: bit18 is op (0=ADD, 1=SUB), bits16-17 is bias, bits20-21 is scale
        s.colorOp.op = static_cast<GXTevOp>(bp_get(value, 1, 18));
        s.colorOp.bias = static_cast<GXTevBias>(bp_get(value, 2, 16));
        s.colorOp.scale = static_cast<GXTevScale>(bp_get(value, 2, 20));
      }
      g_gxState.stateDirty = true;
    }
    return true;
  }

  // TEV alpha combiner stages (0xC1, 0xC3, 0xC5, ... 0xDF)
  if (regId >= 0xC1 && regId <= 0xDF && (regId & 1) == 1) {
    u32 stage = (regId - 0xC1) / 2;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.tevSwapRas = static_cast<GXTevSwapSel>(bp_get(value, 2, 0));
      s.tevSwapTex = static_cast<GXTevSwapSel>(bp_get(value, 2, 2));
      s.alphaPass.d = static_cast<GXTevAlphaArg>(bp_get(value, 3, 4));
      s.alphaPass.c = static_cast<GXTevAlphaArg>(bp_get(value, 3, 7));
      s.alphaPass.b = static_cast<GXTevAlphaArg>(bp_get(value, 3, 10));
      s.alphaPass.a = static_cast<GXTevAlphaArg>(bp_get(value, 3, 13));
      s.alphaOp.clamp = bp_get(value, 1, 19) != 0;
      s.alphaOp.outReg = static_cast<GXTevRegID>(bp_get(value, 2, 22));
      if (bp_get(value, 2, 16) == 3) {
        u32 hwOp = bp_get(value, 1, 18) | (bp_get(value, 2, 20) << 1);
        s.alphaOp.op = static_cast<GXTevOp>(hwOp + 8);
        s.alphaOp.bias = GX_TB_ZERO;
        s.alphaOp.scale = GX_CS_SCALE_1;
      } else {
        s.alphaOp.op = static_cast<GXTevOp>(bp_get(value, 1, 18));
        s.alphaOp.bias = static_cast<GXTevBias>(bp_get(value, 2, 16));
        s.alphaOp.scale = static_cast<GXTevScale>(bp_get(value, 2, 20));
      }
      g_gxState.stateDirty = true;
    }
    return true;
  }

  switch (regId) {
  // genMode (0x00)
  case 0x00: {
    g_gxState.numTexGens = bp_get(value, 4, 0);
    g_gxState.numChans = bp_get(value, 3, 4);
    g_gxState.xfRegValid[0x09] = false;
    g_gxState.xfRegValid[0x3F] = false;
    g_gxState.numTevStages = bp_get(value, 4, 10) + 1;
    u32 hwCull = bp_get(value, 2, 14);
    // Swap front/back to match GX convention
    switch (hwCull) {
    case GX_CULL_FRONT:
      g_gxState.cullMode = GX_CULL_BACK;
      break;
    case GX_CULL_BACK:
      g_gxState.cullMode = GX_CULL_FRONT;
      break;
    default:
      g_gxState.cullMode = static_cast<GXCullMode>(hwCull);
      break;
    }
    g_gxState.numIndStages = bp_get(value, 3, 16);
    g_gxState.stateDirty = true;
    break;
  }

  // BP mask (0x0F) - internal, applies to next BP write
  case 0x0F:
#ifndef NDEBUG
    Log.debug("BP mask set to {:06x}, but selective updates are not implemented", value & 0xFFFFFF);
#endif
    break;

  // TEV indirect stages (0x10-0x1F)
  case 0x10:
  case 0x11:
  case 0x12:
  case 0x13:
  case 0x14:
  case 0x15:
  case 0x16:
  case 0x17:
  case 0x18:
  case 0x19:
  case 0x1A:
  case 0x1B:
  case 0x1C:
  case 0x1D:
  case 0x1E:
  case 0x1F: {
    u32 stage = regId - 0x10;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.indTexStage = static_cast<GXIndTexStageID>(bp_get(value, 2, 0));
      s.indTexFormat = static_cast<GXIndTexFormat>(bp_get(value, 2, 2));
      s.indTexBiasSel = static_cast<GXIndTexBiasSel>(bp_get(value, 3, 4));
      s.indTexAlphaSel = static_cast<GXIndTexAlphaSel>(bp_get(value, 2, 7));
      s.indTexMtxId = static_cast<GXIndTexMtxID>(bp_get(value, 4, 9));
      s.indTexWrapS = static_cast<GXIndTexWrap>(bp_get(value, 3, 13));
      s.indTexWrapT = static_cast<GXIndTexWrap>(bp_get(value, 3, 16));
      s.indTexUseOrigLOD = bp_get(value, 1, 19) != 0;
      s.indTexAddPrev = bp_get(value, 1, 20) != 0;
      g_gxState.stateDirty = true;
    }
    break;
  }

  // Scissor registers (0x20, 0x21)
  case 0x20:
  case 0x21: {
    const u32 scis0 = g_gxState.bpRegCache[0x20];
    const u32 scis1 = g_gxState.bpRegCache[0x21];
    const int32_t tp = static_cast<int32_t>(bp_get(scis0, 11, 0)) - 342;
    const int32_t lf = static_cast<int32_t>(bp_get(scis0, 11, 12)) - 342;
    const int32_t bm = static_cast<int32_t>(bp_get(scis1, 11, 0)) - 342;
    const int32_t rt = static_cast<int32_t>(bp_get(scis1, 11, 12)) - 342;
    const int32_t wd = std::max(rt - lf + 1, 0);
    const int32_t ht = std::max(bm - tp + 1, 0);
    flush_pending_pnmtx_batch();
    set_logical_scissor({lf, tp, wd, ht});
    break;
  }

  // Line/point size (0x22)
  case 0x22: {
    g_gxState.lineWidth = static_cast<u8>(bp_get(value, 8, 0));
    g_gxState.pointSize = static_cast<u8>(bp_get(value, 8, 8));
    g_gxState.lineTexOffset = static_cast<GXTexOffset>(bp_get(value, 3, 16));
    g_gxState.pointTexOffset = static_cast<GXTexOffset>(bp_get(value, 3, 19));
    g_gxState.lineHalfAspect = bp_get(value, 1, 22) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Indirect texture scale (0x25, 0x26)
  case 0x25: {
    if (MaxIndStages > 0) {
      g_gxState.indStages[0].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 0));
      g_gxState.indStages[0].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 4));
    }
    if (MaxIndStages > 1) {
      g_gxState.indStages[1].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 8));
      g_gxState.indStages[1].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 12));
    }
    g_gxState.stateDirty = true;
    break;
  }
  case 0x26: {
    if (MaxIndStages > 2) {
      g_gxState.indStages[2].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 0));
      g_gxState.indStages[2].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 4));
    }
    if (MaxIndStages > 3) {
      g_gxState.indStages[3].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 8));
      g_gxState.indStages[3].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 12));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Indirect texture reference (0x27)
  case 0x27: {
    for (u32 i = 0; i < MaxIndStages; i++) {
      g_gxState.indStages[i].texMapId = static_cast<GXTexMapID>(bp_get(value, 3, i * 6));
      g_gxState.indStages[i].texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, i * 6 + 3));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // TEV order / tref (0x28-0x2F) - 2 stages per register
  case 0x28:
  case 0x29:
  case 0x2A:
  case 0x2B:
  case 0x2C:
  case 0x2D:
  case 0x2E:
  case 0x2F: {
    u32 idx = regId - 0x28;
    u32 stage0 = idx * 2;
    u32 stage1 = idx * 2 + 1;

    // Channel ID reverse mapping from hardware to GX
    static const GXChannelID r2c[] = {GX_COLOR0A0, GX_COLOR1A1,   GX_COLOR0A0,    GX_COLOR1A1,
                                      GX_COLOR0A0, GX_ALPHA_BUMP, GX_ALPHA_BUMPN, GX_COLOR_ZERO};

    if (stage0 < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage0];
      s.texMapId = static_cast<GXTexMapID>(bp_get(value, 3, 0));
      s.texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, 3));
      // bit 6 = tex enable
      if (!bp_get(value, 1, 6)) {
        s.texMapId = GX_TEXMAP_NULL;
      }
      u32 chanHw = bp_get(value, 3, 7);
      s.channelId = (chanHw < 8) ? r2c[chanHw] : GX_COLOR_NULL;
    }
    if (stage1 < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage1];
      s.texMapId = static_cast<GXTexMapID>(bp_get(value, 3, 12));
      s.texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, 15));
      if (!bp_get(value, 1, 18)) {
        s.texMapId = GX_TEXMAP_NULL;
      }
      u32 chanHw = bp_get(value, 3, 19);
      s.channelId = (chanHw < 8) ? r2c[chanHw] : GX_COLOR_NULL;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Z mode (0x40)
  case 0x40: {
    g_gxState.depthCompare = bp_get(value, 1, 0) != 0;
    g_gxState.depthFunc = static_cast<GXCompare>(bp_get(value, 3, 1));
    g_gxState.depthUpdate = bp_get(value, 1, 4) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Blend mode / cmode0 (0x41)
  case 0x41: {
    bool blendEn = bp_get(value, 1, 0) != 0;
    bool logicEn = bp_get(value, 1, 1) != 0;
    bool dither = bp_get(value, 1, 2) != 0;
    g_gxState.colorUpdate = bp_get(value, 1, 3) != 0;
    g_gxState.alphaUpdate = bp_get(value, 1, 4) != 0;
    g_gxState.blendFacDst = static_cast<GXBlendFactor>(bp_get(value, 3, 5));
    g_gxState.blendFacSrc = static_cast<GXBlendFactor>(bp_get(value, 3, 8));
    bool subtract = bp_get(value, 1, 11) != 0;
    g_gxState.blendOp = static_cast<GXLogicOp>(bp_get(value, 4, 12));

    if (subtract) {
      g_gxState.blendMode = GX_BM_SUBTRACT;
    } else if (blendEn) {
      g_gxState.blendMode = GX_BM_BLEND;
    } else if (logicEn) {
      g_gxState.blendMode = GX_BM_LOGIC;
    } else {
      g_gxState.blendMode = GX_BM_NONE;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Dst alpha / cmode1 (0x42)
  case 0x42: {
    u8 alpha = bp_get(value, 8, 0);
    bool enabled = bp_get(value, 1, 8) != 0;
    g_gxState.dstAlpha = enabled ? alpha : UINT32_MAX;
    g_gxState.pixelFmt = decode_pixel_fmt(g_gxState.bpRegCache[0x43], value);
    g_gxState.stateDirty = true;
    break;
  }

  // PE control (0x43) - pixel format, z format, zcomp location
  case 0x43: {
    g_gxState.pixelFmt = decode_pixel_fmt(value, g_gxState.bpRegCache[0x42]);
    g_gxState.zFmt = static_cast<GXZFmt16>(bp_get(value, 3, 3));
    g_gxState.zCompLocBeforeTex = bp_get(value, 1, 6) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Texture copy
  case 0x52: {
    const bool clear = bp_get(value, 1, 11) != 0;
    if (bp_get(value, 1, 14) != 0) {
      Log.warn("STUB: display copy is not implemented");
    } else {
      flush_pending_pnmtx_batch();
      copy_tex(g_gxState.texCopyDest, clear);
      g_gxState.textureDirty++;
      g_gxState.stateDirty = true;
    }
    break;
  }

  // Bounding box clear (0x55, 0x56)
  case 0x55:
  case 0x56:
    bbox_reset();
    break;

  // TLUT load address / execute (0x64, 0x65)
  case 0x64:
    break;
  case 0x65: {
    const auto idx = bp_get(value, 10, 0);
    if (idx < MaxTluts) {
      auto& slot = g_gxState.loadedTluts[idx];
      slot.loadTlut0 = g_gxState.bpRegCache[0x64];
      slot.numEntries = static_cast<u16>(bp_get(value, 10, 10) + 1);
    }
    break;
  }

  // Alpha compare (0xF3)
  case 0xF3: {
    g_gxState.alphaCompare.ref0 = bp_get(value, 8, 0);
    g_gxState.alphaCompare.ref1 = bp_get(value, 8, 8);
    g_gxState.alphaCompare.comp0 = static_cast<GXCompare>(bp_get(value, 3, 16));
    g_gxState.alphaCompare.comp1 = static_cast<GXCompare>(bp_get(value, 3, 19));
    g_gxState.alphaCompare.op = static_cast<GXAlphaOp>(bp_get(value, 2, 22));
    g_gxState.stateDirty = true;
    break;
  }

  // TEV K color/alpha select (0xF6-0xFD)
  case 0xF6:
  case 0xF7:
  case 0xF8:
  case 0xF9:
  case 0xFA:
  case 0xFB:
  case 0xFC:
  case 0xFD: {
    u32 kselIdx = regId - 0xF6;
    // Swap table entries (packed into pairs of ksel registers)
    if (kselIdx < MaxTevSwap * 2) {
      u32 swapIdx = kselIdx / 2;
      if (kselIdx & 1) {
        g_gxState.tevSwapTable[swapIdx].blue = static_cast<GXTevColorChan>(bp_get(value, 2, 0));
        g_gxState.tevSwapTable[swapIdx].alpha = static_cast<GXTevColorChan>(bp_get(value, 2, 2));
      } else {
        g_gxState.tevSwapTable[swapIdx].red = static_cast<GXTevColorChan>(bp_get(value, 2, 0));
        g_gxState.tevSwapTable[swapIdx].green = static_cast<GXTevColorChan>(bp_get(value, 2, 2));
      }
    }
    // K color/alpha selection for 2 stages per register
    u32 stage0 = kselIdx * 2;
    u32 stage1 = kselIdx * 2 + 1;
    if (stage0 < MaxTevStages) {
      g_gxState.tevStages[stage0].kcSel = static_cast<GXTevKColorSel>(bp_get(value, 5, 4));
      g_gxState.tevStages[stage0].kaSel = static_cast<GXTevKAlphaSel>(bp_get(value, 5, 9));
    }
    if (stage1 < MaxTevStages) {
      g_gxState.tevStages[stage1].kcSel = static_cast<GXTevKColorSel>(bp_get(value, 5, 14));
      g_gxState.tevStages[stage1].kaSel = static_cast<GXTevKAlphaSel>(bp_get(value, 5, 19));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Fog A/B parameters (0xEE-0xF0)
  // FOG0 (0xEE): A parameter - sign(1)|exp(8)|mantissa(11) partial IEEE 754 float
  case 0xEE: {
    g_gxState.fog.fog0Raw = value;
    // Reconstruct A = a_encoded * 2^b_s
    u32 a_mant = bp_get(value, 11, 0);
    u32 a_exp = bp_get(value, 8, 11);
    u32 a_sign = bp_get(value, 1, 19);
    u32 a_bits = (a_sign << 31) | (a_exp << 23) | (a_mant << 12);
    float a_encoded;
    std::memcpy(&a_encoded, &a_bits, sizeof(a_encoded));
    u32 b_s = g_gxState.fog.fog2Raw & 0x1F;
    g_gxState.fog.a = std::ldexp(a_encoded, static_cast<int>(b_s));
    g_gxState.stateDirty = true;
    break;
  }
  // FOG1 (0xEF): B mantissa (24-bit)
  case 0xEF: {
    g_gxState.fog.fog1Raw = value;
    u32 b_m = bp_get(value, 24, 0);
    u32 b_s = g_gxState.fog.fog2Raw & 0x1F;
    float B_mant = static_cast<float>(b_m) / 8388638.0f;
    g_gxState.fog.b = std::ldexp(B_mant, static_cast<int>(b_s) - 1);
    g_gxState.stateDirty = true;
    break;
  }
  // FOG2 (0xF0): B shift/exponent (5-bit)
  case 0xF0: {
    g_gxState.fog.fog2Raw = value;
    u32 b_s = bp_get(value, 5, 0);
    // Recompute A with updated b_s
    u32 a_mant = bp_get(g_gxState.fog.fog0Raw, 11, 0);
    u32 a_exp = bp_get(g_gxState.fog.fog0Raw, 8, 11);
    u32 a_sign = bp_get(g_gxState.fog.fog0Raw, 1, 19);
    u32 a_bits = (a_sign << 31) | (a_exp << 23) | (a_mant << 12);
    float a_encoded;
    std::memcpy(&a_encoded, &a_bits, sizeof(a_encoded));
    g_gxState.fog.a = std::ldexp(a_encoded, static_cast<int>(b_s));
    // Recompute B with updated b_s
    u32 b_m = bp_get(g_gxState.fog.fog1Raw, 24, 0);
    float B_mant = static_cast<float>(b_m) / 8388638.0f;
    g_gxState.fog.b = std::ldexp(B_mant, static_cast<int>(b_s) - 1);
    g_gxState.stateDirty = true;
    break;
  }

  // Fog type + C parameter from FOG3 (0xF1)
  case 0xF1: {
    GXFogType fogType = static_cast<GXFogType>(bp_get(value, 3, 21));
    g_gxState.fog.type = fogType;
    // Decode C parameter (same partial float encoding as A)
    u32 c_mant = bp_get(value, 11, 0);
    u32 c_exp = bp_get(value, 8, 11);
    u32 c_sign = bp_get(value, 1, 19);
    u32 c_bits = (c_sign << 31) | (c_exp << 23) | (c_mant << 12);
    std::memcpy(&g_gxState.fog.c, &c_bits, sizeof(g_gxState.fog.c));
    g_gxState.stateDirty = true;
    break;
  }

  // Fog color from FOGCLR (0xF2)
  case 0xF2: {
    u8 b = bp_get(value, 8, 0);
    u8 g = bp_get(value, 8, 8);
    u8 r = bp_get(value, 8, 16);
    g_gxState.fog.color = {
        static_cast<float>(r) / 255.f,
        static_cast<float>(g) / 255.f,
        static_cast<float>(b) / 255.f,
        1.f,
    };
    g_gxState.stateDirty = true;
    break;
  }

  // TEV color registers / K color registers (0xE0-0xE7)
  // RA registers: 0xE0, 0xE2, 0xE4, 0xE6 (even)
  // BG registers: 0xE1, 0xE3, 0xE5, 0xE7 (odd)
  // Bit 23 distinguishes: 0 = TEV color register, 1 = K color register
  case 0xE0:
  case 0xE1:
  case 0xE2:
  case 0xE3:
  case 0xE4:
  case 0xE5:
  case 0xE6:
  case 0xE7: {
    u32 idx = (regId - 0xE0) / 2;
    bool isRA = (regId & 1) == 0;
    bool isKColor = bp_get(value, 1, 23) != 0;

    if (isKColor) {
      // K color register (8-bit components)
      if (idx < GX_MAX_KCOLOR) {
        auto& kc = g_gxState.kcolors[idx];
        if (isRA) {
          kc[0] = static_cast<float>(bp_get(value, 8, 0)) / 255.f;  // R
          kc[3] = static_cast<float>(bp_get(value, 8, 12)) / 255.f; // A
        } else {
          kc[2] = static_cast<float>(bp_get(value, 8, 0)) / 255.f;  // B
          kc[1] = static_cast<float>(bp_get(value, 8, 12)) / 255.f; // G
        }
        g_gxState.stateDirty = true;
      }
    } else {
      // TEV color register (11-bit signed components)
      if (idx < MaxTevRegs) {
        auto& cr = g_gxState.colorRegs[idx];
        if (isRA) {
          // 11-bit signed: sign-extend from 11 bits
          s32 r = bp_get(value, 11, 0);
          if (r & 0x400)
            r |= ~0x7FF; // sign extend
          s32 a = bp_get(value, 11, 12);
          if (a & 0x400)
            a |= ~0x7FF;
          cr[0] = static_cast<float>(r) / 255.f;
          cr[3] = static_cast<float>(a) / 255.f;
        } else {
          s32 b = bp_get(value, 11, 0);
          if (b & 0x400)
            b |= ~0x7FF;
          s32 g = bp_get(value, 11, 12);
          if (g & 0x400)
            g |= ~0x7FF;
          cr[2] = static_cast<float>(b) / 255.f;
          cr[1] = static_cast<float>(g) / 255.f;
        }
        g_gxState.stateDirty = true;
      }
    }
    break;
  }

  // Indirect texture matrices (0x06-0x0E)
  // Each matrix uses 3 consecutive registers (one per row of the 3x2 matrix).
  // Matrix 0: 0x06-0x08, Matrix 1: 0x09-0x0B, Matrix 2: 0x0C-0x0E
  case 0x06:
  case 0x07:
  case 0x08:
  case 0x09:
  case 0x0A:
  case 0x0B:
  case 0x0C:
  case 0x0D:
  case 0x0E: {
    u32 idx = (regId - 0x06) / 3;    // matrix index (0-2)
    u32 column = (regId - 0x06) % 3; // column index (0-2)
    auto& info = g_gxState.indTexMtxs[idx];

    // Decode one packed matrix column: [m[0][column], m[1][column]].
    s32 col0 = bp_get(value, 11, 0);
    if (col0 & 0x400)
      col0 |= ~0x7FF; // sign-extend from 11 bits
    s32 col1 = bp_get(value, 11, 11);
    if (col1 & 0x400)
      col1 |= ~0x7FF;

    auto& packedColumn = column == 0 ? info.mtx.m0 : (column == 1 ? info.mtx.m1 : info.mtx.m2);
    packedColumn.x = static_cast<float>(col0) / 1024.0f;
    packedColumn.y = static_cast<float>(col1) / 1024.0f;

    // Accumulate the indirect matrix scale exponent. The SDK writes two bits per column, but
    // the hardware appears to ignore the top bit from the third column, leaving an effective
    // 5-bit value for adjScale = scaleExp + 17.
    u32 scaleBits = bp_get(value, 2, 22);
    u32 shift = column * 2;
    if (column == 2) {
      info.adjScaleRaw = (info.adjScaleRaw & ~(1u << shift)) | ((scaleBits & 1u) << shift);
    } else {
      info.adjScaleRaw = (info.adjScaleRaw & ~(3u << shift)) | (scaleBits << shift);
    }
    info.scaleExp = static_cast<s8>(info.adjScaleRaw) - 17;

    g_gxState.stateDirty = true;
    break;
  }

  // SU texture coordinate scale registers (0x30-0x3F)
  // Even registers (suTs0): S-axis scale, bias, cyl wrap, line/point offset
  // Odd registers (suTs1): T-axis scale, bias, cyl wrap
  case 0x30:
  case 0x31:
  case 0x32:
  case 0x33:
  case 0x34:
  case 0x35:
  case 0x36:
  case 0x37:
  case 0x38:
  case 0x39:
  case 0x3A:
  case 0x3B:
  case 0x3C:
  case 0x3D:
  case 0x3E:
  case 0x3F: {
    u32 coordIdx = (regId - 0x30) / 2;
    bool isT = (regId & 1) != 0;
    auto& tcs = g_gxState.texCoordScales[coordIdx];
    if (isT) {
      tcs.scaleT = static_cast<u16>(bp_get(value, 16, 0));
      tcs.biasT = bp_get(value, 1, 16) != 0;
      tcs.cylWrapT = bp_get(value, 1, 17) != 0;
    } else {
      tcs.scaleS = static_cast<u16>(bp_get(value, 16, 0));
      tcs.biasS = bp_get(value, 1, 16) != 0;
      tcs.cylWrapS = bp_get(value, 1, 17) != 0;
      tcs.lineOffset = bp_get(value, 1, 18) != 0;
      tcs.pointOffset = bp_get(value, 1, 19) != 0;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Copy clear color (0x4F-0x50) and depth (0x51)
  case 0x4F: {
    u8 r = bp_get(value, 8, 0);
    u8 a = bp_get(value, 8, 8);
    g_gxState.clearColor[0] = static_cast<float>(r) / 255.f;
    g_gxState.clearColor[3] = static_cast<float>(a) / 255.f;
    g_gxState.stateDirty = true;
    break;
  }
  case 0x50: {
    u8 b = bp_get(value, 8, 0);
    u8 g = bp_get(value, 8, 8);
    g_gxState.clearColor[2] = static_cast<float>(b) / 255.f;
    g_gxState.clearColor[1] = static_cast<float>(g) / 255.f;
    g_gxState.stateDirty = true;
    break;
  }
  case 0x51: {
    g_gxState.clearDepth = bp_get(value, 24, 0);
    g_gxState.stateDirty = true;
    break;
  }

  default:
    if (const auto mapping = decode_tex_bp_reg(regId); mapping.has_value()) {
      g_gxState.textureDirty++;
      auto& slot = g_gxState.loadedTextures[mapping->texMapId];
      switch (mapping->kind) {
      case TexBpRegMapping::Kind::Mode0:
        slot.mode0 = value;
        break;
      case TexBpRegMapping::Kind::Mode1:
        slot.mode1 = value;
        break;
      case TexBpRegMapping::Kind::Image0:
        slot.image0 = value;
        slot.mWidth = 0;
        slot.mHeight = 0;
        slot.mFormat = gfx::InvalidTextureFormat;
        break;
      case TexBpRegMapping::Kind::Image3:
        slot.image3 = value;
        break;
      case TexBpRegMapping::Kind::Tlut:
        // TLUT region's TMEM offset
        break;
      case TexBpRegMapping::Kind::Image1:
      case TexBpRegMapping::Kind::Image2:
        // GXTexRegion regs
        break;
      }
      g_gxState.stateDirty = true;
    } else {
#ifndef NDEBUG
      Log.debug("Unhandled BP register 0x{:02X} (value 0x{:06X})", regId, value & 0xFFFFFF);
#endif
    }
    break;
  }
  return true;
}

// CP register handler - decodes CP register writes and updates g_gxState
static void handle_cp(u8 addr, u32 value, bool bigEndian) {
  switch (addr) {
  // VCD low (0x50)
  case 0x50: {
    auto& vd = g_gxState.vtxDesc;
    vd[GX_VA_PNMTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 0));
    vd[GX_VA_TEX0MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 1));
    vd[GX_VA_TEX1MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 2));
    vd[GX_VA_TEX2MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 3));
    vd[GX_VA_TEX3MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 4));
    vd[GX_VA_TEX4MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 5));
    vd[GX_VA_TEX5MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 6));
    vd[GX_VA_TEX6MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 7));
    vd[GX_VA_TEX7MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 8));
    vd[GX_VA_POS] = static_cast<GXAttrType>(bp_get(value, 2, 9));
    vd[GX_VA_NRM] = static_cast<GXAttrType>(bp_get(value, 2, 11));
    vd[GX_VA_CLR0] = static_cast<GXAttrType>(bp_get(value, 2, 13));
    vd[GX_VA_CLR1] = static_cast<GXAttrType>(bp_get(value, 2, 15));
    g_gxState.stateDirty = true;
    g_gxState.clearVtxSizeCache();
    s_vtxDescPackedDirty = true;
    ++s_vtxStateDirtyGen;
    break;
  }

  // VCD high (0x60)
  case 0x60: {
    auto& vd = g_gxState.vtxDesc;
    vd[GX_VA_TEX0] = static_cast<GXAttrType>(bp_get(value, 2, 0));
    vd[GX_VA_TEX1] = static_cast<GXAttrType>(bp_get(value, 2, 2));
    vd[GX_VA_TEX2] = static_cast<GXAttrType>(bp_get(value, 2, 4));
    vd[GX_VA_TEX3] = static_cast<GXAttrType>(bp_get(value, 2, 6));
    vd[GX_VA_TEX4] = static_cast<GXAttrType>(bp_get(value, 2, 8));
    vd[GX_VA_TEX5] = static_cast<GXAttrType>(bp_get(value, 2, 10));
    vd[GX_VA_TEX6] = static_cast<GXAttrType>(bp_get(value, 2, 12));
    vd[GX_VA_TEX7] = static_cast<GXAttrType>(bp_get(value, 2, 14));
    g_gxState.stateDirty = true;
    g_gxState.clearVtxSizeCache();
    s_vtxDescPackedDirty = true;
    ++s_vtxStateDirtyGen;
    break;
  }

  // Matrix index A (0x30)
  case 0x30: {
    const u32 newPnMtx = bp_get(value, 6, 0) / 3;
    if (g_gxState.currentPnMtx != newPnMtx) {
      g_gxState.currentPnMtx = newPnMtx;
      g_gxState.stateDirty = true;
      g_gxState.uniformDirty++;
    }
    break;
  }

  // Matrix index B (0x40)
  case 0x40:
    // Texture matrix indices - used for multi-matrix texgen
    break;

  default:
    // VAT A registers (0x70-0x77)
    if (addr >= 0x70 && addr <= 0x77) {
      u32 fmt = addr - 0x70;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_POS].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 0));
      vf.attrs[GX_VA_POS].type = static_cast<GXCompType>(bp_get(value, 3, 1));
      vf.attrs[GX_VA_POS].frac = static_cast<u8>(bp_get(value, 5, 4));
      const auto nrm_cnt = bp_get(value, 1, 9);
      const auto nrm_nbt3 = bp_get(value, 1, 31);
      vf.attrs[GX_VA_NRM].cnt = static_cast<GXCompCnt>(nrm_nbt3 ? GX_NRM_NBT3 : (nrm_cnt ? GX_NRM_NBT : GX_NRM_XYZ));
      vf.attrs[GX_VA_NRM].type = static_cast<GXCompType>(bp_get(value, 3, 10));
      if (vf.attrs[GX_VA_NRM].type == GX_U8 || vf.attrs[GX_VA_NRM].type == GX_S8) {
        vf.attrs[GX_VA_NRM].frac = 6;
      } else if (vf.attrs[GX_VA_NRM].type == GX_U16 || vf.attrs[GX_VA_NRM].type == GX_S16) {
        vf.attrs[GX_VA_NRM].frac = 14;
      } else {
        vf.attrs[GX_VA_NRM].frac = 0;
      }
      vf.attrs[GX_VA_CLR0].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 13));
      vf.attrs[GX_VA_CLR0].type = static_cast<GXCompType>(bp_get(value, 3, 14));
      vf.attrs[GX_VA_CLR0].frac = 0;
      vf.attrs[GX_VA_CLR1].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 17));
      vf.attrs[GX_VA_CLR1].type = static_cast<GXCompType>(bp_get(value, 3, 18));
      vf.attrs[GX_VA_CLR1].frac = 0;
      vf.attrs[GX_VA_TEX0].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 21));
      vf.attrs[GX_VA_TEX0].type = static_cast<GXCompType>(bp_get(value, 3, 22));
      vf.attrs[GX_VA_TEX0].frac = static_cast<u8>(bp_get(value, 5, 25));
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
      ++s_vtxFmtGen[fmt];
      ++s_vtxStateDirtyGen;
    }
    // VAT B registers (0x80-0x87)
    else if (addr >= 0x80 && addr <= 0x87) {
      u32 fmt = addr - 0x80;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_TEX1].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 0));
      vf.attrs[GX_VA_TEX1].type = static_cast<GXCompType>(bp_get(value, 3, 1));
      vf.attrs[GX_VA_TEX1].frac = static_cast<u8>(bp_get(value, 5, 4));
      vf.attrs[GX_VA_TEX2].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 9));
      vf.attrs[GX_VA_TEX2].type = static_cast<GXCompType>(bp_get(value, 3, 10));
      vf.attrs[GX_VA_TEX2].frac = static_cast<u8>(bp_get(value, 5, 13));
      vf.attrs[GX_VA_TEX3].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 18));
      vf.attrs[GX_VA_TEX3].type = static_cast<GXCompType>(bp_get(value, 3, 19));
      vf.attrs[GX_VA_TEX3].frac = static_cast<u8>(bp_get(value, 5, 22));
      vf.attrs[GX_VA_TEX4].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 27));
      vf.attrs[GX_VA_TEX4].type = static_cast<GXCompType>(bp_get(value, 3, 28));
      // TEX4 frac is in VAT C
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
      ++s_vtxFmtGen[fmt];
      ++s_vtxStateDirtyGen;
    }
    // VAT C registers (0x90-0x97)
    else if (addr >= 0x90 && addr <= 0x97) {
      u32 fmt = addr - 0x90;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_TEX4].frac = static_cast<u8>(bp_get(value, 5, 0));
      vf.attrs[GX_VA_TEX5].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 5));
      vf.attrs[GX_VA_TEX5].type = static_cast<GXCompType>(bp_get(value, 3, 6));
      vf.attrs[GX_VA_TEX5].frac = static_cast<u8>(bp_get(value, 5, 9));
      vf.attrs[GX_VA_TEX6].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 14));
      vf.attrs[GX_VA_TEX6].type = static_cast<GXCompType>(bp_get(value, 3, 15));
      vf.attrs[GX_VA_TEX6].frac = static_cast<u8>(bp_get(value, 5, 18));
      vf.attrs[GX_VA_TEX7].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 23));
      vf.attrs[GX_VA_TEX7].type = static_cast<GXCompType>(bp_get(value, 3, 24));
      vf.attrs[GX_VA_TEX7].frac = static_cast<u8>(bp_get(value, 5, 27));
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
      ++s_vtxFmtGen[fmt];
      ++s_vtxStateDirtyGen;
    }
    // Array base addresses (0xA0-0xAF)
    else if (addr >= 0xA0 && addr <= 0xAF) {
      Log.error("CP_REG_ARRAYBASE_ID is not supported on Aurora. Use GX_AURORA_LOAD_ARRAYBASE instead.");
    }
    // Array strides (0xB0-0xBF)
    else if (addr >= 0xB0 && addr <= 0xBF) {
      u32 attrIdx = addr - 0xB0 + GX_VA_POS;
      if (attrIdx < GX_VA_MAX_ATTR) {
        auto& array = g_gxState.arrays[attrIdx];
        const auto newStride = static_cast<u8>(value);
        if (array.stride != newStride) {
          array.stride = newStride;
          g_gxState.stateDirty = true;
          if (attrIdx >= GX_VA_POS && attrIdx < GX_VA_POS + 12) {
            ++s_arrayStateGen[attrIdx - GX_VA_POS];
          }
          ++s_vtxStateDirtyGen;
        }
      }
    }
    break;
  }
}

// XF register handler - decodes XF (transform unit) register writes and updates g_gxState
static void handle_xf(const u8* data, u32& pos, u32 size, bool bigEndian, bool& pipelineChanged) {
  CHECK(pos + 4 <= size, "XF header read overrun");
  u32 header = read_u32(data + pos, bigEndian);
  pos += 4;

  u32 count = ((header >> 16) & 0xFFFF) + 1;
  u32 addr = header & 0xFFFF;
  u32 dataBytes = count * 4;
  // Log.warn("  xf: addr {:04x} count {} dataBytes {} pos {} -> {}", addr, count, dataBytes, pos, pos + dataBytes);
  CHECK(pos + dataBytes <= size, "XF data read overrun: need {} bytes at pos {}", dataBytes, pos);

  const u8* xfData = data + pos;

  if (copy_xf_data(addr, xfData, count, bigEndian)) {
    // copy_xf_data handled everything (matrices and lights, uniforms only).
  } else if (addr >= 0x1000) {
    // XF registers (0x1000+)
    u32 xfAddr = addr - 0x1000;
    for (u32 i = 0; i < count; i++) {
      u32 reg = xfAddr + i;
      u32 val = read_u32(xfData + i * 4, bigEndian);

      // Skip scalar register writes that haven't changed
      if ((reg <= 0x19 || (reg >= 0x3F && reg <= 0x5F))) {
        if (g_gxState.xfRegValid[reg] && val == g_gxState.xfRegCache[reg]) {
          continue;
        }
        g_gxState.xfRegValid[reg] = true;
        g_gxState.xfRegCache[reg] = val;
      }

      switch (reg) {
      case 0x08:
        // XF vertex specs (numColors, numNormals, numTexCoords) - informational
        break;
      case 0x09:
        // numChans
        g_gxState.numChans = val;
        g_gxState.stateDirty = true;
        pipelineChanged = true;
        break;
      case 0x0A:
        // Ambient color 0
        g_gxState.colorChannelState[GX_COLOR0].ambColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA0].ambColor = unpack_color(val);
        g_gxState.stateDirty = true;
        ++s_curUniState.othAmbGen[0];
        ++s_splitOthUniGen;
        ++s_splitOthAmbGen;
        break;
      case 0x0B:
        // Ambient color 1
        g_gxState.colorChannelState[GX_COLOR1].ambColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA1].ambColor = unpack_color(val);
        g_gxState.stateDirty = true;
        ++s_curUniState.othAmbGen[1];
        ++s_splitOthUniGen;
        ++s_splitOthAmbGen;
        break;
      case 0x0C:
        // Material color 0
        g_gxState.colorChannelState[GX_COLOR0].matColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA0].matColor = unpack_color(val);
        g_gxState.stateDirty = true;
        ++s_curUniState.othMatGen[0];
        ++s_splitOthUniGen;
        ++s_splitOthMatGen;
        break;
      case 0x0D:
        // Material color 1
        g_gxState.colorChannelState[GX_COLOR1].matColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA1].matColor = unpack_color(val);
        g_gxState.stateDirty = true;
        ++s_curUniState.othMatGen[1];
        ++s_splitOthUniGen;
        ++s_splitOthMatGen;
        break;
      case 0x0E:
      case 0x0F:
      case 0x10:
      case 0x11: {
        // Channel control registers
        u32 chanId = reg - 0x0E;
        if (chanId < MaxColorChannels) {
          auto& chan = g_gxState.colorChannelConfig[chanId];
          chan.matSrc = static_cast<GXColorSrc>(bp_get(val, 1, 0));
          chan.lightingEnabled = bp_get(val, 1, 1) != 0;
          u32 lightsLo = bp_get(val, 4, 2);
          chan.ambSrc = static_cast<GXColorSrc>(bp_get(val, 1, 6));
          chan.diffFn = static_cast<GXDiffuseFn>(bp_get(val, 2, 7));
          // Encoding: bit 9 = (attnFn != GX_AF_SPEC), bit 10 = (attnFn != GX_AF_NONE)
          bool bit9 = bp_get(val, 1, 9) != 0;
          bool bit10 = bp_get(val, 1, 10) != 0;
          u32 lightsHi = bp_get(val, 4, 11);
          if (!bit10) {
            chan.attnFn = GX_AF_NONE;
          } else if (!bit9) {
            chan.attnFn = GX_AF_SPEC;
          } else {
            chan.attnFn = GX_AF_SPOT;
          }
          const u32 lightMask = lightsLo | (lightsHi << 4);
          const GX::LightMask nextMask{lightMask};
          if (g_gxState.colorChannelState[chanId].lightMask != nextMask) {
            g_gxState.colorChannelState[chanId].lightMask = nextMask;
            g_gxState.uniformDirty++;
            ++s_splitLightsGen;
          }
          g_gxState.stateDirty = true;
          pipelineChanged = true;
        }
        break;
      }
      case 0x18: {
        // Matrix index A: PnMtx + TexCoord0-3 matrix indices
        g_gxState.currentPnMtx = bp_get(val, 6, 0) / 3;
        for (u32 i = 0; i < 4 && i < MaxTexCoord; i++) {
          auto texMtx = static_cast<GXTexMtx>(bp_get(val, 6, 6 + i * 6));
          assert(texMtx >= 0 && texMtx <= GXTexMtx::GX_IDENTITY);
          if (g_gxState.tcgs[i].mtx != texMtx) {
            g_gxState.tcgs[i].mtx = texMtx;
            pipelineChanged = true;
          }
        }
        g_gxState.stateDirty = true;
        break;
      }
      case 0x19: {
        // Matrix index B: TexCoord4-7 matrix indices
        for (u32 i = 0; i < 4 && (i + 4) < MaxTexCoord; i++) {
          auto texMtx = static_cast<GXTexMtx>(bp_get(val, 6, i * 6));
          if (g_gxState.tcgs[i + 4].mtx != texMtx) {
            g_gxState.tcgs[i + 4].mtx = texMtx;
            pipelineChanged = true;
          }
        }
        g_gxState.stateDirty = true;
        break;
      }
      case 0x1A:
      case 0x1B:
      case 0x1C:
      case 0x1D:
      case 0x1E:
      case 0x1F: {
        // Viewport: sx, sy, sz, ox, oy, oz at XF 0x101A-0x101F
        u32 vpOff = reg - 0x1A;
        if (vpOff == 0 && count >= 6) {
          bool unchanged = true;
          for (u32 k = 0; k < 6; ++k) {
            const u32 w = k == 0 ? val : read_u32(xfData + (i + k) * 4, bigEndian);
            if (!g_gxState.xfRegValid[0x1A + k] || g_gxState.xfRegCache[0x1A + k] != w) {
              unchanged = false;
              break;
            }
          }
          if (unchanged) {
            i += 5;
            continue;
          }
          for (u32 k = 0; k < 6; ++k) {
            g_gxState.xfRegValid[0x1A + k] = true;
            g_gxState.xfRegCache[0x1A + k] = k == 0 ? val : read_u32(xfData + (i + k) * 4, bigEndian);
          }
          const f32 sx = read_f32(xfData + (i + 0) * 4, bigEndian);
          const f32 sy = read_f32(xfData + (i + 1) * 4, bigEndian);
          const f32 sz = read_f32(xfData + (i + 2) * 4, bigEndian);
          const f32 ox = read_f32(xfData + (i + 3) * 4, bigEndian);
          const f32 oy = read_f32(xfData + (i + 4) * 4, bigEndian);
          const f32 oz = read_f32(xfData + (i + 5) * 4, bigEndian);
          const f32 width = sx * 2.0f;
          const f32 height = -sy * 2.0f;
          const f32 left = ox - 340.0f - width / 2.0f;
          const f32 top = oy - 340.0f - height / 2.0f;
          const f32 znear = (oz - sz) / 1.6777215e7f;
          const f32 zfar = oz / 1.6777215e7f;
          const auto& lv = g_gxState.logicalViewport;
          if (lv.left != left || lv.top != top || lv.width != width || lv.height != height ||
              lv.znear != znear || lv.zfar != zfar) {
            flush_pending_pnmtx_batch();
            set_logical_viewport({
                .left = left,
                .top = top,
                .width = width,
                .height = height,
                .znear = znear,
                .zfar = zfar,
            });
            g_gxState.uniformDirty++;
            g_gxState.stateDirty = true;
            ++s_curUniState.vpProjGen;
            ++s_splitOthUniGen;
            ++s_splitOthVpGen;
          }
        }
        break;
      }
      case 0x20:
      case 0x21:
      case 0x22:
      case 0x23:
      case 0x24:
      case 0x25:
      case 0x26: {
        // Projection: 6 params + type at XF 0x1020-0x1026
        u32 projOff = reg - 0x20;
        if (projOff == 0 && count >= 7) {
          bool unchanged = true;
          for (u32 k = 0; k < 7; ++k) {
            const u32 w = k == 0 ? val : read_u32(xfData + (i + k) * 4, bigEndian);
            if (!g_gxState.xfRegValid[0x20 + k] || g_gxState.xfRegCache[0x20 + k] != w) {
              unchanged = false;
              break;
            }
          }
          if (unchanged) {
            i += 6;
            continue;
          }
          for (u32 k = 0; k < 7; ++k) {
            g_gxState.xfRegValid[0x20 + k] = true;
            g_gxState.xfRegCache[0x20 + k] = k == 0 ? val : read_u32(xfData + (i + k) * 4, bigEndian);
          }
          const f32 p0 = read_f32(xfData + (i + 0) * 4, bigEndian);
          const f32 p1 = read_f32(xfData + (i + 1) * 4, bigEndian);
          const f32 p2 = read_f32(xfData + (i + 2) * 4, bigEndian);
          const f32 p3 = read_f32(xfData + (i + 3) * 4, bigEndian);
          const f32 p4 = read_f32(xfData + (i + 4) * 4, bigEndian);
          const f32 p5 = read_f32(xfData + (i + 5) * 4, bigEndian);
          const u32 projType = read_u32(xfData + (i + 6) * 4, bigEndian);
          Mat4x4<float> nextProj{};
          nextProj.m0[0] = p0;
          nextProj.m1[1] = p2;
          nextProj.m2[2] = p4;
          nextProj.m2[3] = p5;
          if (projType == GX_ORTHOGRAPHIC) {
            nextProj.m0[3] = p1;
            nextProj.m1[3] = p3;
            nextProj.m3[3] = 1.0f;
          } else {
            nextProj.m0[2] = p1;
            nextProj.m1[2] = p3;
            nextProj.m3[2] = -1.0f;
          }
          const auto nextType = static_cast<GXProjectionType>(projType);
          if (g_gxState.projType != nextType ||
              std::memcmp(&g_gxState.proj, &nextProj, sizeof(nextProj)) != 0) {
            g_gxState.projType = nextType;
            g_gxState.proj = nextProj;
            g_gxState.stateDirty = true;
            ++s_curUniState.vpProjGen;
            ++s_splitOthUniGen;
            ++s_splitOthProjGen;
          }
        }
        break;
      }
      case 0x3F:
        // numTexGens
        if (g_gxState.numTexGens != static_cast<u8>(val)) {
          g_gxState.numTexGens = static_cast<u8>(val);
          g_gxState.stateDirty = true;
          pipelineChanged = true;
        }
        break;
      default:
        // TexGen config (0x40-0x4F) and post-transform (0x50-0x5F)
        if (reg >= 0x40 && reg <= 0x4F) {
          u32 tcIdx = reg - 0x40;
          if (tcIdx < MaxTexCoord) {
            auto& tcg = g_gxState.tcgs[tcIdx];
            bool proj = bp_get(val, 1, 1) != 0;
            u32 tgType = bp_get(val, 3, 4);
            u32 srcRow = bp_get(val, 5, 7);

            GXTexGenType nextType = tcg.type;
            if (tgType == 0) {
              nextType = proj ? GX_TG_MTX3x4 : GX_TG_MTX2x4;
            } else if (tgType == 1) {
              // Bump mapping: type encodes emboss light
              nextType = static_cast<GXTexGenType>(bp_get(val, 3, 15) + 2);
            } else if (tgType == 2 || tgType == 3) {
              nextType = GX_TG_SRTG;
            }
            // Emboss source texcoord (bits 12-14); 0 for non-bump types
            const u8 nextEmbossSrc = static_cast<u8>(bp_get(val, 3, 12));

            // Decode source from row
            static const GXTexGenSrc rowToSrc[] = {GX_TG_POS,  GX_TG_NRM,  GX_TG_COLOR0, GX_TG_BINRM, GX_TG_TANGENT,
                                                   GX_TG_TEX0, GX_TG_TEX1, GX_TG_TEX2,   GX_TG_TEX3,  GX_TG_TEX4,
                                                   GX_TG_TEX5, GX_TG_TEX6, GX_TG_TEX7};
            const GXTexGenSrc nextSrc = (srcRow < 13) ? rowToSrc[srcRow] : tcg.src;
            if (tcg.type != nextType || tcg.embossSrc != nextEmbossSrc || tcg.src != nextSrc) {
              tcg.type = nextType;
              tcg.embossSrc = nextEmbossSrc;
              tcg.src = nextSrc;
              g_gxState.stateDirty = true;
              pipelineChanged = true;
            }
          }
        } else if (reg >= 0x50 && reg <= 0x5F) {
          u32 tcIdx = reg - 0x50;
          if (tcIdx < MaxTexCoord) {
            auto& tcg = g_gxState.tcgs[tcIdx];
            const auto nextPostMtx = static_cast<GXPTTexMtx>(bp_get(val, 6, 0) + 64);
            const bool nextNorm = bp_get(val, 1, 8) != 0;
            if (tcg.postMtx != nextPostMtx || tcg.normalize != nextNorm) {
              tcg.postMtx = nextPostMtx;
              tcg.normalize = nextNorm;
              g_gxState.stateDirty = true;
              pipelineChanged = true;
            }
          }
        } else {
#ifndef NDEBUG
          Log.debug("Unhandled XF register 0x{:04X} (value 0x{:08X})", reg, val);
#endif
        }
        break;
      }
    }
  }

  pos += dataBytes;
}

static void handle_draw_overrun [[noreturn]] (u32 totalVtxBytes, const u8* data, const u32& pos, u32 size) {
  // Hex dump around the draw command for debugging
  u32 cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  u32 dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  u32 dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (u32 i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, size);
}

// Draw command handler - parses vertices inline and caches results

struct BoundingBox {
  bool armed = false;
  bool unknown = false;
  float left = 0.f, top = 0.f, right = -1.f, bottom = -1.f;
  u16 result[4] = {0x3FF, 0x3FF, 0, 0};
};
BoundingBox sBbox;

bool bbox_read_position(const u8* vtx, GXVtxFmt fmt, float out[3], u32& pnMtx) noexcept {
  const auto& vf = g_gxState.vtxFmts[fmt];
  u32 off = 0;
  pnMtx = g_gxState.currentPnMtx;
  for (int i = GX_VA_PNMTXIDX; i < GX_VA_POS; ++i) {
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT:
      if (i == GX_VA_PNMTXIDX) {
        pnMtx = vtx[off] / 3;
      }
      off += comp_type_size(static_cast<GXAttr>(i), vf.attrs[i].type) *
             comp_cnt_count(static_cast<GXAttr>(i), vf.attrs[i].cnt);
      break;
    case GX_INDEX8:
      off += 1;
      break;
    case GX_INDEX16:
      off += 2;
      break;
    }
  }
  const auto& af = vf.attrs[GX_VA_POS];
  const u32 count = af.cnt == GX_POS_XY ? 2 : 3;
  const u8* src = nullptr;
  bool le = false;
  switch (g_gxState.vtxDesc[GX_VA_POS]) {
  case GX_DIRECT:
    src = vtx + off;
    break;
  case GX_INDEX8:
  case GX_INDEX16: {
    const u32 idx = g_gxState.vtxDesc[GX_VA_POS] == GX_INDEX8 ? vtx[off] : (u32(vtx[off]) << 8 | vtx[off + 1]);
    const auto& arr = g_gxState.arrays[GX_VA_POS];
    if (arr.data == nullptr || arr.stride == 0) {
      return false;
    }
    const u64 elemBytes = u64(count) * comp_type_size(GX_VA_POS, af.type);
    if (arr.size != 0 && u64(idx) * arr.stride + elemBytes > arr.size) {
      return false;
    }
    src = static_cast<const u8*>(arr.data) + u64(idx) * arr.stride;
    le = arr.le;
    break;
  }
  default:
    return false;
  }
  const float scale = 1.f / static_cast<float>(1u << af.frac);
  for (u32 c = 0; c < count; ++c) {
    switch (af.type) {
    case GX_U8:
      out[c] = static_cast<float>(src[c]) * scale;
      break;
    case GX_S8:
      out[c] = static_cast<float>(static_cast<s8>(src[c])) * scale;
      break;
    case GX_U16:
    case GX_S16: {
      u16 v;
      memcpy(&v, src + 2 * c, 2);
      if (!le) {
        v = bswap(v);
      }
      out[c] = (af.type == GX_U16 ? static_cast<float>(v) : static_cast<float>(static_cast<s16>(v))) * scale;
      break;
    }
    case GX_F32: {
      u32 v;
      memcpy(&v, src + 4 * c, 4);
      if (!le) {
        v = bswap(v);
      }
      out[c] = std::bit_cast<float>(v);
      break;
    }
    default:
      return false;
    }
  }
  if (count == 2) {
    out[2] = 0.f;
  }
  return true;
}

void bbox_add_vertices(const u8* data, u32 vtxCount, u32 vtxSize, GXVtxFmt fmt) noexcept {
  if (vtxSize == 0) {
    return;
  }
  const auto& proj = g_gxState.proj;
  const auto& vp = g_gxState.logicalViewport;
  for (u32 i = 0; i < vtxCount && !sBbox.unknown; ++i) {
    float p[3];
    u32 mtx;
    if (!bbox_read_position(data + i * vtxSize, fmt, p, mtx) || mtx >= MaxPnMtx) {
      sBbox.unknown = true;
      break;
    }
    const auto& m = g_gxState.pnMtx[mtx].pos;
    const float v0 = m.m0[0] * p[0] + m.m0[1] * p[1] + m.m0[2] * p[2] + m.m0[3];
    const float v1 = m.m1[0] * p[0] + m.m1[1] * p[1] + m.m1[2] * p[2] + m.m1[3];
    const float v2 = m.m2[0] * p[0] + m.m2[1] * p[1] + m.m2[2] * p[2] + m.m2[3];
    const float cx = proj.m0[0] * v0 + proj.m0[1] * v1 + proj.m0[2] * v2 + proj.m0[3];
    const float cy = proj.m1[0] * v0 + proj.m1[1] * v1 + proj.m1[2] * v2 + proj.m1[3];
    const float cw = proj.m3[0] * v0 + proj.m3[1] * v1 + proj.m3[2] * v2 + proj.m3[3];
    if (!(cw > 1e-6f)) {
      sBbox.unknown = true;
      break;
    }
    const float sx = vp.left + (cx / cw + 1.f) * 0.5f * vp.width;
    const float sy = vp.top + (1.f - cy / cw) * 0.5f * vp.height;
    sBbox.left = std::min(sBbox.left, sx);
    sBbox.right = std::max(sBbox.right, sx);
    sBbox.top = std::min(sBbox.top, sy);
    sBbox.bottom = std::max(sBbox.bottom, sy);
  }
}

static u32 calculate_last_vtx_size(GXVtxFmt fmt) {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      const auto& attrFmt = vtxFmt.attrs[i];
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += (i == GX_VA_NRM && vtxFmt.attrs[i].cnt == GX_NRM_NBT3) ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += (i == GX_VA_NRM && vtxFmt.attrs[i].cnt == GX_NRM_NBT3) ? 6 : 2;
      break;
    }
  }

  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;

  return vtxSize;
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange);
static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices);
// Mali fork: storage vertex fetch is impossible on the target
// (GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS=0), so a draw that native fetch declines —
// instanced lines/points (e.g. GX_LINESTRIP minimap draws), NBT normals, exotic
// component types — is dropped instead of building a storage pipeline that can't
// link and aborts the device. If any dropped geometry turns out to matter, extend
// native fetch to cover it (e.g. instanced lines); texture-vertex fetch is NOT an
// option — it's unusably slow on Mali. Flip to false only to exercise storage on desktop.
static constexpr bool kSkipStorageVertexFetch = true;

static bool try_native_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 vtxSize);
static bool try_native_draw_run(u8 cmd, GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data,
                                u32& pos, u32 size, bool bigEndian);
static bool try_native_draw_indexed(GXVtxFmt fmt, u16 vtxCount, const u8* vtxData, u32 vtxSize, const u8* idxData,
                                    u32 idxBytes, u32 indexCount);

// Draw command handler - parses vertices inline and caches results
static ByteBuffer handle_draw_idx_buf;

static const u8* s_dbgStaticDlPtr = nullptr;
static u32 s_dbgStaticDlSize = 0;

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 size) {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calculate_last_vtx_size(fmt); }

  if (prim != GX_QUADS && prim != GX_TRIANGLES && prim != GX_TRIANGLESTRIP && prim != GX_TRIANGLEFAN &&
      prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS) {
    std::string fullHex;
    const u32 fullHi = std::min<u32>(size, 128u);
    for (u32 i = 0; i < fullHi; ++i) {
      fullHex += fmt::format("{:02x} ", data[i]);
    }
    std::string vd;
    for (u32 a = 0; a < MaxVtxAttr; ++a) {
      if (g_gxState.vtxDesc[a] != GX_NONE) {
        vd += fmt::format("a{}={},", a, static_cast<u32>(g_gxState.vtxDesc[a]));
      }
    }
    Log.error("[BAD-PRIM] prim={} ({:#x}) fmt={} vtxCount={} vtxSize={} pos={} size={} staticDl=({:p},{}) vcd=[{}] bytes[0..{}]=[{}]",
              static_cast<u32>(prim), static_cast<u32>(prim), static_cast<u32>(fmt), vtxCount, vtxSize, pos, size,
              static_cast<const void*>(s_dbgStaticDlPtr), s_dbgStaticDlSize, vd, fullHi, fullHex);
  }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (pos + totalVtxBytes > size)
    UNLIKELY { handle_draw_overrun(totalVtxBytes, data, pos, size); }

  if (sBbox.armed)
    UNLIKELY { bbox_add_vertices(data + pos, vtxCount, vtxSize, fmt); }

  if (g_gxState.cullMode == GX_CULL_ALL) {
    pos += totalVtxBytes;
    return;
  }

  // Native vertex fetch expands supported layouts to hardware attributes on the CPU
  // (advancing pos itself). Unsupported layouts fall through to the storage path.
  if (try_native_draw(prim, fmt, vtxCount, data, pos, vtxSize)) {
    return;
  }

  if (s_staticDlCaptureActive) {
    s_staticDlCaptureAborted = true;
  }
  flush_pending_pnmtx_batch();
  auto* lastDraw = !g_gxState.stateDirty ? gfx::get_last_draw_command<DrawData>() : nullptr;
  // A storage draw must never merge into a native draw's range (incompatible buffer layout).
  const bool canMerge = lastDraw != nullptr && prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS &&
                        lastDraw->instanceCount == 1 && !lastDraw->nativeVertexFetch;

  // Push raw vertex data to buffer. Merged draws must remain byte-contiguous with the previous range.
  gfx::Range vertRange = gfx::push_verts(data + pos, totalVtxBytes, canMerge ? 0 : 4);
  pos += totalVtxBytes;

  // Try to merge with previous draw call
  if (canMerge) {
    // Only if the previous draw call was a single instance draw (no lines/points handling)
    u32 numIndices = 0;
    gfx::Range idxRange;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(handle_draw_idx_buf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(handle_draw_idx_buf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(handle_draw_idx_buf.data(), handle_draw_idx_buf.size(), hadIndexRange ? 0 : 4);
      handle_draw_idx_buf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    ++gfx::g_mergedDrawCallCount;
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
}

static void handle_draw(u8 cmd, const u8* data, u32& pos, u32 size, bool bigEndian) {
  GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);

  CHECK(pos + 2 <= size, "draw vtxCount read overrun");
  u16 vtxCount = read_u16(data + pos, bigEndian);
  pos += 2;

  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calculate_last_vtx_size(fmt); }

  // Fold this draw plus any immediately-following identical draw commands into one native
  // hardware draw (strip topology + primitive restart). Unsupported primitives/layouts fall
  // through to draw_prim (single native, else the storage/software path).
  if (g_gxState.cullMode != GX_CULL_ALL &&
      try_native_draw_run(cmd, prim, fmt, vtxCount, vtxSize, data, pos, size, bigEndian)) {
    return;
  }

  draw_prim(prim, fmt, vtxCount, data, pos, size);
}

static ByteBuffer handle_draw_unmerged_idxBuf;

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    auto& realBuf = handle_draw_unmerged_idxBuf;
    numIndices = prepare_idx_buffer(realBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(realBuf.data(), realBuf.size(), 4);
    realBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) {
  if constexpr (kSkipStorageVertexFetch) {
    // Native fetch declined this layout; the storage fallback would bind vertex SSBOs
    // and can't link on Mali. The caller already consumed the FIFO vertices, so just
    // drop the draw. Warn a few times so we know what geometry is being skipped.
    static Module Log("aurora::gx");
    static int warned = 0;
    if (warned++ < 8) {
      Log.warn("Dropping non-native draw (prim={} fmt={} verts={}) — storage vertex fetch unavailable on Mali",
               static_cast<int>(prim), static_cast<int>(fmt), vtxCount);
    }
    return;
  }
  // Build pipeline, bind groups, and push draw command
  BindGroupRanges ranges{};
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (g_gxState.vtxDesc[i] != GX_INDEX8 && g_gxState.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = g_gxState.arrays[i];
    if (array.cachedRange.size > 0) {
      ranges.vaRanges[i - GX_VA_POS] = array.cachedRange;
    } else {
      const auto range = gfx::push_storage(static_cast<const uint8_t*>(array.data), array.size);
      ranges.vaRanges[i - GX_VA_POS] = range;
      array.cachedRange = range;
    }
  }

  PipelineConfig config{};
  populate_pipeline_config(config, prim, fmt);
  struct PipelineMemoEntry {
    PipelineConfig config{};
    ShaderInfo info{};
    gfx::PipelineRef pipeline{};
    bool used = false;
  };
  static PipelineMemoEntry s_pipeMemo[64]{};
  const uint64_t cfgHash = xxh3_hash(config);
  const uint32_t slot = cfgHash & 63;

  ShaderInfo info;
  gfx::PipelineRef pipeline;
  if (s_pipeMemo[slot].used && std::memcmp(&config, &s_pipeMemo[slot].config, sizeof(PipelineConfig)) == 0) {
    info = s_pipeMemo[slot].info;
    pipeline = s_pipeMemo[slot].pipeline;
  } else {
    info = build_shader_info(config.shaderConfig);
    pipeline = gfx::pipeline_ref(config);
    s_pipeMemo[slot].config = config;
    s_pipeMemo[slot].info = info;
    s_pipeMemo[slot].pipeline = pipeline;
    s_pipeMemo[slot].used = true;
  }

  const auto bindGroups = resolve_and_build_bind_groups(info);

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  gfx::push_draw_command(DrawData{
      .pipeline = pipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = build_uniform(info, vertRange.offset, ranges),
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .bindGroups = bindGroups,
      .dstAlpha = g_gxState.dstAlpha,
  });
}

// ---------------------------------------------------------------------------
// Native vertex fetch.
//
// The storage path software-fetches GX vertices from SSBOs, which (a) miscompiles
// the per-vertex matrix-palette index into slot 0 on AGX/Mali (the "porcupine")
// and (b) cannot link on Mali (0 vertex-stage SSBOs). Here we expand each GX vertex
// on the CPU into real hardware vertex attributes — most importantly pnmtxidx
// becomes a plain Uint32 attribute, so it never flows through the miscompiled
// byte-load — and let the fixed-function vertex input path feed the shader.
// Skinning/projection stay on the GPU. Layouts native fetch can't express fall
// through to the storage path per-draw (try_native_draw* return false).
// ---------------------------------------------------------------------------

// Canonical hardware-attribute byte width per GX attr (must match native_vertex_layout).
static u8 canonical_attr_size(GXAttr attr) noexcept {
  if (attr == GX_VA_PNMTXIDX || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX)) {
    return 4; // Uint32
  }
  if (attr == GX_VA_POS || attr == GX_VA_NRM) {
    return 12; // Float32x3
  }
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    return 4; // Unorm8x4
  }
  return 8; // TEXn: Float32x2
}

static bool native_component_decodable(GXCompType type) noexcept {
  switch (type) {
  case GX_U8:
  case GX_S8:
  case GX_U16:
  case GX_S16:
  case GX_F32:
    return true;
  default:
    return false;
  }
}

static bool native_color_decodable(GXCompType type) noexcept {
  switch (type) {
  case GX_RGB565:
  case GX_RGB8:
  case GX_RGBX8:
  case GX_RGBA4:
  case GX_RGBA6:
  case GX_RGBA8:
    return true;
  default:
    return false;
  }
}

static f32 read_component_as_float(const u8* p, GXCompType type, u8 frac, bool bigEndian) noexcept {
  const f32 scale = 1.0f / static_cast<f32>(1u << frac);
  switch (type) {
  case GX_U8:
    return static_cast<f32>(*p) * scale;
  case GX_S8:
    return static_cast<f32>(static_cast<int8_t>(*p)) * scale;
  case GX_U16:
    return static_cast<f32>(read_u16(p, bigEndian)) * scale;
  case GX_S16:
    return static_cast<f32>(static_cast<int16_t>(read_u16(p, bigEndian))) * scale;
  case GX_F32:
    return std::bit_cast<f32>(read_u32(p, bigEndian));
  default:
    return 0.0f;
  }
}

static u32 expand_bits_to_8(u32 v, u32 bits) noexcept {
  const u32 shifted = v << (8u - bits);
  return shifted | (shifted >> bits);
}

// Decode a GX color to a packed u32 (byte 0 = R, byte 3 = A) for a Unorm8x4 attribute.
static u32 read_color_as_rgba8(const u8* p, GXCompType type, bool bigEndian) noexcept {
  const auto pack = [](u32 r, u32 g, u32 b, u32 a) { return r | (g << 8) | (b << 16) | (a << 24); };
  switch (type) {
  case GX_RGB8:
  case GX_RGBX8:
    return pack(p[0], p[1], p[2], 0xFFu);
  case GX_RGBA8:
    return pack(p[0], p[1], p[2], p[3]);
  case GX_RGB565: {
    const u16 v = read_u16(p, bigEndian);
    return pack(expand_bits_to_8((v >> 11) & 0x1Fu, 5), expand_bits_to_8((v >> 5) & 0x3Fu, 6),
                expand_bits_to_8(v & 0x1Fu, 5), 0xFFu);
  }
  case GX_RGBA4: {
    const u16 v = read_u16(p, bigEndian);
    return pack(expand_bits_to_8((v >> 12) & 0xFu, 4), expand_bits_to_8((v >> 8) & 0xFu, 4),
                expand_bits_to_8((v >> 4) & 0xFu, 4), expand_bits_to_8(v & 0xFu, 4));
  }
  case GX_RGBA6: {
    const u32 v = bigEndian ? (u32(p[0]) << 16) | (u32(p[1]) << 8) | u32(p[2])
                            : (u32(p[2]) << 16) | (u32(p[1]) << 8) | u32(p[0]);
    return pack(expand_bits_to_8((v >> 18) & 0x3Fu, 6), expand_bits_to_8((v >> 12) & 0x3Fu, 6),
                expand_bits_to_8((v >> 6) & 0x3Fu, 6), expand_bits_to_8(v & 0x3Fu, 6));
  }
  default:
    return 0xFFFFFFFFu;
  }
}

// One present GX attr in a native-expanded vertex: how to read it from the FIFO
// record / indexed array, and (via canonical order) where it lands in the output.
struct NativeAttrDesc {
  GXAttr attr;
  u8 attrType;  // GX_DIRECT / GX_INDEX8 / GX_INDEX16
  u8 srcOffset; // byte offset of this attr (direct) or its index (indexed) in the FIFO record
  u8 compType;  // GXCompType of the source component(s)
  u8 cnt;       // source component count
  u8 compSize;  // bytes per source component
  u8 frac;      // fixed-point fraction bits
  u8 arrayStride;
  bool le; // source endianness (direct FIFO = false = big-endian)
};

// Build the native attribute layout for the current vertex format. Returns false
// (declines to the storage path) for anything native cannot faithfully express:
// NBT normals, exotic component types, missing position, >16 attributes, or an
// indexed attr whose source array is not resident.
// Diagnostic: flip kLogNativeDecline to true to log (rate-limited) why a layout
// declined native fetch and fell back to storage. Storage is impossible on Mali
// (GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS=0), so every decline logged here is a draw
// that gets dropped on Mali (see kSkipStorageVertexFetch) — the fix, if it matters,
// is to widen native fetch, not texture fetch. Compile-time so there's no env cruft.
static constexpr bool kLogNativeDecline = true;
static bool native_decline(int reason, GXVtxFmt fmt, int attr) {
  if constexpr (kLogNativeDecline) {
    static Module Log("aurora::gx");
    static const char* const reasons[8] = {"NBT-normal",      "undecodable-type", "nonresident-index-array",
                                            "non-direct-type", ">16-attrs",        "no-position",
                                            "?",               "?"};
    // Dedup by unique declined layout so the boot flood can't exhaust a fixed budget
    // before an in-game dialog/HUD layout first appears — each distinct layout logs once.
    static std::set<std::string> seen;
    const auto& vf = g_gxState.vtxFmts[fmt];
    std::string layout;
    for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
      const auto t = g_gxState.vtxDesc[i];
      if (t == GX_NONE) {
        continue;
      }
      const auto& af = vf.attrs[i];
      layout += fmt::format(" a{}:desc{}/type{}/cnt{}/frac{}", i, static_cast<int>(t), static_cast<int>(af.type),
                            static_cast<int>(af.cnt), static_cast<int>(af.frac));
      if (t == GX_INDEX8 || t == GX_INDEX16) {
        const auto& arr = g_gxState.arrays[i];
        layout += fmt::format("[arrData={} sz={} str={}]", arr.data != nullptr, arr.size, arr.stride);
      }
    }
    const std::string key = fmt::format("r{}f{}{}", reason, static_cast<int>(fmt), layout);
    if (seen.size() < 64 && seen.insert(key).second) {
      Log.warn("[decline-diag] reason={} fmt={} badAttr={} |{}", reasons[(reason < 0 || reason > 7) ? 6 : reason],
               static_cast<int>(fmt), attr, layout);
    }
  }
  return false;
}

static bool build_native_layout(GXVtxFmt fmt, std::vector<NativeAttrDesc>& descs, u8& nativeStride,
                                std::array<AttrConfig, MaxVtxAttr>& nativeAttrs) {
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  descs.clear();
  nativeAttrs = {};
  u8 srcOffset = 0;
  u8 outOffset = 0;
  bool hasPos = false;
  int count = 0;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto type = g_gxState.vtxDesc[i];
    if (type == GX_NONE) {
      continue;
    }
    const auto& attrFmt = vtxFmt.attrs[i];
    const auto cnt = comp_cnt_count(attr, attrFmt.cnt);
    const bool isMtxIdx = attr == GX_VA_PNMTXIDX || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX);
    const bool isColor = attr == GX_VA_CLR0 || attr == GX_VA_CLR1;
    // Native fetch supports standard 3-component normals (cnt 3) and NBT/NBT3 (cnt 9).
    // The shader samples the primary normal (first 3 components) for lighting and reflection.
    if (!isMtxIdx) {
      const auto compType = static_cast<GXCompType>(attrFmt.type);
      if (isColor ? !native_color_decodable(compType) : !native_component_decodable(compType)) {
        return native_decline(1, fmt, i);
      }
    }

    NativeAttrDesc d{};
    d.attr = attr;
    d.attrType = static_cast<u8>(type);
    d.srcOffset = srcOffset;
    d.compType = static_cast<u8>(attrFmt.type);
    d.cnt = cnt;
    d.frac = attrFmt.frac;
    if (type == GX_DIRECT) {
      d.le = false;
      d.compSize = isMtxIdx ? 1 : comp_type_size(attr, attrFmt.type);
      srcOffset += isMtxIdx ? 1 : static_cast<u8>(comp_type_size(attr, attrFmt.type) * cnt);
    } else if (type == GX_INDEX8 || type == GX_INDEX16) {
      const auto& array = g_gxState.arrays[i];
      if (array.data == nullptr || array.stride == 0) {
        return native_decline(2, fmt, i);
      }
      d.le = array.le;
      d.arrayStride = array.stride;
      d.compSize = comp_type_size(attr, attrFmt.type);
      if (attr == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3) {
        srcOffset += (type == GX_INDEX8) ? 3 : 6;
      } else {
        srcOffset += (type == GX_INDEX8) ? 1 : 2;
      }
    } else {
      return native_decline(3, fmt, i);
    }

    auto& na = nativeAttrs[i];
    na.attrType = GX_DIRECT;
    na.offset = outOffset;
    na.cnt = isMtxIdx ? 1 : (isColor ? 1 : (attr == GX_VA_POS || attr == GX_VA_NRM ? 3 : 2));
    na.compType = isMtxIdx ? static_cast<u8>(GX_U8) : (isColor ? static_cast<u8>(GX_RGBA8) : static_cast<u8>(GX_F32));
    outOffset += canonical_attr_size(attr);
    descs.push_back(d);
    if (attr == GX_VA_POS) {
      hasPos = true;
    }
    if (++count > 16) {
      return native_decline(4, fmt, i);
    }
  }
  if (!hasPos) {
    return native_decline(5, fmt, GX_VA_POS);
  }
  nativeStride = outOffset;
  return true;
}

struct AttrLayoutEntry {
  std::array<AttrConfig, MaxVtxAttr> attrs{};
  u8 stride = 0;
};
static std::vector<AttrLayoutEntry> s_attrLayoutPool;

static u16 intern_attr_layout(const std::array<AttrConfig, MaxVtxAttr>& attrs, u8 stride) {
  for (size_t i = 0; i < s_attrLayoutPool.size(); ++i) {
    const auto& e = s_attrLayoutPool[i];
    if (e.stride == stride && std::memcmp(&e.attrs, &attrs, sizeof(attrs)) == 0) {
      return static_cast<u16>(i);
    }
  }
  const u16 id = static_cast<u16>(s_attrLayoutPool.size());
  s_attrLayoutPool.push_back(AttrLayoutEntry{attrs, stride});
  return id;
}

struct CachedNativeLayout {
  u32 vatGen = UINT32_MAX;
  bool success = false;
  u8 nativeStride = 0;
  u16 attrLayoutId = UINT16_MAX;
  std::array<AttrConfig, MaxVtxAttr> nativeAttrs{};
  std::vector<NativeAttrDesc> descs;
  u32 metaVatGen = UINT32_MAX;
  u32 metaArrayGen = UINT32_MAX;
  u32 metaFrame = UINT32_MAX;
  u64 layoutMetaHash = 0;
};
static std::array<CachedNativeLayout, 8> s_cachedNativeLayouts{};

static bool get_native_layout(GXVtxFmt fmt, const std::vector<NativeAttrDesc>*& outDescs, u8& nativeStride,
                              const std::array<AttrConfig, MaxVtxAttr>*& outNativeAttrs, u16& outAttrLayoutId) {
  auto& c = s_cachedNativeLayouts[static_cast<size_t>(fmt)];
  if (c.vatGen != s_vatGen) {
    c.descs.clear();
    c.success = build_native_layout(fmt, c.descs, c.nativeStride, c.nativeAttrs);
    if (c.success) {
      c.attrLayoutId = intern_attr_layout(c.nativeAttrs, c.nativeStride);
    }
    c.vatGen = s_vatGen;
  }
  if (!c.success) return false;
  outDescs = &c.descs;
  nativeStride = c.nativeStride;
  outNativeAttrs = &c.nativeAttrs;
  outAttrLayoutId = c.attrLayoutId;
  return true;
}

// Expand GX vertices (FIFO records) into flat hardware attributes without per-vertex division or append checks.
struct PreparedNativeAttr {
  NativeAttrDesc desc{};
  const u8* arrayData = nullptr;
  u32 maxIndex = UINT32_MAX;
};

static size_t prepare_native_attrs(const std::vector<NativeAttrDesc>& descs, PreparedNativeAttr* outAttrs) noexcept {
  const size_t n = descs.size();
  for (size_t i = 0; i < n; ++i) {
    const auto& d = descs[i];
    outAttrs[i].desc = d;
    if (d.attrType != GX_DIRECT) {
      const auto& array = g_gxState.arrays[d.attr];
      outAttrs[i].arrayData = static_cast<const u8*>(array.data);
      outAttrs[i].maxIndex =
          (array.size != 0 && d.arrayStride != 0) ? (static_cast<u32>(array.size) / d.arrayStride) : UINT32_MAX;
    }
  }
  return n;
}

static void expand_native_vertices_into(const PreparedNativeAttr* attrs, size_t attrCount, const u8* rec, u32 recStride,
                                        u32 vtxCount, u8* dst) noexcept {
  for (u32 v = 0; v < vtxCount; ++v, rec += recStride) {
    for (size_t i = 0; i < attrCount; ++i) {
      const auto& a = attrs[i];
      const auto& d = a.desc;
      const u8* p;
      if (d.attrType == GX_DIRECT) {
        p = rec + d.srcOffset;
      } else {
        u32 index = d.attrType == GX_INDEX8 ? u32(*(rec + d.srcOffset)) : u32(read_u16(rec + d.srcOffset, true));
        if (index >= a.maxIndex) UNLIKELY {
          index = 0; // guard against malformed indices; valid content never hits this
        }
        p = a.arrayData + static_cast<size_t>(index) * d.arrayStride;
      }
      const bool be = !d.le;
      const auto attr = d.attr;
      const auto compType = static_cast<GXCompType>(d.compType);
      if (attr == GX_VA_PNMTXIDX) {
        const u32 val = u32(*p) / 3u; // /3: GX counts matrix rows; shader indexes postex_mtx[in_pnmtxidx] directly
        std::memcpy(dst, &val, sizeof(val));
        dst += sizeof(val);
      } else if (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX) {
        const u32 val = u32(*p); // raw; shader divides by 3
        std::memcpy(dst, &val, sizeof(val));
        dst += sizeof(val);
      } else if (attr == GX_VA_POS || attr == GX_VA_NRM) {
        for (int c = 0; c < 3; ++c) {
          const f32 val = c < d.cnt ? read_component_as_float(p + c * d.compSize, compType, d.frac, be) : 0.0f;
          std::memcpy(dst, &val, sizeof(val));
          dst += sizeof(val);
        }
      } else if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
        const u32 val = read_color_as_rgba8(p, compType, be);
        std::memcpy(dst, &val, sizeof(val));
        dst += sizeof(val);
      } else { // TEXn
        for (int c = 0; c < 2; ++c) {
          const f32 val = c < d.cnt ? read_component_as_float(p + c * d.compSize, compType, d.frac, be) : 0.0f;
          std::memcpy(dst, &val, sizeof(val));
          dst += sizeof(val);
        }
      }
    }
  }
}

static ByteBuffer s_nativeVtxBuf;
static ByteBuffer s_nativeIdxBuf;
static std::vector<NativeAttrDesc> s_nativeDescs;

// ---------------------------------------------------------------------------
// Native geometry cache (M2)
//
// CPU-expanded native vertex/index data is uploaded once into persistent GPU buffers and
// replayed on later frames by content hash. Because skinning stays on the GPU (bone
// matrices are per-draw uniforms, not baked into the verts), a mesh's expanded bytes are
// byte-identical every frame — so steady-state world and character geometry expands +
// uploads exactly once, and every subsequent frame is just a hash lookup plus a draw that
// references the cached range. This is the fix for the "M1: no cache" per-frame re-expand
// + re-upload cost. Keyed by the source FIFO bytes plus the decode layout (and, for
// indexed attributes, the referenced arrays' contents) so two draws collide in the cache
// only when they truly expand to identical geometry.
static constexpr bool kNativeGeomCacheEnabled = true;
// Byte floor below which caching isn't worth the hash/bookkeeping. NOTE: the 2-frame
// recurrence gate in native_geom_resolve_ranges (not this size gate) is what actually keeps
// transient one-shot draws out of the persistent cache, so this only needs to skip trivially
// tiny draws. Lowered 1024 -> 64 so the shadow pre-pass (16B/vert x 4-21 verts = 64-336B) and
// small static props become cacheable and stop re-uploading into the per-frame vertex ring
// every frame -- that ring was ~440KB/frame, ~38% of the per-frame libmali upload volume that
// is the device CPU bottleneck.
static constexpr size_t kNativeGeomCacheMinBytes = 64;

// Unroll triangle strips into TriangleList indices (cached in native geom cache for static
// geometry) so GLES mobile drivers (Mali-G31) never hit GL_PRIMITIVE_RESTART_FIXED_INDEX
// CPU index-scanning slow paths during glDrawElements.
static constexpr bool kStripTopology = false;

// Emit a periodic one-line cache/batching summary to confirm the geometry cache is
// actually hitting on-device (and how effective run-batching is). Perf-tuning scaffolding;
// off by default in shipped builds.
static constexpr bool kLogNativeGeomCacheStats = true;
static constexpr u32 kNativeGeomStatsLogInterval = 30; // frames between summaries
// Per-window counters (reset each log interval).
static u64 s_geomCacheHits = 0;       // draws served from the persistent cache (no expand/upload)
static u64 s_geomCacheMisses = 0;     // draws that had to expand this frame
static u64 s_geomCachePromotes = 0;   // expansions promoted into the persistent cache
static u64 s_geomCacheRingPushes = 0; // expansions that went to the per-frame ring
static u64 s_geomCacheFull = 0;       // promotions that failed (cache exhausted -> reset)
static u64 s_geomRunsBatched = 0;     // native strip/prim runs emitted (raw-FIFO handle_draw)
static u64 s_geomSegmentsBatched = 0; // draw commands folded into those runs
static u64 s_geomDrawsIndexed = 0;    // draws via GX_AURORA_DRAW_INDEXED (dl::optimize output)
static u64 s_geomDrawsSingle = 0;     // draws via GX_AURORA_DRAW_SIZED (single native)
static u32 s_geomStatsLastFrame = 0;

struct NativeGeomKey {
  u64 fifoHash = 0;
  u64 metaHash = 0;
  u32 srcBytes = 0;
  bool operator==(const NativeGeomKey& o) const noexcept {
    return fifoHash == o.fifoHash && metaHash == o.metaHash && srcBytes == o.srcBytes;
  }
};
struct NativeGeomKeyHash {
  size_t operator()(const NativeGeomKey& k) const noexcept {
    return static_cast<size_t>(k.fifoHash ^ (k.metaHash * 0x9e3779b97f4a7c15ull) ^ k.srcBytes);
  }
};
struct NativeGeomEntry {
  gfx::Range vertRange;
  gfx::Range idxRange;
  u32 indexCount = 0;
  std::vector<u8> cpuVerts;
  std::vector<u16> cpuIndices;
};
static size_t s_nativeGeomCpuMirrorBytes = 0;
static constexpr size_t kMaxNativeGeomCpuMirrorEntryBytes = 64 * 1024;
static constexpr size_t kMaxNativeGeomCpuMirrorTotalBytes = 16 * 1024 * 1024;
static constexpr size_t kMaxNativeGeomCacheEntries = 65536;
static constexpr size_t kMaxNativeGeomMergedCacheEntries = 65536;
static absl::flat_hash_map<NativeGeomKey, NativeGeomEntry, NativeGeomKeyHash> s_nativeGeomCache;
// Keys seen but not yet promoted, mapped to (lastFrame, distinct-frame hit count). Content is
// promoted to the persistent cache only once it recurs across enough frames, which keeps
// single-shot, short-lived, and per-frame-animated geometry from churning the cache.
struct NativeGeomSeenInfo {
  u32 lastFrame = 0;
  u8 hits = 0;
};
static absl::flat_hash_map<NativeGeomKey, NativeGeomSeenInfo, NativeGeomKeyHash> s_nativeGeomSeen;
static u32 s_lastGeomAutoResetFrame = 0;
static u32 s_geomFullSinceReset = 0;
static u32 s_nativeGeomCacheGen = 0;
static u32 s_nativeGeomPtrEpoch = 1;
static u64 s_pnmtxMergedBatches = 0;
static u64 s_pnmtxMergedDrawsIn = 0;
static u64 s_pnmtxMergedCacheHits = 0;

// Emit a cache/batching summary every kNativeGeomStatsLogInterval frames. Called once
// per native draw attempt; the frame-delta guard keeps it to one line per interval.
static void native_geom_cache_maybe_log_stats() {
  if constexpr (!kLogNativeGeomCacheStats) {
    return;
  }
  static const bool s_logTiming = std::getenv("TTYD_TIMING_LOG") != nullptr;
  if (!s_logTiming) {
    return;
  }
  const u32 frame = gfx::current_frame();
  if (frame == UINT32_MAX || frame - s_geomStatsLastFrame < kNativeGeomStatsLogInterval) {
    return;
  }
  const u32 frames = frame - s_geomStatsLastFrame;
  s_geomStatsLastFrame = frame;
  const u64 draws = s_geomCacheHits + s_geomCacheMisses;
  Log.info(
      "[geom-cache] {}f: {} draws, {}/{} hit ({:.1f}%), {} promote, {} ring, {} full; "
      "src {} run/{} indexed/{} sized, batched {}<-{} segs ({:.2f}/run), pnmtx {}<-{} ({} hit); {} entries",
      frames, draws, s_geomCacheHits, draws, draws != 0 ? 100.0 * static_cast<double>(s_geomCacheHits) / draws : 0.0,
      s_geomCachePromotes, s_geomCacheRingPushes, s_geomCacheFull, s_geomRunsBatched, s_geomDrawsIndexed,
      s_geomDrawsSingle, s_geomRunsBatched, s_geomSegmentsBatched,
      s_geomRunsBatched != 0 ? static_cast<double>(s_geomSegmentsBatched) / s_geomRunsBatched : 0.0,
      s_pnmtxMergedBatches, s_pnmtxMergedDrawsIn, s_pnmtxMergedCacheHits, s_nativeGeomCache.size());
  const u64 totalUni = s_uniformCacheHits + s_uniformCacheMisses;
  Log.info(
      "[uniform-cache] {} draws, {}/{} hit ({:.1f}%)",
      totalUni, s_uniformCacheHits, totalUni,
      totalUni != 0 ? 100.0 * static_cast<double>(s_uniformCacheHits) / totalUni : 0.0);
  s_uniformCacheHits = 0;
  s_uniformCacheMisses = 0;
  // Per-frame staging->GPU upload volume (last completed frame). This is what feeds the libmali
  // copy worker: ring verts/indices (cache misses), uniforms (every draw, no dedup), and texture
  // uploads. Cached geometry lives in a persistent buffer and does NOT appear in lastVertSize.
  const double kib = 1.0 / 1024.0;
  const u64 uploadTotal = static_cast<u64>(gfx::g_stats.lastVertSize) + gfx::g_stats.lastIndexSize +
                          gfx::g_stats.lastUniformSize + gfx::g_stats.lastTextureUploadSize +
                          gfx::g_stats.lastStorageSize;
  Log.info("[upload] last-frame: vert {:.1f}K, idx {:.1f}K, uni {:.1f}K, tex {:.1f}K, storage {:.1f}K -> total {:.1f}K",
           gfx::g_stats.lastVertSize * kib, gfx::g_stats.lastIndexSize * kib,
           gfx::g_stats.lastUniformSize * kib, gfx::g_stats.lastTextureUploadSize * kib,
           gfx::g_stats.lastStorageSize * kib, static_cast<double>(uploadTotal) * kib);
  s_geomCacheHits = 0;
  s_geomCacheMisses = 0;
  s_geomCachePromotes = 0;
  s_geomCacheRingPushes = 0;
  s_geomCacheFull = 0;
  s_geomRunsBatched = 0;
  s_geomSegmentsBatched = 0;
  s_geomDrawsIndexed = 0;
  s_geomDrawsSingle = 0;
  s_pnmtxMergedBatches = 0;
  s_pnmtxMergedDrawsIn = 0;
  s_pnmtxMergedCacheHits = 0;
}

static constexpr u32 kStaticDrawExtraBit = 0x80000000u;
static constexpr u8 kStaticDrawIndexed = 1u << 0;
static constexpr u8 kStaticDrawStrip = 1u << 1;

struct StaticDlDraw {
  NativeGeomKey key{};
  const NativeGeomEntry* cachedEntry = nullptr;
  u32 cachedPtrEpoch = 0;
  u16 vtxCount = 0;
  u16 attrLayoutId = 0;
  u8 prim = GX_TRIANGLES;
  u8 fmt = GX_VTXFMT0;
  u8 nativeStride = 0;
  u8 flags = 0;

  bool indexed() const noexcept { return (flags & kStaticDrawIndexed) != 0; }
  bool stripTopology() const noexcept { return (flags & kStaticDrawStrip) != 0; }
  GXPrimitive gxPrim() const noexcept { return static_cast<GXPrimitive>(prim); }
  GXVtxFmt gxFmt() const noexcept { return static_cast<GXVtxFmt>(fmt); }
};

struct DlVtxState {
  u64 vtxDescBits = 0;
  u32 discInvalGen = 0;
  u32 usedFmtMask = 0;
  u16 usedArrayMask = 0;
  std::array<u32, MaxVtxFmt> vtxFmtGens{};
  std::array<u32, 12> arrayGens{};
  std::array<VtxFmt, MaxVtxFmt> vtxFmts{};
  std::array<AttrArray, 12> arrays{};
};
static std::vector<DlVtxState> s_dlVtxStates;

static u16 intern_dl_vtx_state(u64 vtxDescBits, u32 discInvalGen, u32 usedFmtMask, u16 usedArrayMask) {
  for (size_t i = 0; i < s_dlVtxStates.size(); ++i) {
    DlVtxState& st = s_dlVtxStates[i];
    if (st.vtxDescBits != vtxDescBits || st.discInvalGen != discInvalGen ||
        st.usedFmtMask != usedFmtMask || st.usedArrayMask != usedArrayMask) {
      continue;
    }
    bool match = true;
    u32 fm = usedFmtMask;
    while (fm != 0) {
      const u32 f = static_cast<u32>(std::countr_zero(fm));
      fm &= fm - 1u;
      if (std::memcmp(&st.vtxFmts[f], &g_gxState.vtxFmts[f], sizeof(VtxFmt)) != 0) {
        match = false;
        break;
      }
    }
    if (!match) continue;
    u32 am = usedArrayMask;
    while (am != 0) {
      const u32 a = static_cast<u32>(std::countr_zero(am));
      am &= am - 1u;
      if (st.arrays[a] != g_gxState.arrays[GX_VA_POS + a]) {
        match = false;
        break;
      }
    }
    if (match) {
      u32 fmUpd = usedFmtMask;
      while (fmUpd != 0) {
        const u32 f = static_cast<u32>(std::countr_zero(fmUpd));
        fmUpd &= fmUpd - 1u;
        st.vtxFmtGens[f] = s_vtxFmtGen[f];
      }
      u32 amUpd = usedArrayMask;
      while (amUpd != 0) {
        const u32 a = static_cast<u32>(std::countr_zero(amUpd));
        amUpd &= amUpd - 1u;
        st.arrayGens[a] = s_arrayStateGen[a];
      }
      return static_cast<u16>(i);
    }
  }
  if (s_dlVtxStates.size() >= UINT16_MAX) {
    return UINT16_MAX;
  }
  const u16 newId = static_cast<u16>(s_dlVtxStates.size());
  DlVtxState& st = s_dlVtxStates.emplace_back();
  st.vtxDescBits = vtxDescBits;
  st.discInvalGen = discInvalGen;
  st.usedFmtMask = usedFmtMask;
  st.usedArrayMask = usedArrayMask;
  for (u32 f = 0; f < MaxVtxFmt; ++f) {
    if ((usedFmtMask & (1u << f)) != 0) {
      st.vtxFmts[f] = g_gxState.vtxFmts[f];
      st.vtxFmtGens[f] = s_vtxFmtGen[f];
    }
  }
  for (u32 a = 0; a < 12; ++a) {
    if ((usedArrayMask & static_cast<u16>(1u << a)) != 0) {
      st.arrays[a] = g_gxState.arrays[GX_VA_POS + a];
      st.arrayGens[a] = s_arrayStateGen[a];
    }
  }
  return newId;
}

static inline bool verify_dl_vtx_state(u16 vtxStateId) noexcept {
  if (vtxStateId >= s_dlVtxStates.size()) UNLIKELY {
    return false;
  }
  if (s_curVtxStateVerifiedGen == s_vtxStateDirtyGen && s_curVtxStateVerifiedId == vtxStateId) LIKELY {
    return true;
  }
  DlVtxState& st = s_dlVtxStates[vtxStateId];
  if (st.discInvalGen != s_gxDiscArrayInvalidateGen || st.vtxDescBits != current_vtx_desc_bits()) {
    return false;
  }
  u32 fm = st.usedFmtMask;
  while (fm != 0) {
    const u32 f = static_cast<u32>(std::countr_zero(fm));
    fm &= fm - 1u;
    if (st.vtxFmtGens[f] != s_vtxFmtGen[f]) {
      if (std::memcmp(&st.vtxFmts[f], &g_gxState.vtxFmts[f], sizeof(VtxFmt)) != 0) {
        return false;
      }
      st.vtxFmtGens[f] = s_vtxFmtGen[f];
    }
  }
  u32 am = st.usedArrayMask;
  while (am != 0) {
    const u32 a = static_cast<u32>(std::countr_zero(am));
    am &= am - 1u;
    if (st.arrayGens[a] != s_arrayStateGen[a]) {
      if (st.arrays[a] != g_gxState.arrays[GX_VA_POS + a]) {
        return false;
      }
      st.arrayGens[a] = s_arrayStateGen[a];
    }
  }
  s_curVtxStateVerifiedGen = s_vtxStateDirtyGen;
  s_curVtxStateVerifiedId = vtxStateId;
  return true;
}

static constexpr u8 kStaticEntryReady = 1u << 0;
static constexpr u8 kStaticEntryMirror = 1u << 1;
static constexpr u8 kStaticEntryBatch = 1u << 2;
static constexpr u8 kStaticEntryCoalesced = 1u << 3;

struct StaticDlEntry {
  const void* dlData = nullptr;
  const void* lastDlData = nullptr;
  uintptr_t minAddr = 0;
  uintptr_t maxEnd = 0;
  u64 head64 = 0;
  u64 lastHead64 = 0;
  u32 size = 0;
  u32 lastDlSize = 0;
  u32 firstDlSize = 0;
  u32 dlGen = 0;
  u32 cachedAllPtrEpoch = 0;
  u32 extraStart = 0;
  u16 vtxStateId = UINT16_MAX;
  u16 drawCount = 0;
  u8 flags = 0;
  StaticDlDraw draw0{};

  bool ready() const noexcept { return (flags & kStaticEntryReady) != 0; }
  bool allHaveMirror() const noexcept { return (flags & kStaticEntryMirror) != 0; }
  bool isBatch() const noexcept { return (flags & kStaticEntryBatch) != 0; }
  bool coalesced() const noexcept { return (flags & kStaticEntryCoalesced) != 0; }
};

static std::vector<StaticDlEntry> s_staticDlPool;
static std::vector<StaticDlDraw> s_staticDlExtraDraws;
static absl::flat_hash_map<const void*, u32> s_staticDlIndex;
struct StaticDlFastSlot {
  const void* key = nullptr;
  u32 idxPlus1 = 0;
};
struct StaticDlFastSet {
  StaticDlFastSlot ways[2]{};
};
static constexpr size_t kStaticDlFastSets = 4096;
static std::array<StaticDlFastSet, kStaticDlFastSets> s_staticDlFast{};
static std::vector<StaticDlDraw> s_staticDlCapturedDraws;
static u32 s_staticDlCapturedFmtMask = 0;

static inline StaticDlDraw& get_static_dl_draw_mut(u32 refIdx) noexcept {
  if ((refIdx & kStaticDrawExtraBit) == 0) {
    return s_staticDlPool[refIdx].draw0;
  }
  return s_staticDlExtraDraws[refIdx & ~kStaticDrawExtraBit];
}

static inline const StaticDlDraw& get_static_dl_draw(u32 refIdx) noexcept {
  return get_static_dl_draw_mut(refIdx);
}

static inline size_t static_dl_fast_set(const void* dlData) noexcept {
  const uintptr_t p = reinterpret_cast<uintptr_t>(dlData);
  const u64 h = static_cast<u64>(p >> 4u) * 11400714819323198485ull;
  return static_cast<size_t>(h >> 52u) & (kStaticDlFastSets - 1u);
}

static inline void invalidate_static_dl_fast() noexcept {
  s_staticDlFast.fill({});
}

static inline void clear_static_dl_cache() noexcept {
  s_staticDlIndex.clear();
  s_staticDlPool.clear();
  s_staticDlExtraDraws.clear();
  s_dlVtxStates.clear();
  s_curVtxStateVerifiedId = UINT16_MAX;
  invalidate_static_dl_fast();
}

static inline u32 find_static_dl_pool_idx(const void* dlData) noexcept {
  auto& set = s_staticDlFast[static_dl_fast_set(dlData)];
  if (set.ways[0].key == dlData) {
    return set.ways[0].idxPlus1 - 1u;
  }
  if (set.ways[1].key == dlData) {
    return set.ways[1].idxPlus1 - 1u;
  }
  const auto it = s_staticDlIndex.find(dlData);
  if (it == s_staticDlIndex.end()) {
    return UINT32_MAX;
  }
  const u32 poolIdx = it->second;
  set.ways[1] = set.ways[0];
  set.ways[0] = {dlData, poolIdx + 1u};
  return poolIdx;
}

// The persistent cache byte regions are recycled on reset (generation bump), which
// invalidates every cached (key -> range) mapping. Drop both maps when that happens.
static inline void native_geom_sync_generation() {
  const u32 gen = gfx::native_geom_cache_generation();
  if (gen != s_nativeGeomCacheGen) UNLIKELY {
    flush_pending_pnmtx_batch();
    s_nativeGeomCacheGen = gen;
    s_nativeGeomCache.clear();
    s_nativeGeomCache.rehash(0);
    ++s_nativeGeomPtrEpoch;
    s_nativeGeomSeen.clear();
    s_nativeGeomSeen.rehash(0);
    clear_static_dl_cache();
    s_nativeGeomCpuMirrorBytes = 0;
    s_geomFullSinceReset = 0;
    prune_unreferenced_textures();
  }
}

// Content hash of an indexed attribute's source array, memoized per (pointer, frame,
// invalidate gen). Frame-scoped so arrays the game rewrites in place between frames (CPU
// envelope skinning deforms positions into the same buffers) are re-hashed each frame;
// GXInvalidateVtxCache handles rewrites within a frame.
struct ArrayHashMemo {
  u32 frame = UINT32_MAX;
  u32 invalGen = UINT32_MAX;
  u32 size = 0;
  u64 hash = 0;
  bool le = true;
  bool immutable = false;
};
static absl::flat_hash_map<const void*, ArrayHashMemo> s_arrayHashMemo;

static u64 native_array_content_hash(const AttrArray& array) {
  if (s_arrayHashMemo.size() > 1024) {
    s_arrayHashMemo.clear();
  }
  const u32 frame = gfx::current_frame();
  const bool isMutableLe = array.le && !array.immutable;
  const u32 targetGen = isMutableLe ? s_gxArrayInvalidateGen : s_gxDiscArrayInvalidateGen;
  auto [it, inserted] = s_arrayHashMemo.try_emplace(array.data);
  auto& memo = it->second;
  const bool frameStale = isMutableLe && (memo.frame != frame);
  if (inserted || frameStale || memo.size != array.size || memo.le != array.le ||
      memo.immutable != array.immutable || memo.invalGen != targetGen) {
    memo.frame = frame;
    memo.invalGen = targetGen;
    memo.size = array.size;
    memo.le = array.le;
    memo.immutable = array.immutable;
    if (is_timing_enabled()) UNLIKELY {
      ++s_drainTiming.arr_hash_calls;
      s_drainTiming.arr_hash_bytes += array.size;
    }
    if (array.size == 0) {
      memo.hash = xxh3_hash(reinterpret_cast<uintptr_t>(array.data));
    } else {
      memo.hash = xxh3_hash_s(array.data, array.size);
    }
  }
  return memo.hash;
}

// Fast hash for geometry meta (layout + counts) avoiding Hasher heap/state overhead
static inline u64 hash_native_geom_meta(GXPrimitive prim, GXVtxFmt fmt, u32 vtxSize, u32 vtxCount,
                                        const std::vector<NativeAttrDesc>& descs,
                                        const void* extraData = nullptr, size_t extraBytes = 0) {
  struct MetaHdr {
    u32 primFmt;
    u32 vtxSize;
    u32 vtxCount;
    u64 layoutMetaHash;
  };
  u64 layoutMetaHash = 0;
  if (!descs.empty()) {
    auto& c = s_cachedNativeLayouts[static_cast<size_t>(fmt)];
    const u32 curFrame = gfx::current_frame();
    if (c.metaVatGen != s_vatGen || c.metaArrayGen != s_gxArrayBindGen || c.metaFrame != curFrame) {
      u64 lh = XXH3_64bits(descs.data(), descs.size() * sizeof(NativeAttrDesc));
      for (const auto& d : descs) {
        if (d.attrType == GX_INDEX8 || d.attrType == GX_INDEX16) {
          const u64 arrHash = native_array_content_hash(g_gxState.arrays[d.attr]);
          lh = xxh3_hash_s(&arrHash, sizeof(arrHash), lh);
        }
      }
      c.layoutMetaHash = lh;
      c.metaVatGen = s_vatGen;
      c.metaArrayGen = s_gxArrayBindGen;
      c.metaFrame = curFrame;
    }
    layoutMetaHash = c.layoutMetaHash;
  }
  const MetaHdr hdr{
    .primFmt = (static_cast<u32>(prim) << 16) | static_cast<u32>(fmt),
    .vtxSize = vtxSize,
    .vtxCount = vtxCount,
    .layoutMetaHash = layoutMetaHash,
  };
  u64 meta = XXH3_64bits(&hdr, sizeof(hdr));
  if (extraData != nullptr && extraBytes > 0) {
    meta = XXH3_64bits_withSeed(extraData, extraBytes, meta);
  }
  return meta;
}

// Fold the decode-relevant layout into the meta hasher: which attrs are present, their
// formats, and for indexed attrs the referenced array's per-frame content hash. Two draws
// with equal FIFO bytes and equal meta hash expand to identical native vertices.
static void native_geom_hash_layout(Hasher& h, const std::vector<NativeAttrDesc>& descs) {
  for (const auto& d : descs) {
    h.update(static_cast<u8>(d.attr));
    h.update(d.attrType);
    h.update(d.compType);
    h.update(d.cnt);
    h.update(d.compSize);
    h.update(d.frac);
    h.update(d.arrayStride);
    h.update(static_cast<u8>(d.le ? 1 : 0));
    if (d.attrType == GX_INDEX8 || d.attrType == GX_INDEX16) {
      h.update(native_array_content_hash(g_gxState.arrays[d.attr]));
    }
  }
}

static const NativeGeomEntry* push_and_cache_native_geom(const NativeGeomKey& key, const u8* vtxPtr, size_t vtxBytes,
                                                          u8 vtxStride, const u8* idxData, size_t idxBytes,
                                                          u32 numIndices, bool mirrorCpu, size_t maxMirrorEntryBytes,
                                                          gfx::Range& outVertRange, gfx::Range& outIdxRange) {
  if (const auto existIt = s_nativeGeomCache.find(key); existIt != s_nativeGeomCache.end()) UNLIKELY {
    auto& existing = existIt->second;
    if (mirrorCpu && existing.cpuVerts.empty() && vtxPtr != nullptr && vtxBytes > 0) {
      const size_t entryBytes = vtxBytes + idxBytes;
      if (entryBytes <= maxMirrorEntryBytes &&
          s_nativeGeomCpuMirrorBytes + entryBytes <= kMaxNativeGeomCpuMirrorTotalBytes) {
        existing.cpuVerts.assign(vtxPtr, vtxPtr + vtxBytes);
        if (idxData != nullptr && idxBytes > 0) {
          const size_t idxCount = idxBytes / sizeof(u16);
          existing.cpuIndices.resize(idxCount);
          std::memcpy(existing.cpuIndices.data(), idxData, idxBytes);
        }
        s_nativeGeomCpuMirrorBytes += entryBytes;
      }
    }
    outVertRange = existing.vertRange;
    outIdxRange = existing.idxRange;
    return &existing;
  }
  const size_t vertAlign = vtxStride > 4 ? static_cast<size_t>(vtxStride) : 4u;
  if (!gfx::can_push_native_cached(vtxBytes, idxBytes, vertAlign)) {
    return nullptr;
  }
  auto [cv, vok] = gfx::push_native_cached_verts(vtxPtr, vtxBytes, vertAlign);
  gfx::Range ci{};
  bool iok = true;
  if (idxData != nullptr && idxBytes > 0) {
    const auto pushed = gfx::push_native_cached_indices(idxData, idxBytes);
    ci = pushed.first;
    iok = pushed.second;
  }
  if (!vok || !iok) {
    return nullptr;
  }
  outVertRange = cv;
  outIdxRange = ci;
  NativeGeomEntry entry{.vertRange = cv, .idxRange = ci, .indexCount = numIndices};
  if (mirrorCpu) {
    const size_t entryBytes = vtxBytes + idxBytes;
    if (entryBytes <= maxMirrorEntryBytes &&
        s_nativeGeomCpuMirrorBytes + entryBytes <= kMaxNativeGeomCpuMirrorTotalBytes) {
      entry.cpuVerts.assign(vtxPtr, vtxPtr + vtxBytes);
      if (idxData != nullptr && idxBytes > 0) {
        const size_t idxCount = idxBytes / sizeof(u16);
        entry.cpuIndices.resize(idxCount);
        std::memcpy(entry.cpuIndices.data(), idxData, idxBytes);
      }
      s_nativeGeomCpuMirrorBytes += entryBytes;
    }
  }
  auto [insIt, ok] = s_nativeGeomCache.emplace(key, std::move(entry));
  ++s_nativeGeomPtrEpoch;
  return &insIt->second;
}

static bool try_promote_native_geom(const NativeGeomKey& key, const u8* vtxPtr, size_t vtxBytes, u8 vtxStride,
                                    const u8* idxData, size_t idxBytes, u32 numIndices, gfx::Range& vertRange,
                                    gfx::Range& idxRange, bool mirrorCpu,
                                    const NativeGeomEntry** outCachedEntry = nullptr, u8 minRecurFrames = 4u) {
  if (kNativeGeomCacheEnabled &&
      (vtxBytes >= kNativeGeomCacheMinBytes || (s_staticDlCaptureActive && vtxBytes > 0))) {
    const u32 frame = gfx::current_frame();
    const auto maybeRequestAutoReset = [frame]() {
      ++s_geomCacheFull;
      ++s_geomFullSinceReset;
      if (s_geomFullSinceReset >= 32u &&
          (s_lastGeomAutoResetFrame == 0 ||
           (frame != UINT32_MAX && frame >= s_lastGeomAutoResetFrame && frame - s_lastGeomAutoResetFrame >= 300u))) {
        gfx::request_native_geometry_cache_reset();
        s_lastGeomAutoResetFrame = frame;
        s_geomFullSinceReset = 0;
      }
    };
    if (s_nativeGeomCache.size() >= kMaxNativeGeomCacheEntries) {
      maybeRequestAutoReset();
      return false;
    }
    auto [seenIt, inserted] = s_nativeGeomSeen.emplace(key, NativeGeomSeenInfo{frame, 1u});
    if (!inserted && seenIt->second.lastFrame != frame) {
      if (frame != UINT32_MAX && frame > seenIt->second.lastFrame && frame - seenIt->second.lastFrame > 60u) {
        seenIt->second.hits = 1u;
      } else if (seenIt->second.hits < 255u) {
        ++seenIt->second.hits;
      }
      seenIt->second.lastFrame = frame;
      if (seenIt->second.hits >= minRecurFrames) {
        if (const NativeGeomEntry* cached =
                push_and_cache_native_geom(key, vtxPtr, vtxBytes, vtxStride, idxData, idxBytes, numIndices, mirrorCpu,
                                           kMaxNativeGeomCpuMirrorEntryBytes, vertRange, idxRange)) {
          if (outCachedEntry != nullptr) {
            *outCachedEntry = cached;
          }
          s_nativeGeomSeen.erase(seenIt);
          ++s_geomCachePromotes;
          return true;
        }
        maybeRequestAutoReset();
      }
    } else if (!inserted) {
      seenIt->second.lastFrame = frame;
    }
    if (s_nativeGeomSeen.size() > 32768u) {
      s_nativeGeomSeen.clear();
    }
  }
  return false;
}

// Resolve the GPU ranges for a freshly expanded batch. On the first cross-frame recurrence
// of identical content the batch is promoted into the persistent cache; otherwise it goes
// to the per-frame ring. `cached` reports which buffers the returned ranges address.
// `idxData`/`idxBytes` may be null/0 for non-indexed (triangle-list) draws.
static void native_geom_resolve_ranges(const NativeGeomKey& key, const u8* vtxPtr, size_t vtxBytes, u8 vtxStride,
                                       const u8* idxData, size_t idxBytes, u32 numIndices, gfx::Range& vertRange,
                                       gfx::Range& idxRange, bool& cached, bool mirrorCpu = true,
                                       u8 minRecurFrames = 4u) {
  if (try_promote_native_geom(key, vtxPtr, vtxBytes, vtxStride, idxData, idxBytes, numIndices, vertRange, idxRange,
                              mirrorCpu, nullptr, minRecurFrames)) {
    cached = true;
    return;
  }
  cached = false;
  const size_t vertAlign = vtxStride > 4 ? static_cast<size_t>(vtxStride) : 4u;
  // Per-frame ring fallback.
  ++s_geomCacheRingPushes;
  vertRange = gfx::push_verts(vtxPtr, vtxBytes, vertAlign);
  if (idxData != nullptr && idxBytes > 0) {
    idxRange = gfx::push_indices(idxData, idxBytes, 4);
  }
}

namespace {
static constexpr bool kEnablePnMtxBatch = true;

struct NativePipeMemoEntry {
  uint64_t cfgHash = 0;
  uint32_t gen = 0;
  PipelineConfig config{};
  gfx::PipelineRef pipeline{};
  std::array<gfx::PipelineRef, 4> mergedPipelines{};
  std::array<gfx::PipelineRef, 4> mergedListPipelines{};
  uint64_t shaderHash = 0;
  bool valid = false;
  u8 hasMergedMask = 0;
  u8 hasMergedListMask = 0;
};
static constexpr size_t kNativePipeMemoSize = 512;
static std::array<NativePipeMemoEntry, kNativePipeMemoSize> s_nativePipeMemo{};

static gfx::PipelineRef resolve_pipeline_cached(const PipelineConfig& cfg, bool& outReady,
                                                uint64_t& outShaderHash, size_t* outMemoIdx = nullptr,
                                                uint64_t* outCfgHash = nullptr) noexcept {
  const uint32_t pipeGen = gfx::pipeline_cache_gen();
  const uint64_t cfgHash = xxh3_hash(cfg);
  if (outCfgHash != nullptr) *outCfgHash = cfgHash;
  const size_t slot = (cfgHash ^ (cfgHash >> 16)) & (kNativePipeMemoSize - 1);
  auto& pipeMemo = s_nativePipeMemo[slot];
  if (pipeMemo.valid && pipeMemo.gen == pipeGen && pipeMemo.cfgHash == cfgHash &&
      std::memcmp(&pipeMemo.config, &cfg, sizeof(PipelineConfig)) == 0) {
    outReady = true;
    outShaderHash = pipeMemo.shaderHash;
    if (outMemoIdx != nullptr) *outMemoIdx = slot;
    return pipeMemo.pipeline;
  }
  const auto tr0 = timing_now();
  gfx::PipelineRef pipeline = gfx::pipeline_ref_status(cfg, outReady);
  const auto tr1 = timing_now();
  s_drainTiming.ms_pipe_ref += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
  if (outReady) {
    outShaderHash = XXH3_64bits(&cfg.shaderConfig, sizeof(ShaderConfig));
    pipeMemo.cfgHash = cfgHash;
    pipeMemo.gen = pipeGen;
    pipeMemo.config = cfg;
    pipeMemo.pipeline = pipeline;
    pipeMemo.mergedPipelines = {};
    pipeMemo.mergedListPipelines = {};
    pipeMemo.shaderHash = outShaderHash;
    pipeMemo.valid = true;
    pipeMemo.hasMergedMask = 0;
    pipeMemo.hasMergedListMask = 0;
    if (outMemoIdx != nullptr) *outMemoIdx = slot;
  } else {
    outShaderHash = 0;
    if (outMemoIdx != nullptr) *outMemoIdx = SIZE_MAX;
  }
  return pipeline;
}

static bool shader_uses_fixed_pnmtx_texgen(const ShaderInfo& info, const ShaderConfig& cfg) noexcept {
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = cfg.tcgs[i];
    if (tcg.type == GX_TG_MTX2x4 || tcg.type == GX_TG_MTX3x4) {
      if (info.indexAttr.test(GX_VA_TEX0MTXIDX + i)) {
        return true;
      }
      if (tcg.mtx != GX_IDENTITY && tcg.mtx < GX_TEXMTX0) {
        return true;
      }
    }
  }
  return false;
}

static uint32_t s_lastPipelineDirty = UINT32_MAX;
static GXPrimitive s_lastPrim = static_cast<GXPrimitive>(UINT32_MAX);
static GXVtxFmt s_lastFmt = static_cast<GXVtxFmt>(UINT32_MAX);
static bool s_lastStripTopology = false;
static uint8_t s_lastNativeStride = 0;
static u32 s_lastUniformFrame = UINT32_MAX;
static bool s_forceTexResolveAfterBatch = false;
static PipelineConfig s_currentConfig{};

static void resolve_single_native_draw_state(GXPrimitive prim, GXVtxFmt fmt,
                                             const std::array<AttrConfig, MaxVtxAttr>& nativeAttrs, u8 nativeStride,
                                             bool stripTopology, bool preCfgPopulated, bool preConfigSame,
                                             bool preBgPopulated) noexcept {
  const bool forceTexResolve = s_forceTexResolveAfterBatch;
  s_forceTexResolveAfterBatch = false;
  const u32 curFrame = gfx::current_frame();
  const auto resolveBindGroupsTimed = [&]() noexcept -> GXBindGroups {
    const auto t_res0 = timing_now();
    const GXBindGroups resolved = resolve_and_build_bind_groups(s_lastInfo);
    const auto t_res1 = timing_now();
    s_drainTiming.ms_tex_resolve += std::chrono::duration<double, std::milli>(t_res1 - t_res0).count();
    ++s_drainTiming.bg_calls;
    return resolved;
  };

  const bool samePipeline =
      preConfigSame ||
      (!preCfgPopulated && s_hasLastDrawState &&
       (s_lastPipelineDirty == g_gxState.pipelineDirty) &&
       (s_lastPrim == prim ||
        (s_lastConfig.shaderConfig.lineMode == 0 &&
         (prim == GX_TRIANGLES || prim == GX_TRIANGLESTRIP) &&
         (s_lastPrim == GX_TRIANGLES || s_lastPrim == GX_TRIANGLESTRIP))) &&
       (s_lastFmt == fmt) &&
       (s_lastStripTopology == stripTopology || s_lastConfig.shaderConfig.lineMode == 0) &&
       (s_lastNativeStride == nativeStride));
  const PipelineConfig* configPtr = nullptr;
  if (samePipeline) {
    configPtr = &s_lastConfig;
  } else if (preCfgPopulated) {
    s_lastPipelineDirty = g_gxState.pipelineDirty;
    s_lastPrim = prim;
    s_lastFmt = fmt;
    s_lastStripTopology = stripTopology;
    s_lastNativeStride = nativeStride;
    configPtr = &s_currentConfig;
  } else {
    const auto tp0 = timing_now();
    s_currentConfig = {};
    populate_pipeline_config(s_currentConfig, prim, fmt, /*fillAttrs=*/false);
    s_currentConfig.shaderConfig.attrs = nativeAttrs;
    s_currentConfig.shaderConfig.vtxStride = nativeStride;
    s_currentConfig.shaderConfig.nativeVertexFetch = 1;
    s_currentConfig.triangleStripTopology = stripTopology ? 1u : 0u;
    const auto tp1 = timing_now();
    s_drainTiming.ms_pipe_cfg += std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    s_lastPipelineDirty = g_gxState.pipelineDirty;
    s_lastPrim = prim;
    s_lastFmt = fmt;
    s_lastStripTopology = stripTopology;
    s_lastNativeStride = nativeStride;
    configPtr = &s_currentConfig;
  }

  const bool sameConfig =
      samePipeline ||
      (!preCfgPopulated && s_hasLastDrawState && std::memcmp(configPtr, &s_lastConfig, sizeof(PipelineConfig)) == 0);
  if (sameConfig) {
    const bool texChanged = preBgPopulated || forceTexResolve || (s_lastTextureDirty != g_gxState.textureDirty);
    if (texChanged) {
      s_lastBindGroups = preBgPopulated ? s_candidateBg : resolveBindGroupsTimed();
      s_lastTextureDirty = g_gxState.textureDirty;
    }
    const bool needUniformRebuild = texChanged || s_lastUniformFrame != curFrame ||
                                   s_lastUniformDirty != g_gxState.uniformDirty || s_lastUniformRange.size == 0;
    if (!needUniformRebuild) {
      ++s_uniformCacheHits;
    } else {
      BindGroupRanges ranges{};
      const auto tu0 = timing_now();
      s_lastUniformRange = build_uniform(s_lastInfo, 0, ranges);
      const auto tu1 = timing_now();
      s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
      s_lastUniformDirty = g_gxState.uniformDirty;
      s_lastUniformFrame = curFrame;
      ++s_uniformCacheMisses;
    }
    return;
  }

  uint64_t cachedShaderHash = 0;
  bool pipeReady = false;
  size_t memoIdx = SIZE_MAX;
  uint64_t cfgHash = 0;
  s_lastPipeline = resolve_pipeline_cached(*configPtr, pipeReady, cachedShaderHash, &memoIdx, &cfgHash);
  s_lastPipeMemoIdx = memoIdx;
  s_lastCfgHash = cfgHash;
  s_lastConfig = *configPtr;
  s_hasLastDrawState = true;

  const bool sameShader = std::memcmp(&s_lastConfig.shaderConfig, &s_lastShaderConfig, sizeof(ShaderConfig)) == 0;
  if (sameShader) {
    const bool texChanged = forceTexResolve || (s_lastTextureDirty != g_gxState.textureDirty);
    if (texChanged) {
      s_lastBindGroups = resolveBindGroupsTimed();
      s_lastTextureDirty = g_gxState.textureDirty;
    }
    const bool needUniformRebuild = texChanged || s_lastUniformFrame != curFrame ||
                                   s_lastUniformDirty != g_gxState.uniformDirty || s_lastUniformRange.size == 0;
    if (!needUniformRebuild) {
      ++s_uniformCacheHits;
    } else {
      BindGroupRanges ranges{};
      const auto tu0 = timing_now();
      s_lastUniformRange = build_uniform(s_lastInfo, 0, ranges);
      const auto tu1 = timing_now();
      s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
      s_lastUniformDirty = g_gxState.uniformDirty;
      s_lastUniformFrame = curFrame;
      ++s_uniformCacheMisses;
    }
  } else {
    const uint64_t shaderHash =
        cachedShaderHash != 0 ? cachedShaderHash : XXH3_64bits(&s_lastConfig.shaderConfig, sizeof(ShaderConfig));
    const size_t slot = (shaderHash ^ (shaderHash >> 16)) & (kShaderDrawCacheSize - 1);
    auto& entry = s_shaderDrawCache[slot];

    if (entry.shaderHash == shaderHash) {
      s_lastInfo = entry.info;
      const bool texChanged = forceTexResolve || (entry.textureDirty != g_gxState.textureDirty);
      if (!texChanged) {
        s_lastBindGroups = entry.bindGroups;
      } else {
        s_lastBindGroups = resolveBindGroupsTimed();
        entry.bindGroups = s_lastBindGroups;
        entry.textureDirty = g_gxState.textureDirty;
      }
      const bool needUniformRebuild = texChanged || entry.frame != curFrame ||
                                     entry.uniformDirty != g_gxState.uniformDirty || entry.uniformRange.size == 0;
      if (!needUniformRebuild) {
        s_lastUniformRange = entry.uniformRange;
        ++s_uniformCacheHits;
      } else {
        BindGroupRanges ranges{};
        const auto tu0 = timing_now();
        s_lastUniformRange = build_uniform(s_lastInfo, 0, ranges);
        const auto tu1 = timing_now();
        s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
        entry.uniformRange = s_lastUniformRange;
        entry.uniformDirty = g_gxState.uniformDirty;
        entry.textureDirty = g_gxState.textureDirty;
        entry.frame = curFrame;
        ++s_uniformCacheMisses;
      }
    } else {
      s_lastInfo = build_shader_info(s_lastConfig.shaderConfig);
      s_lastBindGroups = resolveBindGroupsTimed();
      BindGroupRanges ranges{};
      const auto tu0 = timing_now();
      s_lastUniformRange = build_uniform(s_lastInfo, 0, ranges);
      const auto tu1 = timing_now();
      s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();

      entry.shaderHash = shaderHash;
      entry.info = s_lastInfo;
      entry.bindGroups = s_lastBindGroups;
      entry.uniformRange = s_lastUniformRange;
      entry.textureDirty = g_gxState.textureDirty;
      entry.uniformDirty = g_gxState.uniformDirty;
      entry.frame = curFrame;
      ++s_uniformCacheMisses;
    }

    s_lastShaderConfig = s_lastConfig.shaderConfig;
    s_lastTextureDirty = g_gxState.textureDirty;
    s_lastUniformDirty = g_gxState.uniformDirty;
    s_lastUniformFrame = curFrame;
  }
}

static void emit_native_draw_command(gfx::PipelineRef pipeline, gfx::Range vertRange, gfx::Range idxRange,
                                     gfx::Range uniformRange, u32 vtxCount, u32 numIndices,
                                     const GXBindGroups& bindGroups, u32 dstAlpha, bool cachedGeometry,
                                     uint32_t uniDirty, u32 texMtxGen, u32 lightsGen, u32 bpUniGen,
                                     u32 othUniGen, u8 lineWidth = 6) noexcept {
  if (is_timing_enabled()) UNLIKELY {
    if (!s_splitHasPrev) {
      s_splitHasPrev = true;
    } else {
      ++s_drainTiming.split_pairs;
      const bool sPipe = (pipeline == s_splitPrevPipe);
      const bool sTex = (bindGroups.textureBindGroup == s_splitPrevTex);
      const bool sDa = (dstAlpha == s_splitPrevDstAlpha);
      const bool cTm = (texMtxGen != s_splitPrevTexMtxGen);
      const bool cLi = (lightsGen != s_splitPrevLightsGen);
      const bool cBp = (bpUniGen != s_splitPrevBpUniGen);
      const bool cOth = (othUniGen != s_splitPrevOthUniGen);
      const bool sameNonPnUni = sDa && !cTm && !cLi && !cBp && !cOth;
      if (sPipe) ++s_drainTiming.split_same_pipe;
      if (sTex) ++s_drainTiming.split_same_tex;
      if (sPipe && sTex) {
        ++s_drainTiming.split_same_both;
        if (sameNonPnUni) {
          if (uniDirty == s_splitPrevUniformDirty) ++s_drainTiming.split_same_all;
          else ++s_drainTiming.split_only_pnmtx;
        }
      } else if (sPipe && !sTex && sameNonPnUni) {
        ++s_drainTiming.split_only_tex;
      } else if (!sPipe && sTex && sameNonPnUni) {
        ++s_drainTiming.split_only_pipe;
      }
      if (!sPipe) {
        const auto& a = s_splitPrevConfig;
        const auto& bcfg = s_lastConfig;
        const bool dCull = (a.cullMode != bcfg.cullMode);
        const bool dZ = (a.depthCompare != bcfg.depthCompare || a.depthUpdate != bcfg.depthUpdate || a.depthFunc != bcfg.depthFunc);
        const bool dBlend = (a.blendMode != bcfg.blendMode || a.blendFacSrc != bcfg.blendFacSrc || a.blendFacDst != bcfg.blendFacDst ||
                             a.blendOp != bcfg.blendOp || a.alphaUpdate != bcfg.alphaUpdate || a.colorUpdate != bcfg.colorUpdate ||
                             a.dstAlpha != bcfg.dstAlpha);
        const bool dTopo = (a.triangleStripTopology != bcfg.triangleStripTopology || a.shaderConfig.lineMode != bcfg.shaderConfig.lineMode);
        const bool dFog = (a.shaderConfig.fogType != bcfg.shaderConfig.fogType);
        const bool dAttrs = (a.shaderConfig.vtxStride != bcfg.shaderConfig.vtxStride ||
                             std::memcmp(&a.shaderConfig.attrs, &bcfg.shaderConfig.attrs, sizeof(a.shaderConfig.attrs)) != 0);
        const bool dTev = (a.shaderConfig.tevStageCount != bcfg.shaderConfig.tevStageCount ||
                           a.shaderConfig.numIndStages != bcfg.shaderConfig.numIndStages ||
                           std::memcmp(&a.shaderConfig.tevSwapTable, &bcfg.shaderConfig.tevSwapTable, sizeof(a.shaderConfig.tevSwapTable)) != 0 ||
                           std::memcmp(&a.shaderConfig.tevStages, &bcfg.shaderConfig.tevStages, sizeof(a.shaderConfig.tevStages)) != 0 ||
                           std::memcmp(&a.shaderConfig.indStages, &bcfg.shaderConfig.indStages, sizeof(a.shaderConfig.indStages)) != 0);
        const bool dChan = (std::memcmp(&a.shaderConfig.colorChannels, &bcfg.shaderConfig.colorChannels, sizeof(a.shaderConfig.colorChannels)) != 0);
        const bool dTcg = (std::memcmp(&a.shaderConfig.tcgs, &bcfg.shaderConfig.tcgs, sizeof(a.shaderConfig.tcgs)) != 0);
        const bool dAcmp = (std::memcmp(&a.shaderConfig.alphaCompare, &bcfg.shaderConfig.alphaCompare, sizeof(a.shaderConfig.alphaCompare)) != 0);
        const bool dPaper = (a.shaderConfig.edgeAA != bcfg.shaderConfig.edgeAA ||
                             a.shaderConfig.shadowFill != bcfg.shaderConfig.shadowFill ||
                             a.shaderConfig.sdfMaps != bcfg.shaderConfig.sdfMaps);
        if (dCull) ++s_drainTiming.pd_cull;
        if (dZ) ++s_drainTiming.pd_z;
        if (dBlend) ++s_drainTiming.pd_blend;
        if (dTopo) ++s_drainTiming.pd_topo;
        if (dFog) ++s_drainTiming.pd_fog;
        if (dAttrs) ++s_drainTiming.pd_attrs;
        if (dTev) ++s_drainTiming.pd_tev;
        if (dChan) ++s_drainTiming.pd_chan;
        if (dTcg) ++s_drainTiming.pd_tcg;
        if (dAcmp) ++s_drainTiming.pd_acmp;
        if (dPaper) ++s_drainTiming.pd_paper;
        if (sTex && sameNonPnUni) {
          if (dCull) ++s_drainTiming.op_cull;
          if (dZ) ++s_drainTiming.op_z;
          if (dBlend) ++s_drainTiming.op_blend;
          if (dTopo) ++s_drainTiming.op_topo;
          if (dFog) ++s_drainTiming.op_fog;
          if (dAttrs) ++s_drainTiming.op_attrs;
          if (dTev) ++s_drainTiming.op_tev;
          if (dChan) ++s_drainTiming.op_chan;
          if (dTcg) ++s_drainTiming.op_tcg;
          if (dAcmp) ++s_drainTiming.op_acmp;
          if (dPaper) ++s_drainTiming.op_paper;
        }
      }
      if (cTm) ++s_drainTiming.split_chg_texmtx;
      if (cLi) ++s_drainTiming.split_chg_lights;
      if (cBp) {
        ++s_drainTiming.split_chg_bp_uni;
        if (s_splitBpFogGen != s_splitPrevBpFogGen) ++s_drainTiming.bp_fog;
        if (s_splitBpTevColGen != s_splitPrevBpTevColGen) ++s_drainTiming.bp_tcol;
        if (s_splitBpSuGen != s_splitPrevBpSuGen) ++s_drainTiming.bp_su;
        if (s_splitBpIndGen != s_splitPrevBpIndGen) ++s_drainTiming.bp_ind;
        if (s_splitBpLineGen != s_splitPrevBpLineGen) ++s_drainTiming.bp_line;
      }
      if (cOth) {
        ++s_drainTiming.split_chg_oth_uni;
        if (s_splitOthAmbGen != s_splitPrevOthAmbGen) ++s_drainTiming.oth_amb;
        if (s_splitOthMatGen != s_splitPrevOthMatGen) ++s_drainTiming.oth_mat;
        if (s_splitOthVpGen != s_splitPrevOthVpGen) ++s_drainTiming.oth_vp;
        if (s_splitOthProjGen != s_splitPrevOthProjGen) ++s_drainTiming.oth_proj;
        if (s_splitOthTexDimGen != s_splitPrevOthTexDimGen) ++s_drainTiming.oth_tdim;
        if (s_splitOthOffGen != s_splitPrevOthOffGen) ++s_drainTiming.oth_off;
      }
    }
    s_splitPrevPipe = pipeline;
    s_splitPrevTex = bindGroups.textureBindGroup;
    s_splitPrevDstAlpha = dstAlpha;
    s_splitPrevUniformDirty = uniDirty;
    s_splitPrevTexMtxGen = texMtxGen;
    s_splitPrevLightsGen = lightsGen;
    s_splitPrevBpUniGen = bpUniGen;
    s_splitPrevOthUniGen = othUniGen;
    s_splitPrevOthAmbGen = s_splitOthAmbGen;
    s_splitPrevOthMatGen = s_splitOthMatGen;
    s_splitPrevOthVpGen = s_splitOthVpGen;
    s_splitPrevOthProjGen = s_splitOthProjGen;
    s_splitPrevOthTexDimGen = s_splitOthTexDimGen;
    s_splitPrevOthOffGen = s_splitOthOffGen;
    s_splitPrevBpFogGen = s_splitBpFogGen;
    s_splitPrevBpTevColGen = s_splitBpTevColGen;
    s_splitPrevBpSuGen = s_splitBpSuGen;
    s_splitPrevBpIndGen = s_splitBpIndGen;
    s_splitPrevBpLineGen = s_splitBpLineGen;
    s_splitPrevConfig = s_lastConfig;
  }

  const auto tc0 = timing_now();
  gfx::push_draw_command(DrawData{
      .pipeline = pipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = uniformRange,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = 1,
      .bindGroups = bindGroups,
      .dstAlpha = dstAlpha,
      .nativeVertexFetch = true,
      .cachedGeometry = cachedGeometry,
      .lineWidth = lineWidth,
  });
  const auto tc1 = timing_now();
  s_drainTiming.ms_push_cmd += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
}

static constexpr u8 kItemStaticPool = 1u << 0;
static constexpr u8 kItemCached = 1u << 1;
static constexpr u8 kItemIndexed = 1u << 2;
static constexpr u8 kItemStripTopology = 1u << 3;
static constexpr u8 kItemPromoTried = 1u << 4;

struct PendingPnMtxItem {
  u32 refIdx = 0; // StaticDlDraw ref (with kStaticDrawExtraBit) if kItemStaticPool, else b.extras index
  u16 vtxCount = 0;
  u8 slotIdx = 0;
  u8 texSlot = 0;
  u8 flags = 0;
};
static_assert(sizeof(PendingPnMtxItem) == 12);

struct PendingPnMtxExtra {
  NativeGeomKey key{};
  const NativeGeomEntry* cachedEntry = nullptr;
  u32 cachedPtrEpoch = 0;
  gfx::Range vertRange{};
  gfx::Range idxRange{};
  u32 numIndices = 0;
  u32 scratchVtxOff = 0;
  u32 scratchVtxBytes = 0;
  u32 scratchIdxOff = 0;
  u32 scratchIdxCount = 0;
};

struct PendingPnMtxBatch {
  bool active = false;
  GXPrimitive prim = GX_TRIANGLES;
  GXVtxFmt fmt = GX_VTXFMT0;
  bool stripTopology = false;
  bool mixedTopology = false;
  std::array<AttrConfig, MaxVtxAttr> nativeAttrs{};
  u16 attrLayoutId = UINT16_MAX;
  u8 nativeStride = 0;
  uint32_t pipelineDirty = 0;
  uint32_t textureDirty = 0;
  u32 dstAlpha = 0;
  u32 texMtxGen = 0;
  u32 lightsGen = 0;
  u32 bpUniGen = 0;
  u32 othUniGen = 0;
  u64 uniFastGen = UINT64_MAX;
  BatchUniState uni{};
  uint32_t finalUniformDirty = 0;

  std::array<Mat3x4<float>, MaxPnMtx> posMtxs{};
  std::array<Mat3x4<float>, MaxPnMtx> nrmMtxs{};
  std::array<u32, MaxPnMtx> slotMtxGens{};
  u32 mtxCount = 0;
  u32 lastMtxIdx = UINT32_MAX;
  u32 lastMtxGen = 0;
  u32 lastMtxSlot = UINT32_MAX;
  bool allowMultiTex = false;
  u32 lastTexSlot = 0;
  u32 texSlotCount = 1;
  std::array<gfx::BindGroupRef, MaxTextures> texSlotBindGroups{};
  std::array<f32, MaxTextures> texSlotBiases{};
  static constexpr u64 kBatchFhSeed = 0x9e3779b97f4a7c15ull;
  static constexpr u64 kBatchMhSeed = 0x504e4d54584d5247ull;

  u32 totalVtxCount = 0;
  bool allCached = true;
  u64 batchFh = kBatchFhSeed;
  u64 batchMh = kBatchMhSeed;

  std::vector<PendingPnMtxItem> items;
  std::vector<PendingPnMtxExtra> extras;
  std::vector<u8> scratchVerts;
  std::vector<u16> scratchIdxs;
};
static PendingPnMtxBatch s_pendingPnMtxBatch;

static inline u64 batch_rotl64(u64 x, unsigned r) noexcept {
  return (x << r) | (x >> (64u - r));
}

static inline u64 batch_mix64(u64 x) noexcept {
  x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31u);
}

static inline void batch_hash_append_cached(u64& fh, u64& mh, const NativeGeomKey& key, u32 slotIdx,
                                            u32 texSlot, bool stripTopology) noexcept {
  const u32 slotTag = (slotIdx & 0xFFu) | ((texSlot & 0xFFu) << 8u) | (stripTopology ? 0x10000u : 0u);
  const u64 m1 = batch_mix64(key.fifoHash ^ batch_rotl64(key.metaHash, 19u) ^
                             (static_cast<u64>(key.srcBytes) << 32u) ^ slotTag);
  const u64 m2 = batch_mix64(key.metaHash ^ (static_cast<u64>(slotTag) * 0xff51afd7ed558ccdull) ^
                             (static_cast<u64>(key.srcBytes) * 0xc4ceb9fe1a85ec53ull));
  fh = batch_rotl64(fh, 13u) * 0x9e3779b97f4a7c15ull + m1;
  mh = (batch_rotl64(mh, 17u) * 0xbf58476d1ce4e5b9ull) ^ m2;
}
} // namespace

static u32 append_unrolled_strip_indices(ByteBuffer& dst, const u16* srcIdxs, u32 srcIdxCount, u16 vtxCount,
                                         u16 vtxBase) noexcept {
  u32 emitted = 0;
  const u32 n = srcIdxs != nullptr ? srcIdxCount : static_cast<u32>(vtxCount);
  if (n < 3) {
    return 0;
  }
  dst.reserve_extra(static_cast<size_t>(n - 2) * 3u * sizeof(u16));
  int runLen = 0;
  u16 prev2 = 0;
  u16 prev1 = 0;
  bool flip = false;
  for (u32 i = 0; i < n; ++i) {
    const u16 idx = srcIdxs != nullptr ? srcIdxs[i] : static_cast<u16>(i);
    if (idx == 0xFFFFu) {
      runLen = 0;
      flip = false;
      continue;
    }
    if (runLen == 0) {
      prev2 = idx;
      runLen = 1;
    } else if (runLen == 1) {
      prev1 = idx;
      runLen = 2;
    } else {
      const u16 a = flip ? prev1 : prev2;
      const u16 bv = flip ? prev2 : prev1;
      const u16 c = idx;
      if (a != bv && bv != c && c != a) {
        dst.append<u16>(static_cast<u16>(vtxBase + a));
        dst.append<u16>(static_cast<u16>(vtxBase + bv));
        dst.append<u16>(static_cast<u16>(vtxBase + c));
        emitted += 3;
      }
      prev2 = prev1;
      prev1 = idx;
      flip = !flip;
    }
  }
  return emitted;
}

static void flush_pending_pnmtx_batch() {
  auto& b = s_pendingPnMtxBatch;
  if (!b.active) {
    return;
  }
  b.active = false;
  if (b.items.empty()) {
    return;
  }
  DrainScopeTimer _t_flush(s_drainTiming.ms_flush);
  if (b.items.size() == 1) {
    if (is_timing_enabled()) UNLIKELY ++s_drainTiming.bat_single;
    const auto& item0 = b.items[0];
    if ((item0.flags & kItemStaticPool) != 0) {
      StaticDlDraw& sd = get_static_dl_draw_mut(item0.refIdx);
      const NativeGeomEntry* ge = sd.cachedEntry;
      if (ge == nullptr || sd.cachedPtrEpoch != s_nativeGeomPtrEpoch) UNLIKELY {
        const auto cit = s_nativeGeomCache.find(sd.key);
        if (cit == s_nativeGeomCache.end()) UNLIKELY {
          b.items.clear();
          b.extras.clear();
          b.scratchVerts.clear();
          b.scratchIdxs.clear();
          return;
        }
        ge = &cit->second;
        sd.cachedEntry = ge;
        sd.cachedPtrEpoch = s_nativeGeomPtrEpoch;
      }
      emit_native_draw_command(s_lastPipeline, ge->vertRange, ge->idxRange, s_lastUniformRange, item0.vtxCount,
                               ge->indexCount, s_lastBindGroups, b.dstAlpha, /*cachedGeometry=*/true,
                               b.finalUniformDirty, b.texMtxGen, b.lightsGen, b.bpUniGen, b.othUniGen);
    } else {
      PendingPnMtxExtra& ex0 = b.extras[item0.refIdx];
      bool cached0 = (item0.flags & kItemCached) != 0;
      if (!cached0) {
        const bool indexed0 = (item0.flags & kItemIndexed) != 0;
        const u8* vtxPtr = b.scratchVerts.data() + ex0.scratchVtxOff;
        const u8* idxPtr = indexed0 ? reinterpret_cast<const u8*>(b.scratchIdxs.data() + ex0.scratchIdxOff) : nullptr;
        const size_t idxBytes = indexed0 ? static_cast<size_t>(ex0.scratchIdxCount) * sizeof(u16) : 0u;
        if ((item0.flags & kItemPromoTried) != 0) {
          const size_t vertAlign = b.nativeStride > 4 ? static_cast<size_t>(b.nativeStride) : 4u;
          ++s_geomCacheRingPushes;
          ex0.vertRange = gfx::push_verts(vtxPtr, ex0.scratchVtxBytes, vertAlign);
          if (idxPtr != nullptr && idxBytes > 0) {
            ex0.idxRange = gfx::push_indices(idxPtr, idxBytes, 4);
          }
        } else {
          native_geom_resolve_ranges(ex0.key, vtxPtr, ex0.scratchVtxBytes, b.nativeStride, idxPtr, idxBytes,
                                     ex0.numIndices, ex0.vertRange, ex0.idxRange, cached0, /*mirrorCpu=*/true,
                                     static_cast<u8>(s_staticDlCaptureActive ? 2u : 4u));
        }
      }
      emit_native_draw_command(s_lastPipeline, ex0.vertRange, ex0.idxRange, s_lastUniformRange, item0.vtxCount,
                               ex0.numIndices, s_lastBindGroups, b.dstAlpha, cached0, b.finalUniformDirty,
                               b.texMtxGen, b.lightsGen, b.bpUniGen, b.othUniGen);
    }
    b.items.clear();
    b.extras.clear();
    b.scratchVerts.clear();
    b.scratchIdxs.clear();
    return;
  }

  if (is_timing_enabled()) UNLIKELY ++s_drainTiming.bat_merged;
  ++s_pnmtxMergedBatches;
  s_pnmtxMergedDrawsIn += b.items.size();

  const bool outStrip = b.stripTopology && !b.mixedTopology;
  const bool needListPipeMemo = b.mixedTopology && b.stripTopology;
  const u8 batchTexTier = b.texSlotCount <= 1 ? 0u : (b.texSlotCount <= 2 ? 1u : (b.texSlotCount <= 4 ? 2u : 3u));
  const u8 tierBit = static_cast<u8>(1u << batchTexTier);
  const u8 mergedStride = static_cast<u8>(b.nativeStride + sizeof(u32));
  bool haveMergedPipe = false;
  gfx::PipelineRef mergedPipeline{};
  const uint32_t pipeGen = gfx::pipeline_cache_gen();
  const size_t memoIdx = s_lastPipeMemoIdx;
  if (memoIdx < kNativePipeMemoSize) {
    const auto& memo = s_nativePipeMemo[memoIdx];
    if (memo.valid && memo.gen == pipeGen && memo.pipeline == s_lastPipeline && memo.cfgHash == s_lastCfgHash) {
      if (!needListPipeMemo && (memo.hasMergedMask & tierBit) != 0) {
        mergedPipeline = memo.mergedPipelines[batchTexTier];
        haveMergedPipe = true;
      } else if (needListPipeMemo && (memo.hasMergedListMask & tierBit) != 0) {
        mergedPipeline = memo.mergedListPipelines[batchTexTier];
        haveMergedPipe = true;
      }
    }
  }
  if (!haveMergedPipe) {
    s_currentConfig = s_lastConfig;
    s_currentConfig.triangleStripTopology = outStrip ? 1u : 0u;
    s_currentConfig.shaderConfig.texBatchTier = batchTexTier;
    auto& mergedAttrs = s_currentConfig.shaderConfig.attrs;
    for (int a = GX_VA_PNMTXIDX + 1; a < MaxVtxAttr; ++a) {
      if (mergedAttrs[a].attrType != GX_NONE) {
        mergedAttrs[a].offset = static_cast<u8>(mergedAttrs[a].offset + sizeof(u32));
      }
    }
    mergedAttrs[GX_VA_PNMTXIDX].attrType = GX_DIRECT;
    mergedAttrs[GX_VA_PNMTXIDX].cnt = 1;
    mergedAttrs[GX_VA_PNMTXIDX].compType = static_cast<u8>(GX_U8);
    mergedAttrs[GX_VA_PNMTXIDX].offset = 0;
    s_currentConfig.shaderConfig.vtxStride = mergedStride;

    bool pipeReady = false;
    uint64_t mergedShaderHash = 0;
    mergedPipeline = resolve_pipeline_cached(s_currentConfig, pipeReady, mergedShaderHash);
    if (pipeReady && memoIdx < kNativePipeMemoSize) {
      auto& memo = s_nativePipeMemo[memoIdx];
      if (memo.valid && memo.gen == pipeGen && memo.pipeline == s_lastPipeline && memo.cfgHash == s_lastCfgHash) {
        if (!needListPipeMemo) {
          memo.mergedPipelines[batchTexTier] = mergedPipeline;
          memo.hasMergedMask |= tierBit;
        } else {
          memo.mergedListPipelines[batchTexTier] = mergedPipeline;
          memo.hasMergedListMask |= tierBit;
        }
      }
    }
  }

  const auto tu0 = timing_now();
  const gfx::Range mergedUniformRange =
      build_merged_uniform(s_lastInfo, s_lastUniformRange.offset, s_lastUniformRange.size,
                           b.posMtxs.data(), b.nrmMtxs.data(), b.mtxCount,
                           b.texSlotBiases.data(), b.texSlotCount);
  const auto tu1 = timing_now();
  s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();

  const GXBindGroups mergedBindGroups =
      (b.texSlotCount > 1) ? build_batched_tex0_bind_groups(b.texSlotBindGroups.data(), b.texSlotCount)
                           : s_lastBindGroups;

  gfx::Range mergedVertRange{};
  gfx::Range mergedIdxRange{};
  u32 mergedNumIndices = 0;
  bool mergedCached = false;

  NativeGeomKey mergedKey{};
  if (b.allCached) {
    const u64 hdrHash = batch_mix64(
        (static_cast<u64>(b.items.size()) << 32u) |
        (static_cast<u64>(outStrip ? 1u : 0u) << 16u) |
        (static_cast<u64>(b.mixedTopology ? 2u : 0u) << 16u) |
        (static_cast<u64>(b.nativeStride) << 8u) |
        static_cast<u64>(b.totalVtxCount));
    mergedKey = {b.batchFh ^ hdrHash,
                 b.batchMh ^ (static_cast<u64>(b.totalVtxCount) * 0x94d049bb133111ebull),
                 b.totalVtxCount * static_cast<u32>(mergedStride)};
    if (const auto hitIt = s_nativeGeomCache.find(mergedKey); hitIt != s_nativeGeomCache.end()) {
      ++s_pnmtxMergedCacheHits;
      mergedVertRange = hitIt->second.vertRange;
      mergedIdxRange = hitIt->second.idxRange;
      mergedNumIndices = hitIt->second.indexCount;
      mergedCached = true;
      if (needListPipeMemo) {
        s_lastConfig.triangleStripTopology = 0u;
      }
      if (b.texSlotCount > 1) {
        s_forceTexResolveAfterBatch = true;
      }
      emit_native_draw_command(mergedPipeline, mergedVertRange, mergedIdxRange, mergedUniformRange, b.totalVtxCount,
                               mergedNumIndices, mergedBindGroups, b.dstAlpha, mergedCached, b.finalUniformDirty,
                               b.texMtxGen, b.lightsGen, b.bpUniGen, b.othUniGen);
      b.items.clear();
      b.extras.clear();
      b.scratchVerts.clear();
      b.scratchIdxs.clear();
      return;
    }
  }

  bool anyIndexed = outStrip || b.mixedTopology;
  for (const auto& it : b.items) {
    if ((it.flags & kItemIndexed) != 0) {
      anyIndexed = true;
      break;
    }
  }

  static ByteBuffer s_mergedVtxBuf;
  static ByteBuffer s_mergedIdxBuf;
  s_mergedVtxBuf.clear();
  const size_t mergedVtxTotalBytes = static_cast<size_t>(b.totalVtxCount) * mergedStride;
  s_mergedVtxBuf.reserve_extra(mergedVtxTotalBytes);
  s_mergedIdxBuf.clear();

  u16 vtxBase = 0;
  for (size_t s = 0; s < b.items.size(); ++s) {
    const auto& it = b.items[s];
    const bool itStatic = (it.flags & kItemStaticPool) != 0;
    const bool itCached = itStatic || ((it.flags & kItemCached) != 0);
    const bool itIndexed = (it.flags & kItemIndexed) != 0;
    const bool itStrip = (it.flags & kItemStripTopology) != 0;
    const u8* srcVerts = nullptr;
    const u16* srcIdxs = nullptr;
    u32 srcIdxCount = 0;
    if (itCached) {
      const NativeGeomEntry* ge = nullptr;
      const NativeGeomKey* keyPtr = nullptr;
      if (itStatic) {
        StaticDlDraw& sd = get_static_dl_draw_mut(it.refIdx);
        keyPtr = &sd.key;
        if (sd.cachedPtrEpoch == s_nativeGeomPtrEpoch) {
          ge = sd.cachedEntry;
        }
        if (ge == nullptr) UNLIKELY {
          const auto cit = s_nativeGeomCache.find(*keyPtr);
          if (cit != s_nativeGeomCache.end()) {
            ge = &cit->second;
            sd.cachedEntry = ge;
            sd.cachedPtrEpoch = s_nativeGeomPtrEpoch;
          }
        }
      } else {
        PendingPnMtxExtra& ex = b.extras[it.refIdx];
        keyPtr = &ex.key;
        if (ex.cachedPtrEpoch == s_nativeGeomPtrEpoch) {
          ge = ex.cachedEntry;
        }
        if (ge == nullptr) UNLIKELY {
          const auto cit = s_nativeGeomCache.find(*keyPtr);
          if (cit != s_nativeGeomCache.end()) {
            ge = &cit->second;
            ex.cachedEntry = ge;
            ex.cachedPtrEpoch = s_nativeGeomPtrEpoch;
          }
        }
      }
      const size_t needVtxBytes = static_cast<size_t>(it.vtxCount) * b.nativeStride;
      if (ge != nullptr && ge->cpuVerts.size() >= needVtxBytes) LIKELY {
        srcVerts = ge->cpuVerts.data();
        srcIdxs = ge->cpuIndices.data();
        srcIdxCount = static_cast<u32>(ge->cpuIndices.size());
      }
    } else {
      const PendingPnMtxExtra& ex = b.extras[it.refIdx];
      const size_t needVtxBytes = static_cast<size_t>(it.vtxCount) * b.nativeStride;
      if (static_cast<size_t>(ex.scratchVtxOff) + needVtxBytes <= b.scratchVerts.size()) LIKELY {
        srcVerts = b.scratchVerts.data() + ex.scratchVtxOff;
      }
      srcIdxs = (itIndexed && static_cast<size_t>(ex.scratchIdxOff) + ex.scratchIdxCount <= b.scratchIdxs.size())
                    ? (b.scratchIdxs.data() + ex.scratchIdxOff)
                    : nullptr;
      srcIdxCount = srcIdxs != nullptr ? ex.scratchIdxCount : 0u;
    }

    const u32 slotVal = static_cast<u32>(it.slotIdx) | (static_cast<u32>(it.texSlot) << 4u);
    if (srcVerts != nullptr) LIKELY {
      for (u16 v = 0; v < it.vtxCount; ++v) {
        s_mergedVtxBuf.append<u32>(slotVal);
        const u8* vp = srcVerts + static_cast<size_t>(v) * b.nativeStride;
        s_mergedVtxBuf.append(vp, b.nativeStride);
      }
    } else {
      for (u16 v = 0; v < it.vtxCount; ++v) {
        s_mergedVtxBuf.append<u32>(slotVal);
        s_mergedVtxBuf.append_zeroes(b.nativeStride);
      }
    }

    if (anyIndexed) {
      if (outStrip) {
        if (s > 0) {
          s_mergedIdxBuf.append<u16>(0xFFFFu);
          ++mergedNumIndices;
        }
        if (itIndexed && srcIdxs != nullptr) {
          for (u32 ii = 0; ii < srcIdxCount; ++ii) {
            const u16 idx = srcIdxs[ii];
            s_mergedIdxBuf.append<u16>(idx == 0xFFFFu ? 0xFFFFu : static_cast<u16>(idx + vtxBase));
          }
          mergedNumIndices += srcIdxCount;
        } else {
          for (u16 v = 0; v < it.vtxCount; ++v) {
            s_mergedIdxBuf.append<u16>(static_cast<u16>(vtxBase + v));
          }
          mergedNumIndices += it.vtxCount;
        }
      } else if (itStrip) {
        mergedNumIndices += append_unrolled_strip_indices(
            s_mergedIdxBuf, (itIndexed && srcIdxs != nullptr) ? srcIdxs : nullptr, srcIdxCount, it.vtxCount, vtxBase);
      } else {
        if (itIndexed && srcIdxs != nullptr) {
          for (u32 ii = 0; ii < srcIdxCount; ++ii) {
            s_mergedIdxBuf.append<u16>(static_cast<u16>(srcIdxs[ii] + vtxBase));
          }
          mergedNumIndices += srcIdxCount;
        } else {
          for (u16 v = 0; v < it.vtxCount; ++v) {
            s_mergedIdxBuf.append<u16>(static_cast<u16>(vtxBase + v));
          }
          mergedNumIndices += it.vtxCount;
        }
      }
    }
    vtxBase = static_cast<u16>(vtxBase + it.vtxCount);
  }

  const u8* idxPtr = anyIndexed ? s_mergedIdxBuf.data() : nullptr;
  const size_t idxBytes = anyIndexed ? s_mergedIdxBuf.size() : 0u;
  if (b.allCached && s_nativeGeomCache.size() < kMaxNativeGeomMergedCacheEntries) {
    native_geom_resolve_ranges(mergedKey, s_mergedVtxBuf.data(), s_mergedVtxBuf.size(), mergedStride, idxPtr, idxBytes,
                               mergedNumIndices, mergedVertRange, mergedIdxRange, mergedCached, /*mirrorCpu=*/false,
                               8u);
  } else {
    const size_t vertAlign = mergedStride > 4 ? static_cast<size_t>(mergedStride) : 4u;
    ++s_geomCacheRingPushes;
    mergedVertRange = gfx::push_verts(s_mergedVtxBuf.data(), s_mergedVtxBuf.size(), vertAlign);
    if (idxPtr != nullptr && idxBytes > 0) {
      mergedIdxRange = gfx::push_indices(idxPtr, idxBytes, 4);
    }
  }
  s_mergedIdxBuf.clear();

  if (needListPipeMemo) {
    s_lastConfig.triangleStripTopology = 0u;
  }
  if (b.texSlotCount > 1) {
    s_forceTexResolveAfterBatch = true;
  }
  emit_native_draw_command(mergedPipeline, mergedVertRange, mergedIdxRange, mergedUniformRange, b.totalVtxCount,
                           mergedNumIndices, mergedBindGroups, b.dstAlpha, mergedCached, b.finalUniformDirty,
                           b.texMtxGen, b.lightsGen, b.bpUniGen, b.othUniGen);
  b.items.clear();
  b.extras.clear();
  b.scratchVerts.clear();
  b.scratchIdxs.clear();
}

static bool pnmtx_slot_equals(const Mat3x4<float>& a, const Mat3x4<float>& b) noexcept {
  return std::memcmp(&a, &b, sizeof(Mat3x4<float>)) == 0;
}

static inline bool try_fit_pnmtx_tex_and_mtx(PendingPnMtxBatch& b, u32 mtxIdx, const PnMtx& curMtx, u32 curMtxGen,
                                             u32& outSlotIdx, u32& outTexSlot, bool& outPreBgPopulated,
                                             bool& outFlTex, bool& outFlMtx) noexcept {
  bool sameTex = (g_gxState.textureDirty == b.textureDirty);
  u32 texSlot = b.lastTexSlot;
  bool texFit = sameTex;
  bool needAllocTexSlot = false;
  bool updateSingleTexDirty = false;
  f32 newTexSlotBias = 0.f;
  if (!sameTex) {
    const auto t_res0 = timing_now();
    s_candidateBg = resolve_and_build_bind_groups(s_lastInfo);
    const auto t_res1 = timing_now();
    s_drainTiming.ms_tex_resolve += std::chrono::duration<double, std::milli>(t_res1 - t_res0).count();
    ++s_drainTiming.bg_calls;
    const gfx::BindGroupRef candBg = s_candidateBg.textureBindGroup;
    const f32 candBias = b.allowMultiTex ? current_tex_lod_bias(0) : 0.f;
    if (candBg == b.texSlotBindGroups[b.lastTexSlot] &&
        (!b.allowMultiTex || b.texSlotBiases[b.lastTexSlot] == candBias)) {
      sameTex = true;
      texFit = true;
      b.textureDirty = g_gxState.textureDirty;
      if (b.texSlotCount == 1 && b.lastTexSlot == 0) {
        updateSingleTexDirty = true;
      }
    } else if (b.allowMultiTex && candBg != 0) {
      u32 foundSlot = UINT32_MAX;
      for (u32 ts = 0; ts < b.texSlotCount; ++ts) {
        if (b.texSlotBindGroups[ts] == candBg && b.texSlotBiases[ts] == candBias) {
          foundSlot = ts;
          break;
        }
      }
      if (foundSlot != UINT32_MAX) {
        texSlot = foundSlot;
        texFit = true;
      } else if (b.texSlotCount < 4u) {
        texSlot = b.texSlotCount;
        needAllocTexSlot = true;
        newTexSlotBias = candBias;
        texFit = true;
      }
    }
    if (!texFit) {
      outPreBgPopulated = true;
      outFlTex = true;
      return false;
    }
  }

  u32 slotIdx = UINT32_MAX;
  if (b.lastMtxIdx == mtxIdx && b.lastMtxGen == curMtxGen) {
    slotIdx = b.lastMtxSlot;
  } else {
    for (u32 s = 0; s < b.mtxCount; ++s) {
      if (b.slotMtxGens[s] == curMtxGen ||
          (b.posMtxs[s].m0[3] == curMtx.pos.m0[3] &&
           b.posMtxs[s].m1[3] == curMtx.pos.m1[3] &&
           b.posMtxs[s].m2[3] == curMtx.pos.m2[3] &&
           pnmtx_slot_equals(b.posMtxs[s], curMtx.pos) &&
           (!s_lastInfo.usesNormals || pnmtx_slot_equals(b.nrmMtxs[s], curMtx.nrm)))) {
        slotIdx = s;
        break;
      }
    }
  }
  bool needAllocMtxSlot = false;
  if (slotIdx == UINT32_MAX && b.mtxCount < MaxPnMtx) {
    slotIdx = b.mtxCount;
    needAllocMtxSlot = true;
  }
  if (slotIdx == UINT32_MAX) {
    if (!sameTex) {
      outPreBgPopulated = true;
    }
    outFlMtx = true;
    return false;
  }

  if (updateSingleTexDirty) {
    s_lastBindGroups = s_candidateBg;
    s_lastTextureDirty = g_gxState.textureDirty;
  }
  if (!sameTex) {
    if (needAllocTexSlot) {
      b.texSlotBindGroups[texSlot] = s_candidateBg.textureBindGroup;
      b.texSlotBiases[texSlot] = newTexSlotBias;
      ++b.texSlotCount;
    }
    b.textureDirty = g_gxState.textureDirty;
    b.lastTexSlot = texSlot;
  }
  if (needAllocMtxSlot) {
    ++b.mtxCount;
    b.posMtxs[slotIdx] = curMtx.pos;
    b.nrmMtxs[slotIdx] = curMtx.nrm;
    b.slotMtxGens[slotIdx] = curMtxGen;
  }
  b.lastMtxIdx = mtxIdx;
  b.lastMtxGen = curMtxGen;
  b.lastMtxSlot = slotIdx;
  outSlotIdx = slotIdx;
  outTexSlot = texSlot;
  return true;
}

static inline void record_pnmtx_item(PendingPnMtxBatch& b, u16 vtxCount, u32 slotIdx, u32 texSlot,
                                     bool cached, bool indexed, bool stripTopology, bool promoTried,
                                     u32 staticPoolRefIdx, const NativeGeomKey& key,
                                     const NativeGeomEntry* cachedEntry, u32 cachedEntryEpoch,
                                     const gfx::Range& vertRange, const gfx::Range& idxRange, u32 numIndices,
                                     const u8* vtxPtr, size_t vtxBytes, const u8* idxPtr, size_t idxBytes) {
  PendingPnMtxItem& item = b.items.emplace_back();
  item.vtxCount = vtxCount;
  item.slotIdx = static_cast<u8>(slotIdx);
  item.texSlot = static_cast<u8>(texSlot);
  item.flags = (cached ? kItemCached : 0u) | (indexed ? kItemIndexed : 0u) |
               (stripTopology ? kItemStripTopology : 0u) | (promoTried ? kItemPromoTried : 0u);
  if (staticPoolRefIdx != UINT32_MAX) {
    item.flags |= kItemStaticPool;
    item.refIdx = staticPoolRefIdx;
    return;
  }
  item.refIdx = static_cast<u32>(b.extras.size());
  PendingPnMtxExtra& ex = b.extras.emplace_back();
  ex.key = key;
  ex.cachedEntry = cachedEntry;
  ex.cachedPtrEpoch = cached ? cachedEntryEpoch : 0u;
  ex.vertRange = vertRange;
  ex.idxRange = idxRange;
  ex.numIndices = numIndices;
  if (!cached) {
    ex.scratchVtxOff = static_cast<u32>(b.scratchVerts.size());
    ex.scratchVtxBytes = static_cast<u32>(vtxBytes);
    b.scratchVerts.resize(ex.scratchVtxOff + vtxBytes);
    std::memcpy(b.scratchVerts.data() + ex.scratchVtxOff, vtxPtr, vtxBytes);
    if (indexed && idxPtr != nullptr && idxBytes > 0) {
      ex.scratchIdxOff = static_cast<u32>(b.scratchIdxs.size());
      ex.scratchIdxCount = static_cast<u32>(idxBytes / sizeof(u16));
      b.scratchIdxs.resize(ex.scratchIdxOff + ex.scratchIdxCount);
      std::memcpy(b.scratchIdxs.data() + ex.scratchIdxOff, idxPtr, idxBytes);
    }
  }
}

static void submit_native_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount,
                                  const std::array<AttrConfig, MaxVtxAttr>& nativeAttrs, u16 attrLayoutId,
                                  u8 nativeStride, const NativeGeomKey& key, const NativeGeomEntry* cachedEntry,
                                  const u8* vtxPtr, size_t vtxBytes, const u8* idxPtr, size_t idxBytes, u32 numIndices,
                                  bool indexed, bool stripTopology, u32 staticPoolRefIdx = UINT32_MAX,
                                  bool preBgPopulatedArg = false, bool preCfgSameArg = false) {
  if (vtxCount == 0) {
    return;
  }
  if (s_staticDlCaptureActive) UNLIKELY {
    const u8 drawFlags = static_cast<u8>((indexed ? kStaticDrawIndexed : 0u) |
                                         (stripTopology ? kStaticDrawStrip : 0u));
    s_staticDlCapturedDraws.push_back(StaticDlDraw{
        key,
        nullptr,
        0u,
        vtxCount,
        attrLayoutId,
        static_cast<u8>(prim),
        static_cast<u8>(fmt),
        nativeStride,
        drawFlags,
    });
    s_staticDlCapturedFmtMask |= static_cast<u32>(1u << fmt);
  }

  bool cached = (cachedEntry != nullptr);
  u32 cachedEntryEpoch = cached ? s_nativeGeomPtrEpoch : 0u;
  gfx::Range cachedVertRange = cached ? cachedEntry->vertRange : gfx::Range{};
  gfx::Range cachedIdxRange = cached ? cachedEntry->idxRange : gfx::Range{};
  u32 effectiveNumIndices = cached ? cachedEntry->indexCount : numIndices;
  const size_t minMirrorVtxBytes = static_cast<size_t>(vtxCount) * nativeStride;
  bool hasCpuMirror = cached && cachedEntry->cpuVerts.size() >= minMirrorVtxBytes &&
                      (!indexed || cachedEntry->cpuIndices.size() >= effectiveNumIndices);

  const bool baseEligible = kEnablePnMtxBatch && vtxCount <= 0xFFFEu &&
                            nativeAttrs[GX_VA_PNMTXIDX].attrType == GX_NONE &&
                            prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS;

  // Promote recurring uncached draws early when possible so static batches turn allCached faster.
  bool promoTried = false;
  if (baseEligible && !cached && vtxBytes <= kMaxNativeGeomCpuMirrorEntryBytes) {
    promoTried = true;
    const NativeGeomEntry* promotedEntry = nullptr;
    if (try_promote_native_geom(key, vtxPtr, vtxBytes, nativeStride, idxPtr, idxBytes, numIndices,
                                cachedVertRange, cachedIdxRange, /*mirrorCpu=*/true, &promotedEntry,
                                static_cast<u8>(s_staticDlCaptureActive ? 2u : 4u))) {
      cached = true;
      cachedEntry = promotedEntry;
      cachedEntryEpoch = s_nativeGeomPtrEpoch;
      effectiveNumIndices = promotedEntry != nullptr ? promotedEntry->indexCount : numIndices;
      hasCpuMirror = (promotedEntry != nullptr && promotedEntry->cpuVerts.size() >= minMirrorVtxBytes &&
                      (!indexed || promotedEntry->cpuIndices.size() >= effectiveNumIndices));
    }
  }

  DrainScopeTimer _t_sub(s_drainTiming.ms_submit);
  const bool itemBatchable = baseEligible && (!cached || hasCpuMirror);
  auto& b = s_pendingPnMtxBatch;
  const u32 mtxIdx = std::min<u32>(g_gxState.currentPnMtx, MaxPnMtx - 1);
  const PnMtx& curMtx = g_gxState.pnMtx[mtxIdx];
  const u32 curMtxGen = s_pnMtxGen[mtxIdx];

  bool preCfgPopulated = false;
  bool preConfigSame = preCfgSameArg;
  bool preBgPopulated = preBgPopulatedArg;

  const u64 curFastUniGen = (static_cast<u64>(s_splitBpUniGen) << 32) | s_splitOthUniGen;
  UniMismatch uniMismatch = UniMismatch::None;
  const bool sameUniFast =
      !preCfgSameArg && !preBgPopulatedArg && b.active && itemBatchable &&
      ((b.uniFastGen == curFastUniGen && (!s_lastInfo.lightingEnabled || b.lightsGen == s_splitLightsGen)) ||
       ((uniMismatch = batch_uniforms_mismatch(b.uni, b.lightsGen, s_lastInfo, b.allowMultiTex)) == UniMismatch::None));
  if (sameUniFast && b.uniFastGen != curFastUniGen) {
    b.uniFastGen = curFastUniGen;
  }

  if (!preCfgSameArg && !preBgPopulatedArg && b.active && itemBatchable &&
      (b.stripTopology == stripTopology || s_lastConfig.shaderConfig.lineMode == 0) &&
      b.nativeStride == nativeStride &&
      b.dstAlpha == g_gxState.dstAlpha &&
      b.texMtxGen == s_splitTexMtxGen &&
      sameUniFast &&
      b.totalVtxCount + vtxCount <= 0xFFFEu &&
      b.attrLayoutId == attrLayoutId) {
    const u32 curStripU32 = stripTopology ? 1u : 0u;
    const bool topoMatchesLastCfg = (curStripU32 == s_lastConfig.triangleStripTopology);
    const bool lineZero = (s_lastConfig.shaderConfig.lineMode == 0);
    const bool primTriCompatible =
        (prim == b.prim) ||
        (lineZero && (prim == GX_TRIANGLES || prim == GX_TRIANGLESTRIP) &&
         (b.prim == GX_TRIANGLES || b.prim == GX_TRIANGLESTRIP));
    bool cfgPopulated = false;
    bool samePipe = (g_gxState.pipelineDirty == b.pipelineDirty && fmt == b.fmt &&
                     (topoMatchesLastCfg ? (prim == b.prim) : primTriCompatible));
    if (!samePipe) {
      const auto tp0 = timing_now();
      s_currentConfig = {};
      populate_pipeline_config(s_currentConfig, prim, fmt, /*fillAttrs=*/false);
      s_currentConfig.shaderConfig.attrs = nativeAttrs;
      s_currentConfig.shaderConfig.vtxStride = nativeStride;
      s_currentConfig.shaderConfig.nativeVertexFetch = 1;
      const bool canIgnoreTopo = (s_currentConfig.shaderConfig.lineMode == 0 && s_lastConfig.shaderConfig.lineMode == 0);
      if (canIgnoreTopo) {
        s_currentConfig.triangleStripTopology = s_lastConfig.triangleStripTopology;
      } else {
        s_currentConfig.triangleStripTopology = curStripU32;
      }
      const auto tp1 = timing_now();
      s_drainTiming.ms_pipe_cfg += std::chrono::duration<double, std::milli>(tp1 - tp0).count();
      cfgPopulated = true;
      if (std::memcmp(&s_currentConfig, &s_lastConfig, sizeof(PipelineConfig)) == 0) {
        samePipe = true;
        if (topoMatchesLastCfg || canIgnoreTopo) {
          b.pipelineDirty = g_gxState.pipelineDirty;
          b.prim = prim;
          b.fmt = fmt;
          s_lastPipelineDirty = g_gxState.pipelineDirty;
          s_lastPrim = prim;
          s_lastFmt = fmt;
        }
      } else {
        s_currentConfig.triangleStripTopology = curStripU32;
        preCfgPopulated = true;
      }
    }
    if (samePipe) {
      u32 slotIdx = 0;
      u32 texSlot = b.lastTexSlot;
      bool flTex = false;
      bool flMtx = false;
      if (try_fit_pnmtx_tex_and_mtx(b, mtxIdx, curMtx, curMtxGen, slotIdx, texSlot, preBgPopulated, flTex, flMtx)) {
        if (!topoMatchesLastCfg) {
          b.mixedTopology = true;
        }
        if (topoMatchesLastCfg || lineZero) {
          b.pipelineDirty = g_gxState.pipelineDirty;
          b.prim = prim;
        }
        b.textureDirty = g_gxState.textureDirty;
        b.totalVtxCount += vtxCount;
        b.finalUniformDirty = g_gxState.uniformDirty;
        const bool itemIndexed = cached ? (effectiveNumIndices != 0) : indexed;
        const u32 refId = (cached && hasCpuMirror) ? staticPoolRefIdx : UINT32_MAX;
        record_pnmtx_item(b, vtxCount, slotIdx, texSlot, cached, itemIndexed, stripTopology, promoTried,
                          refId, key, cachedEntry, cachedEntryEpoch, cachedVertRange, cachedIdxRange,
                          effectiveNumIndices, vtxPtr, vtxBytes, idxPtr, idxBytes);
        if (!cached) {
          b.allCached = false;
        } else if (b.allCached) {
          batch_hash_append_cached(b.batchFh, b.batchMh, key, slotIdx, texSlot, stripTopology);
        }
        if (is_timing_enabled()) UNLIKELY ++s_drainTiming.bat_ok;
        return;
      }
      if (topoMatchesLastCfg) {
        preConfigSame = true;
      } else {
        if (!cfgPopulated) {
          s_currentConfig = s_lastConfig;
        }
        s_currentConfig.triangleStripTopology = curStripU32;
        preCfgPopulated = true;
      }
      if (is_timing_enabled()) UNLIKELY {
        if (flMtx) ++s_drainTiming.fl_mtx;
        else if (flTex) ++s_drainTiming.fl_tex;
      }
    } else if (is_timing_enabled()) UNLIKELY {
      ++s_drainTiming.fl_pipe;
    }
  } else if (!preCfgSameArg && !preBgPopulatedArg && b.active && is_timing_enabled()) UNLIKELY {
    if (!itemBatchable) ++s_drainTiming.fl_nonbat;
    else if ((b.stripTopology != stripTopology && s_lastConfig.shaderConfig.lineMode != 0) ||
             b.nativeStride != nativeStride ||
             b.attrLayoutId != attrLayoutId) ++s_drainTiming.fl_topo;
    else if (b.dstAlpha != g_gxState.dstAlpha) ++s_drainTiming.fl_da;
    else if (b.texMtxGen != s_splitTexMtxGen) ++s_drainTiming.fl_tm;
    else if (uniMismatch == UniMismatch::Lights) ++s_drainTiming.fl_li;
    else if (uniMismatch == UniMismatch::Bp) ++s_drainTiming.fl_bp;
    else if (uniMismatch == UniMismatch::Oth) ++s_drainTiming.fl_oth;
    else ++s_drainTiming.fl_vtx;
  }

  flush_pending_pnmtx_batch();
  if (cached && cachedEntryEpoch != s_nativeGeomPtrEpoch) UNLIKELY {
    const auto cit = s_nativeGeomCache.find(key);
    if (cit != s_nativeGeomCache.end()) {
      cachedEntry = &cit->second;
      cachedEntryEpoch = s_nativeGeomPtrEpoch;
      cachedVertRange = cachedEntry->vertRange;
      cachedIdxRange = cachedEntry->idxRange;
      effectiveNumIndices = cachedEntry->indexCount;
      hasCpuMirror = cachedEntry->cpuVerts.size() >= minMirrorVtxBytes &&
                     (!indexed || cachedEntry->cpuIndices.size() >= effectiveNumIndices);
    } else {
      cached = false;
      cachedEntry = nullptr;
      cachedEntryEpoch = 0u;
      hasCpuMirror = false;
    }
  }

  resolve_single_native_draw_state(prim, fmt, nativeAttrs, nativeStride, stripTopology, preCfgPopulated,
                                   preConfigSame, preBgPopulated);

  if (!itemBatchable || shader_uses_fixed_pnmtx_texgen(s_lastInfo, s_lastConfig.shaderConfig)) {
    if (is_timing_enabled() && itemBatchable) UNLIKELY ++s_drainTiming.skip_fixtg;
    if (!cached) {
      if (promoTried) {
        const size_t vertAlign = nativeStride > 4 ? static_cast<size_t>(nativeStride) : 4u;
        ++s_geomCacheRingPushes;
        cachedVertRange = gfx::push_verts(vtxPtr, vtxBytes, vertAlign);
        if (idxPtr != nullptr && idxBytes > 0) {
          cachedIdxRange = gfx::push_indices(idxPtr, idxBytes, 4);
        }
      } else {
        native_geom_resolve_ranges(key, vtxPtr, vtxBytes, nativeStride, idxPtr, idxBytes, effectiveNumIndices,
                                   cachedVertRange, cachedIdxRange, cached, /*mirrorCpu=*/true,
                                   static_cast<u8>(s_staticDlCaptureActive ? 2u : 4u));
      }
    }
    const u8 lineWidth = (prim == GX_LINES || prim == GX_LINESTRIP)
                             ? (g_gxState.lineWidth != 0 ? g_gxState.lineWidth : static_cast<u8>(6))
                             : static_cast<u8>(6);
    emit_native_draw_command(s_lastPipeline, cachedVertRange, cachedIdxRange, s_lastUniformRange, vtxCount,
                             effectiveNumIndices, s_lastBindGroups, g_gxState.dstAlpha, cached,
                             g_gxState.uniformDirty, s_splitTexMtxGen, s_splitLightsGen, s_splitBpUniGen,
                             s_splitOthUniGen, lineWidth);
    return;
  }

  if (b.items.capacity() < 256) {
    b.items.reserve(256);
    b.extras.reserve(256);
    b.scratchVerts.reserve(65536);
    b.scratchIdxs.reserve(16384);
  }

  b.active = true;
  b.prim = prim;
  b.fmt = fmt;
  b.stripTopology = stripTopology;
  b.mixedTopology = false;
  b.nativeAttrs = nativeAttrs;
  b.attrLayoutId = attrLayoutId;
  b.nativeStride = nativeStride;
  b.pipelineDirty = g_gxState.pipelineDirty;
  b.textureDirty = g_gxState.textureDirty;
  b.dstAlpha = g_gxState.dstAlpha;
  b.texMtxGen = s_splitTexMtxGen;
  b.lightsGen = s_splitLightsGen;
  b.bpUniGen = s_splitBpUniGen;
  b.othUniGen = s_splitOthUniGen;
  b.uniFastGen = curFastUniGen;
  b.uni = s_curUniState;
  b.finalUniformDirty = g_gxState.uniformDirty;
  b.posMtxs[0] = curMtx.pos;
  b.nrmMtxs[0] = curMtx.nrm;
  b.slotMtxGens[0] = curMtxGen;
  b.mtxCount = 1;
  b.lastMtxIdx = mtxIdx;
  b.lastMtxGen = curMtxGen;
  b.lastMtxSlot = 0;
  b.allowMultiTex = (s_lastInfo.sampledTextures.count() == 1 &&
                     s_lastInfo.sampledTextures.test(0) &&
                     !s_lastInfo.usesTexSize &&
                     !s_lastInfo.usesTexCoordScales &&
                     s_lastConfig.shaderConfig.sdfMaps == 0 &&
                     s_lastBindGroups.textureBindGroup != 0);
  b.lastTexSlot = 0;
  b.texSlotCount = 1;
  b.texSlotBindGroups[0] = s_lastBindGroups.textureBindGroup;
  b.texSlotBiases[0] = b.allowMultiTex ? current_tex_lod_bias(0) : 0.f;
  b.totalVtxCount = vtxCount;
  b.allCached = cached;
  b.batchFh = PendingPnMtxBatch::kBatchFhSeed;
  b.batchMh = PendingPnMtxBatch::kBatchMhSeed;
  if (cached) {
    batch_hash_append_cached(b.batchFh, b.batchMh, key, 0u, 0u, stripTopology);
  }
  b.items.clear();
  b.extras.clear();
  b.scratchVerts.clear();
  b.scratchIdxs.clear();

  const bool item0Indexed = cached ? (effectiveNumIndices != 0) : indexed;
  const u32 refId = (cached && hasCpuMirror) ? staticPoolRefIdx : UINT32_MAX;
  record_pnmtx_item(b, vtxCount, 0u, 0u, cached, item0Indexed, stripTopology, promoTried,
                    refId, key, cachedEntry, cachedEntryEpoch, cachedVertRange, cachedIdxRange,
                    effectiveNumIndices, vtxPtr, vtxBytes, idxPtr, idxBytes);
}

// Expand `vtxCount` raw FIFO vertices and push a native draw, serving steady-state
// geometry from the persistent content-hash cache (no re-expand, no re-upload).
static bool try_native_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 vtxSize) {
  if (prim != GX_TRIANGLES && prim != GX_TRIANGLESTRIP && prim != GX_TRIANGLEFAN && prim != GX_QUADS &&
      prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS) {
    return false;
  }
  u8 nativeStride = 0;
  const std::array<AttrConfig, MaxVtxAttr>* nativeAttrsPtr = nullptr;
  u16 attrLayoutId = 0;
  const std::vector<NativeAttrDesc>* descsPtr = nullptr;
  if (!get_native_layout(fmt, descsPtr, nativeStride, nativeAttrsPtr, attrLayoutId)) {
    return false;
  }
  const auto& nativeAttrs = *nativeAttrsPtr;
  const auto& descs = *descsPtr;
  ++s_geomDrawsSingle;
  const u32 srcBytes = static_cast<u32>(vtxCount) * vtxSize;
  if (((prim == GX_LINES || prim == GX_LINESTRIP) && vtxCount < 2) || (prim == GX_POINTS && vtxCount == 0)) {
    pos += srcBytes;
    return true;
  }
  const u16 drawVtxCount = (prim == GX_LINES) ? static_cast<u16>(vtxCount & ~1u) : vtxCount;

  // Content-hash key: source FIFO bytes + decode layout (+ indexed array contents). A hit
  // replays the cached GPU ranges without touching the CPU-side vertex data at all.
  NativeGeomKey key{};
  if constexpr (kNativeGeomCacheEnabled) {
    const auto th0 = timing_now();
    const u64 fifoHash = XXH3_64bits(data + pos, srcBytes);
    if (is_timing_enabled()) UNLIKELY s_drainTiming.fifo_hash_bytes += srcBytes;
    const u64 metaHash = hash_native_geom_meta(prim, fmt, vtxSize, vtxCount, descs);
    key = {fifoHash, metaHash, srcBytes};
    native_geom_sync_generation();
    const auto th1 = timing_now();
    s_drainTiming.ms_geom_hash += std::chrono::duration<double, std::milli>(th1 - th0).count();
    if (const auto it = s_nativeGeomCache.find(key); it != s_nativeGeomCache.end()) {
      ++s_geomCacheHits;
      pos += srcBytes;
      submit_native_gx_draw(prim, fmt, drawVtxCount, nativeAttrs, attrLayoutId, nativeStride, key, &it->second, nullptr,
                            0, nullptr, 0, it->second.indexCount, it->second.indexCount != 0, /*stripTopology=*/false);
      return true;
    }
    ++s_geomCacheMisses;
  }

  // Miss — expand the run into flat native attributes.
  s_nativeVtxBuf.clear();
  if (vtxCount > 0) {
    std::array<PreparedNativeAttr, MaxVtxAttr> fastAttrs{};
    const size_t attrCount = prepare_native_attrs(descs, fastAttrs.data());
    u8* dst = s_nativeVtxBuf.alloc_write(static_cast<size_t>(vtxCount) * nativeStride);
    expand_native_vertices_into(fastAttrs.data(), attrCount, data + pos, vtxSize, vtxCount, dst);
  }
  pos += srcBytes;

  // Strips/fans/quads/line-strips get a generated index buffer; plain triangles, lines, and points draw non-indexed.
  const bool indexed = (prim != GX_TRIANGLES && prim != GX_LINES && prim != GX_POINTS);
  u32 numIndices = 0;
  s_nativeIdxBuf.clear();
  if (prim == GX_LINESTRIP) {
    s_nativeIdxBuf.reserve_extra(static_cast<size_t>(vtxCount - 1u) * 2u * sizeof(u16));
    for (u16 v = 0; v + 1u < vtxCount; ++v) {
      s_nativeIdxBuf.append<u16>(v);
      s_nativeIdxBuf.append<u16>(static_cast<u16>(v + 1u));
    }
    numIndices = static_cast<u32>(vtxCount - 1u) * 2u;
  } else if (indexed) {
    numIndices = prepare_idx_buffer(s_nativeIdxBuf, prim, 0, vtxCount);
  }

  submit_native_gx_draw(prim, fmt, drawVtxCount, nativeAttrs, attrLayoutId, nativeStride, key, nullptr,
                        s_nativeVtxBuf.data(), s_nativeVtxBuf.size(), indexed ? s_nativeIdxBuf.data() : nullptr,
                        indexed ? s_nativeIdxBuf.size() : 0, numIndices, indexed, /*stripTopology=*/false);
  s_nativeIdxBuf.clear();
  return true;
}

// Batched native path for raw-FIFO draw commands. Display lists emit runs of back-to-back
// identical draw commands (same VAT byte ⇒ same primitive + layout; no state change can
// sit between adjacent draws), so this collapses a whole run into ONE hardware draw — the
// biggest CPU-side win, since per-draw pipeline/bind-group/uniform submission dominates
// strip-heavy scenes. Triangle strips keep strip topology with 0xffff primitive-restart
// separators rather than being unrolled to a triangle list. `pos` enters at the first
// draw's vertex data (cmd + vtxCount already consumed) and is advanced past the whole
// consumed run on success. Serves steady-state geometry from the persistent content-hash
// cache (no re-expand, no re-upload).
static bool try_native_draw_run(u8 cmd, GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data,
                                u32& pos, u32 size, bool bigEndian) {
  if (prim != GX_TRIANGLES && prim != GX_TRIANGLESTRIP && prim != GX_TRIANGLEFAN && prim != GX_QUADS) {
    return false;
  }
  if (vtxSize == 0 || vtxCount == 0) {
    return false;
  }
  u8 nativeStride = 0;
  const std::array<AttrConfig, MaxVtxAttr>* nativeAttrsPtr = nullptr;
  u16 attrLayoutId = 0;
  const std::vector<NativeAttrDesc>* descsPtr = nullptr;
  if (!get_native_layout(fmt, descsPtr, nativeStride, nativeAttrsPtr, attrLayoutId)) {
    return false;
  }
  const auto& nativeAttrs = *nativeAttrsPtr;
  const auto& descs = *descsPtr;

  // Scan forward across back-to-back identical draw commands (including 0x00 CP_CMD_NOP
  // 32-byte display-list padding bytes between packed strip calls in mapdrv/animdrv).
  // FIFO layout per draw is [cmd][vtxCount:u16][vtxData]; require the same cmd byte and a whole vertex block.
  static std::vector<u16> segCounts;
  static std::vector<u32> segOffsets;
  if (segCounts.capacity() < 2048) segCounts.reserve(2048);
  if (segOffsets.capacity() < 2048) segOffsets.reserve(2048);
  segCounts.clear();
  segOffsets.clear();
  segCounts.push_back(vtxCount);
  segOffsets.push_back(pos);
  u32 batchVtxCount = vtxCount;
  u32 scan = pos + static_cast<u32>(vtxCount) * vtxSize;
  const u16 minVtx = prim == GX_QUADS ? 4 : 3;
  const auto ts0 = timing_now();
  while (true) {
    u32 nextCmd = scan;
    while (nextCmd < size && data[nextCmd] == CP_CMD_NOP) {
      ++nextCmd;
    }
    scan = nextCmd;
    if (nextCmd + 3 > size || data[nextCmd] != cmd) {
      break;
    }
    const u16 nextVtxCount = read_u16(data + nextCmd + 1, bigEndian);
    // Cap the batch below 0xffff verts so no index collides with the 0xffff restart marker.
    if (nextVtxCount < minVtx || batchVtxCount + nextVtxCount >= 0xffffu) {
      break;
    }
    const u32 nextVtxBytes = static_cast<u32>(nextVtxCount) * vtxSize;
    if (nextCmd + 3 + nextVtxBytes > size) {
      break;
    }
    segCounts.push_back(nextVtxCount);
    segOffsets.push_back(nextCmd + 3);
    batchVtxCount += nextVtxCount;
    scan = nextCmd + 3 + nextVtxBytes;
  }
  const auto ts1 = timing_now();
  s_drainTiming.ms_scan += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

  if (sBbox.armed)
    UNLIKELY {
      for (size_t s = 0; s < segCounts.size(); ++s) {
        bbox_add_vertices(data + segOffsets[s], segCounts[s], vtxSize, fmt);
      }
    }

  ++s_geomRunsBatched;
  s_geomSegmentsBatched += segCounts.size();
  const bool stripTopology = prim == GX_TRIANGLESTRIP && kStripTopology;
  const u32 srcBytes = batchVtxCount * vtxSize;
  const u32 lastSegEnd = segOffsets.back() + static_cast<u32>(segCounts.back()) * vtxSize;

  // Content-hash key: FIFO bytes of every segment + decode layout + the run's segment
  // structure (+ indexed array contents). A hit replays cached GPU ranges with no CPU work.
  NativeGeomKey key{};
  if constexpr (kNativeGeomCacheEnabled) {
    const auto th0 = timing_now();
    const u64 fifoHash = XXH3_64bits(data + pos, lastSegEnd - pos);
    if (is_timing_enabled()) UNLIKELY s_drainTiming.fifo_hash_bytes += (lastSegEnd - pos);
    const u64 metaHash = hash_native_geom_meta(prim, fmt, vtxSize, batchVtxCount, descs,
                                               segCounts.data(), segCounts.size() * sizeof(u16));
    key = {fifoHash, metaHash, srcBytes};
    native_geom_sync_generation();
    const auto th1 = timing_now();
    s_drainTiming.ms_geom_hash += std::chrono::duration<double, std::milli>(th1 - th0).count();
    if (const auto it = s_nativeGeomCache.find(key); it != s_nativeGeomCache.end()) {
      ++s_geomCacheHits;
      pos = scan;
      submit_native_gx_draw(prim, fmt, static_cast<u16>(batchVtxCount), nativeAttrs, attrLayoutId, nativeStride, key,
                            &it->second, nullptr, 0, nullptr, 0, it->second.indexCount, it->second.indexCount != 0,
                            stripTopology);
      return true;
    }
    ++s_geomCacheMisses;
  }

  // Miss — expand every segment into flat native attributes (contiguous in one buffer).
  const auto t_miss0 = timing_now();
  const auto te0 = timing_now();
  s_nativeVtxBuf.clear();
  if (batchVtxCount > 0) {
    std::array<PreparedNativeAttr, MaxVtxAttr> fastAttrs{};
    const size_t attrCount = prepare_native_attrs(descs, fastAttrs.data());
    u8* dst = s_nativeVtxBuf.alloc_write(static_cast<size_t>(batchVtxCount) * nativeStride);
    for (size_t s = 0; s < segCounts.size(); ++s) {
      const u16 count = segCounts[s];
      const u8* segData = data + segOffsets[s];
      expand_native_vertices_into(fastAttrs.data(), attrCount, segData, vtxSize, count, dst);
      dst += static_cast<size_t>(count) * nativeStride;
    }
  }
  const auto te1 = timing_now();
  s_drainTiming.ms_geom_expand += std::chrono::duration<double, std::milli>(te1 - te0).count();
  pos = scan;

  // Index topology for the run: strips stay strips (restart-separated); a plain triangle
  // list draws non-indexed; fans/quads (and strips when strip topology is off) unroll to a
  // per-segment triangle list.
  s_nativeIdxBuf.clear();
  u32 numIndices = 0;
  bool indexed = true;
  if (stripTopology) {
    numIndices = build_triangle_strip_batch_topology_indices(s_nativeIdxBuf, segCounts);
  } else if (prim == GX_TRIANGLES) {
    indexed = false; // numIndices stays 0 -> non-indexed Draw(batchVtxCount)
  } else {
    u16 segStart = 0;
    for (const u16 count : segCounts) {
      numIndices += prepare_idx_buffer(s_nativeIdxBuf, prim, segStart, count);
      segStart = static_cast<u16>(segStart + count);
    }
  }

  submit_native_gx_draw(prim, fmt, static_cast<u16>(batchVtxCount), nativeAttrs, attrLayoutId, nativeStride, key,
                        nullptr, s_nativeVtxBuf.data(), s_nativeVtxBuf.size(),
                        indexed ? s_nativeIdxBuf.data() : nullptr, indexed ? s_nativeIdxBuf.size() : 0, numIndices,
                        indexed, stripTopology);
  s_nativeIdxBuf.clear();
  const auto t_miss1 = timing_now();
  s_drainTiming.ms_geom_miss += std::chrono::duration<double, std::milli>(t_miss1 - t_miss0).count();
  return true;
}

// Emit ONE GX_AURORA_DRAW_INDEXED normally (no merge): expand the (already deduped) vertices and use
// the supplied triangle-list indices, both served from the content-hash cache on recurrence.
static bool native_emit_indexed(GXVtxFmt fmt, u16 vtxCount, const u8* vtxData, u32 vtxSize, const u8* idxData,
                                u32 idxBytes, u32 indexCount) {
  u8 nativeStride = 0;
  const std::array<AttrConfig, MaxVtxAttr>* nativeAttrsPtr = nullptr;
  u16 attrLayoutId = 0;
  const std::vector<NativeAttrDesc>* descsPtr = nullptr;
  if (!get_native_layout(fmt, descsPtr, nativeStride, nativeAttrsPtr, attrLayoutId)) {
    return false;
  }
  const auto& nativeAttrs = *nativeAttrsPtr;
  const auto& descs = *descsPtr;
  const u32 srcBytes = static_cast<u32>(vtxCount) * vtxSize;

  NativeGeomKey key{};
  if constexpr (kNativeGeomCacheEnabled) {
    const auto th0 = timing_now();
    const u64 fifoHash = XXH3_64bits(vtxData, srcBytes);
    if (is_timing_enabled()) UNLIKELY s_drainTiming.fifo_hash_bytes += srcBytes;
    const u64 metaHash = hash_native_geom_meta(GX_TRIANGLES, fmt, vtxSize, vtxCount, descs, idxData, idxBytes);
    key = {fifoHash, metaHash, srcBytes};
    native_geom_sync_generation();
    const auto th1 = timing_now();
    s_drainTiming.ms_geom_hash += std::chrono::duration<double, std::milli>(th1 - th0).count();
    if (const auto it = s_nativeGeomCache.find(key); it != s_nativeGeomCache.end()) {
      ++s_geomCacheHits;
      submit_native_gx_draw(GX_TRIANGLES, fmt, vtxCount, nativeAttrs, attrLayoutId, nativeStride, key, &it->second,
                            nullptr, 0, nullptr, 0, it->second.indexCount, it->second.indexCount != 0,
                            /*stripTopology=*/false);
      return true;
    }
    ++s_geomCacheMisses;
  }

  s_nativeVtxBuf.clear();
  if (vtxCount > 0) {
    std::array<PreparedNativeAttr, MaxVtxAttr> fastAttrs{};
    const size_t attrCount = prepare_native_attrs(descs, fastAttrs.data());
    u8* dst = s_nativeVtxBuf.alloc_write(static_cast<size_t>(vtxCount) * nativeStride);
    expand_native_vertices_into(fastAttrs.data(), attrCount, vtxData, vtxSize, vtxCount, dst);
  }

  submit_native_gx_draw(GX_TRIANGLES, fmt, vtxCount, nativeAttrs, attrLayoutId, nativeStride, key, nullptr,
                        s_nativeVtxBuf.data(), s_nativeVtxBuf.size(), idxData, idxBytes, indexCount, indexCount != 0,
                        /*stripTopology=*/false);
  return true;
}

// Native path for GX_AURORA_DRAW_INDEXED: expand the marker's vertices into hardware attributes and
// emit one native draw (served from the content-hash cache on recurrence).
static bool try_native_draw_indexed(GXVtxFmt fmt, u16 vtxCount, const u8* vtxData, u32 vtxSize, const u8* idxData,
                                    u32 idxBytes, u32 indexCount) {
  ++s_geomDrawsIndexed;
  return native_emit_indexed(fmt, vtxCount, vtxData, vtxSize, idxData, idxBytes, indexCount);
}

std::string read_string(const u8* data, u32& pos, u32 size, bool bigEndian) {
  CHECK(pos + 2 <= size, "Aurora string length read overrun");
  const u16 length = read_u16(data + pos, bigEndian);
  pos += 2;

  CHECK(pos + length <= size, "Aurora string read overrun");
  std::string str(reinterpret_cast<const char*>(data) + pos, length);
  pos += length;
  return str;
}

static inline bool try_append_cached_static_dl_draw(u32 staticPoolRefIdx, const StaticDlDraw& d, u32 mtxIdx,
                                                    const PnMtx& curMtx, u32 curMtxGen, u64 curFastUniGen,
                                                    bool& outPreBgPopulated, bool& outPrePipeSame) noexcept {
  if (d.vtxCount == 0 || d.vtxCount > 0xFFFEu) {
    return false;
  }
  auto& b = s_pendingPnMtxBatch;
  if (!b.active) {
    return false;
  }
  if (b.nativeStride != d.nativeStride || b.attrLayoutId != d.attrLayoutId ||
      b.totalVtxCount + d.vtxCount > 0xFFFEu ||
      b.dstAlpha != g_gxState.dstAlpha || b.texMtxGen != s_splitTexMtxGen) {
    return false;
  }
  if ((b.uniFastGen != curFastUniGen || (s_lastInfo.lightingEnabled && b.lightsGen != s_splitLightsGen)) &&
      batch_uniforms_mismatch(b.uni, b.lightsGen, s_lastInfo, b.allowMultiTex) != UniMismatch::None) {
    return false;
  }
  const bool stripTopo = d.stripTopology();
  const GXPrimitive prim = d.gxPrim();
  const bool lineZero = (s_lastConfig.shaderConfig.lineMode == 0);
  if (!lineZero && b.stripTopology != stripTopo) {
    return false;
  }
  if (g_gxState.pipelineDirty != b.pipelineDirty || d.gxFmt() != b.fmt) {
    return false;
  }
  const bool topoMatchesLastCfg = ((stripTopo ? 1u : 0u) == s_lastConfig.triangleStripTopology);
  if (prim != b.prim &&
      !(lineZero && (prim == GX_TRIANGLES || prim == GX_TRIANGLESTRIP) &&
        (b.prim == GX_TRIANGLES || b.prim == GX_TRIANGLESTRIP))) {
    return false;
  }
  if (!topoMatchesLastCfg && !lineZero) {
    return false;
  }

  u32 slotIdx = 0;
  u32 texSlot = b.lastTexSlot;
  bool flTex = false;
  bool flMtx = false;
  if (!try_fit_pnmtx_tex_and_mtx(b, mtxIdx, curMtx, curMtxGen, slotIdx, texSlot, outPreBgPopulated, flTex, flMtx)) {
    outPrePipeSame = topoMatchesLastCfg;
    if (is_timing_enabled()) UNLIKELY {
      if (flMtx) ++s_drainTiming.fl_mtx;
      else if (flTex) ++s_drainTiming.fl_tex;
    }
    return false;
  }

  if (!topoMatchesLastCfg) {
    b.mixedTopology = true;
  }
  if (topoMatchesLastCfg || lineZero) {
    b.pipelineDirty = g_gxState.pipelineDirty;
    b.prim = prim;
  }
  b.uniFastGen = curFastUniGen;
  b.totalVtxCount += d.vtxCount;
  b.finalUniformDirty = g_gxState.uniformDirty;
  if (b.allCached) {
    batch_hash_append_cached(b.batchFh, b.batchMh, d.key, slotIdx, texSlot, stripTopo);
  }
  PendingPnMtxItem& item = b.items.emplace_back();
  item.refIdx = staticPoolRefIdx;
  item.vtxCount = d.vtxCount;
  item.slotIdx = static_cast<u8>(slotIdx);
  item.texSlot = static_cast<u8>(texSlot);
  item.flags = kItemStaticPool | kItemCached | (d.indexed() ? kItemIndexed : 0u) |
               (stripTopo ? kItemStripTopology : 0u);
  return true;
}

static inline u64 dl_head64(const u8* dlData, u32 dlSize) noexcept {
  u64 head64 = 0;
  if (dlData != nullptr && dlSize > 0) {
    if (dlSize >= 8) {
      std::memcpy(&head64, dlData, 8);
    } else {
      std::memcpy(&head64, dlData, dlSize);
    }
  }
  return head64;
}

static bool replay_static_dl_entry(u32 poolIdx, StaticDlEntry& entry) {
  if (!verify_dl_vtx_state(entry.vtxStateId)) {
    return false;
  }
  const u32 ptrEpoch = s_nativeGeomPtrEpoch;
  bool allCached = true;
  if (entry.cachedAllPtrEpoch != ptrEpoch) UNLIKELY {
    bool allMirror = true;
    for (u32 i = 0; i < entry.drawCount; ++i) {
      StaticDlDraw& d = (i == 0) ? entry.draw0 : s_staticDlExtraDraws[entry.extraStart + (i - 1u)];
      if (d.cachedPtrEpoch != ptrEpoch || d.cachedEntry == nullptr) {
        const auto git = s_nativeGeomCache.find(d.key);
        if (git == s_nativeGeomCache.end()) {
          allCached = false;
          break;
        }
        d.cachedEntry = &git->second;
        d.cachedPtrEpoch = ptrEpoch;
      }
      const size_t minVtxBytes = static_cast<size_t>(d.vtxCount) * d.nativeStride;
      allMirror &= (d.cachedEntry->cpuVerts.size() >= minVtxBytes) &&
                   (!d.indexed() || d.cachedEntry->cpuIndices.size() >= d.cachedEntry->indexCount);
    }
    if (allCached) {
      entry.cachedAllPtrEpoch = ptrEpoch;
      if (allMirror) {
        entry.flags |= kStaticEntryMirror;
      } else {
        entry.flags &= static_cast<u8>(~kStaticEntryMirror);
      }
    }
  }
  if (!allCached) {
    return false;
  }
  const u32 mtxIdx = std::min<u32>(g_gxState.currentPnMtx, MaxPnMtx - 1);
  const PnMtx& curMtx = g_gxState.pnMtx[mtxIdx];
  const u32 curMtxGen = s_pnMtxGen[mtxIdx];
  const u64 curFastUniGen = (static_cast<u64>(s_splitBpUniGen) << 32) | s_splitOthUniGen;
  const bool hasMirror = entry.allHaveMirror();
  const bool timing = is_timing_enabled();
  const u32 drawCount = entry.drawCount;
  const u32 extraStart = entry.extraStart;
  for (u32 i = 0; i < drawCount; ++i) {
    const u32 refIdx = (i == 0) ? poolIdx : (kStaticDrawExtraBit | (extraStart + (i - 1u)));
    StaticDlDraw& d =
        (i == 0) ? s_staticDlPool[poolIdx].draw0 : s_staticDlExtraDraws[extraStart + (i - 1u)];
    ++s_geomCacheHits;
    bool preBg = false;
    bool prePipeSame = false;
    if (hasMirror && try_append_cached_static_dl_draw(refIdx, d, mtxIdx, curMtx, curMtxGen, curFastUniGen, preBg,
                                                      prePipeSame)) {
      if (timing) UNLIKELY ++s_drainTiming.bat_ok;
      continue;
    }
    const NativeGeomEntry* ge = d.cachedEntry;
    if (ge == nullptr || d.cachedPtrEpoch != s_nativeGeomPtrEpoch) UNLIKELY {
      const auto cit = s_nativeGeomCache.find(d.key);
      if (cit == s_nativeGeomCache.end()) UNLIKELY {
        return false;
      }
      ge = &cit->second;
      d.cachedEntry = ge;
      d.cachedPtrEpoch = s_nativeGeomPtrEpoch;
    }
    submit_native_gx_draw(d.gxPrim(), d.gxFmt(), d.vtxCount, s_attrLayoutPool[d.attrLayoutId].attrs,
                          d.attrLayoutId, d.nativeStride, d.key, ge, nullptr, 0, nullptr, 0, ge->indexCount,
                          d.indexed(), d.stripTopology(), refIdx, preBg, prePipeSame);
  }
  return true;
}

static void store_static_dl_entry(const void* cacheKey, const void* dlData, u32 firstDlSize, u64 head64,
                                  const void* lastDlData, u32 lastDlSize, u64 lastHead64, uintptr_t minAddr,
                                  uintptr_t maxEnd, u32 sizeOrCount, u32 dlGen, bool isBatch, u32 usedFmtMask,
                                  std::vector<StaticDlDraw>& draws) {
  if (draws.empty()) {
    return;
  }
  u16 usedArrayMask = 0;
  bool hasMutableArr = false;
  for (u32 a = 0; a < 12; ++a) {
    const GXAttrType at = g_gxState.vtxDesc[GX_VA_POS + a];
    if (at == GX_INDEX8 || at == GX_INDEX16) {
      usedArrayMask |= static_cast<u16>(1u << a);
      const AttrArray& arr = g_gxState.arrays[GX_VA_POS + a];
      if (arr.le && !arr.immutable) {
        hasMutableArr = true;
      }
    }
  }
  if (hasMutableArr) {
    return;
  }

  const u32 ptrEpoch = s_nativeGeomPtrEpoch;
  for (auto& d : draws) {
    const auto git = s_nativeGeomCache.find(d.key);
    if (git == s_nativeGeomCache.end()) {
      return;
    }
    d.cachedEntry = &git->second;
    d.cachedPtrEpoch = ptrEpoch;
    if (git->second.indexCount != 0) {
      d.flags |= kStaticDrawIndexed;
    } else {
      d.flags &= static_cast<u8>(~kStaticDrawIndexed);
    }
  }

  if (draws.size() > 8192u) {
    return;
  }

  const u32 finalEpoch = s_nativeGeomPtrEpoch;
  bool allMirror = true;
  for (auto& d : draws) {
    if (d.cachedEntry == nullptr || d.cachedPtrEpoch != finalEpoch) {
      const auto git = s_nativeGeomCache.find(d.key);
      if (git == s_nativeGeomCache.end()) {
        return;
      }
      d.cachedEntry = &git->second;
      d.cachedPtrEpoch = finalEpoch;
    }
    const size_t minVtxBytes = static_cast<size_t>(d.vtxCount) * d.nativeStride;
    allMirror &= (d.cachedEntry->cpuVerts.size() >= minVtxBytes) &&
                 (!d.indexed() || d.cachedEntry->cpuIndices.size() >= d.cachedEntry->indexCount);
  }

  const u16 vtxStateId = intern_dl_vtx_state(current_vtx_desc_bits(), s_gxDiscArrayInvalidateGen,
                                             usedFmtMask, usedArrayMask);
  if (vtxStateId == UINT16_MAX) {
    return;
  }

  const auto existIt = s_staticDlIndex.find(cacheKey);
  const u32 existingIdx = (existIt != s_staticDlIndex.end()) ? existIt->second : UINT32_MAX;
  if (existIt != s_staticDlIndex.end()) {
    flush_pending_pnmtx_batch();
  } else if (s_staticDlPool.size() >= 16384u || s_staticDlExtraDraws.size() >= 131072u) {
    flush_pending_pnmtx_batch();
    clear_static_dl_cache();
  }
  if (s_staticDlPool.capacity() == 0) {
    s_staticDlPool.reserve(4096);
    s_staticDlExtraDraws.reserve(4096);
  }
  u32 storePoolIdx = existingIdx;
  if (storePoolIdx == UINT32_MAX) {
    storePoolIdx = static_cast<u32>(s_staticDlPool.size());
    s_staticDlPool.emplace_back();
    s_staticDlIndex.emplace(cacheKey, storePoolIdx);
  }
  StaticDlEntry& entry = s_staticDlPool[storePoolIdx];
  entry.dlData = dlData;
  entry.lastDlData = lastDlData;
  entry.minAddr = minAddr;
  entry.maxEnd = maxEnd;
  entry.head64 = head64;
  entry.lastHead64 = lastHead64;
  entry.size = sizeOrCount;
  entry.firstDlSize = firstDlSize;
  entry.lastDlSize = lastDlSize;
  entry.dlGen = dlGen;
  entry.cachedAllPtrEpoch = finalEpoch;
  entry.vtxStateId = vtxStateId;
  entry.drawCount = static_cast<u16>(draws.size());
  entry.flags = static_cast<u8>(kStaticEntryReady | (allMirror ? kStaticEntryMirror : 0u) |
                                (isBatch ? kStaticEntryBatch : 0u) | kStaticEntryCoalesced);
  entry.draw0 = draws[0];
  if (draws.size() > 1) {
    entry.extraStart = static_cast<u32>(s_staticDlExtraDraws.size());
    for (size_t i = 1; i < draws.size(); ++i) {
      s_staticDlExtraDraws.push_back(draws[i]);
    }
  } else {
    entry.extraStart = 0;
  }
  auto& fastSet = s_staticDlFast[static_dl_fast_set(cacheKey)];
  if (fastSet.ways[0].key == cacheKey) {
    fastSet.ways[0].idxPlus1 = storePoolIdx + 1u;
  } else {
    fastSet.ways[1] = fastSet.ways[0];
    fastSet.ways[0] = {cacheKey, storePoolIdx + 1u};
  }
}

static void handle_static_dl(const u8* dlData, u32 dlSize, u32 dlGen) {
  if (!sBbox.armed && g_gxState.cullMode == GX_CULL_ALL) {
    return;
  }

  native_geom_sync_generation();

  const u64 head64 = dl_head64(dlData, dlSize);

  if (!sBbox.armed && !s_staticDlCaptureActive && dlData != nullptr && dlSize > 0) {
    const u32 poolIdx = find_static_dl_pool_idx(dlData);
    if (poolIdx != UINT32_MAX) {
      StaticDlEntry& entry = s_staticDlPool[poolIdx];
      if (entry.ready() && !entry.isBatch() && entry.dlData == dlData && entry.size == dlSize &&
          entry.dlGen == dlGen && entry.head64 == head64 && entry.drawCount > 0 &&
          replay_static_dl_entry(poolIdx, entry)) {
        return;
      }
    }
  }

  const bool prevActive = s_staticDlCaptureActive;
  const bool prevAborted = s_staticDlCaptureAborted;
  auto prevDraws = std::move(s_staticDlCapturedDraws);
  const u32 prevFmtMask = s_staticDlCapturedFmtMask;
  const u8* prevDbgPtr = s_dbgStaticDlPtr;
  const u32 prevDbgSize = s_dbgStaticDlSize;
  s_dbgStaticDlPtr = dlData;
  s_dbgStaticDlSize = dlSize;

  const bool canCapture = !sBbox.armed && !prevActive && g_gxState.cullMode != GX_CULL_ALL &&
                          dlData != nullptr && dlSize > 0;
  s_staticDlCaptureActive = canCapture;
  s_staticDlCaptureAborted = false;
  s_staticDlCapturedDraws.clear();
  s_staticDlCapturedFmtMask = 0;

  process_stream(dlData, dlSize, true, false);
  s_dbgStaticDlPtr = prevDbgPtr;
  s_dbgStaticDlSize = prevDbgSize;

  const bool aborted = s_staticDlCaptureAborted;
  auto draws = std::move(s_staticDlCapturedDraws);
  const u32 usedFmtMask = s_staticDlCapturedFmtMask;

  s_staticDlCaptureActive = prevActive;
  s_staticDlCaptureAborted = prevAborted;
  s_staticDlCapturedDraws = std::move(prevDraws);
  s_staticDlCapturedFmtMask = prevFmtMask;

  if (!canCapture || aborted || draws.empty()) {
    return;
  }
  store_static_dl_entry(dlData, dlData, dlSize, head64, dlData, dlSize, head64,
                        reinterpret_cast<uintptr_t>(dlData), reinterpret_cast<uintptr_t>(dlData) + dlSize,
                        dlSize, dlGen, /*isBatch=*/false, usedFmtMask, draws);
}

static void handle_static_dl_batch(const void* batchKey, const u8* itemsData, u32 count, u32 dlGen,
                                   bool bigEndian) {
  if (!sBbox.armed && g_gxState.cullMode == GX_CULL_ALL) {
    return;
  }
  if (count == 0) {
    return;
  }

  native_geom_sync_generation();

  const u8* const firstPtr =
      reinterpret_cast<const u8*>(static_cast<uintptr_t>(read_u64(itemsData, bigEndian)));
  const u32 firstSize = read_u32(itemsData + 8u, bigEndian);
  const u8* const lastItem = itemsData + static_cast<size_t>(count - 1u) * 12u;
  const u8* const lastPtr =
      reinterpret_cast<const u8*>(static_cast<uintptr_t>(read_u64(lastItem, bigEndian)));
  const u32 lastSize = read_u32(lastItem + 8u, bigEndian);
  const u64 firstHead64 = dl_head64(firstPtr, firstSize);
  const u64 lastHead64 = dl_head64(lastPtr, lastSize);

  if (!sBbox.armed && !s_staticDlCaptureActive && batchKey != nullptr) {
    const u32 poolIdx = find_static_dl_pool_idx(batchKey);
    if (poolIdx != UINT32_MAX) {
      StaticDlEntry& entry = s_staticDlPool[poolIdx];
      if (entry.ready() && entry.isBatch() && entry.size == count && entry.dlGen == dlGen &&
          entry.dlData == firstPtr && entry.firstDlSize == firstSize && entry.head64 == firstHead64 &&
          entry.lastDlData == lastPtr && entry.lastDlSize == lastSize && entry.lastHead64 == lastHead64 &&
          entry.drawCount > 0 && replay_static_dl_entry(poolIdx, entry)) {
        return;
      }
    }
  }

  const bool prevActive = s_staticDlCaptureActive;
  const bool prevAborted = s_staticDlCaptureAborted;
  auto prevDraws = std::move(s_staticDlCapturedDraws);
  const u32 prevFmtMask = s_staticDlCapturedFmtMask;
  const u8* prevDbgPtr = s_dbgStaticDlPtr;
  const u32 prevDbgSize = s_dbgStaticDlSize;

  const bool canCapture = !sBbox.armed && !prevActive && g_gxState.cullMode != GX_CULL_ALL && batchKey != nullptr;
  s_staticDlCaptureActive = canCapture;
  s_staticDlCaptureAborted = false;
  s_staticDlCapturedDraws.clear();
  s_staticDlCapturedFmtMask = 0;

  uintptr_t minAddr = UINTPTR_MAX;
  uintptr_t maxEnd = 0;
  for (u32 i = 0; i < count; ++i) {
    const u8* item = itemsData + static_cast<size_t>(i) * 12u;
    const u8* subPtr = reinterpret_cast<const u8*>(static_cast<uintptr_t>(read_u64(item, bigEndian)));
    const u32 subSize = read_u32(item + 8u, bigEndian);
    if (subPtr != nullptr && subSize > 0) {
      const uintptr_t p = reinterpret_cast<uintptr_t>(subPtr);
      if (p < minAddr) minAddr = p;
      if (p + subSize > maxEnd) maxEnd = p + subSize;
      s_dbgStaticDlPtr = subPtr;
      s_dbgStaticDlSize = subSize;
      process_stream(subPtr, subSize, true, false);
    }
  }
  s_dbgStaticDlPtr = prevDbgPtr;
  s_dbgStaticDlSize = prevDbgSize;

  const bool aborted = s_staticDlCaptureAborted;
  auto draws = std::move(s_staticDlCapturedDraws);
  const u32 usedFmtMask = s_staticDlCapturedFmtMask;

  s_staticDlCaptureActive = prevActive;
  s_staticDlCaptureAborted = prevAborted;
  s_staticDlCapturedDraws = std::move(prevDraws);
  s_staticDlCapturedFmtMask = prevFmtMask;

  if (!canCapture || aborted || draws.empty()) {
    return;
  }
  if (minAddr == UINTPTR_MAX) {
    minAddr = 0;
  }
  store_static_dl_entry(batchKey, firstPtr, firstSize, firstHead64, lastPtr, lastSize, lastHead64,
                        minAddr, maxEnd, count, dlGen, /*isBatch=*/true, usedFmtMask, draws);
}

static void invalidate_dl_memory_range(uintptr_t rLo, uintptr_t rHi) {
  if (rHi <= rLo) {
    return;
  }
  auto overlapsRange = [rLo, rHi](const void* ptr, u32 bytes) -> bool {
    if (ptr == nullptr) {
      return false;
    }
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = bytes != 0u ? (p + static_cast<uintptr_t>(bytes)) : (p + 1u);
    return p < rHi && end > rLo;
  };
  absl::erase_if(s_arrayHashMemo, [rLo, rHi](const auto& kv) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(kv.first);
    return p >= rLo && p < rHi;
  });
  bool erasedDl = false;
  absl::erase_if(s_staticDlIndex, [rLo, rHi, &erasedDl](const auto& kv) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(kv.first);
    bool hit = (p >= rLo && p < rHi);
    if (!hit && kv.second < s_staticDlPool.size()) {
      const StaticDlEntry& e = s_staticDlPool[kv.second];
      if (e.isBatch() && e.maxEnd > e.minAddr) {
        hit = (e.minAddr < rHi && e.maxEnd > rLo);
      }
    }
    if (hit) {
      if (kv.second < s_staticDlPool.size()) {
        s_staticDlPool[kv.second].flags = 0;
        s_staticDlPool[kv.second].drawCount = 0;
      }
      erasedDl = true;
      return true;
    }
    return false;
  });
  if (erasedDl) {
    flush_pending_pnmtx_batch();
    invalidate_static_dl_fast();
  }
  bool vtxStateOverlap = false;
  for (DlVtxState& st : s_dlVtxStates) {
    u32 am = st.usedArrayMask;
    bool hit = false;
    for (u32 a = 0; am != 0u; ++a, am >>= 1u) {
      if ((am & 1u) != 0u && overlapsRange(st.arrays[a].data, st.arrays[a].size)) {
        hit = true;
        break;
      }
    }
    if (hit) {
      st.discInvalGen = UINT32_MAX - 1u;
      vtxStateOverlap = true;
    }
  }
  bool boundOverlap = false;
  for (const auto& arr : g_gxState.arrays) {
    if (overlapsRange(arr.data, arr.size)) {
      boundOverlap = true;
      break;
    }
  }
  if (boundOverlap) {
    ++s_gxDiscArrayInvalidateGen;
    ++s_gxArrayBindGen;
  }
  if (boundOverlap || vtxStateOverlap) {
    ++s_vtxStateDirtyGen;
  }
}

void handle_aurora(const u8* data, u32& pos, u32 size, bool bigEndian) {
  ZoneScoped;
  CHECK(pos + 2 <= size, "Aurora cmd read overrun");
  u16 subCmd = read_u16(data + pos, bigEndian);
  pos += 2;

  // Setting of vertex array bases.
  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    CHECK(pos + 24 <= size, "GX_AURORA_LOAD_VIEWPORT_RENDER read overrun");
    const f32 left = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 top = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 width = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 height = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 nearZ = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 farZ = read_f32(data + pos, bigEndian);
    pos += 4;
    const auto& rv = g_gxState.renderViewport;
    if (rv.left != left || rv.top != top || rv.width != width || rv.height != height ||
        rv.znear != nearZ || rv.zfar != farZ) {
      flush_pending_pnmtx_batch();
      for (int k = 0; k < 6; ++k) {
        g_gxState.xfRegValid[0x1A + k] = false;
      }
      set_render_viewport({
          .left = left,
          .top = top,
          .width = width,
          .height = height,
          .znear = nearZ,
          .zfar = farZ,
      });
      g_gxState.uniformDirty++;
      ++s_curUniState.vpProjGen;
      ++s_splitOthUniGen;
      ++s_splitOthVpGen;
    }
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    CHECK(pos + 16 <= size, "GX_AURORA_LOAD_SCISSOR_RENDER read overrun");
    const int32_t left = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t top = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t width = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t height = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    flush_pending_pnmtx_batch();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_SET_PROJECTION_SCALE_X) {
    CHECK(pos + 4 <= size, "GX_AURORA_SET_PROJECTION_SCALE_X read overrun");
    const f32 scale = read_f32(data + pos, bigEndian);
    pos += 4;
    if (g_gxState.projScaleX != scale) {
      flush_pending_pnmtx_batch();
      g_gxState.projScaleX = scale;
      g_gxState.pipelineDirty++;
      g_gxState.uniformDirty++;
      g_gxState.stateDirty = true;
      ++s_curUniState.vpProjGen;
      ++s_splitOthUniGen;
      ++s_splitOthProjGen;
    }
  } else if (subCmd == GX_AURORA_SET_PAPER) {
    CHECK(pos + 10 <= size, "GX_AURORA_SET_PAPER read overrun");
    const u8 edges = data[pos];
    pos += 1;
    const bool shadow = data[pos] != 0;
    pos += 1;
    const auto sdf = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    if (edges != g_gxState.paperEdges || shadow != g_gxState.paperShadow || sdf != g_gxState.paperSdf) {
      g_gxState.paperEdges = edges;
      g_gxState.paperShadow = shadow;
      g_gxState.paperSdf = sdf;
      u8 sdfMaps = 0;
      for (u32 i = 0; i < MaxTextures; ++i) {
        if (sdf != nullptr && g_gxState.loadedTextures[i].data == sdf) {
          sdfMaps |= static_cast<u8>(1u << i);
        }
      }
      g_gxState.paperSdfMaps = sdfMaps;
      g_gxState.pipelineDirty++;
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    CHECK(pos + 64 <= size, "GX_AURORA_LOAD_PROJECTION_FULL read overrun");
    Mat4x4<float> nextProj{};
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        nextProj[r][c] = read_f32(data + pos, bigEndian);
        pos += 4;
      }
    }
    if (std::memcmp(&g_gxState.proj, &nextProj, sizeof(nextProj)) != 0) {
      flush_pending_pnmtx_batch();
      for (int k = 0; k < 7; ++k) {
        g_gxState.xfRegValid[0x20 + k] = false;
      }
      g_gxState.proj = nextProj;
      g_gxState.uniformDirty++;
      g_gxState.stateDirty = true;
      ++s_curUniState.vpProjGen;
      ++s_splitOthUniGen;
      ++s_splitOthProjGen;
    }
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    CHECK(pos + 13 <= size, "GX_AURORA_LOAD_ARRAYBASE read overrun");
    u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;

    u64 arrayAddr = read_u64(data + pos, bigEndian);
    pos += 8;
    u32 arraySize = read_u32(data + pos, bigEndian);
    pos += 4;
    const u8 leByte = data[pos];
    const bool le = (leByte != 0);
    const bool immutable = (leByte == 2);
    pos += 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le || array.immutable != immutable) {
      if (array.le != le || (array.data == nullptr) != (newData == nullptr)) {
        g_gxState.pipelineDirty++;
        s_vatGen++;
      }
      ++s_gxArrayBindGen;
      if (attrIdx >= GX_VA_POS && attrIdx < GX_VA_POS + 12) {
        ++s_arrayStateGen[attrIdx - GX_VA_POS];
      }
      ++s_vtxStateDirtyGen;
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.immutable = immutable;
      // Only drop the cached upload when the backing array actually changes.
      array.cachedRange = {};
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    CHECK(pos + 34 <= size, "GX_AURORA_LOAD_TEXOBJ read overrun");
    const auto texMapId = data[pos];
    pos += 1;
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const void* const newData = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    const u32 newWidth = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 newHeight = read_u32(data + pos, bigEndian);
    pos += 4;
    const GXTexFmt newFormat = static_cast<GXTexFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    const GXTlut newTlut = static_cast<GXTlut>(read_u32(data + pos, bigEndian));
    pos += 4;
    const bool newMip = data[pos] != 0;
    pos += 1;
    const u32 newTexObjId = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 newTexDataVersion = read_u32(data + pos, bigEndian);
    pos += 4;
    const bool dimChanged = slot.mWidth != newWidth || slot.mHeight != newHeight;
    const bool metaChanged = slot.no_cache() || slot.data != newData || dimChanged || slot.mFormat != newFormat ||
                             slot.tlut != newTlut || ((slot.flags & 1u) != 0) != newMip ||
                             slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion;
    const u8 bit = static_cast<u8>(1u << texMapId);
    const u8 sdfMaps = g_gxState.paperSdf != nullptr && newData == g_gxState.paperSdf
                           ? static_cast<u8>(g_gxState.paperSdfMaps | bit)
                           : static_cast<u8>(g_gxState.paperSdfMaps & ~bit);
    const bool sdfChanged = sdfMaps != g_gxState.paperSdfMaps;
    if (metaChanged || sdfChanged) {
      if (metaChanged) {
        g_gxState.textureDirty++;
      }
      if (dimChanged) {
        g_gxState.uniformDirty++;
        ++s_curUniState.texDimGen[texMapId];
        ++s_splitOthUniGen;
        ++s_splitOthTexDimGen;
      }
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      if (newMip) {
        slot.flags |= 1u;
      } else {
        slot.flags &= ~1u;
      }
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      slot.set_no_cache(false); // Reset no-cache flag
      if (sdfChanged) {
        g_gxState.paperSdfMaps = sdfMaps;
        g_gxState.pipelineDirty++;
      }
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    CHECK(pos + 23 <= size, "GX_AURORA_LOAD_TLUT read overrun");
    const auto idx = data[pos];
    pos += 1;
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    const void* tlutData = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    const auto fmt = static_cast<GXTlutFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    const u16 numEntries = read_u16(data + pos, bigEndian);
    pos += 2;
    const u32 tlutObjId = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 tlutDataVersion = read_u32(data + pos, bigEndian);
    pos += 4;
    auto& slot = g_gxState.loadedTluts[idx];
    const bool tlutChanged = slot.no_cache() || slot.data != tlutData || slot.format != fmt ||
                             slot.numEntries != numEntries || slot.tlutObjId != tlutObjId ||
                             slot.tlutDataVersion != tlutDataVersion;
    if (tlutChanged) {
      g_gxState.textureDirty++;
      slot.data = tlutData;
      slot.format = fmt;
      slot.numEntries = numEntries;
      slot.tlutObjId = tlutObjId;
      slot.tlutDataVersion = tlutDataVersion;
      slot.set_no_cache(false); // Reset no-cache flag
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    CHECK(pos + 20 <= size, "GX2_SET_POLYGON_OFFSET read overrun");
    g_gxState.frontOffset = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.frontScale = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.backOffset = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.backScale = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.clamp = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.pipelineDirty++;
    g_gxState.stateDirty = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    CHECK(pos + 16 <= size, "GX_AURORA_LOAD_COPY_SRC read overrun");
    const int32_t left = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t top = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t width = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t height = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    CHECK(pos + 13 <= size, "GX_AURORA_LOAD_COPY_DST read overrun");
    g_gxState.texCopyDstWidth = read_u32(data + pos, bigEndian);
    pos += 4;
    g_gxState.texCopyDstHeight = read_u32(data + pos, bigEndian);
    pos += 4;
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    CHECK(pos + 8 <= size, "GX_AURORA_LOAD_COPY_DEST read overrun");
    g_gxState.texCopyDest = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    flush_pending_pnmtx_batch();
    gfx::depth_peek::request_snapshot();
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    CHECK(pos + 8 <= size, "GX_AURORA_BEGIN_OFFSCREEN read overrun");
    const u32 width = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 height = read_u32(data + pos, bigEndian);
    pos += 4;
    flush_pending_pnmtx_batch();
    gfx::begin_offscreen(width, height);
    g_gxState.pipelineDirty++;
    g_gxState.uniformDirty++;
    g_gxState.stateDirty = true;
    ++s_curUniState.offGen;
    ++s_splitOthUniGen;
    ++s_splitOthOffGen;
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    flush_pending_pnmtx_batch();
    gfx::end_offscreen();
    g_gxState.pipelineDirty++;
    g_gxState.uniformDirty++;
    g_gxState.stateDirty = true;
    ++s_curUniState.offGen;
    ++s_splitOthUniGen;
    ++s_splitOthOffGen;
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    CHECK(pos + 4 <= size, "GX_AURORA_DESTROY_TEXOBJ read overrun");
    evict_texture_object(read_u32(data + pos, bigEndian));
    pos += 4;
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    CHECK(pos + 4 <= size, "GX_AURORA_DESTROY_TLUT read overrun");
    evict_tlut_object(read_u32(data + pos, bigEndian));
    pos += 4;
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    CHECK(pos + 8 <= size, "GX_AURORA_DESTROY_COPY_TEX read overrun");
    flush_pending_pnmtx_batch();
    evict_copy_texture(reinterpret_cast<const void*>(read_u64(data + pos, bigEndian)));
    pos += 8;
  } else if (subCmd == GX_AURORA_ALIAS_COPY_TEX) {
    CHECK(pos + 16 <= size, "GX_AURORA_ALIAS_COPY_TEX read overrun");
    const auto* src = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    const auto* dest = reinterpret_cast<const void*>(read_u64(data + pos + 8, bigEndian));
    flush_pending_pnmtx_batch();
    alias_copy_texture(src, dest);
    pos += 16;
  } else if (subCmd == GX_AURORA_FORGET_MEMORY) {
    CHECK(pos + 16 <= size, "GX_AURORA_FORGET_MEMORY read overrun");
    const auto* lo = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    const auto* hi = reinterpret_cast<const void*>(read_u64(data + pos + 8, bigEndian));
    forget_memory(lo, hi);
    const uintptr_t loAddr = reinterpret_cast<uintptr_t>(lo);
    const uintptr_t hiAddr = reinterpret_cast<uintptr_t>(hi);
    invalidate_dl_memory_range(loAddr, hiAddr);
    pos += 16;
  } else if (subCmd == GX_AURORA_MOVE_MEMORY) {
    CHECK(pos + 24 <= size, "GX_AURORA_MOVE_MEMORY read overrun");
    const auto* from = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    const auto* to = reinterpret_cast<const void*>(read_u64(data + pos + 8, bigEndian));
    const u64 sz = read_u64(data + pos + 16, bigEndian);
    move_memory(from, to, static_cast<size_t>(sz));
    const uintptr_t fromAddr = reinterpret_cast<uintptr_t>(from);
    const uintptr_t toAddr = reinterpret_cast<uintptr_t>(to);
    const size_t moveBytes = static_cast<size_t>(sz);
    invalidate_dl_memory_range(fromAddr, fromAddr + moveBytes);
    invalidate_dl_memory_range(toAddr, toAddr + moveBytes);
    pos += 24;
  } else if (subCmd == GX_AURORA_INVAL_DISC_ARRAYS) {
    ++s_gxDiscArrayInvalidateGen;
    ++s_gxArrayBindGen;
    ++s_vtxStateDirtyGen;
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    CHECK(pos + 5 <= size, "GX_AURORA_DRAW_SIZED read overrun");
    u8 cmd = data[pos];
    pos += 1;
    u32 byteLen = read_u32(data + pos, bigEndian);
    pos += 4;
    GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calculate_last_vtx_size(fmt);
      }
      ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
             "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), data, pos, size);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    CHECK(pos + 7 <= size, "GX_AURORA_DRAW_INDEXED read overrun");
    const u8 cmd = data[pos];
    pos += 1;
    const u16 vtxCount = read_u16(data + pos, bigEndian);
    pos += 2;
    const u32 indexCount = read_u32(data + pos, bigEndian);
    pos += 4;
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
           static_cast<u32>(prim));
    const u32 idxBytes = indexCount * static_cast<u32>(sizeof(u16));
    CHECK(pos + idxBytes <= size, "GX_AURORA_DRAW_INDEXED index data overrun");
    // Index data is always host-endian. The native path pushes it (to the geometry cache
    // or the per-frame ring) itself; the storage fallback pushes it to the ring below.
    const u8* idxData = data + pos;
    pos += idxBytes;
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calculate_last_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    CHECK(pos + totalVtxBytes <= size, "GX_AURORA_DRAW_INDEXED vertex data overrun");
    const u8* vtxData = data + pos;
    pos += totalVtxBytes;
    if (sBbox.armed && indexCount != 0) {
      bbox_add_vertices(vtxData, vtxCount, vtxSize, fmt);
    }
    if (indexCount != 0 && g_gxState.cullMode != GX_CULL_ALL) {
      if (!try_native_draw_indexed(fmt, vtxCount, vtxData, vtxSize, idxData, idxBytes, indexCount)) {
        if (s_staticDlCaptureActive) {
          s_staticDlCaptureAborted = true;
        }
        flush_pending_pnmtx_batch();
        const gfx::Range idxRange = gfx::push_indices(idxData, idxBytes, 4);
        const gfx::Range vertRange = gfx::push_verts(vtxData, totalVtxBytes, 4);
        push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
      }
    }
  } else if (subCmd == GX_AURORA_CALL_STATIC_DL) {
    CHECK(pos + 16 <= size, "GX_AURORA_CALL_STATIC_DL read overrun");
    const u64 dlAddr = read_u64(data + pos, bigEndian);
    pos += 8;
    const u32 dlSize = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 dlGen = read_u32(data + pos, bigEndian);
    pos += 4;
    {
      DrainScopeTimer _t_sdl(s_drainTiming.ms_static_dl);
      handle_static_dl(reinterpret_cast<const u8*>(static_cast<uintptr_t>(dlAddr)), dlSize, dlGen);
    }
  } else if (subCmd == GX_AURORA_CALL_STATIC_DL_BATCH) {
    CHECK(pos + 16 <= size, "GX_AURORA_CALL_STATIC_DL_BATCH header overrun");
    const u64 batchAddr = read_u64(data + pos, bigEndian);
    pos += 8;
    const u32 count = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 dlGen = read_u32(data + pos, bigEndian);
    pos += 4;
    CHECK(count <= (size - pos) / 12u, "GX_AURORA_CALL_STATIC_DL_BATCH items overrun");
    const u8* itemsData = data + pos;
    pos += count * 12u;
    {
      DrainScopeTimer _t_sdl(s_drainTiming.ms_static_dl);
      handle_static_dl_batch(reinterpret_cast<const void*>(static_cast<uintptr_t>(batchAddr)), itemsData, count, dlGen,
                             bigEndian);
    }
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    flush_pending_pnmtx_batch();
    auto label = read_string(data, pos, size, bigEndian);
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    flush_pending_pnmtx_batch();
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    flush_pending_pnmtx_batch();
    auto label = read_string(data, pos, size, bigEndian);
    gfx::insert_debug_marker(std::move(label));
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

} // namespace aurora::gx::fifo

namespace aurora::gx {
void bbox_reset() noexcept {
  auto& b = fifo::sBbox;
  b.armed = true;
  b.unknown = false;
  b.left = b.top = std::numeric_limits<float>::max();
  b.right = b.bottom = -std::numeric_limits<float>::max();
  b.result[0] = 0x3FF;
  b.result[1] = 0x3FF;
  b.result[2] = 0;
  b.result[3] = 0;
}

void bbox_read(u16* left, u16* top, u16* right, u16* bottom) noexcept {
  auto& b = fifo::sBbox;
  if (b.armed && !b.unknown && b.left <= b.right && b.top <= b.bottom) {
    const auto [fbW, fbH] = logical_fb_size();
    float maxW = fbW > 0 ? static_cast<float>(fbW) : 640.0f;
    float maxH = fbH > 0 ? static_cast<float>(fbH) : 480.0f;
    float l = std::clamp(b.left, 0.0f, maxW);
    float t = std::clamp(b.top, 0.0f, maxH);
    float r = std::clamp(b.right, 0.0f, maxW);
    float bot = std::clamp(b.bottom, 0.0f, maxH);
    if (left) *left = static_cast<u16>(std::floor(l));
    if (top) *top = static_cast<u16>(std::floor(t));
    if (right) *right = static_cast<u16>(std::ceil(r));
    if (bottom) *bottom = static_cast<u16>(std::ceil(bot));
  } else {
    if (left) *left = 0x3FF;
    if (top) *top = 0x3FF;
    if (right) *right = 0;
    if (bottom) *bottom = 0;
  }
  b.armed = false;
}
} // namespace aurora::gx
