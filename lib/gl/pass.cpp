#include "pass.hpp"

#include "gl_core.hpp"
#include "state.hpp"

#include <absl/container/flat_hash_map.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// The PassEncoder issues GL directly against the render worker's current context
// (aurora's own command list already provides the WebGPU CommandEncoder ordering).
// All methods here run on the render worker.
//
// Phase 3 adds GX geometry submission: the texture/uniform bind groups, the native
// vertex layout (a VAO per present-attr set, S5 integer matrix indices) and the
// indexed/non-indexed draws. Viewport/scissor (S1b Y-flip), blend constant and the
// attribute-less clear/present Draw were wired in Phase 2.

namespace aurora::gl {
namespace {
GLenum topology_gl(PrimitiveTopology topology) {
  switch (topology) {
  case PrimitiveTopology::PointList:
    return GL_POINTS;
  case PrimitiveTopology::LineList:
    return GL_LINES;
  case PrimitiveTopology::LineStrip:
    return GL_LINE_STRIP;
  case PrimitiveTopology::TriangleList:
    return GL_TRIANGLES;
  case PrimitiveTopology::TriangleStrip:
    return GL_TRIANGLE_STRIP;
  }
  return GL_TRIANGLES;
}

// Canonical GX native-fetch attribute order, mirrored from command_processor.cpp
// build_native_layout()/canonical_attr_size(). These are GXAttr enum values; bit i of
// gl::Pipeline::vertexLayout means "attr i is present". Present attrs get packed
// `layout(location)` indices (0,1,2,... in this order) matching the GLSL emitter's
// addInput() and the CPU vertex expansion. Kept as plain ints so pass.cpp needs no GX
// headers; the enum numbering is load-bearing and fixed across the codebase.
namespace canon {
constexpr int kPnMtxIdx = 0;   // GX_VA_PNMTXIDX
constexpr int kTexMtxIdx0 = 1; // GX_VA_TEX0MTXIDX
constexpr int kTexMtxIdx7 = 8; // GX_VA_TEX7MTXIDX
constexpr int kPos = 9;        // GX_VA_POS
constexpr int kNrm = 10;       // GX_VA_NRM
constexpr int kClr0 = 11;      // GX_VA_CLR0
constexpr int kClr1 = 12;      // GX_VA_CLR1
constexpr int kTex0 = 13;      // GX_VA_TEX0
constexpr int kCount = 21;     // through GX_VA_TEX7
} // namespace canon

struct AttrSpec {
  GLint comps;
  GLenum type;
  bool normalized;
  bool integer;
  uint32_t size; // bytes in the expanded vertex record (canonical_attr_size)
};

AttrSpec attr_spec(int attr) {
  if (attr == canon::kPnMtxIdx || (attr >= canon::kTexMtxIdx0 && attr <= canon::kTexMtxIdx7)) {
    return {1, GL_UNSIGNED_INT, false, true, 4}; // Uint32 matrix index (S5)
  }
  if (attr == canon::kPos || attr == canon::kNrm) {
    return {3, GL_FLOAT, false, false, 12};
  }
  if (attr == canon::kClr0 || attr == canon::kClr1) {
    return {4, GL_UNSIGNED_BYTE, true, false, 4}; // Unorm8x4, byte0=R
  }
  return {2, GL_FLOAT, false, false, 8}; // TEXn Float32x2
}

// VAO per present-attr layout (worker-only; VAOs are container objects, not shared
// across contexts). Enable-mask + attrib formats live in the VAO; the per-draw base
// offset is re-specified each draw (the vertex data sits at a dynamic ring offset).
absl::flat_hash_map<uint32_t, GLuint> g_vaoCache;

struct GxVaoLayout {
  GLuint vao = 0;
  uint32_t stride = 0;
  GLuint ibufId = 0;
  GLuint fallbackVao = 0;
  GLuint fallbackIbufId = 0;
  uint64_t fallbackBaseOffset = UINT64_MAX;
};
absl::flat_hash_map<uint64_t, GxVaoLayout> g_gxVaoCache;

GLuint get_or_create_vao(uint32_t mask, bool& created) {
  const auto it = g_vaoCache.find(mask);
  if (it != g_vaoCache.end()) {
    created = false;
    return it->second;
  }
  GLuint vao = 0;
  gl.GenVertexArrays(1, &vao);
  g_vaoCache.emplace(mask, vao);
  created = true;
  return vao;
}

static GLuint s_lastVao = 0;
static GLuint s_lastVbuf = 0;
static uint64_t s_lastBaseOffset = UINT64_MAX;
static GLuint s_lastIbuf = 0;

void bind_gx_layout_fallback(GxVaoLayout& layout, uint32_t mask, const Buffer& vbuf, uint64_t baseOffset,
                             const Buffer& ibuf) {
  if (layout.fallbackVao == 0) {
    gl.GenVertexArrays(1, &layout.fallbackVao);
    bind_vertex_array(layout.fallbackVao);
    GLuint location = 0;
    for (int i = 0; i < canon::kCount; ++i) {
      if ((mask & (1u << i)) != 0) {
        gl.EnableVertexAttribArray(location++);
      }
    }
  } else {
    bind_vertex_array(layout.fallbackVao);
  }
  if (layout.fallbackBaseOffset == baseOffset && layout.fallbackIbufId == ibuf.id) {
    return;
  }
  gl.BindBuffer(GL_ARRAY_BUFFER, vbuf.id);
  uint32_t offset = 0;
  GLuint location = 0;
  for (int i = 0; i < canon::kCount; ++i) {
    if ((mask & (1u << i)) == 0) {
      continue;
    }
    const auto spec = attr_spec(i);
    const auto* ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(baseOffset + offset));
    if (spec.integer) {
      gl.VertexAttribIPointer(location, spec.comps, spec.type, static_cast<GLsizei>(layout.stride), ptr);
    } else {
      gl.VertexAttribPointer(location, spec.comps, spec.type, spec.normalized ? GL_TRUE : GL_FALSE,
                             static_cast<GLsizei>(layout.stride), ptr);
    }
    offset += spec.size;
    ++location;
  }
  if (ibuf.id != 0 && layout.fallbackIbufId != ibuf.id) {
    gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf.id);
    layout.fallbackIbufId = ibuf.id;
  }
  layout.fallbackBaseOffset = baseOffset;
}

struct FastGxVaoCache {
  struct Entry {
    uint64_t key = 0;
    GxVaoLayout* layout = nullptr;
  };
  Entry entries[64]{};
};
FastGxVaoCache s_fastGxVao{};

void reset_fast_gx_vao_cache() noexcept { s_fastGxVao = {}; }

GxVaoLayout& get_or_create_gx_layout(uint32_t mask, const Buffer& vbuf, const Buffer& ibuf) {
  const uint64_t key = (static_cast<uint64_t>(vbuf.id) << 32) | mask;
  const size_t slot = static_cast<size_t>(key ^ (key >> 6) ^ (key >> 32)) & 63u;
  auto& cached = s_fastGxVao.entries[slot];
  if (cached.layout != nullptr && cached.key == key) {
    return *cached.layout;
  }

  auto it = g_gxVaoCache.find(key);
  if (it != g_gxVaoCache.end()) {
    cached.key = key;
    cached.layout = &it->second;
    return it->second;
  }

  uint32_t stride = 0;
  for (int i = 0; i < canon::kCount; ++i) {
    if ((mask & (1u << i)) != 0) {
      stride += attr_spec(i).size;
    }
  }

  GLuint vao = 0;
  gl.GenVertexArrays(1, &vao);
  bind_vertex_array(vao);

  gl.BindBuffer(GL_ARRAY_BUFFER, vbuf.id);
  uint32_t offset = 0;
  GLuint location = 0;
  for (int i = 0; i < canon::kCount; ++i) {
    if ((mask & (1u << i)) == 0) {
      continue;
    }
    const auto spec = attr_spec(i);
    gl.EnableVertexAttribArray(location);
    const auto* ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(offset));
    if (spec.integer) {
      gl.VertexAttribIPointer(location, spec.comps, spec.type, static_cast<GLsizei>(stride), ptr);
    } else {
      gl.VertexAttribPointer(location, spec.comps, spec.type, spec.normalized ? GL_TRUE : GL_FALSE,
                             static_cast<GLsizei>(stride), ptr);
    }
    offset += spec.size;
    ++location;
  }
  if (ibuf.id != 0) {
    gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf.id);
  }

  auto res = g_gxVaoCache.emplace(key, GxVaoLayout{.vao = vao, .stride = stride, .ibufId = ibuf.id});
  // Reset TLS/static pointers since emplace() may rehash g_gxVaoCache.
  s_fastGxVao = {};
  s_fastGxVao.entries[slot].key = key;
  s_fastGxVao.entries[slot].layout = &res.first->second;
  return res.first->second;
}

// RmlUi geometry VAO: interleaved Rml::Vertex records (stride 20) with position
// (Float32x2 @0, loc0), tex_coord (Float32x2 @12, loc1) and colour (Unorm8x4 @8, loc2) --
// byte-identical to the WebGPU VertexBufferLayout it replaces (offsets are offsetof on
// Rml::Vertex, which orders position, colour, tex_coord). Attributes are re-pointed each
// draw because the geometry sits at a dynamic vertex-ring offset.
void bind_rml_layout(const Buffer& vbuf, uint64_t baseOffset, const Buffer& ibuf) {
  constexpr uint32_t kRmlStride = 20; // sizeof(Rml::Vertex)
  constexpr uintptr_t kPosOffset = 0;
  constexpr uintptr_t kColourOffset = 8;
  constexpr uintptr_t kTexCoordOffset = 12;
  bool created = false;
  const GLuint vao = get_or_create_vao(kRmlGeometryVertexLayout, created);
  bind_vertex_array(vao);
  gl.BindBuffer(GL_ARRAY_BUFFER, vbuf.id);
  if (created) {
    gl.EnableVertexAttribArray(0);
    gl.EnableVertexAttribArray(1);
    gl.EnableVertexAttribArray(2);
  }
  gl.VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, kRmlStride,
                         reinterpret_cast<const void*>(baseOffset + kPosOffset));
  gl.VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, kRmlStride,
                         reinterpret_cast<const void*>(baseOffset + kTexCoordOffset));
  gl.VertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, kRmlStride,
                         reinterpret_cast<const void*>(baseOffset + kColourOffset));
  gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf.id);
}
} // namespace

void reset_pass_vao_cache() noexcept {
  for (const auto& [_, layout] : g_gxVaoCache) {
    if (layout.vao != 0) {
      gl.DeleteVertexArrays(1, &layout.vao);
    }
    if (layout.fallbackVao != 0) {
      gl.DeleteVertexArrays(1, &layout.fallbackVao);
    }
  }
  g_gxVaoCache.clear();
  reset_fast_gx_vao_cache();
  for (const auto& [_, vao] : g_vaoCache) {
    if (vao != 0) {
      gl.DeleteVertexArrays(1, &vao);
    }
  }
  g_vaoCache.clear();
  s_lastVao = 0;
  s_lastVbuf = 0;
  s_lastBaseOffset = UINT64_MAX;
  s_lastIbuf = 0;
}

void PassEncoder::SetPipeline(const Pipeline& pipeline) {
  if (!m_hasPipeline || m_pipeline.program != pipeline.program) {
    use_program(pipeline.program);
  }
  if (!m_hasPipeline || !(m_pipeline.state == pipeline.state)) {
    apply_baked_state(pipeline.state);
  }
  m_pipeline = pipeline;
  m_hasPipeline = true;
}

void PassEncoder::SetUniformBuffer0(GLuint buffer, uint32_t offset, uint32_t size) {
  bind_uniform_range(0, buffer, offset, size);
}

void PassEncoder::SetBindGroup(uint32_t index, const BindingSet& set, size_t dynamicOffsetCount,
                               const uint32_t* dynamicOffsets) {
  (void)index;
  if (set.bufferCount == 0) {
    // Texture group (GX group 2 / RmlUi): unit i == GX slot i (S3). Sampler uniforms
    // were bound to these units at link time.
    bind_texture_set(set.textures.data(), set.useActiveTexMask ? set.activeTexMask : 0xFFu);
    return;
  }
  // Uniform group (GX group 1): one buffer bound with a per-draw dynamic offset.
  size_t dynIdx = 0;
  for (uint8_t j = 0; j < set.bufferCount; ++j) {
    const auto& b = set.buffers[j];
    uint32_t offset = b.offset;
    if (b.dynamic && dynIdx < dynamicOffsetCount) {
      offset += dynamicOffsets[dynIdx++];
    }
    bind_uniform_range(b.binding, b.buffer, offset, b.size);
  }
}

void PassEncoder::SetVertexBuffer(uint32_t slot, const Buffer& buffer, uint64_t offset, uint64_t size) {
  (void)slot;
  m_vertexBuffer = buffer;
  m_vertexOffset = offset;
  m_vertexSize = size;
}

void PassEncoder::SetIndexBuffer(const Buffer& buffer, IndexFormat format, uint64_t offset, uint64_t size) {
  (void)size;
  m_indexBuffer = buffer;
  m_indexFormat = format;
  m_indexOffset = offset;
}

void PassEncoder::SetViewport(float x, float y, float width, float height, float minDepth, float maxDepth) {
  // WebGPU/GX viewports are top-left origin; GL is bottom-left. Flip once, here (S1b).
  const auto gx = static_cast<GLint>(x);
  const auto gw = static_cast<GLsizei>(width);
  const auto gh = static_cast<GLsizei>(height);
  const auto gy = static_cast<GLint>(static_cast<float>(m_target.height) - (y + height));
  set_viewport_gl(gx, gy, gw, gh, minDepth, maxDepth);
}

void PassEncoder::SetScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
  const auto gy = static_cast<GLint>(m_target.height - (y + height));
  set_scissor_gl(static_cast<GLint>(x), gy, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
}

void PassEncoder::SetBlendConstant(const Color* color) {
  if (color == nullptr) {
    return;
  }
  set_blend_color(static_cast<float>(color->r), static_cast<float>(color->g), static_cast<float>(color->b),
                  static_cast<float>(color->a));
}

void PassEncoder::SetStencilReference(uint32_t reference) {
  m_stencilRef = reference;
  set_stencil_reference(reference);
}

void PassEncoder::SetLineWidth(float width) {
  set_line_width(width);
}

static uint32_t s_passStatDraws = 0;
static uint32_t s_passStatDrawArrays = 0;
static uint32_t s_passStatDrawIdx = 0;
static uint32_t s_passStatFallback = 0;
static uint32_t s_passStatBase0 = 0;
static uint32_t s_passStatBaseN = 0;
static uint32_t s_passStatEabBind = 0;
static uint32_t s_passStatPr = 0;
static uint32_t s_passStatSlow = 0;
static double s_passStatLookupMs = 0.0;
static double s_passStatGlMs = 0.0;
static double s_passStatArrMs = 0.0;
static double s_passStatB0Ms = 0.0;
static double s_passStatBNMs = 0.0;
static double s_passStatPrMs = 0.0;
static double s_passStatMaxMs = 0.0;

static inline bool pass_timing_detail() noexcept {
  static const bool enabled = std::getenv("TTYD_TIMING_DETAIL") != nullptr;
  return enabled;
}

static inline void pass_stat_tick(bool isIdx, bool fallback, bool base0, bool eabBind, bool pr,
                                  double lookupMs, double glMs) {
  ++s_passStatDraws;
  if (isIdx) ++s_passStatDrawIdx; else { ++s_passStatDrawArrays; s_passStatArrMs += glMs; }
  if (fallback) ++s_passStatFallback;
  if (isIdx && !fallback) {
    if (base0) { ++s_passStatBase0; s_passStatB0Ms += glMs; }
    else { ++s_passStatBaseN; s_passStatBNMs += glMs; }
  }
  if (pr) { ++s_passStatPr; s_passStatPrMs += glMs; }
  if (eabBind) ++s_passStatEabBind;
  if (glMs > 0.05) ++s_passStatSlow;
  if (glMs > s_passStatMaxMs) s_passStatMaxMs = glMs;
  s_passStatLookupMs += lookupMs;
  s_passStatGlMs += glMs;
  if (s_passStatDraws >= 470) {
    std::fprintf(stderr,
                 "[PASS-DETAIL/470d] arr=%u(%.2fms) b0=%u(%.2fms) bN=%u(%.2fms) pr=%u(%.2fms) fall=%u eab=%u slow=%u max=%.2fms lookup=%.2fms gl=%.2fms\n",
                 s_passStatDrawArrays, s_passStatArrMs, s_passStatBase0, s_passStatB0Ms,
                 s_passStatBaseN, s_passStatBNMs, s_passStatPr, s_passStatPrMs,
                 s_passStatFallback, s_passStatEabBind, s_passStatSlow, s_passStatMaxMs,
                 s_passStatLookupMs, s_passStatGlMs);
    s_passStatDraws = 0;
    s_passStatDrawArrays = 0;
    s_passStatDrawIdx = 0;
    s_passStatFallback = 0;
    s_passStatBase0 = 0;
    s_passStatBaseN = 0;
    s_passStatEabBind = 0;
    s_passStatPr = 0;
    s_passStatSlow = 0;
    s_passStatLookupMs = 0.0;
    s_passStatGlMs = 0.0;
    s_passStatArrMs = 0.0;
    s_passStatB0Ms = 0.0;
    s_passStatBNMs = 0.0;
    s_passStatPrMs = 0.0;
    s_passStatMaxMs = 0.0;
  }
}

void PassEncoder::Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
  (void)firstInstance;
  if (!m_hasPipeline || m_pipeline.program == 0 || vertexCount == 0) {
    return;
  }
  const bool logDetail = pass_timing_detail();
  const auto t0 = logDetail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  bool usedFallback = false;
  if (m_pipeline.vertexLayout == kRmlGeometryVertexLayout) {
    bind_rml_layout(m_vertexBuffer, m_vertexOffset, m_indexBuffer);
  } else if (m_pipeline.vertexLayout != 0) {
    auto& layout = get_or_create_gx_layout(m_pipeline.vertexLayout, m_vertexBuffer, m_indexBuffer);
    if (__builtin_expect(layout.stride > 0 && (m_vertexOffset % layout.stride == 0), 1)) {
      bind_vertex_array(layout.vao);
      firstVertex += static_cast<uint32_t>(m_vertexOffset / layout.stride);
    } else {
      bind_gx_layout_fallback(layout, m_pipeline.vertexLayout, m_vertexBuffer, m_vertexOffset, m_indexBuffer);
      usedFallback = true;
    }
  } else {
    // Attribute-less draw (clear / present / RmlUi fullscreen triangle via gl_VertexID).
    bind_vertex_array(0);
  }
  const auto t1 = logDetail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const GLenum mode = topology_gl(m_pipeline.state.topology);
  if (instanceCount > 1) {
    gl.DrawArraysInstanced(mode, static_cast<GLint>(firstVertex), static_cast<GLsizei>(vertexCount),
                           static_cast<GLsizei>(instanceCount));
  } else {
    gl.DrawArrays(mode, static_cast<GLint>(firstVertex), static_cast<GLsizei>(vertexCount));
  }
  if (logDetail) [[unlikely]] {
    const auto t2 = std::chrono::steady_clock::now();
    pass_stat_tick(false, usedFallback, false, false, m_pipeline.state.primitiveRestart,
                   std::chrono::duration<double, std::milli>(t1 - t0).count(),
                   std::chrono::duration<double, std::milli>(t2 - t1).count());
  }
}

void PassEncoder::DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex,
                              uint32_t firstInstance) {
  (void)baseVertex; // native fetch bakes the vertex base into the attrib pointers
  (void)firstInstance;
  if (!m_hasPipeline || m_pipeline.program == 0 || indexCount == 0) {
    return;
  }
  const bool logDetail = pass_timing_detail();
  const auto t0 = logDetail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const GLenum mode = topology_gl(m_pipeline.state.topology);
  const bool u32 = m_indexFormat == IndexFormat::Uint32;
  const GLenum idxType = u32 ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT;
  // glDrawElements' last arg is a byte offset into the element buffer bound in the VAO.
  const uintptr_t idxBytes = m_indexOffset + static_cast<uintptr_t>(firstIndex) * (u32 ? 4u : 2u);
  const auto* idxPtr = reinterpret_cast<const void*>(idxBytes);

  bool usedFallback = false;
  bool base0 = false;
  GLuint rangeEnd = 0;
  bool eabBind = false;
  if (m_pipeline.vertexLayout == kRmlGeometryVertexLayout) {
    bind_rml_layout(m_vertexBuffer, m_vertexOffset, m_indexBuffer);
  } else {
    auto& layout = get_or_create_gx_layout(m_pipeline.vertexLayout, m_vertexBuffer, m_indexBuffer);
    const bool aligned = layout.stride > 0 && (m_vertexOffset % layout.stride == 0);
    const bool canBaseVertex = aligned && (m_vertexOffset == 0 ||
        (instanceCount > 1 ? gl.DrawElementsInstancedBaseVertex != nullptr : gl.DrawElementsBaseVertex != nullptr));
    if (__builtin_expect(canBaseVertex, 1)) {
      bind_vertex_array(layout.vao);
      if (m_indexBuffer.id != 0 && layout.ibufId != m_indexBuffer.id) {
        gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer.id);
        layout.ibufId = m_indexBuffer.id;
        eabBind = true;
      }
      const GLint baseVert = m_vertexOffset == 0 ? 0 : static_cast<GLint>(m_vertexOffset / layout.stride);
      base0 = (baseVert == 0);
      // The draw's indices address the bound vertex slice from its start (the base vertex), so
      // they lie in [0, slice vertices - 1]: given to the driver, it doesn't scan the indices for
      // their range on the CPU (the Mali blob did, at ~55 us a draw: most of the render worker's
      // time in Rogueport). $TTYD_GL_NO_RANGE=1: the plain draws, as before.
      static const bool s_noRange = std::getenv("TTYD_GL_NO_RANGE") != nullptr;
      rangeEnd = 0;
      if (!s_noRange && instanceCount <= 1 && m_vertexSize != kWholeSize && layout.stride > 0 &&
          m_vertexSize >= layout.stride) {
        rangeEnd = static_cast<GLuint>(m_vertexSize / layout.stride - 1);
      }
      if (baseVert != 0) {
        const auto t1 = logDetail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (instanceCount > 1) {
          gl.DrawElementsInstancedBaseVertex(mode, static_cast<GLsizei>(indexCount), idxType, idxPtr,
                                             static_cast<GLsizei>(instanceCount), baseVert);
        } else if (rangeEnd != 0 && gl.DrawRangeElementsBaseVertex != nullptr) {
          gl.DrawRangeElementsBaseVertex(mode, 0, rangeEnd, static_cast<GLsizei>(indexCount), idxType, idxPtr,
                                         baseVert);
        } else {
          gl.DrawElementsBaseVertex(mode, static_cast<GLsizei>(indexCount), idxType, idxPtr, baseVert);
        }
        if (logDetail) [[unlikely]] {
          const auto t2 = std::chrono::steady_clock::now();
          pass_stat_tick(true, false, false, eabBind, m_pipeline.state.primitiveRestart,
                         std::chrono::duration<double, std::milli>(t1 - t0).count(),
                         std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        return;
      }
    } else {
      bind_gx_layout_fallback(layout, m_pipeline.vertexLayout, m_vertexBuffer, m_vertexOffset, m_indexBuffer);
      usedFallback = true;
    }
  }
  const auto t1 = logDetail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  if (instanceCount > 1) {
    gl.DrawElementsInstanced(mode, static_cast<GLsizei>(indexCount), idxType, idxPtr,
                             static_cast<GLsizei>(instanceCount));
  } else if (base0 && !usedFallback && rangeEnd != 0 && gl.DrawRangeElements != nullptr) {
    gl.DrawRangeElements(mode, 0, rangeEnd, static_cast<GLsizei>(indexCount), idxType, idxPtr);
  } else {
    gl.DrawElements(mode, static_cast<GLsizei>(indexCount), idxType, idxPtr);
  }
  if (logDetail) [[unlikely]] {
    const auto t2 = std::chrono::steady_clock::now();
    pass_stat_tick(true, usedFallback, base0, eabBind, m_pipeline.state.primitiveRestart,
                   std::chrono::duration<double, std::milli>(t1 - t0).count(),
                   std::chrono::duration<double, std::milli>(t2 - t1).count());
  }
}

void PassEncoder::End() {
  // Store-op handling (StoreOp::Discard -> glInvalidateFramebuffer, S7) is applied in
  // gfx::common render() where the pass's per-attachment store ops are known.
}

// KHR_debug markers (glPushDebugGroup source enum + gating) land with the debug-
// output plumbing in a later phase; these are gated behind AURORA_GFX_DEBUG_GROUPS
// at the call sites and stay no-ops for now.
void PassEncoder::PushDebugGroup(const char* label) { (void)label; }
void PassEncoder::PopDebugGroup() {}
void PassEncoder::InsertDebugMarker(const char* label) { (void)label; }

} // namespace aurora::gl
