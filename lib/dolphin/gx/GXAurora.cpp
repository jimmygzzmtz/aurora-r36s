#include "dolphin/gx/GXAurora.h"

#include <limits>

#include "__gx.h"
#include "gx.hpp"
#include "../../window.hpp"

#include "../../gfx/common.hpp"
#include "../../gx/fifo.hpp"

static void GXWriteString(const char* label) {
  auto length = strlen(label);

  if (length > std::numeric_limits<u16>::max()) {
    Log.warn("Debug marker size over u16 max, truncating");
    length = std::numeric_limits<u16>::max();
  }

  GX_WRITE_U16(length);
  GX_WRITE_DATA(label, length);
}

void GXPushDebugGroup(const char* label) {
  GX_WRITE_AURORA(GX_AURORA_DEBUG_GROUP_PUSH);
  GXWriteString(label);
}

void GXPopDebugGroup() { GX_WRITE_AURORA(GX_AURORA_DEBUG_GROUP_POP); }

void GXInsertDebugMarker(const char* label) {
  GX_WRITE_AURORA(GX_AURORA_DEBUG_MARKER_INSERT);
  GXWriteString(label);
}

void AuroraSetViewportPolicy(AuroraViewportPolicy policy) {
  g_gxState.viewportPolicy = policy;
  aurora::window::set_frame_buffer_aspect_fit(policy == AURORA_VIEWPORT_FIT);
}

void AuroraSetDisplayAspect(f32 aspect) {}

void AuroraGetRenderSize(u32* width, u32* height) {
  const auto windowSize = aurora::window::get_window_size();
  if (width != nullptr) {
    *width = windowSize.fb_width;
  }
  if (height != nullptr) {
    *height = windowSize.fb_height;
  }
}

void GXSetViewportRender(f32 left, f32 top, f32 wd, f32 ht, f32 nearz, f32 farz) {
  if (!aurora::gx::fifo::in_display_list()) {
    __gxShadow.vpEpoch = 0;
  }
  GX_WRITE_AURORA(GX_AURORA_LOAD_VIEWPORT_RENDER);
  GX_WRITE_F32(left);
  GX_WRITE_F32(top);
  GX_WRITE_F32(wd);
  GX_WRITE_F32(ht);
  GX_WRITE_F32(nearz);
  GX_WRITE_F32(farz);
}

void GXSetScissorRender(u32 left, u32 top, u32 wd, u32 ht) {
  if (!aurora::gx::fifo::in_display_list()) {
    __gxShadow.suScisEpoch = 0;
  }
  GX_WRITE_AURORA(GX_AURORA_LOAD_SCISSOR_RENDER);
  GX_WRITE_U32(left);
  GX_WRITE_U32(top);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
}

void GXSetProjectionFull(const void* mtx) {
  if (!aurora::gx::fifo::in_display_list()) {
    __gxShadow.projEpoch = 0;
  }
  const f32* values = reinterpret_cast<const f32*>(mtx);
  GX_WRITE_AURORA(GX_AURORA_LOAD_PROJECTION_FULL);
  for (int i = 0; i < 16; ++i) {
    GX_WRITE_F32(values[i]);
  }
}

void GXSetProjectionScaleX(f32 scale) {
  GX_WRITE_AURORA(GX_AURORA_SET_PROJECTION_SCALE_X);
  GX_WRITE_F32(scale);
}

// aurora/paper.h: kept here, sent whole on each change
#include "aurora/paper.h"
static u8 s_paperEdges = 0;
static u8 s_paperShadow = 0;
static const void* s_paperSdf = nullptr;
static void write_paper() {
  GX_WRITE_AURORA(GX_AURORA_SET_PAPER);
  GX_WRITE_U8(s_paperEdges);
  GX_WRITE_U8(s_paperShadow);
  GX_WRITE_U64(reinterpret_cast<u64>(s_paperSdf));
}
void aurora_set_paper_edges(int on) {
  if (s_paperEdges != (on != 0)) {
    s_paperEdges = on != 0;
    write_paper();
  }
}
void aurora_set_paper_distance_field(const void* texels) {
  if (s_paperSdf != texels) {
    s_paperSdf = texels;
    write_paper();
  }
}
void aurora_set_paper_shadow(int on) {
  if (s_paperShadow != (on != 0)) {
    s_paperShadow = on != 0;
    write_paper();
  }
}

void GX2SetPolygonOffset(f32 mFrontOffset, f32 mFrontScale, f32 mBackOffset, f32 mBackScale, f32 mClamp) {
  GX_WRITE_AURORA(GX2_SET_POLYGON_OFFSET);
  GX_WRITE_F32(mFrontOffset);
  GX_WRITE_F32(mFrontScale);
  GX_WRITE_F32(mBackOffset);
  GX_WRITE_F32(mBackScale);
  GX_WRITE_F32(mClamp);
}

void GXCreateFrameBuffer(u32 width, u32 height) {
  GX_WRITE_AURORA(GX_AURORA_BEGIN_OFFSCREEN);
  GX_WRITE_U32(width);
  GX_WRITE_U32(height);
}

void GXRestoreFrameBuffer() {
  GX_WRITE_AURORA(GX_AURORA_END_OFFSCREEN);
}

// LM compatibility stubs for GXAurora
extern "C" {
void AuroraCaptureBegin(void* buf, uint32_t size) { (void)buf; (void)size; }
uint32_t AuroraCaptureEnd() { return 0; }
uint32_t AuroraCapturePos() { return 0; }
void AuroraCaptureReplay(const void* buf, uint32_t size) { (void)buf; (void)size; }
void AuroraReplay(const void* data, uint32_t size) { (void)data; (void)size; }
void AuroraCaptureJoin(uint8_t join) { (void)join; }
uint64_t AuroraFifoStreamPos(void) { return 0; }
void AuroraSetFifoErrorCallback(AuroraFifoErrorCallback callback) { (void)callback; }
void AuroraSetDrawProbe(AuroraDrawProbeFn fn) { (void)fn; }
void AuroraGXSync(void) {
  if (!aurora::gx::fifo::in_display_list() && aurora::gfx::in_frame()) {
    aurora::gx::fifo::drain();
  }
}
void AuroraGXKick(uint32_t minBytes) { aurora::gx::fifo::kick_if_over(minBytes); }

void GXSetDrawClassAurora(uint8_t drawClass) { (void)drawClass; }
void GXSceneEndAurora(void) {}
void GXSetWireframeOverrideAurora(uint8_t mode) { (void)mode; }
uint8_t GXGetWireframeOverrideAurora(void) { return 0; }
void GXSetWireframeAurora(uint8_t enable, uint32_t rgba) { (void)enable; (void)rgba; }
void GXSetCullOverrideAurora(uint8_t twoSided) { (void)twoSided; }
void GXSaveStateAurora(uint8_t slot) { (void)slot; }
void GXRestoreStateAurora(uint8_t slot) { (void)slot; }
void GXSetCopyGenerationAurora(uint8_t generationAndFlags) { (void)generationAndFlags; }
void GXSetProbeTagAurora(uint32_t tag) { (void)tag; }
void GXSetMaterialHintAurora(uint8_t hint) { (void)hint; }
void GXSetCopyClamp(GXFBClamp clamp) { (void)clamp; }
void GXSetShadowPassAurora(uint8_t pass) { (void)pass; }
void GXSetAuroraAccuracy(uint32_t flags) { (void)flags; }
uint32_t GXGetAuroraAccuracy(void) { return 0; }
void GXSetAuroraTexGenBeyondCount(uint32_t mode) { (void)mode; }
}

namespace aurora::gx::fifo {
uint64_t stream_position() { return 0; }
uint64_t processed_position() { return 0; }
uint64_t published_position() { return 0; }
}

namespace aurora::gfx {
struct ExtraLight;
void set_extra_lights(const ExtraLight* lights, uint32_t count) { (void)lights; (void)count; }
}

void AuroraForgetMemory(const void* lo, const void* hi) {
  if (lo == nullptr || hi <= lo) {
    return;
  }
  __gx_shadow_invalidate_tex();
  __gx_shadow_invalidate_arrays();
  __gx_dl_forget_memory(lo, hi);
  GX_WRITE_AURORA(GX_AURORA_FORGET_MEMORY);
  GX_WRITE_U64(reinterpret_cast<u64>(lo));
  GX_WRITE_U64(reinterpret_cast<u64>(hi));
}

void AuroraMoveMemory(const void* from, const void* to, u32 size) {
  if (from == nullptr || to == nullptr || from == to || size == 0) {
    return;
  }
  __gx_shadow_invalidate_tex();
  __gx_shadow_invalidate_arrays();
  __gx_dl_forget_memory(from, static_cast<const u8*>(from) + size);
  __gx_dl_forget_memory(to, static_cast<const u8*>(to) + size);
  GX_WRITE_AURORA(GX_AURORA_MOVE_MEMORY);
  GX_WRITE_U64(reinterpret_cast<u64>(from));
  GX_WRITE_U64(reinterpret_cast<u64>(to));
  GX_WRITE_U64(static_cast<u64>(size));
}

void AuroraInvalidateDiscArrays(void) {
  __gx_shadow_invalidate_arrays();
  __gx_dl_invalidate_cache();
  GX_WRITE_AURORA(GX_AURORA_INVAL_DISC_ARRAYS);
}

#include "aurora/display.h"

extern "C" {
void GXSetProjectionTransform(f32 scaleX, f32 scaleY, f32 offsetX, f32 offsetY) {
  (void)scaleX; (void)scaleY; (void)offsetX; (void)offsetY;
}
void AuroraSetTextureAnisotropy(u8 maxAniso) { (void)maxAniso; }
void AuroraSetTextureLodBias(f32 bias) { (void)bias; }
void AuroraSetSharpTextures(GXBool sharp) { (void)sharp; }
void AuroraSetPresentBackdrop(GXBool enabled) { (void)enabled; }
}

extern "C" {
AuroraReadbackStream AuroraReadbackCreateStream(void) {
  static AuroraReadbackStream s_nextStream = 1;
  return s_nextStream++;
}

void AuroraReadbackDestroyStream(AuroraReadbackStream stream) {
  (void)stream;
}

GXBool AuroraReadbackCaptureCopies(AuroraReadbackStream stream, const AuroraReadbackCopySrc* srcs, u32 count) {
  (void)stream; (void)srcs; (void)count;
  return GX_FALSE;
}

GXBool AuroraReadbackCapturePeeks(AuroraReadbackStream stream, const AuroraReadbackPeekRequest* peeks, u32 count) {
  (void)stream; (void)peeks; (void)count;
  return GX_FALSE;
}

GXBool AuroraReadbackGet(AuroraReadbackStream stream, AuroraReadbackDest* dests, u32 count) {
  (void)stream;
  if (dests != nullptr) {
    for (u32 i = 0; i < count; ++i) {
      dests[i].found = GX_FALSE;
    }
  }
  return GX_FALSE;
}
}

extern "C" {
void AuroraSetPresentArea(float left, float top, float right, float bottom) {
  (void)left; (void)top; (void)right; (void)bottom;
}

void AuroraSetDrawsBlocking(bool blocking) {
  (void)blocking;
}

void AuroraFlushTextureCaches(void) {
  if (aurora::gx::fifo::threaded()) {
    aurora::gx::fifo::drain(); // (the GX thread owns these caches while it runs)
  }
  __gx_shadow_invalidate_tex();
  aurora::gx::clear_static_texture_cache();
  aurora::gx::clear_copy_texture_cache();
}
}
