#include <aurora/aurora.h>

#ifdef AURORA_ENABLE_GX
#include "gfx/common.hpp"
#include "gfx/render_worker.hpp"
#include "gx/fifo.hpp"

namespace aurora::gx {
// Declared in gx/gx.hpp, which this TU doesn't otherwise need.
void expire_texture_object_caches() noexcept;
} // namespace aurora::gx
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"
#include "webgpu/sdl2shim_present.hpp"
#include "gl/pass.hpp"
#include "gl/state.hpp"
#include <optional>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "gamepad.hpp"
#include "internal.hpp"
#include "window.hpp"
#include <aurora/time.hpp>

#include <SDL3/SDL_filesystem.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#include <magic_enum.hpp>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
Module Log("aurora");

#ifdef AURORA_ENABLE_GX
// The present-blit viewport helper (set_present_viewport) and its scissor clamp are
// gone with the wgpu Surface present path; Phase 2 reintroduces viewport setting on
// the GL present blit via gl::PassEncoder.
#endif

#ifdef AURORA_ENABLE_GX
// The hand-rolled backend serves OpenGLES; NULL is the SDL_Renderer fallback for
// headless / imgui-only. The Dawn D3D/Metal/Vulkan/WebGPU options are gone.
constexpr std::array PreferredBackendOrder{
    BACKEND_OPENGLES,
    BACKEND_NULL,
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_INFO, "Aurora", "aurora::initialize: desiredBackend=%d", (int)selectedBackend);
#endif
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
#ifdef __ANDROID__
      __android_log_print(ANDROID_LOG_ERROR, "Aurora", "aurora::initialize: webgpu::initialize failed for desiredBackend=%d", (int)selectedBackend);
#endif
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
#ifdef __ANDROID__
      __android_log_print(ANDROID_LOG_INFO, "Aurora", "aurora::initialize: trying backend %d", (int)selectedBackend);
#endif
      if (!window::create_window(selectedBackend)) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "Aurora", "aurora::initialize: create_window failed for backend %d: %s", (int)selectedBackend, SDL_GetError());
#endif
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "Aurora", "aurora::initialize: webgpu::initialize SUCCEEDED for backend %d", (int)selectedBackend);
#endif
        break;
      } else {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "Aurora", "aurora::initialize: webgpu::initialize FAILED for backend %d: %s", (int)selectedBackend, SDL_GetError());
#endif
        window::destroy_window();
      }
    }
  }

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when there is no GL backend (headless/NULL).
  if (webgpu::g_backendType == BACKEND_NULL) {
    ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();

#ifdef AURORA_ENABLE_GX
  gfx::initialize();

  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
#ifdef AURORA_ENABLE_GX
  // gpu_synchronize (not render_worker::synchronize) so a worker parked on an EFB slot is released.
  gfx::gpu_synchronize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  gamepad::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    gamepad::initialize();
  }
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  {
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    // The GL backend has no wgpu::Surface: it renders offscreen and presents by
    // swapping the window's default framebuffer (desktop) or handing an EFB slot to
    // the main thread (device). Both are always ready once the context exists.
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    return false;
  }
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  auto t0 = std::chrono::steady_clock::now();
  gx::fifo::drain();
  auto t1 = std::chrono::steady_clock::now();
  const double d_wait = 0.0;
  // After the drain so this frame's binds are stamped/current; before gfx::end_frame so
  // any evicted textures land in this frame's deferred-destroy snapshot.
  gx::expire_texture_object_caches();
  gfx::finish();
  auto t2 = std::chrono::steady_clock::now();
  auto t_pres0 = std::chrono::steady_clock::now();
  if (webgpu::sdl2shim_present::active()) {
    webgpu::sdl2shim_present::flush_present();
  }
  auto t_pres1 = std::chrono::steady_clock::now();
  auto imguiDrawData = imgui::freeze();

  const auto& presentSource = webgpu::present_source();
  // Size the UI from the NATIVE present surface (window drawable), not surfaceConfiguration (the
  // internalResolutionScale-downscaled EFB). Otherwise the UI would render at the game's internal
  // resolution and get upscaled with the scene -- at 0.5x that made the UI huge and cramped. This
  // is the same centered content rect composite_ui_overlay/present_frame blit into, so the UI
  // target maps 1:1 and stays crisp and correctly-sized at any internal resolution scale.
  const auto viewport = webgpu::calculate_present_viewport(webgpu::present_surface_width(),
                                                           webgpu::present_surface_height(),
                                                           presentSource.size.width, presentSource.size.height);

  gl::Texture rmlTexture{};
  gl::Sampler rmlSampler{};
  bool rmlOverlay = false;
#if AURORA_ENABLE_RMLUI
  if (rmlui::is_initialized()) {
    auto rmlFrame = rmlui::record_frame(viewport);
    rmlTexture = rmlFrame.texture;
    rmlSampler = rmlFrame.sampler;
    rmlOverlay = rmlFrame.overlay;
  }
#endif

  gfx::end_frame([rmlTexture, rmlSampler, rmlOverlay, imguiDrawData = std::move(imguiDrawData)]() {
    // Runs on the render worker (which owns the GL context). Composite the finished scene, the
    // RmlUi overlay and the imgui overlay into the present target, then swap (desktop) or hand the
    // EFB slot to the main thread (device).
    if (webgpu::sdl2shim_present::active()) {
      // Device (SDL2-shim EFB path): acquire a slot, composite into its worker FBO, publish. The
      // main thread scans it out via flush_present() and owns after_present(); the worker never
      // swaps (Mali kmsdrm page flips are display-thread-bound -- a worker swap shows nothing).
      auto frame = webgpu::sdl2shim_present::acquire();
      if (frame) {
        const uint32_t targetFbo = frame->fbo;
        webgpu::present_frame(targetFbo);
        if (rmlTexture.id != 0) {
          webgpu::composite_ui_overlay(rmlTexture, rmlSampler, rmlOverlay, targetFbo);
        }
        {
          gl::PassEncoder uiPass(gl::PassTarget{
              .fbo = targetFbo,
              .width = webgpu::present_surface_width(),
              .height = webgpu::present_surface_height(),
          });
          imgui::render(uiPass, imguiDrawData);
        }
        // The present composite issued raw, out-of-band GL; drop the state-cache shadow so the next
        // frame's first pass re-establishes everything (desktop does this inside present_swap()).
        gl::reset_state_cache();
        webgpu::sdl2shim_present::publish(frame->slot);
      } else {
        webgpu::sdl2shim_present::publish_empty();
      }
    } else {
      // Desktop: composite into the window's default framebuffer and swap from the worker.
      webgpu::present_frame();
      if (rmlTexture.id != 0) {
        webgpu::composite_ui_overlay(rmlTexture, rmlSampler, rmlOverlay);
      }
      {
        gl::PassEncoder uiPass(gl::PassTarget{
            .fbo = 0,
            .width = webgpu::g_graphicsConfig.surfaceConfiguration.width,
            .height = webgpu::g_graphicsConfig.surfaceConfiguration.height,
        });
        imgui::render(uiPass, imguiDrawData);
      }
      webgpu::present_swap();
      gfx::after_present();
    }
    gfx::after_submit();

    TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

    TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(gfx::g_stats.queuedPipelines));
    TracyPlot("aurora: createdPipelines", static_cast<int64_t>(gfx::g_stats.createdPipelines));
    TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
    TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
    TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
    TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
    TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
    TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(gfx::g_stats.lastTextureUploadSize));
  });

  auto t3 = std::chrono::steady_clock::now();
  static const bool s_logTiming = std::getenv("TTYD_TIMING_LOG") != nullptr;
  static int s_aurora_prof = 0;
  if (s_logTiming && ++s_aurora_prof >= 30) {
    s_aurora_prof = 0;
    const double d_drain = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double d_finish = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double d_pres = std::chrono::duration<double, std::milli>(t_pres1 - t_pres0).count();
    const double d_end = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::fprintf(stdout, "[AURORA-SUB] drain: %.1fms (wait: %.1fms) | pres: %.1fms | finish: %.1fms | gfx_end: %.1fms\n",
                 d_drain, d_wait, d_pres, d_finish, d_end);
    std::fflush(stdout);
  }

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
void aurora_set_frame_buffer_scale(float scale) {
  aurora::window::set_frame_buffer_scale(scale);
}
AuroraWindowSize aurora_window_size() {
  return aurora::window::get_window_size();
}
void aurora_apply_frame_buffer_resize() {}
void aurora_gpu_frame_time(uint64_t* lastNs, uint64_t* meanNs, uint64_t* maxNs, uint64_t* count) {
  if (lastNs != nullptr) *lastNs = 0;
  if (meanNs != nullptr) *meanNs = 0;
  if (maxNs != nullptr) *maxNs = 0;
  if (count != nullptr) *count = 0;
}
void aurora_capture_frame(const char* path) {
  (void)path;
}
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
void aurora_flush_present() {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::sdl2shim_present::flush_present();
#endif
}
bool aurora_try_flush_present() {
#ifdef AURORA_ENABLE_GX
  return aurora::webgpu::sdl2shim_present::try_flush_present();
#else
  return false;
#endif
}
void aurora_set_double_buffering(bool enable) {
#ifdef AURORA_ENABLE_GX
  aurora::gx::fifo::set_double_buffering(enable);
#else
  (void)enable;
#endif
}
bool aurora_get_double_buffering() {
#ifdef AURORA_ENABLE_GX
  return aurora::gx::fifo::double_buffering_enabled();
#else
  return false;
#endif
}
void aurora_fifo_finish_frame_producer() {
#ifdef AURORA_ENABLE_GX
  aurora::gx::fifo::finish_frame_producer();
#endif
}
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
void aurora_set_timescale(float scale) { aurora::time::set_scale(scale); }
float aurora_get_timescale() { return aurora::time::scale(); }

void aurora_set_resampler(AuroraSampler sampler) {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::set_resampler(sampler);
#else
  (void)sampler;
#endif
}

bool aurora_present_waits_for_vblank() {
  return aurora::g_config.vsync;
}

extern "C" void aurora_gfx_pool_stats(uint32_t* peakBytes, uint32_t* reservedBytes, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (peakBytes != nullptr) peakBytes[i] = 0;
    if (reservedBytes != nullptr) reservedBytes[i] = 0;
  }
}

extern "C" void aurora_gfx_texture_stats(uint64_t* srcBytes, uint64_t* uploadedBytes, uint64_t* count) {
  if (srcBytes != nullptr) *srcBytes = 0;
  if (uploadedBytes != nullptr) *uploadedBytes = 0;
  if (count != nullptr) *count = 0;
}


// LM compatibility stubs for headless / GLES port
extern "C" {
SDL_Window* aurora_get_window() { return aurora::window::get_sdl_window(); }
AuroraWindowSize aurora_get_window_size() {
  return aurora::window::get_sdl_window() != nullptr ? aurora::window::get_window_size() : AuroraWindowSize{};
}
uint32_t aurora_get_msaa_support() { return 1; }
uint32_t aurora_get_msaa() { return 1; }
void aurora_request_screenshot(const char* path) { (void)path; }
void aurora_present_timing_enable(bool enable) { (void)enable; }
void aurora_set_gpu_keep_warm(uint32_t units) { (void)units; }
uint64_t aurora_frame_counter() { static uint64_t s_fc = 0; return ++s_fc; }
size_t aurora_present_timings(void* out, size_t max) { (void)out; (void)max; return 0; }
uint64_t aurora_present_timing_timeouts() { return 0; }
size_t aurora_describe_activity(char* out, size_t size, uint32_t minMs) {
  if (out && size > 0) out[0] = 0;
  return 0;
}
bool aurora_wait_frame_done(uint64_t frame, int64_t timeoutNs) { (void)frame; (void)timeoutNs; return true; }
void aurora_set_thread_hooks(void* start, void* tick) { (void)start; (void)tick; }
void aurora_imgui_set_frame_callback(void* callback) { (void)callback; }

// Enhanced lighting & efb_ram stubs
void aurora_efb_ram_set_enabled(bool enabled) { (void)enabled; }
bool aurora_efb_ram_enabled(void) { return false; }
void aurora_efb_ram_set_scaled_depth_reads(bool enabled) { (void)enabled; }
void aurora_efb_ram_set_readback_all(bool enabled) { (void)enabled; }
void aurora_efb_ram_cpu_read(const void* ptr, uint32_t size) { (void)ptr; (void)size; }
void aurora_efb_ram_guest_copy(const void* dest, uint32_t width, uint32_t height, uint32_t format) {
  (void)dest; (void)width; (void)height; (void)format;
}
void aurora_efb_ram_watch(const void* ptr, uint32_t size) { (void)ptr; (void)size; }
uint32_t aurora_efb_ram_apply(void) { return 0; }
void aurora_efb_ram_set_writeback_callback(void* fn, void* userdata) { (void)fn; (void)userdata; }
void aurora_efb_ram_get_stats(void* out) {
  if (out) std::memset(out, 0, 64);
}

void aurora_enhanced_set_config(const void* config) { (void)config; }
void aurora_enhanced_beam_flare(float strength) { (void)strength; }
uint32_t aurora_enhanced_last_light_count(void) { return 0; }
void aurora_enhanced_shadow_stats(uint32_t* lights, uint32_t* casters, uint32_t* skipped) {
  if (lights) *lights = 0;
  if (casters) *casters = 0;
  if (skipped) *skipped = 0;
}
void aurora_postfx_set_chain(const char* wgsl, const void* passes, uint32_t passCount) {
  (void)wgsl; (void)passes; (void)passCount;
}
void aurora_postfx_set_uniforms(const void* data, uint32_t size) { (void)data; (void)size; }
bool aurora_postfx_active() { return false; }
uint32_t aurora_pipeline_cache_seed(const char* path) { (void)path; return 0; }
bool aurora_pipeline_cache_export(const char* path) { (void)path; return false; }
void aurora_pipeline_cache_get_stats(void* out) { if (out) std::memset(out, 0, 64); }
void aurora_pipeline_cache_set_log(bool enabled) { (void)enabled; }
void aurora_pipeline_cache_set_wait(bool enabled) { (void)enabled; }
void aurora_pipeline_cache_set_threads(uint32_t threads) { (void)threads; }
void aurora_pipeline_cache_set_boost(bool enabled) { (void)enabled; }
void aurora_pipeline_cache_note_crash(void) {}
uint32_t aurora_pipeline_cache_describe_stuck(uint64_t minNs, char* buf, size_t bufSize) {
  (void)minNs; if (buf && bufSize > 0) buf[0] = 0; return 0;
}
}

#include "aurora/display.h"
#include "aurora/hdr.h"
#include "aurora/color.h"
#include "aurora/picture.h"
#include "aurora/upscale.h"

extern "C" {
void aurora_set_supersample(int factor) { (void)factor; }
void aurora_set_post_filter(int mode) { (void)mode; }
void aurora_set_vrr(bool enabled) { (void)enabled; }

bool aurora_hdr_supported(void) { return false; }
void aurora_set_hdr(bool enabled, float paperWhiteNits, float highlightBoost) {
  (void)enabled; (void)paperWhiteNits; (void)highlightBoost;
}
bool aurora_hdr_active(void) { return false; }

void aurora_set_color_matrix(const float* matrix) { (void)matrix; }

void aurora_set_picture_effects(const AuroraPictureEffects* effects) { (void)effects; }
int aurora_capture_picture(AuroraPictureCallback callback, void* user) {
  (void)callback; (void)user;
  return 0;
}
void AuroraSetTextureUpscale(unsigned factor, AuroraTextureUpscaler upscaler) {
  (void)factor; (void)upscaler;
}
}
