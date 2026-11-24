#include "gl_core.hpp"

#include "../internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace aurora::gl {
namespace {
Module Log("aurora::gl");

bool g_loaded = false;
void* g_libGLESv2 = nullptr;
void* g_libEGL = nullptr;

using EglGetProcAddressFn = void* (*)(const char*);
EglGetProcAddressFn g_eglGetProcAddress = nullptr;

void* dlsym_lib(void*& handle, const char* const* names, const char* sym) {
  if (handle == nullptr) {
    for (const char* const* n = names; *n != nullptr; ++n) {
#ifdef _WIN32
      handle = reinterpret_cast<void*>(LoadLibraryA(*n));
#else
      handle = dlopen(*n, RTLD_NOW | RTLD_GLOBAL);
#endif
      if (handle != nullptr) {
        break;
      }
    }
  }
#ifdef _WIN32
  return handle != nullptr ? reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), sym)) : nullptr;
#else
  return handle != nullptr ? dlsym(handle, sym) : nullptr;
#endif
}
} // namespace

// Resolution chain, mirroring gpu.cpp's sdl2shim_egl_get_proc: the shim's own
// getProc first; then eglGetProcAddress for extensions; then direct dlsym against
// libGLESv2 / libEGL for core symbols that eglGetProcAddress refuses.
void* resolve(ProcAddressFn getProc, const char* name) {
  if (getProc != nullptr) {
    if (void* p = getProc(name)) {
      return p;
    }
  }
#ifdef _WIN32
  static const char* const kEgl[] = {"libEGL.dll", "libEGL_angle.dll", nullptr};
#else
  static const char* const kEgl[] = {"libEGL.so.1", "libEGL.so", "libmali.so", "libMali.so", nullptr};
#endif
  if (g_eglGetProcAddress == nullptr) {
    g_eglGetProcAddress = reinterpret_cast<EglGetProcAddressFn>(dlsym_lib(g_libEGL, kEgl, "eglGetProcAddress"));
  }
  if (g_eglGetProcAddress != nullptr) {
    if (void* p = g_eglGetProcAddress(name)) {
      return p;
    }
  }
#ifdef _WIN32
  static const char* const kGles[] = {"libGLESv2.dll", "libGLESv2_angle.dll", nullptr};
#else
  static const char* const kGles[] = {"libGLESv3.so", "libGLESv2.so", "libGLESv2.so.2", nullptr};
#endif
  if (void* p = dlsym_lib(g_libGLESv2, kGles, name)) {
    return p;
  }
  return dlsym_lib(g_libEGL, kEgl, name);
}

GlProcTable gl;

// $TTYD_TIMING_LOG: each hot GL entry point wrapped with a timer, so the log says which calls the
// driver spends the render worker's time in (the RK3326's Mali blob: most of it). One [GL-PROF]
// line per 60 frames (prof_frame, from present_frame): the costliest calls, per frame.
namespace prof {
struct Stat {
  const char* name = nullptr;
  std::atomic<uint64_t> ns{0};
  std::atomic<uint32_t> calls{0};
};
Stat g_stats[96];
std::atomic<int> g_count{0};
bool g_on = false;
inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
} // namespace prof

#define GLPROF(member)                                                                                      \
  do {                                                                                                      \
    using Fn = decltype(gl.member);                                                                         \
    static Fn s_orig = nullptr;                                                                             \
    static int s_idx = -1;                                                                                  \
    if (gl.member == nullptr || s_idx >= 0) {                                                               \
      break;                                                                                                \
    }                                                                                                       \
    s_idx = prof::g_count.fetch_add(1);                                                                     \
    prof::g_stats[s_idx].name = #member;                                                                    \
    s_orig = gl.member;                                                                                     \
    gl.member = [](auto... args) {                                                                          \
      const uint64_t t0 = prof::now_ns();                                                                   \
      if constexpr (std::is_void_v<decltype(s_orig(args...))>) {                                             \
        s_orig(args...);                                                                                    \
        prof::g_stats[s_idx].ns.fetch_add(prof::now_ns() - t0, std::memory_order_relaxed);                  \
        prof::g_stats[s_idx].calls.fetch_add(1, std::memory_order_relaxed);                                 \
      } else {                                                                                              \
        auto result = s_orig(args...);                                                                      \
        prof::g_stats[s_idx].ns.fetch_add(prof::now_ns() - t0, std::memory_order_relaxed);                  \
        prof::g_stats[s_idx].calls.fetch_add(1, std::memory_order_relaxed);                                 \
        return result;                                                                                      \
      }                                                                                                     \
    };                                                                                                      \
  } while (0)

static void install_prof_wrappers() {
  GLPROF(DrawElements);
  GLPROF(DrawElementsBaseVertex);
  GLPROF(DrawRangeElements);
  GLPROF(DrawRangeElementsBaseVertex);
  GLPROF(DrawArrays);
  GLPROF(DrawElementsInstanced);
  GLPROF(UseProgram);
  GLPROF(BindTexture);
  GLPROF(ActiveTexture);
  GLPROF(BindSampler);
  GLPROF(BindBufferRange);
  GLPROF(BindBufferBase);
  GLPROF(BindBuffer);
  GLPROF(BufferSubData);
  GLPROF(BufferData);
  GLPROF(MapBufferRange);
  GLPROF(UnmapBuffer);
  GLPROF(FlushMappedBufferRange);
  GLPROF(BindVertexArray);
  GLPROF(VertexAttribPointer);
  GLPROF(VertexAttribIPointer);
  GLPROF(EnableVertexAttribArray);
  GLPROF(DisableVertexAttribArray);
  GLPROF(BindFramebuffer);
  GLPROF(Enable);
  GLPROF(Disable);
  GLPROF(BlendFuncSeparate);
  GLPROF(BlendEquationSeparate);
  GLPROF(BlendColor);
  GLPROF(ColorMask);
  GLPROF(DepthMask);
  GLPROF(DepthFunc);
  GLPROF(CullFace);
  GLPROF(FrontFace);
  GLPROF(PolygonOffset);
  GLPROF(Viewport);
  GLPROF(Scissor);
  GLPROF(StencilFuncSeparate);
  GLPROF(Clear);
  GLPROF(ClearBufferfv);
  GLPROF(InvalidateFramebuffer);
  GLPROF(TexSubImage2D);
  GLPROF(TexImage2D);
  GLPROF(TexStorage2D);
  GLPROF(ClientWaitSync);
  GLPROF(FenceSync);
  GLPROF(BlitFramebuffer);
  GLPROF(LineWidth);
  GLPROF(Uniform1i);
  GLPROF(DrawBuffers);
  GLPROF(Flush);
  GLPROF(GenerateMipmap);
  GLPROF(CompressedTexSubImage2D);
  GLPROF(TexParameteri);
  GLPROF(SamplerParameteri);
  prof::g_on = true;
  Log.info("[gl] GL call timing on ($TTYD_TIMING_LOG): [GL-PROF] lines per 60 frames");
}
#undef GLPROF

bool load(ProcAddressFn getProc) {
  bool missing = false;

  // REQ: a core ES 3.0 entry point the render path cannot run without; a null
  // fails load(). OPT: device-only (EGL interop) or feature-gated (compressed
  // textures, MSAA, mapped buffers, KHR_debug) -- callers null-check these.
#define REQ(member, name)                                                                                    \
  gl.member = reinterpret_cast<decltype(gl.member)>(resolve(getProc, name));                                 \
  if (gl.member == nullptr) {                                                                                \
    Log.warn("[gl] missing required entry point {}", name);                                                  \
    missing = true;                                                                                          \
  }
#define OPT(member, name) gl.member = reinterpret_cast<decltype(gl.member)>(resolve(getProc, name));

  // Buffers
  REQ(GenBuffers, "glGenBuffers")
  REQ(DeleteBuffers, "glDeleteBuffers")
  REQ(BindBuffer, "glBindBuffer")
  REQ(BufferData, "glBufferData")
  REQ(BufferSubData, "glBufferSubData")
  REQ(BindBufferRange, "glBindBufferRange")
  REQ(BindBufferBase, "glBindBufferBase")
  OPT(MapBufferRange, "glMapBufferRange")
  OPT(UnmapBuffer, "glUnmapBuffer")
  OPT(FlushMappedBufferRange, "glFlushMappedBufferRange")
  OPT(CopyBufferSubData, "glCopyBufferSubData")
  // GL_EXT_buffer_storage on GLES (libmali/PowerVR); the ARB/core name is the desktop spelling.
  OPT(BufferStorage, "glBufferStorageEXT")
  if (gl.BufferStorage == nullptr) {
    OPT(BufferStorage, "glBufferStorage")
  }

  // Vertex arrays
  REQ(GenVertexArrays, "glGenVertexArrays")
  REQ(DeleteVertexArrays, "glDeleteVertexArrays")
  REQ(BindVertexArray, "glBindVertexArray")
  REQ(EnableVertexAttribArray, "glEnableVertexAttribArray")
  REQ(DisableVertexAttribArray, "glDisableVertexAttribArray")
  REQ(VertexAttribPointer, "glVertexAttribPointer")
  REQ(VertexAttribIPointer, "glVertexAttribIPointer")

  // Textures
  REQ(GenTextures, "glGenTextures")
  REQ(DeleteTextures, "glDeleteTextures")
  OPT(IsTexture, "glIsTexture")
  REQ(BindTexture, "glBindTexture")
  REQ(ActiveTexture, "glActiveTexture")
  REQ(TexStorage2D, "glTexStorage2D")
  REQ(TexImage2D, "glTexImage2D")
  REQ(TexSubImage2D, "glTexSubImage2D")
  OPT(CompressedTexImage2D, "glCompressedTexImage2D")
  OPT(CompressedTexSubImage2D, "glCompressedTexSubImage2D")
  REQ(CopyTexSubImage2D, "glCopyTexSubImage2D")
  REQ(TexParameteri, "glTexParameteri")
  REQ(TexParameterf, "glTexParameterf")
  REQ(GenerateMipmap, "glGenerateMipmap")
  REQ(PixelStorei, "glPixelStorei")

  // Sampler objects
  REQ(GenSamplers, "glGenSamplers")
  REQ(DeleteSamplers, "glDeleteSamplers")
  REQ(BindSampler, "glBindSampler")
  REQ(SamplerParameteri, "glSamplerParameteri")
  REQ(SamplerParameterf, "glSamplerParameterf")

  // Framebuffers / renderbuffers
  REQ(GenFramebuffers, "glGenFramebuffers")
  REQ(DeleteFramebuffers, "glDeleteFramebuffers")
  REQ(BindFramebuffer, "glBindFramebuffer")
  REQ(FramebufferTexture2D, "glFramebufferTexture2D")
  REQ(FramebufferRenderbuffer, "glFramebufferRenderbuffer")
  REQ(CheckFramebufferStatus, "glCheckFramebufferStatus")
  REQ(BlitFramebuffer, "glBlitFramebuffer")
  REQ(InvalidateFramebuffer, "glInvalidateFramebuffer")
  REQ(DrawBuffers, "glDrawBuffers")
  REQ(ReadBuffer, "glReadBuffer")
  REQ(ReadPixels, "glReadPixels")
  REQ(GenRenderbuffers, "glGenRenderbuffers")
  REQ(DeleteRenderbuffers, "glDeleteRenderbuffers")
  REQ(BindRenderbuffer, "glBindRenderbuffer")
  REQ(RenderbufferStorage, "glRenderbufferStorage")
  OPT(RenderbufferStorageMultisample, "glRenderbufferStorageMultisample")

  // Clears
  REQ(ClearColor, "glClearColor")
  REQ(ClearDepthf, "glClearDepthf")
  REQ(ClearStencil, "glClearStencil")
  REQ(Clear, "glClear")
  REQ(ClearBufferfv, "glClearBufferfv")
  REQ(ClearBufferfi, "glClearBufferfi")
  REQ(ClearBufferiv, "glClearBufferiv")
  REQ(ClearBufferuiv, "glClearBufferuiv")

  // Shaders / programs
  REQ(CreateShader, "glCreateShader")
  REQ(ShaderSource, "glShaderSource")
  REQ(CompileShader, "glCompileShader")
  REQ(GetShaderiv, "glGetShaderiv")
  REQ(GetShaderInfoLog, "glGetShaderInfoLog")
  REQ(DeleteShader, "glDeleteShader")
  REQ(CreateProgram, "glCreateProgram")
  REQ(AttachShader, "glAttachShader")
  REQ(BindAttribLocation, "glBindAttribLocation")
  REQ(LinkProgram, "glLinkProgram")
  REQ(GetProgramiv, "glGetProgramiv")
  REQ(GetProgramInfoLog, "glGetProgramInfoLog")
  REQ(UseProgram, "glUseProgram")
  REQ(DeleteProgram, "glDeleteProgram")
  REQ(GetUniformLocation, "glGetUniformLocation")
  REQ(Uniform1i, "glUniform1i")
  REQ(GetUniformBlockIndex, "glGetUniformBlockIndex")
  REQ(UniformBlockBinding, "glUniformBlockBinding")
  REQ(GetActiveUniformBlockiv, "glGetActiveUniformBlockiv")
  // Program binary (ES 3.0 core, but OPT: the binary cache degrades to source
  // compile when absent, and the ProgramParameteri hint is inert on drivers that
  // do not need it). glProgramBinary is the base spelling; some GLES stacks only
  // export the OES alias.
  OPT(GetProgramBinary, "glGetProgramBinary")
  if (gl.GetProgramBinary == nullptr) {
    OPT(GetProgramBinary, "glGetProgramBinaryOES")
  }
  OPT(ProgramBinary, "glProgramBinary")
  if (gl.ProgramBinary == nullptr) {
    OPT(ProgramBinary, "glProgramBinaryOES")
  }
  OPT(ProgramParameteri, "glProgramParameteri")

  // Fixed-function state
  REQ(Enable, "glEnable")
  REQ(Disable, "glDisable")
  REQ(BlendFuncSeparate, "glBlendFuncSeparate")
  REQ(BlendEquationSeparate, "glBlendEquationSeparate")
  REQ(BlendColor, "glBlendColor")
  REQ(ColorMask, "glColorMask")
  REQ(DepthMask, "glDepthMask")
  REQ(DepthFunc, "glDepthFunc")
  REQ(DepthRangef, "glDepthRangef")
  REQ(CullFace, "glCullFace")
  REQ(FrontFace, "glFrontFace")
  REQ(PolygonOffset, "glPolygonOffset")
  REQ(LineWidth, "glLineWidth")
  REQ(Viewport, "glViewport")
  REQ(Scissor, "glScissor")
  REQ(StencilFuncSeparate, "glStencilFuncSeparate")
  REQ(StencilOpSeparate, "glStencilOpSeparate")
  REQ(StencilMaskSeparate, "glStencilMaskSeparate")

  // Draw
  REQ(DrawArrays, "glDrawArrays")
  REQ(DrawElements, "glDrawElements")
  OPT(DrawArraysInstanced, "glDrawArraysInstanced")
  OPT(DrawElementsInstanced, "glDrawElementsInstanced")
  OPT(DrawRangeElements, "glDrawRangeElements")
  OPT(DrawRangeElementsBaseVertex, "glDrawRangeElementsBaseVertex")
  if (gl.DrawRangeElementsBaseVertex == nullptr) {
    OPT(DrawRangeElementsBaseVertex, "glDrawRangeElementsBaseVertexEXT")
  }
  if (gl.DrawRangeElementsBaseVertex == nullptr) {
    OPT(DrawRangeElementsBaseVertex, "glDrawRangeElementsBaseVertexOES")
  }
  OPT(DrawElementsBaseVertex, "glDrawElementsBaseVertex")
  if (gl.DrawElementsBaseVertex == nullptr) {
    OPT(DrawElementsBaseVertex, "glDrawElementsBaseVertexEXT")
  }
  if (gl.DrawElementsBaseVertex == nullptr) {
    OPT(DrawElementsBaseVertex, "glDrawElementsBaseVertexOES")
  }
  OPT(DrawElementsInstancedBaseVertex, "glDrawElementsInstancedBaseVertex")
  if (gl.DrawElementsInstancedBaseVertex == nullptr) {
    OPT(DrawElementsInstancedBaseVertex, "glDrawElementsInstancedBaseVertexEXT")
  }
  if (gl.DrawElementsInstancedBaseVertex == nullptr) {
    OPT(DrawElementsInstancedBaseVertex, "glDrawElementsInstancedBaseVertexOES")
  }

  // Sync / query
  REQ(FenceSync, "glFenceSync")
  REQ(ClientWaitSync, "glClientWaitSync")
  REQ(WaitSync, "glWaitSync")
  REQ(DeleteSync, "glDeleteSync")
  REQ(Finish, "glFinish")
  REQ(Flush, "glFlush")
  REQ(GetError, "glGetError")
  REQ(GetString, "glGetString")
  REQ(GetStringi, "glGetStringi")
  REQ(GetIntegerv, "glGetIntegerv")
  REQ(GetFloatv, "glGetFloatv")

  // Debug (KHR_debug; optional)
  OPT(DebugMessageCallback, "glDebugMessageCallback")
  OPT(DebugMessageControl, "glDebugMessageControl")
  OPT(PushDebugGroup, "glPushDebugGroup")
  OPT(PopDebugGroup, "glPopDebugGroup")
  OPT(ObjectLabel, "glObjectLabel")

  // EGL (device path only -- absent on the desktop SDL_GL path, so all OPT)
  OPT(eglGetCurrentContext, "eglGetCurrentContext")
  OPT(eglGetCurrentDisplay, "eglGetCurrentDisplay")
  OPT(eglGetCurrentSurface, "eglGetCurrentSurface")
  OPT(eglMakeCurrent, "eglMakeCurrent")
  OPT(eglCreateContext, "eglCreateContext")
  OPT(eglDestroyContext, "eglDestroyContext")
  OPT(eglCreatePbufferSurface, "eglCreatePbufferSurface")
  OPT(eglDestroySurface, "eglDestroySurface")
  OPT(eglChooseConfig, "eglChooseConfig")
  OPT(eglGetConfigAttrib, "eglGetConfigAttrib")
  OPT(eglGetError, "eglGetError")
  OPT(eglQueryString, "eglQueryString")
  OPT(eglCreateImageKHR, "eglCreateImageKHR")
  OPT(eglDestroyImageKHR, "eglDestroyImageKHR")
  OPT(eglCreateSyncKHR, "eglCreateSyncKHR")
  OPT(eglDestroySyncKHR, "eglDestroySyncKHR")
  OPT(eglClientWaitSyncKHR, "eglClientWaitSyncKHR")
  OPT(eglWaitSyncKHR, "eglWaitSyncKHR")
  OPT(eglSwapInterval, "eglSwapInterval")
  OPT(glEGLImageTargetTexture2DOES, "glEGLImageTargetTexture2DOES")

#undef REQ
#undef OPT

  if (std::getenv("TTYD_TIMING_LOG") != nullptr) {
    install_prof_wrappers();
  }
  g_loaded = !missing;
  if (missing) {
    Log.error("[gl] entry-point table incomplete; GL backend cannot initialize");
  }
  return g_loaded;
}

bool loaded() { return g_loaded; }

} // namespace aurora::gl

namespace aurora::gl {
void prof_frame() noexcept {
  static int s_frames = 0;
  if (!prof::g_on || ++s_frames < 60) {
    return;
  }
  s_frames = 0;
  struct Row {
    const char* name;
    double ms;
    uint32_t calls;
  };
  Row rows[96];
  int n = 0;
  const int count = std::min(prof::g_count.load(), 96);
  double total = 0;
  for (int i = 0; i < count; ++i) {
    const uint64_t ns = prof::g_stats[i].ns.exchange(0, std::memory_order_relaxed);
    const uint32_t calls = prof::g_stats[i].calls.exchange(0, std::memory_order_relaxed);
    if (calls != 0) {
      rows[n++] = {prof::g_stats[i].name, ns / 1e6 / 60.0, calls / 60};
      total += ns / 1e6 / 60.0;
    }
  }
  std::sort(rows, rows + n, [](const Row& a, const Row& b) { return a.ms > b.ms; });
  char line[1024];
  int len = std::snprintf(line, sizeof(line), "[GL-PROF] per frame: %.2f ms in GL |", total);
  for (int i = 0; i < n && i < 14 && len < static_cast<int>(sizeof(line)); ++i) {
    len += std::snprintf(line + len, sizeof(line) - len, " %s %.2fms/%u", rows[i].name, rows[i].ms, rows[i].calls);
  }
  std::printf("%s\n", line);
  std::fflush(stdout);
}
} // namespace aurora::gl
