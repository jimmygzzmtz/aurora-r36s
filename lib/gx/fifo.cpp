#include "fifo.hpp"
#include "command_processor.hpp"
#include "../gfx/common.hpp"
#include "../internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#if defined(__linux__)
#include <pthread.h>
#endif
#include <mutex>
#include <thread>
#include <vector>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
} // namespace detail

struct FifoSlot {
  uint8_t* data = nullptr;
  uint32_t size = 0;
  uint32_t capacity = 0;
};

static FifoSlot sSlots[2]{};
static uint32_t sProducerSlot = 0;
static uint32_t sConsumerSlot = 0;
static std::atomic<uint64_t> sFramesProduced{0};
static std::atomic<uint64_t> sFramesConsumed{0};
static std::mutex sFifoMutex;
static std::condition_variable sProducerCv;
static std::condition_variable sConsumerCv;
static std::atomic<bool> sDoubleBuffering{false};
static std::atomic<bool> sStopping{false};

// Async end-of-frame drain thread state: drains the previous producer slot while the main thread
// concurrently runs flush_present().
static std::thread sAsyncDrainThread;
static std::mutex sAsyncDrainMutex;
static std::condition_variable sAsyncDrainCv;
static bool sAsyncDrainBusy = false;
static bool sAsyncDrainStop = false;
static uint32_t sAsyncConsumeSlot = 0;
static double sAsyncDrainLastMs = 0.0;

static void async_drain_worker() {
  for (;;) {
    uint32_t slot = 0;
    {
      std::unique_lock<std::mutex> lock(sAsyncDrainMutex);
      sAsyncDrainCv.wait(lock, [] { return sAsyncDrainBusy || sAsyncDrainStop; });
      if (sAsyncDrainStop && !sAsyncDrainBusy) {
        return;
      }
      slot = sAsyncConsumeSlot;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t sz = sSlots[slot].size;
    if (sz > 0) {
      process(sSlots[slot].data, sz, true);
      sSlots[slot].size = 0;
    }
    const auto t1 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(sAsyncDrainMutex);
      sAsyncDrainLastMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      sAsyncDrainBusy = false;
    }
    sAsyncDrainCv.notify_all();
  }
}

// --- The GX thread ($TTYD_GX_THREAD=1) ---------------------------------------------------------
// The game thread writes the FIFO as always; at command boundaries it hands the bytes written so
// far to this thread (kick: the buffer goes to the queue, the writer takes a fresh one from a
// pool) and goes on with the game while they are translated -- the GameCube's own model, where
// the GPU reads the FIFO while the CPU runs on. The game already keeps what its draws read alive
// until a draw sync (GXSetDrawSync -> AuroraGXSync) or the frame's end, and both drain(), which
// here waits for this thread to finish everything handed to it. Every other path that touched
// the command processor from the game thread drains first (create_pass, resolve_pass, custom
// draws, the bounding box, the texture-cache flush), so the processor only ever runs on one
// thread at a time. On the RK3326 this moved about 45% of the game thread's work in Rogueport
// onto an idle core.
bool __gx_in_begin() noexcept; // GXVert.cpp: inside GXBegin/GXEnd (a kick would split a draw)

namespace {
bool sThreadedWanted = false;
std::thread sGxThread;
std::thread::id sGxThreadId;
std::mutex sGxMutex;
std::condition_variable sGxWork; // the GX thread waits for a chunk
std::condition_variable sGxIdle; // drain() waits for the GX thread
std::deque<FifoSlot> sGxQueue;
std::vector<FifoSlot> sGxFree;
bool sGxBusy = false;
bool sGxStop = false;
bool sGxFresh = true; // the next chunk starts a stream piece (after a drain): process() resets its caches
// The EFB pass is handed to the render worker in pieces of this many commands while it is recorded
// (gfx::stream_efb_pass), so it renders the frame alongside the game instead of after it.
// $TTYD_PASS_STREAM=0: off; a number: the piece size.
uint32_t sStreamCommands = 0;

// $TTYD_TIMING_LOG: what the GX thread got and did, one [GX-THREAD] line per 60 drains
struct GxStats {
  std::atomic<uint32_t> kicks{0}, refusedBegin{0}, refusedFrame{0}, refusedEmpty{0}, chunks{0}, fresh{0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> waitUs{0};
  uint32_t syncs = 0;
} sGxStats;
bool gx_stats_on() {
  static const bool on = std::getenv("TTYD_TIMING_LOG") != nullptr;
  return on;
}

void gx_thread_main() {
#if defined(__linux__)
  pthread_setname_np(pthread_self(), "aurora-gx");
#endif
  for (;;) {
    FifoSlot chunk;
    bool fresh = false;
    {
      std::unique_lock lock(sGxMutex);
      sGxWork.wait(lock, [] { return sGxStop || !sGxQueue.empty(); });
      if (sGxQueue.empty()) {
        return; // stopping, nothing left
      }
      chunk = sGxQueue.front();
      sGxQueue.pop_front();
      sGxBusy = true;
      fresh = sGxFresh;
      sGxFresh = false;
    }
    process_chunk(chunk.data, chunk.size, fresh);
    if (sStreamCommands != 0 && gfx::efb_pass_holds(sStreamCommands)) {
      process_flush(); // (the pending batched draw belongs to the pass being handed over)
      gfx::stream_efb_pass(sStreamCommands);
    }
    if (gx_stats_on()) {
      sGxStats.chunks.fetch_add(1, std::memory_order_relaxed);
      sGxStats.bytes.fetch_add(chunk.size, std::memory_order_relaxed);
      if (fresh) {
        sGxStats.fresh.fetch_add(1, std::memory_order_relaxed);
      }
    }
    bool idle = false;
    {
      std::lock_guard lock(sGxMutex);
      idle = sGxQueue.empty();
    }
    if (idle) {
      process_flush(); // (the pending PnMtx batch submitted before anyone can see the processor idle)
    }
    {
      std::lock_guard lock(sGxMutex);
      chunk.size = 0;
      sGxFree.push_back(chunk);
      sGxBusy = false;
    }
    sGxIdle.notify_all();
  }
}

bool on_gx_thread() { return sGxThread.joinable() && std::this_thread::get_id() == sGxThreadId; }
} // namespace

bool threaded() { return sThreadedWanted; }

// (outside a frame the stream waits for the next one, as AuroraGXSync's drain does)
// force (drain): hand over even inside an open GX_AUTO draw, as the single-threaded drain()
// processed the buffer whatever state the writer was in.
static bool kick_impl(bool force) {
  if (!sThreadedWanted || on_gx_thread()) {
    return false;
  }
  const bool inBegin = !force && __gx_in_begin();
  if (detail::sBufferSize == 0 || inBegin || !gfx::in_frame()) {
    if (gx_stats_on()) {
      (detail::sBufferSize == 0 ? sGxStats.refusedEmpty : inBegin ? sGxStats.refusedBegin : sGxStats.refusedFrame)
          .fetch_add(1, std::memory_order_relaxed);
    }
    return false;
  }
  if (gx_stats_on()) {
    sGxStats.kicks.fetch_add(1, std::memory_order_relaxed);
  }
  FifoSlot next{};
  {
    std::lock_guard lock(sGxMutex);
    if (!sGxThread.joinable()) {
      sGxStop = false;
      sGxThread = std::thread(gx_thread_main);
      sGxThreadId = sGxThread.get_id();
      Log.info("GX thread started: the FIFO is translated beside the game");
    }
    sGxQueue.push_back(FifoSlot{detail::sBufferData, detail::sBufferSize, detail::sBufferCapacity});
    if (!sGxFree.empty()) {
      next = sGxFree.back();
      sGxFree.pop_back();
    }
  }
  sGxWork.notify_one();
  if (next.data == nullptr) {
    next.capacity = 256 * 1024;
    next.data = static_cast<uint8_t*>(std::malloc(next.capacity));
  }
  detail::sBufferData = next.data;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = next.capacity;
  sSlots[sProducerSlot].data = next.data; // (write_data_grow keeps the slot in step with the buffer)
  sSlots[sProducerSlot].capacity = next.capacity;
  return true;
}

bool kick() { return kick_impl(false); }

void kick_if_over(uint32_t bytes) {
  if (detail::sBufferSize >= bytes) {
    kick();
  }
}

// The game thread: everything written so far translated, and the GX thread idle.
static void gx_thread_sync() {
  kick_impl(true);
  const auto t0 = std::chrono::steady_clock::now();
  {
    std::unique_lock lock(sGxMutex);
    sGxIdle.wait(lock, [] { return sGxQueue.empty() && !sGxBusy; });
    sGxFresh = true;
  }
  if (gx_stats_on()) {
    sGxStats.waitUs += static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    if (++sGxStats.syncs >= 60) {
      std::printf("[GX-THREAD] 60 drains: %u kicks (refused: %u empty, %u in GXBegin, %u outside a frame), "
                  "%u chunks (%u fresh), %.1f KB translated, drains waited %.2f ms each, buffer now %u B\n",
                  sGxStats.kicks.exchange(0), sGxStats.refusedEmpty.exchange(0), sGxStats.refusedBegin.exchange(0),
                  sGxStats.refusedFrame.exchange(0), sGxStats.chunks.exchange(0), sGxStats.fresh.exchange(0),
                  sGxStats.bytes.exchange(0) / 1024.0, sGxStats.waitUs.exchange(0) / 60.0 / 1000.0,
                  detail::sBufferSize);
      std::fflush(stdout);
      sGxStats.syncs = 0;
    }
  }
}

static void gx_thread_stop() {
  std::thread thread;
  {
    std::lock_guard lock(sGxMutex);
    sGxStop = true;
    thread = std::move(sGxThread);
  }
  sGxWork.notify_all();
  if (thread.joinable()) {
    thread.join();
  }
  std::lock_guard lock(sGxMutex);
  for (FifoSlot& s : sGxQueue) {
    std::free(s.data);
  }
  for (FifoSlot& s : sGxFree) {
    std::free(s.data);
  }
  sGxQueue.clear();
  sGxFree.clear();
  sGxBusy = false;
  sGxFresh = true;
}

void init() {
  constexpr uint32_t initialCapacity = 512 * 1024;
  if (const char* want = std::getenv("TTYD_GX_THREAD"); want != nullptr && want[0] != '0') {
    sThreadedWanted = true;
    sStreamCommands = 96;
    if (const char* stream = std::getenv("TTYD_PASS_STREAM"); stream != nullptr) {
      sStreamCommands = static_cast<uint32_t>(std::strtoul(stream, nullptr, 10));
    }
  }
  {
    std::lock_guard<std::mutex> alock(sAsyncDrainMutex);
    sAsyncDrainStop = false;
    sAsyncDrainBusy = false;
    sAsyncDrainLastMs = 0.0;
  }
  std::lock_guard<std::mutex> lock(sFifoMutex);
  sStopping.store(false, std::memory_order_relaxed);
  for (int i = 0; i < 2; ++i) {
    std::free(sSlots[i].data);
    sSlots[i].data = static_cast<uint8_t*>(std::malloc(initialCapacity));
    sSlots[i].size = 0;
    sSlots[i].capacity = initialCapacity;
  }
  sProducerSlot = 0;
  sConsumerSlot = 0;
  sFramesProduced.store(0, std::memory_order_relaxed);
  sFramesConsumed.store(0, std::memory_order_relaxed);

  detail::sBufferData = sSlots[0].data;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = initialCapacity;
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
}

void shutdown() {
  if (sThreadedWanted) {
    gx_thread_sync();
    gx_thread_stop();
  }
  std::thread threadToJoin;
  {
    std::lock_guard<std::mutex> lock(sAsyncDrainMutex);
    sAsyncDrainStop = true;
    threadToJoin = std::move(sAsyncDrainThread);
  }
  sAsyncDrainCv.notify_all();
  if (threadToJoin.joinable()) {
    threadToJoin.join();
  }

  sStopping.store(true, std::memory_order_release);
  sProducerCv.notify_all();
  sConsumerCv.notify_all();
  std::lock_guard<std::mutex> lock(sFifoMutex);
  for (int i = 0; i < 2; ++i) {
    std::free(sSlots[i].data);
    sSlots[i].data = nullptr;
    sSlots[i].size = 0;
    sSlots[i].capacity = 0;
  }
  detail::sBufferData = nullptr;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = 0;
}

void set_double_buffering(bool enable) {
  if (sDoubleBuffering.load(std::memory_order_acquire) == enable) return;
  if (enable) {
    std::unique_lock<std::mutex> lock(sFifoMutex);
    sDoubleBuffering.store(true, std::memory_order_release);
    Log.info("Double-buffered GX FIFO enabled");
  } else {
    std::unique_lock<std::mutex> lock(sFifoMutex);
    while (sFramesProduced.load(std::memory_order_acquire) > sFramesConsumed.load(std::memory_order_acquire) &&
           !sStopping.load(std::memory_order_relaxed)) {
      sProducerCv.wait(lock);
    }
    sDoubleBuffering.store(false, std::memory_order_release);
    Log.info("Double-buffered GX FIFO disabled");
  }
}

bool double_buffering_enabled() {
  return sDoubleBuffering.load(std::memory_order_relaxed);
}

void finish_frame_producer() {
  if (!sDoubleBuffering.load(std::memory_order_acquire)) {
    return;
  }
  sSlots[sProducerSlot].size = detail::sBufferSize;
  sSlots[sProducerSlot].data = detail::sBufferData;
  sSlots[sProducerSlot].capacity = detail::sBufferCapacity;

  std::unique_lock<std::mutex> lock(sFifoMutex);
  const uint64_t produced = ++sFramesProduced;
  sConsumerCv.notify_all();

  while (produced - sFramesConsumed.load(std::memory_order_acquire) > 1 &&
         !sStopping.load(std::memory_order_relaxed)) {
    sProducerCv.wait(lock);
  }
  if (sStopping.load(std::memory_order_relaxed)) return;

  sProducerSlot = 1 - sProducerSlot;
  detail::sBufferData = sSlots[sProducerSlot].data;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = sSlots[sProducerSlot].capacity;
}

void write_data_grow(const void* data, uint32_t length) {
  uint32_t needed = detail::sBufferSize + length;
  uint32_t newCap = std::max(detail::sBufferCapacity * 2, needed);
  detail::sBufferData = static_cast<uint8_t*>(std::realloc(detail::sBufferData, newCap));
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize = needed;
  detail::sBufferCapacity = newCap;
  sSlots[sProducerSlot].data = detail::sBufferData;
  sSlots[sProducerSlot].capacity = newCap;
}

void patch_u32(const uint32_t offset, const uint32_t val) {
  ASSERT(!detail::sInDisplayList && offset + sizeof(uint32_t) <= detail::sBufferSize,
         "fifo::patch_u32: invalid patch offset {} (buffer size {})", offset, detail::sBufferSize);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + offset, &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  uint32_t bytesWritten = detail::sDlWritePos;
  uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain() {
  if (sThreadedWanted) {
    if (!on_gx_thread()) {
      gx_thread_sync(); // (on the GX thread itself: the stream is already being processed in order)
    }
    return;
  }
  if (!sDoubleBuffering.load(std::memory_order_acquire)) {
    drain_async_wait();
    if (detail::sBufferSize == 0) {
      return;
    }
    process(detail::sBufferData, detail::sBufferSize, true);
    detail::sBufferSize = 0;
    return;
  }

  std::unique_lock<std::mutex> lock(sFifoMutex);
  while (sFramesProduced.load(std::memory_order_acquire) <= sFramesConsumed.load(std::memory_order_acquire) &&
         !sStopping.load(std::memory_order_relaxed)) {
    sConsumerCv.wait(lock);
  }
  if (sStopping.load(std::memory_order_relaxed)) return;

  const uint32_t consumeSlot = sConsumerSlot;
  const uint8_t* consumeData = sSlots[consumeSlot].data;
  const uint32_t consumeSize = sSlots[consumeSlot].size;
  lock.unlock();

  if (consumeSize > 0) {
    process(consumeData, consumeSize, true);
  }

  lock.lock();
  sSlots[consumeSlot].size = 0;
  sConsumerSlot = 1 - sConsumerSlot;
  ++sFramesConsumed;
  sProducerCv.notify_all();
}

void drain_async_start() {
  if (sDoubleBuffering.load(std::memory_order_acquire) || detail::sInDisplayList) {
    return;
  }
  {
    std::unique_lock<std::mutex> lock(sAsyncDrainMutex);
    sAsyncDrainCv.wait(lock, [] { return !sAsyncDrainBusy; });
    if (detail::sBufferSize == 0) {
      sAsyncDrainLastMs = 0.0;
      return;
    }
    if (!sAsyncDrainThread.joinable() && !sAsyncDrainStop) {
      sAsyncDrainThread = std::thread(async_drain_worker);
    }
  }
  const uint32_t consumeSlot = sProducerSlot;
  const uint32_t nextSlot = 1u - sProducerSlot;
  if (sSlots[nextSlot].data == nullptr) {
    drain();
    return;
  }
  sSlots[consumeSlot].data = detail::sBufferData;
  sSlots[consumeSlot].size = detail::sBufferSize;
  sSlots[consumeSlot].capacity = detail::sBufferCapacity;

  sProducerSlot = nextSlot;
  detail::sBufferData = sSlots[nextSlot].data;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = sSlots[nextSlot].capacity;

  {
    std::lock_guard<std::mutex> lock(sAsyncDrainMutex);
    sAsyncConsumeSlot = consumeSlot;
    sAsyncDrainLastMs = 0.0;
    sAsyncDrainBusy = true;
  }
  sAsyncDrainCv.notify_one();
}

double drain_async_wait() {
  std::unique_lock<std::mutex> lock(sAsyncDrainMutex);
  sAsyncDrainCv.wait(lock, [] { return !sAsyncDrainBusy; });
  return sAsyncDrainLastMs;
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }
void clear_buffer() { detail::sBufferSize = 0; }

} // namespace aurora::gx::fifo
