#pragma once

#include "../internal.hpp"

namespace aurora::gx::fifo {

// Process a buffer of GX FIFO commands
void process(const uint8_t* data, uint32_t size, bool bigEndian);

// The GX thread's pieces of process() (fifo.cpp, $TTYD_GX_THREAD): one chunk of a stream cut at
// command boundaries. fresh: the first chunk after the stream was synchronized (the draw-state
// caches reset, as process() does); later chunks continue its state and its pending PnMtx batch.
// process_flush() ends the stream piece (the pending batch submitted) before going idle.
void process_chunk(const uint8_t* data, uint32_t size, bool fresh);
void process_flush();

} // namespace aurora::gx::fifo
