<div align="center">
  <img src="assets/aurora.png" alt="Logo" width="640">
</div>
<br/>

> **aurora-r36s** is a fork for ARM Linux handhelds with a Rockchip RK3326 and a Mali-G31 GPU (the R36S
> and similar), running through PortMaster. It builds on [jbcgames/aurora](https://github.com/jbcgames/aurora)'s
> `rk3326-gles` branch (a native OpenGL ES 3.0 backend in place of Dawn), which builds on
> [encounter/aurora](https://github.com/encounter/aurora). It powers the
> [Paper Cut R36S port](https://github.com/jimmygzzmtz/paper-cut). The changes are in Aurora's GX and GL layers, not in
> any one game, so other Aurora-based GameCube ports on these devices can use them too.
>
> What this fork adds (branch `r36s`):
> - **GX thread** (`TTYD_GX_THREAD=1`): GX FIFO translation runs on its own core while the game runs, as the
>   GameCube's GPU read the FIFO. Games hand it work with `AuroraGXKick()`; every existing sync point waits for it.
> - **Streamed render passes**: with the GX thread, the EFB pass goes to the render worker in pieces while it is being
>   recorded, so a frame is drawn alongside the game instead of after it (`TTYD_PASS_STREAM`).
> - **Mali draw cost**: indexed draws pass their index range (`glDrawRangeElements*`), so the Mali driver stops reading
>   every draw's indices on the CPU (`TTYD_GL_NO_RANGE=1` to compare).
> - **Readable Dear ImGui text**: the font atlas is re-uploaded whenever it changes (glyphs drew as black boxes).
> - **Present on the frame tick** (`aurora_try_flush_present`) and a CPU fence wait before the swap.
> - **Measuring**: `TTYD_TIMING_LOG=1` prints per-frame GL call costs (`[GL-PROF]`), GX thread statistics
>   (`[GX-THREAD]`) and swap timing (`[SWAP]`).
> - The SDL3-over-SDL2 shim location is the `AURORA_SDL3_SHIM_DIR` CMake option.

Aurora is a source-level GameCube & Wii compatibility layer intended for use with game decompilation projects.

Originally developed for use in [Metaforce](https://github.com/AxioDL/metaforce), a Metroid Prime reverse engineering project.
It now powers several completed source ports, including [Dusklight](https://github.com/TwilitRealm/dusklight).

### Features

- Application layer using SDL3
  - Runs on Windows, Linux, macOS, iOS, tvOS, Android
- GX compatibility layer
  - Graphics API support: D3D12, Vulkan, Metal
  - Highly accurate and performant GX implementation
  - Robust pipeline cache system with "transferable" cache support for releases
  - Dolphin-compatible texture pack support
  - Widescreen & resolution scaling support
  - Custom APIs for offscreen rendering
- PAD compatibility layer
  - Utilizes `SDL_Gamepad` for wide controller support, including GameCube controller adapters
  - Layered input routing system for advanced input handling and remapping
  - Automatically saves and loads controller bindings and port mappings
  - Gyro, keyboard & mouse support
- DVD compatibility layer
  - Utilizes [nod](https://github.com/encounter/nod) to support all GameCube/Wii disc image types, including RVZ
- CARD compatibility layer
  - Full compatibility with Dolphin `.gci` and `.raw` for game saves
- [Dear ImGui](https://github.com/ocornut/imgui) built-in for simple debug UIs
- [RmlUi](https://github.com/mikke89/RmlUi) built-in for full-fledged HTML/CSS-based UIs

### Graphics

The GX compatibility layer is built on top of [WebGPU](https://www.w3.org/TR/webgpu/), a cross-platform graphics API
abstraction layer. WebGPU allows targeting all major platforms simultaneously with minimal overhead. The WebGPU
implementation used is Chromium's [Dawn](https://dawn.googlesource.com/dawn/).

![Screenshot](assets/screenshot.png)

### Building

See [docs/building.md](docs/building.md) for build instructions, CMake integration, and configuration options.

### License

Aurora is licensed under the [MIT License](LICENSE).
