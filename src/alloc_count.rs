//! Test-only allocation and engine-call counters. The global allocator wraps
//! `System` and is compiled out of non-test builds.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Process-wide counting allocator; tallies live on the calling thread.
pub struct Counting;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
    static SET_SKY: Cell<u32> = const { Cell::new(0) };
    static UNIFORMS: Cell<u32> = const { Cell::new(0) };
    static SETTINGS_APPLY: Cell<u32> = const { Cell::new(0) };
    static TEX_LAYERS: Cell<u32> = const { Cell::new(0) };
    static CELL_READS: Cell<u64> = const { Cell::new(0) };
    static CELL_WRITES: Cell<u64> = const { Cell::new(0) };
}

/// Allocations since the last [`reset`].
pub fn alloc_count() -> u64 {
    ALLOCS.with(Cell::get)
}

/// Bytes requested since the last [`reset`].
pub fn alloc_bytes() -> u64 {
    BYTES.with(Cell::get)
}

/// Zero the allocation and engine-call counters for this thread.
pub fn reset() {
    ALLOCS.with(|c| c.set(0));
    BYTES.with(|c| c.set(0));
    SET_SKY.with(|c| c.set(0));
    UNIFORMS.with(|c| c.set(0));
    SETTINGS_APPLY.with(|c| c.set(0));
    TEX_LAYERS.with(|c| c.set(0));
    CELL_READS.with(|c| c.set(0));
    CELL_WRITES.with(|c| c.set(0));
}

/// A `CellStore::block_at` on a loaded world (scheduler → chunk).
pub fn note_cell_read() {
    CELL_READS.with(|c| c.set(c.get() + 1));
}

/// A `CellStore::set_block` on a loaded world (scheduler → chunk).
pub fn note_cell_write() {
    CELL_WRITES.with(|c| c.set(c.get() + 1));
}

/// Chunk reads since the last [`reset`].
pub fn cell_reads() -> u64 {
    CELL_READS.with(Cell::get)
}

/// Chunk writes since the last [`reset`].
pub fn cell_writes() -> u64 {
    CELL_WRITES.with(Cell::get)
}

fn bump(layout: Layout) {
    ALLOCS.with(|c| c.set(c.get() + 1));
    BYTES.with(|c| c.set(c.get() + layout.size() as u64));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump(layout);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump(layout);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        BYTES.with(|c| c.set(c.get() + new_size as u64));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Engine entry points a quiet frame must not keep hitting.
#[derive(Clone, Copy, Debug)]
pub enum EngineCall {
    SetSky,
    FrameUniforms,
    SettingsApply,
    TexLayers,
}

/// Record one game-side engine push (skipped cache hits do not call this).
pub fn note_engine(call: EngineCall) {
    match call {
        EngineCall::SetSky => SET_SKY.with(|c| c.set(c.get() + 1)),
        EngineCall::FrameUniforms => UNIFORMS.with(|c| c.set(c.get() + 1)),
        EngineCall::SettingsApply => SETTINGS_APPLY.with(|c| c.set(c.get() + 1)),
        EngineCall::TexLayers => TEX_LAYERS.with(|c| c.set(c.get() + 1)),
    }
}

/// Engine-call counts since the last [`reset`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineCalls {
    pub set_sky: u32,
    pub uniforms: u32,
    pub settings_apply: u32,
    pub tex_layers: u32,
}

/// Snapshot of [`note_engine`] tallies for this thread.
pub fn engine_calls() -> EngineCalls {
    EngineCalls {
        set_sky: SET_SKY.with(Cell::get),
        uniforms: UNIFORMS.with(Cell::get),
        settings_apply: SETTINGS_APPLY.with(Cell::get),
        tex_layers: TEX_LAYERS.with(Cell::get),
    }
}
