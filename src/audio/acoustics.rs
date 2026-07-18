//! Pure acoustic logic: window sampling, DDA occlusion trace, and the response
//! nonlinearity. No `&mut`, no statics, no authority — every function is total
//! and depends only on its arguments.

use glam::{DVec3, IVec3, UVec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Response {
    World,
    Ui,
    Ambient,
    Voice,
}

#[derive(Clone, Copy, Debug)]
pub enum Medium {
    Air,
    Water,
}

#[derive(Clone, Copy, Debug)]
pub struct Listener {
    pub pos: DVec3, // eye position, world space (f64, 1 voxel = 1 m)
    pub yaw: f32,
    pub pitch: f32,
    pub medium: Medium,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Open,
    Solid { absorption: u8 }, // absorption 0..=255 per metre
    Unloaded,
}

/// Absorption charged to an `Unloaded` cell. Unknown geometry is treated as
/// lightly occluding rather than transparent: far/streamed-out terrain should
/// dampen, not leak, sound — but not as hard as fully solid rock, so a listener
/// near the loaded edge still hears through a thin unloaded margin. Keeps
/// `trace` total by charging a constant instead of branching to a panic.
const UNLOADED_ABSORPTION: u8 = 16;

pub struct AcousticWindow {
    origin: IVec3,
    size: UVec3,
    cells: Box<[Cell]>,
}

impl AcousticWindow {
    pub fn new(origin: IVec3, size: UVec3, cells: Box<[Cell]>) -> Result<Self, WindowError> {
        if size.x > MAX_WINDOW_DIM || size.y > MAX_WINDOW_DIM || size.z > MAX_WINDOW_DIM {
            return Err(WindowError::OverLimit);
        }
        let count = size.x as usize * size.y as usize * size.z as usize;
        if count != cells.len() {
            return Err(WindowError::SizeMismatch);
        }
        Ok(Self { origin, size, cells })
    }

    /// Cell at a world coordinate; outside the window it reads `Unloaded`.
    pub fn cell(&self, world: IVec3) -> Cell {
        let local = world - self.origin;
        if local.x < 0
            || local.y < 0
            || local.z < 0
            || local.x as u32 >= self.size.x
            || local.y as u32 >= self.size.y
            || local.z as u32 >= self.size.z
        {
            return Cell::Unloaded;
        }
        // x fastest, then y, then z.
        let idx = local.x as usize
            + local.y as usize * self.size.x as usize
            + local.z as usize * self.size.x as usize * self.size.y as usize;
        self.cells[idx]
    }
}

pub const MAX_WINDOW_DIM: u32 = 96; // 2 × radius 48 m — profile before raising

#[derive(Debug)]
pub enum WindowError {
    SizeMismatch,
    OverLimit,
}

#[derive(Clone, Copy, Debug)]
pub struct Coords {
    pub distance: f32,
    pub occlusion: f32,
    pub medium: Medium,
}

/// Coordinates that have already passed smoothing. `respond`/`audibility` accept
/// nothing else, so a caller cannot feed raw per-frame `trace` output into the
/// nonlinearity — enforced by the type rather than a doc obligation. Minted only
/// by the runtime's `Smoothed` cell, or `coincident` for non-spatial cues.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedCoords(Coords);

impl SmoothedCoords {
    /// Crate-visible so the runtime's smoother (the sole time-varying mint) can build one.
    pub(crate) fn new(distance: f32, occlusion: f32, medium: Medium) -> Self {
        Self(Coords { distance, occlusion, medium })
    }
    /// Distance 0: smoothing is the identity, so this is a valid smoothed value
    /// with nothing to integrate (UI cues).
    pub(crate) fn coincident(medium: Medium) -> Self {
        Self::new(0.0, 0.0, medium)
    }
}

fn cell_absorption(cell: Cell) -> u32 {
    match cell {
        Cell::Open => 0,
        Cell::Solid { absorption } => absorption as u32,
        Cell::Unloaded => UNLOADED_ABSORPTION as u32,
    }
}

/// Amanatides–Woo DDA from listener to source accumulating absorption × thickness.
/// `occlusion` is in full-absorption-metres: Σ (absorption/255) × (metres spent in
/// that cell). Total on `Unloaded` and on degenerate rays — never panics.
pub fn trace(win: &AcousticWindow, from: DVec3, to: DVec3, medium: Medium) -> Coords {
    let delta = to - from;
    let len = delta.length();
    let distance = len as f32;

    // Degenerate or non-finite ray: no traversal, zero occlusion.
    if !len.is_finite() || len <= f64::EPSILON || !from.is_finite() || !to.is_finite() {
        return Coords {
            distance: if distance.is_finite() { distance } else { 0.0 },
            occlusion: 0.0,
            medium,
        };
    }

    let dir = delta / len;
    let mut cell = from.floor().as_ivec3();

    // Per-axis DDA setup; a zero component never crosses a boundary (tMax = ∞).
    let mut step = [0i32; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    let f = [from.x, from.y, from.z];
    let d = [dir.x, dir.y, dir.z];
    let c = [cell.x, cell.y, cell.z];
    for a in 0..3 {
        if d[a] > 0.0 {
            step[a] = 1;
            t_max[a] = ((c[a] + 1) as f64 - f[a]) / d[a];
            t_delta[a] = 1.0 / d[a];
        } else if d[a] < 0.0 {
            step[a] = -1;
            t_max[a] = (c[a] as f64 - f[a]) / d[a];
            t_delta[a] = -1.0 / d[a];
        }
    }

    let mut occ = 0.0f64;
    let mut t = 0.0f64;
    // Bound the walk by the number of cells the segment can cross; guarantees
    // termination even if the endpoints sit far outside the window.
    let cap = (len.ceil() as usize) * 3 + 16;
    for _ in 0..cap {
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        let seg_end = t_max[axis].min(len);
        let thickness = seg_end - t;
        if thickness > 0.0 {
            occ += cell_absorption(win.cell(cell)) as f64 * thickness / 255.0;
        }
        if t_max[axis] >= len {
            break;
        }
        t = t_max[axis];
        cell[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }

    Coords {
        distance,
        occlusion: occ as f32,
        medium,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dsp {
    pub gain: f32,
    pub lowpass_hz: f32,
    pub pan: Option<[f32; 3]>,
}

// --- fixed response curves; tune constants behind these names ---
const OCCL_K: f32 = 0.08;
const LP_K: f32 = 0.12;
const LP_MAX_HZ: f32 = 20_000.0;
const LP_MIN_HZ: f32 = 200.0;
const WATER_LP_HZ: f32 = 1_200.0;
const WATER_GAIN: f32 = 0.7;

fn ref_dist(r: Response) -> f32 {
    match r {
        Response::World => 8.0,
        Response::Ambient => 16.0,
        Response::Voice => 10.0,
        Response::Ui => 8.0, // unused: Ui bypasses distance
    }
}

fn dist_gain(r: Response, d: f32) -> f32 {
    let x = d / ref_dist(r);
    1.0 / (1.0 + x * x)
}

fn occl_gain(o: f32) -> f32 {
    (-OCCL_K * o).exp()
}

fn occl_lp(o: f32) -> f32 {
    (LP_MAX_HZ * (-LP_K * o).exp()).clamp(LP_MIN_HZ, LP_MAX_HZ)
}

fn is_water(m: Medium) -> bool {
    matches!(m, Medium::Water)
}

/// Yaw/pitch orthonormal basis: forward, right, up. Right-handed; +yaw turns
/// toward +X, +pitch tilts toward +Y. Panning reads listener-local components.
fn listener_basis(l: &Listener) -> (DVec3, DVec3, DVec3) {
    let (yaw, pitch) = (l.yaw as f64, l.pitch as f64);
    let forward = DVec3::new(
        pitch.cos() * yaw.sin(),
        pitch.sin(),
        pitch.cos() * yaw.cos(),
    )
    .normalize_or_zero();
    // `forward × Y` collapses to 0 at pitch ±90° (forward ∥ Y), which would center
    // every pan. Fall back to the yaw-only horizontal right vector (`forward_flat ×
    // Y`, independent of pitch) so looking straight up/down still resolves L/R.
    let cross = forward.cross(DVec3::Y);
    let right = if cross.length_squared() < 1e-12 {
        DVec3::new(-yaw.cos(), 0.0, yaw.sin())
    } else {
        cross.normalize()
    };
    let up = right.cross(forward);
    (forward, right, up)
}

/// The spatial gain base shared by `respond` and `audibility`: the
/// distance×occlusion attenuation before any authored/water/clamp factors. Because
/// both the ranking bound and the applied level read the SAME curve here, the
/// audibility quantifier cannot drift off the response curve. Ui is non-spatial (1).
fn spatial_base(r: Response, distance: f32, occlusion: f32) -> f32 {
    match r {
        Response::Ui => 1.0,
        _ => dist_gain(r, distance) * occl_gain(occlusion),
    }
}

/// Fixed response curves per Response class. Takes [`SmoothedCoords`] by
/// construction, so raw `trace` output can never reach it.
pub fn respond(r: Response, sc: SmoothedCoords, listener: &Listener, source: Option<DVec3>) -> Dsp {
    let Coords { distance, occlusion, medium } = sc.0;
    if let Response::Ui = r {
        return Dsp { gain: 1.0, lowpass_hz: LP_MAX_HZ, pan: None };
    }

    let media_differ = is_water(listener.medium) != is_water(medium);
    let any_water = is_water(listener.medium) || is_water(medium);

    let water_gain = if media_differ { WATER_GAIN } else { 1.0 };
    let gain = (spatial_base(r, distance, occlusion) * water_gain).clamp(0.0, 1.0);

    let mut lowpass_hz = occl_lp(occlusion);
    if any_water {
        lowpass_hz = lowpass_hz.min(WATER_LP_HZ);
    }

    // Pan is None when coincident (< 0.5 m) or non-spatial.
    let pan = source.and_then(|s| {
        if distance < 0.5 {
            return None;
        }
        let dir = (s - listener.pos).normalize_or_zero();
        if dir == DVec3::ZERO {
            return None;
        }
        let (forward, right, up) = listener_basis(listener);
        Some([
            dir.dot(right) as f32,
            dir.dot(up) as f32,
            dir.dot(forward) as f32,
        ])
    });

    Dsp { gain, lowpass_hz, pan }
}

/// Cheap audibility upper bound for ranking: `spatial_base(r,c) * gain`, the
/// SAME `spatial_base` `respond` uses, so the ranking curve can't drift off the
/// applied curve. Admissibility: a clip layer's applied level is
/// `respond().gain · occ_gain · lgain`, where `respond().gain ≤ spatial_base`
/// (water ≤ 1, clamp only lowers). The caller passes `gain = occ_gain · max_lgain`
/// with `max_lgain = max` over the cue's layer-gain ranges, and every layer's
/// `lgain ≤ max_lgain` (authored gain is bounded to (0, 4] at load), so
/// `applied ≤ spatial_base · occ_gain · max_lgain = audibility`. Emitters/voice
/// carry a single authored gain and pass it directly, so ranking never starves a
/// louder group.
pub fn audibility(r: Response, sc: SmoothedCoords, gain: f32) -> f32 {
    spatial_base(r, sc.0.distance, sc.0.occlusion) * gain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(size: u32, fill: Cell) -> AcousticWindow {
        let n = (size * size * size) as usize;
        AcousticWindow::new(IVec3::ZERO, UVec3::splat(size), vec![fill; n].into_boxed_slice())
            .unwrap()
    }

    #[test]
    fn trace_is_total() {
        let unloaded = window(4, Cell::Unloaded);
        let open = window(4, Cell::Open);

        // All-Unloaded window: finite, non-negative occlusion.
        let c = trace(&unloaded, DVec3::new(0.5, 0.5, 0.5), DVec3::new(3.5, 3.5, 3.5), Medium::Air);
        assert!(c.occlusion.is_finite() && c.occlusion >= 0.0);
        assert!(c.distance.is_finite());

        // from == to (zero length): zero occlusion, zero distance.
        let c = trace(&open, DVec3::splat(1.5), DVec3::splat(1.5), Medium::Air);
        assert_eq!(c.distance, 0.0);
        assert_eq!(c.occlusion, 0.0);

        // Corner-grazing exact diagonal (DDA tie on all three axes).
        let c = trace(&unloaded, DVec3::ZERO, DVec3::new(4.0, 4.0, 4.0), Medium::Air);
        assert!(c.occlusion.is_finite() && c.occlusion >= 0.0);

        // Endpoints far outside the window: still total, all cells read Unloaded.
        let c = trace(&open, DVec3::new(-50.0, -50.0, -50.0), DVec3::new(50.0, 50.0, 50.0), Medium::Water);
        assert!(c.occlusion.is_finite() && c.occlusion >= 0.0);
        assert!(matches!(c.medium, Medium::Water));
    }

    // audibility is an admissible upper bound on the applied level
    // (respond().gain * authored_gain), unconditionally over gain ∈ [0, 4].
    #[test]
    fn audibility_dominates_applied_gain() {
        let listener = Listener { pos: DVec3::ZERO, yaw: 0.3, pitch: -0.2, medium: Medium::Air };
        let responses = [Response::World, Response::Ambient, Response::Voice, Response::Ui];
        for &r in &responses {
            for &dist in &[0.0f32, 0.4, 2.0, 8.0, 40.0] {
                for &occl in &[0.0f32, 1.0, 5.0, 25.0] {
                    for &medium in &[Medium::Air, Medium::Water] {
                        let sc = SmoothedCoords::new(dist, occl, medium);
                        for &g in &[0.0f32, 0.25, 1.0, 4.0] {
                            let ub = audibility(r, sc, g);
                            let applied =
                                respond(r, sc, &listener, Some(DVec3::new(dist as f64, 0.0, 0.0))).gain * g;
                            assert!(
                                ub + 1e-5 >= applied,
                                "audibility {ub} < applied gain {applied} (r={r:?}, d={dist}, o={occl}, g={g})"
                            );
                        }
                    }
                }
            }
        }
    }
}
