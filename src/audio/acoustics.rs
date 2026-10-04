//! Pure acoustic logic: window sampling, DDA occlusion trace, and the response
//! nonlinearity. No `&mut`, no statics, no authority — every function is total
//! and depends only on its arguments.

use glam::{DQuat, UVec3};
use voxel_engine::{DVec3, IVec3};

use crate::camera::{direction_from_angles, rotate};
use crate::math::BLOCK_METERS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Response {
    World,
    Ui,
    Ambient,
    Voice,
}

#[derive(Clone, Copy, Debug)]
pub struct Listener {
    pub pos: DVec3, // eye position in world-space blocks; responses use metres
    pub yaw: f32,
    pub pitch: f32,
    /// Body frame the yaw and pitch are measured in. Identity keeps the old world basis.
    pub frame: DQuat,
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
    /// Where the cells live relative to the physical world (identity off round worlds).
    frame: WindowFrame,
}

/// The local map from physical positions (listener, sources) into the frame a window's cells are
/// addressed in. On a round world the window samples storage cells around the eye's storage
/// position, so a physical point maps to `cell_at + to_cells · (p − phys_at)`: the chart's
/// embedding linearised at the capture point, exact enough across a window of 2 × 47 blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowFrame {
    pub phys_at: DVec3,
    pub cell_at: DVec3,
    pub to_cells: glam::DMat3,
}

impl WindowFrame {
    pub const IDENTITY: Self = Self {
        phys_at: DVec3::ZERO,
        cell_at: DVec3::ZERO,
        to_cells: glam::DMat3::IDENTITY,
    };

    #[inline]
    pub fn map(&self, p: DVec3) -> DVec3 {
        self.cell_at + self.to_cells * (p - self.phys_at)
    }
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
        Ok(Self {
            origin,
            size,
            cells,
            frame: WindowFrame::IDENTITY,
        })
    }

    /// The same cells, addressed through `frame` (see [`WindowFrame`]).
    pub fn with_frame(mut self, frame: WindowFrame) -> Self {
        self.frame = frame;
        self
    }

    pub fn frame(&self) -> &WindowFrame {
        &self.frame
    }

    /// Recover the cell buffer so a later capture can refill it in place.
    pub(crate) fn into_cells(self) -> Box<[Cell]> {
        self.cells
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
}

/// Coordinates that have already passed smoothing. `respond`/`audibility` accept
/// nothing else, so a caller cannot feed raw per-frame `trace` output into the
/// nonlinearity — enforced by the type rather than a doc obligation. Minted only
/// by the runtime's `Smoothed` cell, or `coincident` for non-spatial cues.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedCoords(Coords);

impl SmoothedCoords {
    /// Crate-visible so the runtime's smoother (the sole time-varying mint) can build one.
    pub(crate) fn new(distance: f32, occlusion: f32) -> Self {
        Self(Coords {
            distance,
            occlusion,
        })
    }
    /// Distance 0: smoothing is the identity, so this is a valid smoothed value
    /// with nothing to integrate (UI cues).
    pub(crate) fn coincident() -> Self {
        Self::new(0.0, 0.0)
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
///
/// `from` and `to` are physical positions; the distance is measured between them and the walk runs
/// through the window's own frame ([`WindowFrame`]). Only the part of the segment inside the window
/// is walked cell by cell: everything outside reads `Unloaded`, so it is charged in closed form. The
/// work is bounded by the window's size whatever the endpoints (a source a billion blocks away once
/// walked a billion cells on the main thread).
pub fn trace(win: &AcousticWindow, from: DVec3, to: DVec3) -> Coords {
    let len_phys = (to - from).length();
    let distance = (len_phys * BLOCK_METERS) as f32;
    let (from, to) = (win.frame.map(from), win.frame.map(to));
    let delta = to - from;
    let len = delta.length();

    // Degenerate or non-finite ray: no traversal, zero occlusion.
    if !len.is_finite() || len <= f64::EPSILON || !from.is_finite() || !to.is_finite() {
        return Coords {
            distance: if distance.is_finite() { distance } else { 0.0 },
            occlusion: 0.0,
        };
    }

    let dir = delta / len;
    // The segment's parameter range inside the window box (slab test); outside it every cell is
    // `Unloaded`.
    let lo = win.origin.as_dvec3();
    let hi = lo + win.size.as_dvec3();
    let (mut t0, mut t1) = (0.0f64, len);
    for a in 0..3 {
        let (f, d) = (from[a], dir[a]);
        if d == 0.0 {
            if f < lo[a] || f >= hi[a] {
                t1 = -1.0;
            }
            continue;
        }
        let (ta, tb) = ((lo[a] - f) / d, (hi[a] - f) / d);
        t0 = t0.max(ta.min(tb));
        t1 = t1.min(ta.max(tb));
    }
    let unloaded = UNLOADED_ABSORPTION as f64 * BLOCK_METERS / 255.0;
    if t1 <= t0 {
        return Coords {
            distance,
            occlusion: (len * unloaded) as f32,
        };
    }
    let outside = t0 + (len - t1);
    let (from, len) = (from + dir * t0, t1 - t0);
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

    let mut occ = outside * unloaded;
    let mut t = 0.0f64;
    // Bound the walk by the number of cells the clipped segment can cross (at most the window's
    // three dimensions) so it always terminates.
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
            occ += cell_absorption(win.cell(cell)) as f64 * thickness * BLOCK_METERS / 255.0;
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

/// The camera's canonical yaw/pitch basis: yaw zero looks +X and positive yaw
/// turns toward +Z. Panning reads listener-local components.
fn listener_basis(l: &Listener) -> (DVec3, DVec3, DVec3) {
    let forward = rotate(l.frame, direction_from_angles(l.yaw, l.pitch));
    let (sin_yaw, cos_yaw) = (l.yaw as f64).sin_cos();
    // Yaw-only right stays defined while looking straight up/down.
    let right = rotate(l.frame, DVec3::new(-sin_yaw, 0.0, cos_yaw));
    let up = right.cross(forward);
    (forward, right, up)
}

/// The spatial gain base shared by `respond` and `audibility`: the
/// distance×occlusion attenuation before any authored/clamp factors. Because
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
    let Coords {
        distance,
        occlusion,
    } = sc.0;
    if let Response::Ui = r {
        return Dsp {
            gain: 1.0,
            lowpass_hz: LP_MAX_HZ,
            pan: None,
        };
    }

    let gain = spatial_base(r, distance, occlusion).clamp(0.0, 1.0);
    let lowpass_hz = occl_lp(occlusion);

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

    Dsp {
        gain,
        lowpass_hz,
        pan,
    }
}

/// Cheap audibility upper bound for ranking: `spatial_base(r,c) * gain`, the
/// SAME `spatial_base` `respond` uses, so the ranking curve can't drift off the
/// applied curve. Admissibility: a clip layer's applied level is
/// `respond().gain · occ_gain · lgain`, where `respond().gain ≤ spatial_base`
/// (the clamp only lowers). The caller passes `gain = occ_gain · max_lgain`
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
    use glam::DQuat;

    fn window(size: u32, fill: Cell) -> AcousticWindow {
        let n = (size * size * size) as usize;
        AcousticWindow::new(
            IVec3::ZERO,
            UVec3::splat(size),
            vec![fill; n].into_boxed_slice(),
        )
        .unwrap()
    }

    #[test]
    fn trace_is_total() {
        let unloaded = window(4, Cell::Unloaded);
        let open = window(4, Cell::Open);

        // All-Unloaded window: finite, non-negative occlusion.
        let c = trace(
            &unloaded,
            DVec3::new(0.5, 0.5, 0.5),
            DVec3::new(3.5, 3.5, 3.5),
        );
        assert!(c.occlusion.is_finite() && c.occlusion >= 0.0);
        assert!(c.distance.is_finite());

        // from == to (zero length): zero occlusion, zero distance.
        let c = trace(&open, DVec3::splat(1.5), DVec3::splat(1.5));
        assert_eq!(c.distance, 0.0);
        assert_eq!(c.occlusion, 0.0);

        // Corner-grazing exact diagonal (DDA tie on all three axes).
        let c = trace(&unloaded, DVec3::ZERO, DVec3::new(4.0, 4.0, 4.0));
        assert!(c.occlusion.is_finite() && c.occlusion >= 0.0);

        // Endpoints far outside the window: still total, all cells read Unloaded.
        let c = trace(
            &open,
            DVec3::new(-50.0, -50.0, -50.0),
            DVec3::new(50.0, 50.0, 50.0),
        );
        assert!(c.occlusion.is_finite() && c.occlusion >= 0.0);
    }

    #[test]
    fn trace_reports_shared_world_scale_in_metres() {
        let open = window(4, Cell::Open);
        let c = trace(&open, DVec3::new(0.5, 0.5, 0.5), DVec3::new(1.5, 0.5, 0.5));
        assert!((c.distance - BLOCK_METERS as f32).abs() < 1e-6);
    }

    /// Midpoint-rule reference for `trace`'s occlusion in the window's own frame.
    fn occlusion_by_sampling(win: &AcousticWindow, from: DVec3, to: DVec3, steps: usize) -> f64 {
        let (from, to) = (win.frame.map(from), win.frame.map(to));
        let dt = (to - from).length() / steps as f64;
        (0..steps)
            .map(|i| {
                let p = from + (to - from) * ((i as f64 + 0.5) / steps as f64);
                cell_absorption(win.cell(p.floor().as_ivec3())) as f64 * dt * BLOCK_METERS / 255.0
            })
            .sum()
    }

    /// A window of open cells with a few walls of different absorption.
    fn walled(size: u32) -> AcousticWindow {
        let n = size as usize;
        let cells: Vec<Cell> = (0..n * n * n)
            .map(|i| match (i % n, (i / n) % n, i / (n * n)) {
                (3, _, _) => Cell::Solid { absorption: 255 },
                (_, 5, z) if z > 2 => Cell::Solid { absorption: 90 },
                (6, y, 4) if y < 6 => Cell::Unloaded,
                _ => Cell::Open,
            })
            .collect();
        AcousticWindow::new(IVec3::new(-2, -1, 0), UVec3::splat(size), cells.into_boxed_slice()).unwrap()
    }

    #[test]
    fn segments_leaving_the_window_charge_the_outside_as_unloaded() {
        let win = walled(8);
        let cases = [
            (DVec3::new(-1.3, 2.2, 3.7), DVec3::new(5.1, 4.9, 6.2)),
            (DVec3::new(-20.0, 3.5, 4.5), DVec3::new(30.0, 3.5, 4.5)),
            (DVec3::new(0.25, -9.0, 2.5), DVec3::new(1.75, 12.0, 6.5)),
            (DVec3::new(-30.0, -30.0, -30.0), DVec3::new(-20.0, -25.0, -21.0)),
            (DVec3::new(1.5, 2.5, 3.5), DVec3::new(4.5, 5.5, 6.5)),
        ];
        for (from, to) in cases {
            let got = trace(&win, from, to).occlusion as f64;
            let want = occlusion_by_sampling(&win, from, to, 400_000);
            assert!((got - want).abs() <= 1e-3 * want.max(1.0), "{from} -> {to}: {got} vs {want}");
        }
    }

    /// A source a billion blocks away (a storage cell traced from a physical listener) costs a
    /// window's worth of steps, not a billion: the outside is charged in closed form.
    #[test]
    fn a_source_a_billion_blocks_away_is_traced_in_closed_form() {
        let open = window(8, Cell::Open);
        let (from, to) = (DVec3::splat(4.0), DVec3::new(1.1e9, 1.5e7, 1.2e8));
        let start = std::time::Instant::now();
        let c = trace(&open, from, to);
        assert!(start.elapsed() < std::time::Duration::from_millis(20), "took {:?}", start.elapsed());
        let len = (to - from).length();
        let inside = 4.0 * len / (to.x - from.x);
        let want = (len - inside) * UNLOADED_ABSORPTION as f64 * BLOCK_METERS / 255.0;
        assert!((c.occlusion as f64 - want).abs() <= 1e-6 * want, "{} vs {want}", c.occlusion);
    }

    /// On a round world the cells are storage cells while listener and source are physical: the
    /// window's frame carries them over, so a wall between them is heard and the distance stays
    /// physical.
    #[test]
    fn the_window_frame_maps_physical_endpoints_onto_its_cells() {
        let x0 = 1_100_000_000;
        // A wall at storage x = x0 + 4.
        let cells: Vec<Cell> = (0..512)
            .map(|i| if i % 8 == 4 { Cell::Solid { absorption: 255 } } else { Cell::Open })
            .collect();
        // Physical X runs along storage Z, Y along Y, and physical -Z along storage +X with one storage
        // cell every 1.02 physical blocks.
        let to_cells = glam::DMat3::from_cols(DVec3::Z, DVec3::Y, DVec3::new(-1.0 / 1.02, 0.0, 0.0));
        let listener = DVec3::new(10.0, 50.0, -20.0);
        let frame = WindowFrame { phys_at: listener, cell_at: DVec3::new(x0 as f64 + 2.5, 3.5, 3.5), to_cells };
        let win = AcousticWindow::new(IVec3::new(x0, 0, 0), UVec3::splat(8), cells.into_boxed_slice())
            .unwrap()
            .with_frame(frame);
        let behind_wall = listener - DVec3::Z * (3.0 * 1.02);
        let c = trace(&win, listener, behind_wall);
        assert!((c.distance as f64 - 3.0 * 1.02 * BLOCK_METERS).abs() < 1e-5, "physical distance: {}", c.distance);
        assert!((c.occlusion as f64 - BLOCK_METERS).abs() < 1e-6, "one full wall cell: {}", c.occlusion);
        let in_front = listener - DVec3::Z * 1.02;
        assert_eq!(trace(&win, listener, in_front).occlusion, 0.0, "nothing between");
    }

    #[test]
    fn panning_uses_the_camera_yaw_convention() {
        fn pan(listener: Listener, source: DVec3) -> [f32; 3] {
            respond(
                Response::World,
                SmoothedCoords::new(2.0, 0.0),
                &listener,
                Some(source),
            )
            .pan
            .expect("non-coincident source")
        }

        let listener = Listener {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            frame: DQuat::IDENTITY,
        };
        assert!(
            pan(listener, DVec3::X)[0].abs() < 1e-6,
            "front must be centered"
        );
        assert!(pan(listener, DVec3::Z)[0] > 0.99, "+Z is right at yaw zero");
        assert!(
            pan(listener, -DVec3::Z)[0] < -0.99,
            "-Z is left at yaw zero"
        );

        let turned = Listener {
            yaw: std::f32::consts::FRAC_PI_2,
            ..listener
        };
        assert!(
            pan(turned, DVec3::Z)[0].abs() < 1e-6,
            "turned front must center"
        );
        assert!(
            pan(turned, -DVec3::X)[0] > 0.99,
            "-X is right after quarter-turn"
        );
    }

    // audibility is an admissible upper bound on the applied level
    // (respond().gain * authored_gain), unconditionally over gain ∈ [0, 4].
    #[test]
    fn audibility_dominates_applied_gain() {
        let listener = Listener {
            pos: DVec3::ZERO,
            yaw: 0.3,
            pitch: -0.2,
            frame: DQuat::IDENTITY,
        };
        let responses = [
            Response::World,
            Response::Ambient,
            Response::Voice,
            Response::Ui,
        ];
        for &r in &responses {
            for &dist in &[0.0f32, 0.4, 2.0, 8.0, 40.0] {
                for &occl in &[0.0f32, 1.0, 5.0, 25.0] {
                    let sc = SmoothedCoords::new(dist, occl);
                    for &g in &[0.0f32, 0.25, 1.0, 4.0] {
                        let ub = audibility(r, sc, g);
                        let applied =
                            respond(r, sc, &listener, Some(DVec3::new(dist as f64, 0.0, 0.0))).gain
                                * g;
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
