//! Best-effort host, GPU, display, and source-control inventory for reports.

use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use std::fs;
use std::path::Path;
use std::process::Command;

use ash::vk;
use voxel_engine::Engine;

use crate::settings::Settings;

use super::json::Json;

const BUILD_GAME_REV: Option<&str> = option_env!("WATT_BUILD_GAME_REV");
const BUILD_GAME_DIRTY: Option<&str> = option_env!("WATT_BUILD_GAME_DIRTY");
const BUILD_ENGINE_REV: Option<&str> = option_env!("WATT_BUILD_ENGINE_REV");
const BUILD_ENGINE_DIRTY: Option<&str> = option_env!("WATT_BUILD_ENGINE_DIRTY");

#[derive(Default)]
pub(super) struct SystemInfo {
    hostname: Option<String>,
    os: Option<String>,
    kernel: Option<String>,
    architecture: String,
    cpu: CpuInfo,
    memory_total_bytes: Option<u64>,
    memory_limit_bytes: Option<u64>,
    gpu: GpuProbe,
    displays: Vec<MonitorInfo>,
    session_type: Option<String>,
    desktop: Option<String>,
}

impl SystemInfo {
    pub(super) fn collect() -> Self {
        Self {
            hostname: read_trimmed("/etc/hostname"),
            os: os_pretty_name(),
            kernel: read_trimmed("/proc/sys/kernel/osrelease"),
            architecture: std::env::consts::ARCH.to_string(),
            cpu: CpuInfo::collect(),
            memory_total_bytes: meminfo_kib("MemTotal").map(|kib| kib * 1024),
            memory_limit_bytes: cgroup_memory_limit(),
            gpu: GpuProbe::collect(),
            displays: connected_monitors(),
            session_type: std::env::var("XDG_SESSION_TYPE").ok(),
            desktop: std::env::var("XDG_CURRENT_DESKTOP").ok(),
        }
    }

    pub(super) fn to_json(&self) -> Json {
        Json::object(vec![
            ("hostname", Json::optional_str(self.hostname.as_deref())),
            ("os", Json::optional_str(self.os.as_deref())),
            ("kernel", Json::optional_str(self.kernel.as_deref())),
            ("architecture", Json::from(self.architecture.as_str())),
            ("cpu", self.cpu.to_json()),
            (
                "memory_total_bytes",
                Json::optional_u64(self.memory_total_bytes),
            ),
            (
                "memory_limit_bytes",
                Json::optional_u64(self.memory_limit_bytes),
            ),
            ("gpu", self.gpu.to_json()),
            (
                "connected_displays",
                Json::array(self.displays.iter().map(MonitorInfo::to_json).collect()),
            ),
            (
                "session_type",
                Json::optional_str(self.session_type.as_deref()),
            ),
            ("desktop", Json::optional_str(self.desktop.as_deref())),
        ])
    }

    pub(super) fn gpu_name(&self) -> &str {
        self.gpu.used.as_deref().unwrap_or("unknown")
    }
}

#[derive(Default)]
struct CpuInfo {
    model: Option<String>,
    vendor: Option<String>,
    logical_cores: usize,
    physical_cores: Option<usize>,
    governors: Vec<String>,
    boost: Option<bool>,
}

impl CpuInfo {
    fn collect() -> Self {
        let text = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let field = |name: &str| {
            text.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == name).then(|| value.trim().to_string())
            })
        };
        let mut physical = BTreeSet::new();
        for block in text.split("\n\n") {
            let mut package = None;
            let mut core = None;
            for line in block.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                match key.trim() {
                    "physical id" => package = Some(value.trim().to_string()),
                    "core id" => core = Some(value.trim().to_string()),
                    _ => {}
                }
            }
            if let (Some(package), Some(core)) = (package, core) {
                physical.insert((package, core));
            }
        }
        let governors = glob_values("/sys/devices/system/cpu", "cpu", "cpufreq/scaling_governor");
        let boost = read_trimmed("/sys/devices/system/cpu/cpufreq/boost")
            .and_then(|v| match v.as_str() {
                "0" => Some(false),
                "1" => Some(true),
                _ => None,
            })
            .or_else(|| {
                read_trimmed("/sys/devices/system/cpu/intel_pstate/no_turbo").and_then(|v| match v
                    .as_str()
                {
                    "0" => Some(true),
                    "1" => Some(false),
                    _ => None,
                })
            });
        Self {
            model: field("model name").or_else(|| field("Hardware")),
            vendor: field("vendor_id"),
            logical_cores: std::thread::available_parallelism().map_or(1, usize::from),
            physical_cores: (!physical.is_empty()).then_some(physical.len()),
            governors,
            boost,
        }
    }

    fn to_json(&self) -> Json {
        Json::object(vec![
            ("model", Json::optional_str(self.model.as_deref())),
            ("vendor", Json::optional_str(self.vendor.as_deref())),
            ("logical_cores_available", Json::from(self.logical_cores)),
            (
                "physical_cores",
                self.physical_cores.map_or(Json::Null, Json::from),
            ),
            (
                "scaling_governors",
                Json::array(
                    self.governors
                        .iter()
                        .map(|v| Json::from(v.as_str()))
                        .collect(),
                ),
            ),
            ("boost_enabled", self.boost.map_or(Json::Null, Json::from)),
        ])
    }
}

#[derive(Default)]
struct GpuProbe {
    used: Option<String>,
    selection_method: String,
    devices: Vec<GpuInfo>,
    error: Option<String>,
}

struct GpuInfo {
    name: String,
    device_type: String,
    vendor_id: u32,
    device_id: u32,
    api_version: String,
    driver_name: Option<String>,
    driver_info: Option<String>,
    driver_id: i32,
    local_memory_bytes: u64,
    max_msaa: u32,
    engine_candidate: bool,
    likely_used: bool,
}

impl GpuProbe {
    fn collect() -> Self {
        if let Ok(name) = std::env::var("WATT_BENCH_GPU")
            && !name.trim().is_empty()
        {
            let mut probed = probe_vulkan().unwrap_or_default();
            probed.used = Some(name.trim().to_string());
            probed.selection_method = "WATT_BENCH_GPU override".into();
            return probed;
        }
        probe_vulkan().unwrap_or_else(|error| Self {
            used: None,
            selection_method: "unavailable".into(),
            devices: Vec::new(),
            error: Some(error),
        })
    }

    fn to_json(&self) -> Json {
        Json::object(vec![
            ("used", Json::optional_str(self.used.as_deref())),
            (
                "selection_method",
                Json::from(self.selection_method.as_str()),
            ),
            ("probe_error", Json::optional_str(self.error.as_deref())),
            (
                "devices",
                Json::array(self.devices.iter().map(GpuInfo::to_json).collect()),
            ),
        ])
    }
}

impl GpuInfo {
    fn to_json(&self) -> Json {
        Json::object(vec![
            ("name", Json::from(self.name.as_str())),
            ("type", Json::from(self.device_type.as_str())),
            ("vendor_id", Json::from(self.vendor_id)),
            ("device_id", Json::from(self.device_id)),
            ("vulkan_api", Json::from(self.api_version.as_str())),
            (
                "driver_name",
                Json::optional_str(self.driver_name.as_deref()),
            ),
            (
                "driver_info",
                Json::optional_str(self.driver_info.as_deref()),
            ),
            ("driver_id", Json::from(self.driver_id)),
            (
                "device_local_memory_bytes",
                Json::from(self.local_memory_bytes),
            ),
            ("engine_candidate", Json::from(self.engine_candidate)),
            ("likely_used", Json::from(self.likely_used)),
        ])
    }
}

fn probe_vulkan() -> Result<GpuProbe, String> {
    let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("load Vulkan: {e}"))?;
    let app_name = CString::new("project_watt_cubed_benchmark").expect("static CString");
    let app = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .api_version(vk::API_VERSION_1_3);
    let create = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&create, None) }
        .map_err(|e| format!("create metadata Vulkan instance: {e:?}"))?;
    let result = (|| {
        let physical = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("enumerate Vulkan devices: {e:?}"))?;
        let mut devices = Vec::new();
        for pd in physical {
            let properties = unsafe { instance.get_physical_device_properties(pd) };
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
            unsafe { instance.get_physical_device_properties2(pd, &mut properties2) };
            let memory = unsafe { instance.get_physical_device_memory_properties(pd) };
            let local_memory_bytes = memory.memory_heaps[..memory.memory_heap_count as usize]
                .iter()
                .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|heap| heap.size)
                .sum();
            let max_msaa = max_sample_count(
                properties.limits.framebuffer_color_sample_counts
                    & properties.limits.framebuffer_depth_sample_counts,
            );
            let engine_candidate = engine_feature_candidate(&instance, pd, properties.api_version);
            devices.push(GpuInfo {
                name: c_char_string(&properties.device_name).unwrap_or_else(|| "unknown".into()),
                device_type: device_type_name(properties.device_type).into(),
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                api_version: version_string(properties.api_version),
                driver_name: c_char_string(&driver.driver_name),
                driver_info: c_char_string(&driver.driver_info),
                driver_id: driver.driver_id.as_raw(),
                local_memory_bytes,
                max_msaa,
                engine_candidate,
                likely_used: false,
            });
        }
        let best_score = devices
            .iter()
            .filter(|d| d.engine_candidate)
            .map(|d| gpu_score(&d.device_type))
            .max();
        let best: Vec<usize> = best_score.map_or_else(Vec::new, |score| {
            devices
                .iter()
                .enumerate()
                .filter(|(_, d)| d.engine_candidate && gpu_score(&d.device_type) == score)
                .map(|(i, _)| i)
                .collect()
        });
        let (used, method) = if best.len() == 1 {
            devices[best[0]].likely_used = true;
            (
                Some(devices[best[0]].name.clone()),
                "unique highest-priority renderer-compatible Vulkan device".to_string(),
            )
        } else if best.is_empty() {
            (
                None,
                "no renderer-compatible Vulkan candidate found".to_string(),
            )
        } else {
            (
                None,
                "ambiguous: multiple equal-priority renderer-compatible devices; set WATT_BENCH_GPU"
                    .to_string(),
            )
        };
        Ok(GpuProbe {
            used,
            selection_method: method,
            devices,
            error: None,
        })
    })();
    unsafe { instance.destroy_instance(None) };
    result
}

fn engine_feature_candidate(instance: &ash::Instance, pd: vk::PhysicalDevice, api: u32) -> bool {
    if vk::api_version_major(api) < 1
        || (vk::api_version_major(api) == 1 && vk::api_version_minor(api) < 3)
    {
        return false;
    }
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f13)
        .push_next(&mut f12);
    unsafe { instance.get_physical_device_features2(pd, &mut features) };
    if f13.dynamic_rendering != vk::TRUE
        || f13.synchronization2 != vk::TRUE
        || f12.draw_indirect_count != vk::TRUE
    {
        return false;
    }
    let extensions = unsafe { instance.enumerate_device_extension_properties(pd) };
    let Ok(extensions) = extensions else {
        return false;
    };
    let has = |wanted: &CStr| {
        extensions.iter().any(|ext| {
            ext.extension_name_as_c_str()
                .is_ok_and(|name| name == wanted)
        })
    };
    let graphics = unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .any(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    graphics && has(ash::khr::swapchain::NAME) && has(ash::khr::push_descriptor::NAME)
}

fn c_char_string<const N: usize>(raw: &[std::ffi::c_char; N]) -> Option<String> {
    let value = unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_string();
    (!value.is_empty()).then_some(value)
}

fn max_sample_count(flags: vk::SampleCountFlags) -> u32 {
    for (bit, n) in [
        (vk::SampleCountFlags::TYPE_8, 8),
        (vk::SampleCountFlags::TYPE_4, 4),
        (vk::SampleCountFlags::TYPE_2, 2),
    ] {
        if flags.contains(bit) {
            return n;
        }
    }
    1
}

/// Device-local heap, framebuffer MSAA ceiling, and the largest connected
/// display — the startup VRAM guard's one probe (same Vulkan path as the
/// benchmark report).
pub(crate) fn graphics_caps() -> (crate::render_config::DeviceCaps, (u32, u32)) {
    let gpu = GpuProbe::collect();
    let chosen = gpu
        .devices
        .iter()
        .find(|d| d.likely_used)
        .or_else(|| {
            gpu.devices
                .iter()
                .filter(|d| d.engine_candidate)
                .max_by_key(|d| d.local_memory_bytes)
        })
        .or_else(|| gpu.devices.iter().max_by_key(|d| d.local_memory_bytes));
    let caps = crate::render_config::DeviceCaps {
        device_local_memory_bytes: chosen.map(|d| d.local_memory_bytes),
        max_msaa: chosen.map(|d| d.max_msaa).unwrap_or(8),
    };
    (caps, largest_display_extent())
}

fn largest_display_extent() -> (u32, u32) {
    connected_monitors()
        .iter()
        .filter_map(|m| m.preferred_mode.map(|(w, h, _)| (w, h)))
        .max_by_key(|&(w, h)| w.saturating_mul(h))
        .unwrap_or((3840, 2160))
}

fn device_type_name(kind: vk::PhysicalDeviceType) -> &'static str {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn gpu_score(kind: &str) -> u8 {
    match kind {
        "discrete" => 100,
        "integrated" => 50,
        "virtual" => 20,
        _ => 10,
    }
}

fn version_string(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

#[derive(Default)]
struct MonitorInfo {
    connector: String,
    enabled: bool,
    manufacturer: Option<String>,
    model: Option<String>,
    serial: Option<u32>,
    physical_mm: Option<(u32, u32)>,
    preferred_mode: Option<(u32, u32, f64)>,
    modes: Vec<String>,
}

impl MonitorInfo {
    fn to_json(&self) -> Json {
        Json::object(vec![
            ("connector", Json::from(self.connector.as_str())),
            ("enabled", Json::from(self.enabled)),
            (
                "manufacturer",
                Json::optional_str(self.manufacturer.as_deref()),
            ),
            ("model", Json::optional_str(self.model.as_deref())),
            ("serial", self.serial.map_or(Json::Null, Json::from)),
            (
                "physical_size_mm",
                self.physical_mm.map_or(Json::Null, |(w, h)| {
                    Json::object(vec![("width", Json::from(w)), ("height", Json::from(h))])
                }),
            ),
            (
                "edid_preferred_mode",
                self.preferred_mode.map_or(Json::Null, |(w, h, hz)| {
                    Json::object(vec![
                        ("width", Json::from(w)),
                        ("height", Json::from(h)),
                        ("refresh_hz", Json::number(hz)),
                    ])
                }),
            ),
            (
                "advertised_modes",
                Json::array(self.modes.iter().map(|v| Json::from(v.as_str())).collect()),
            ),
        ])
    }
}

fn connected_monitors() -> Vec<MonitorInfo> {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };
    let mut monitors = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.contains('-') {
            continue;
        }
        let path = entry.path();
        if read_trimmed(path.join("status")).as_deref() != Some("connected") {
            continue;
        }
        let edid = fs::read(path.join("edid")).unwrap_or_default();
        let parsed = parse_edid(&edid);
        let modes = fs::read_to_string(path.join("modes"))
            .unwrap_or_default()
            .lines()
            .take(32)
            .map(str::to_string)
            .collect();
        monitors.push(MonitorInfo {
            connector: name,
            enabled: read_trimmed(path.join("enabled")).as_deref() == Some("enabled"),
            manufacturer: parsed.manufacturer,
            model: parsed.model,
            serial: parsed.serial,
            physical_mm: parsed.physical_mm,
            preferred_mode: parsed.preferred_mode,
            modes,
        });
    }
    monitors.sort_by(|a, b| a.connector.cmp(&b.connector));
    monitors
}

#[derive(Default)]
struct EdidInfo {
    manufacturer: Option<String>,
    model: Option<String>,
    serial: Option<u32>,
    physical_mm: Option<(u32, u32)>,
    preferred_mode: Option<(u32, u32, f64)>,
}

fn parse_edid(edid: &[u8]) -> EdidInfo {
    if edid.len() < 128 || edid[..8] != [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00] {
        return EdidInfo::default();
    }
    let code = u16::from_be_bytes([edid[8], edid[9]]);
    let manufacturer: String = [10, 5, 0]
        .into_iter()
        .map(|shift| (((code >> shift) & 0x1f) as u8 + b'A' - 1) as char)
        .collect();
    let manufacturer = manufacturer
        .chars()
        .all(|c| c.is_ascii_uppercase())
        .then_some(manufacturer);
    let serial = u32::from_le_bytes([edid[12], edid[13], edid[14], edid[15]]);
    let physical_mm = (edid[21] != 0 && edid[22] != 0)
        .then_some((u32::from(edid[21]) * 10, u32::from(edid[22]) * 10));
    let mut model = None;
    let mut preferred_mode = None;
    for offset in [54usize, 72, 90, 108] {
        let d = &edid[offset..offset + 18];
        if d[0] == 0 && d[1] == 0 && d[3] == 0xfc {
            let value = String::from_utf8_lossy(&d[5..18])
                .trim_matches(['\0', '\n', '\r', ' '])
                .to_string();
            if !value.is_empty() {
                model = Some(value);
            }
        } else if preferred_mode.is_none() {
            let pixel_clock = u64::from(u16::from_le_bytes([d[0], d[1]])) * 10_000;
            if pixel_clock != 0 {
                let h_active = u32::from(d[2]) | (u32::from(d[4] & 0xf0) << 4);
                let h_blank = u32::from(d[3]) | (u32::from(d[4] & 0x0f) << 8);
                let v_active = u32::from(d[5]) | (u32::from(d[7] & 0xf0) << 4);
                let v_blank = u32::from(d[6]) | (u32::from(d[7] & 0x0f) << 8);
                let total = u64::from(h_active + h_blank) * u64::from(v_active + v_blank);
                if total != 0 {
                    preferred_mode = Some((h_active, v_active, pixel_clock as f64 / total as f64));
                }
            }
        }
    }
    EdidInfo {
        manufacturer,
        model,
        serial: (serial != 0).then_some(serial),
        physical_mm,
        preferred_mode,
    }
}

pub(super) fn display_json(eng: &Engine, settings: &Settings, system: Option<&SystemInfo>) -> Json {
    let monitors = system.map_or(&[][..], |info| info.displays.as_slice());
    let width = eng.screen_width().max(1) as u32;
    let height = eng.screen_height().max(1) as u32;
    let scale = eng.render_scale();
    let enabled: Vec<&MonitorInfo> = monitors.iter().filter(|m| m.enabled).collect();
    let matching: Vec<&MonitorInfo> = enabled
        .iter()
        .copied()
        .filter(|m| {
            m.preferred_mode
                .is_some_and(|(w, h, _)| (w, h) == (width, height))
        })
        .collect();
    let (connector, model, selection) = if let [monitor] = matching.as_slice() {
        (
            Some(monitor.connector.as_str()),
            monitor.model.as_deref(),
            "unique enabled DRM connector matching the window dimensions",
        )
    } else if let [monitor] = enabled.as_slice() {
        (
            Some(monitor.connector.as_str()),
            monitor.model.as_deref(),
            "only enabled DRM connector",
        )
    } else if enabled.is_empty() {
        (None, None, "window dimensions known; monitor unavailable")
    } else {
        (None, None, "ambiguous: multiple enabled DRM connectors")
    };
    Json::object(vec![
        ("window_width", Json::from(width)),
        ("window_height", Json::from(height)),
        (
            "render_width",
            Json::from(((width as f32 * scale) as u32).max(1)),
        ),
        (
            "render_height",
            Json::from(((height as f32 * scale) as u32).max(1)),
        ),
        ("render_scale", Json::number(f64::from(scale))),
        ("fullscreen", Json::from(eng.fullscreen())),
        ("vsync", Json::from(eng.vsync())),
        ("target_fps", Json::from(eng.target_fps())),
        ("msaa", Json::from(eng.msaa())),
        ("fov_degrees", Json::number(f64::from(settings.fov))),
        ("monitor_connector", Json::optional_str(connector)),
        ("monitor_model", Json::optional_str(model)),
        ("monitor_selection_method", Json::from(selection)),
    ])
}

pub(super) fn settings_json(s: &Settings, eng: &Engine) -> Json {
    Json::object(vec![
        ("preset", Json::from(s.preset.label().to_ascii_lowercase())),
        ("render_distance", Json::from(s.render_distance)),
        ("vertical_distance", Json::from(s.vertical_distance)),
        ("lod_enabled", Json::from(s.lod2)),
        ("lod_levels", Json::from(s.lod_levels)),
        ("lod_detail", Json::from(s.lod_detail)),
        ("stream_hz", Json::from(s.stream_hz)),
        ("physics_hz", Json::from(s.physics_hz)),
        ("sky_hz", Json::from(s.sky_hz)),
        ("mod_hz", Json::from(s.mod_hz)),
        ("simulation", Json::from(s.simulation)),
        ("mod_logic", Json::from(s.mod_logic)),
        ("autosave", Json::from(s.autosave)),
        ("lighting", Json::from(s.lighting)),
        ("ambient_occlusion", Json::from(s.ao)),
        ("occlusion", Json::from(s.occlusion)),
        ("face_culling", Json::from(eng.cull_faces())),
        (
            "hud_mode",
            Json::from(format!("{:?}", s.hud_mode).to_ascii_lowercase()),
        ),
        ("minimap", Json::from(s.minimap)),
        ("mod_hud", Json::from(s.mod_hud)),
        ("player_models", Json::from(s.player_models)),
        ("name_tags", Json::from(s.name_tags)),
        (
            "render_lanes",
            Json::object(vec![
                ("taa", Json::from(s.taa)),
                ("fog", Json::from(s.fog)),
                ("blocklight", Json::from(s.blocklight)),
                ("ambient", Json::from(s.ambient)),
                ("sunlight", Json::from(s.sunlight)),
                ("exposure", Json::from(s.exposure)),
                ("bloom", Json::from(s.bloom)),
                ("godrays", Json::from(s.godrays)),
                ("shadows", Json::from(s.shadows)),
                ("sky", Json::from(s.sky)),
                ("clouds", Json::from(s.clouds)),
                ("weather", Json::from(s.weather)),
                ("stars", Json::from(s.stars)),
                ("day_night", Json::from(s.day_night)),
                ("vrs", Json::from(s.vrs)),
                ("water_animation", Json::from(s.water_anim)),
                ("vignette", Json::from(s.vignette)),
            ]),
        ),
    ])
}

pub(super) fn software_json() -> Json {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let renderer = project.parent().map(|p| p.join("voxel-engine"));
    let project_status = git_status(project);
    let renderer_status = renderer.as_deref().and_then(git_status);
    let project_revision = std::env::var("WATT_BENCH_GAME_REV")
        .ok()
        .or_else(|| BUILD_GAME_REV.map(str::to_owned))
        .or_else(|| git_revision(project));
    let renderer_revision = std::env::var("WATT_BENCH_ENGINE_REV")
        .ok()
        .or_else(|| BUILD_ENGINE_REV.map(str::to_owned))
        .or_else(|| renderer.as_deref().and_then(git_revision));
    let project_dirty = project_status
        .as_ref()
        .map(|status| status.dirty)
        .or_else(|| build_dirty(BUILD_GAME_DIRTY));
    let renderer_dirty = renderer_status
        .as_ref()
        .map(|status| status.dirty)
        .or_else(|| build_dirty(BUILD_ENGINE_DIRTY));
    Json::object(vec![
        ("package", Json::from(env!("CARGO_PKG_NAME"))),
        ("version", Json::from(env!("CARGO_PKG_VERSION"))),
        (
            "build_profile",
            Json::from(if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }),
        ),
        ("target_os", Json::from(std::env::consts::OS)),
        ("target_arch", Json::from(std::env::consts::ARCH)),
        (
            "project_revision",
            Json::optional_str(project_revision.as_deref()),
        ),
        (
            "project_worktree_dirty",
            project_dirty.map_or(Json::Null, Json::from),
        ),
        (
            "project_changed_paths",
            project_status
                .as_ref()
                .map_or(Json::Null, |s| Json::from(s.changed_paths)),
        ),
        (
            "renderer_revision",
            Json::optional_str(renderer_revision.as_deref()),
        ),
        (
            "renderer_worktree_dirty",
            renderer_dirty.map_or(Json::Null, Json::from),
        ),
        (
            "renderer_changed_paths",
            renderer_status
                .as_ref()
                .map_or(Json::Null, |s| Json::from(s.changed_paths)),
        ),
    ])
}

fn build_dirty(value: Option<&str>) -> Option<bool> {
    value.and_then(|value| match value {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    })
}

struct GitStatus {
    dirty: bool,
    changed_paths: usize,
}

/// Best-effort dirty-state probe. It runs after measurement, and absence of
/// `git` is represented as JSON null rather than incorrectly claiming clean.
fn git_status(worktree: &Path) -> Option<GitStatus> {
    let output = Command::new("git")
        .arg("-C")
        .arg(worktree)
        .args(["status", "--porcelain=v1", "--untracked-files=normal"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let changed_paths = String::from_utf8_lossy(&output.stdout).lines().count();
    Some(GitStatus {
        dirty: changed_paths != 0,
        changed_paths,
    })
}

fn git_revision(worktree: &Path) -> Option<String> {
    let dot_git = worktree.join(".git");
    let git_dir = if dot_git.is_dir() {
        dot_git
    } else {
        let marker = fs::read_to_string(dot_git).ok()?;
        let relative = marker.trim().strip_prefix("gitdir:")?.trim();
        worktree.join(relative)
    };
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    if !head.starts_with("ref: ") {
        return Some(head.to_string());
    }
    let reference = &head[5..];
    fs::read_to_string(git_dir.join(reference))
        .ok()
        .map(|s| s.trim().to_string())
        .or_else(|| {
            fs::read_to_string(git_dir.join("packed-refs"))
                .ok()?
                .lines()
                .find_map(|line| {
                    let (hash, name) = line.split_once(' ')?;
                    (name == reference).then(|| hash.to_string())
                })
        })
}

pub(super) fn resident_bytes() -> Option<u64> {
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // Linux page size is overwhelmingly 4096; `/proc/self/status` gives an
    // exact KiB value and is preferred when present.
    meminfo_path_kib("/proc/self/status", "VmRSS")
        .map(|kib| kib * 1024)
        .or(Some(pages * 4096))
}

fn meminfo_kib(key: &str) -> Option<u64> {
    meminfo_path_kib("/proc/meminfo", key)
}

fn meminfo_path_kib(path: impl AsRef<Path>, key: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name != key {
            return None;
        }
        value.split_whitespace().next()?.parse().ok()
    })
}

fn cgroup_memory_limit() -> Option<u64> {
    let value = read_trimmed("/sys/fs/cgroup/memory.max")?;
    if value == "max" {
        return None;
    }
    value.parse().ok()
}

fn os_pretty_name() -> Option<String> {
    fs::read_to_string("/etc/os-release")
        .ok()?
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix("PRETTY_NAME=")?;
            Some(value.trim_matches('"').replace("\\\"", "\""))
        })
}

fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    let value = fs::read_to_string(path).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn glob_values(root: &str, prefix: &str, suffix: &str) -> Vec<String> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut values = BTreeSet::new();
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().starts_with(prefix) {
            continue;
        }
        if let Some(value) = read_trimmed(entry.path().join(suffix)) {
            values.insert(value);
        }
    }
    values.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::parse_edid;

    #[test]
    fn parses_a_minimal_edid_identity_and_timing() {
        let mut edid = [0u8; 128];
        edid[..8].copy_from_slice(&[0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00]);
        // Manufacturer ABC.
        edid[8..10].copy_from_slice(&0x0443u16.to_be_bytes());
        edid[12..16].copy_from_slice(&1234u32.to_le_bytes());
        edid[21] = 60;
        edid[22] = 34;
        // 1920x1080 @ ~60 Hz detailed timing, 148.5 MHz.
        let d = &mut edid[54..72];
        d[0..2].copy_from_slice(&14850u16.to_le_bytes());
        d[2] = 0x80;
        d[3] = 0x18;
        d[4] = 0x71;
        d[5] = 0x38;
        d[6] = 0x2d;
        d[7] = 0x40;
        let parsed = parse_edid(&edid);
        assert_eq!(parsed.manufacturer.as_deref(), Some("ABC"));
        assert_eq!(parsed.serial, Some(1234));
        assert_eq!(parsed.physical_mm, Some((600, 340)));
        let (w, h, hz) = parsed.preferred_mode.unwrap();
        assert_eq!((w, h), (1920, 1080));
        assert!((hz - 60.0).abs() < 0.1);
    }
}
