# processRRFS.py
import os
import shutil
import datetime
import glob
import json
import math
import sys
import re
import time
import threading
import contextlib
import functools
import concurrent.futures
import urllib.request
import numpy as np
import xarray as xr
import cv2
import boto3
import contourpy
import gzip
import struct
from pyproj import CRS, Transformer

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ.setdefault("TQDM_DISABLE", "1")
sys.stdout.reconfigure(line_buffering=True)

# RRFS forecast hours: the 00/06/12/18z cycles run hourly out to F84; every other cycle stops at F18.
EXTENDED_RUNS = {"00", "06", "12", "18"}
PREFER_EXTENDED_RUNS = False
EXTENDED_LOOKBACK_HOURS = 7
NORMAL_LOOKBACK_HOURS = 3
MODEL_NAME = "rrfs"

# Live-mode settings
POLL_SECONDS = 20
STALL_TIMEOUT_MINUTES = 45
MAX_STEPS_PER_BATCH = 6
MAX_STEP_ATTEMPTS = 3
LATEST_MIN_FRAMES = 1
STAGE_DIR = "run_rrfs_stage"


def forecast_steps_for_run(run_hour):
    return list(range(0, 85)) if str(run_hour).zfill(2) in EXTENDED_RUNS else list(range(0, 19))

MAX_TEXTURE_SIZE = 4096
MAX_CONCURRENT_WORKERS = int(os.environ.get("MAX_WORKERS", "12"))
GZIP_LEVEL = 6
GZIP_THREADS = 4
MAX_CONCURRENT_PARAMS = 2

CONFIG_FILE_PATH = os.path.join("config", "parametersRRFS.json")
DEFAULT_HF_REPO = "PhillyWeatherGuy/baroclinic-rrfs-data"

CONTOUR_COORD_SCALE = 1000
CONTOUR_FORMAT = "CTV2"
INT16_MAX = 32767

CLIMATOLOGY_NAME = "hgt.mon.ltm.1991-2020.nc"
_HERE = os.path.dirname(os.path.abspath(__file__))
CLIMATOLOGY_CANDIDATES = [
    os.path.join(_HERE, CLIMATOLOGY_NAME),
    os.path.join(_HERE, "climatology", CLIMATOLOGY_NAME),
    CLIMATOLOGY_NAME,
    os.path.join("climatology", CLIMATOLOGY_NAME),
]
CLIMATOLOGY_CACHE = {}

# ---------------------------------------------------------------------------
# RRFS native grid (Lambert conformal, 3 km). Identical to HRRR: 1799 x 1059.
# ---------------------------------------------------------------------------
NX, NY, DX = 1799, 1059, 3000.0
RRFS_LCC = CRS.from_proj4("+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +R=6371229 +units=m +no_defs")
_TO_LCC = Transformer.from_crs("EPSG:4326", RRFS_LCC, always_xy=True)
_TO_LL = Transformer.from_crs(RRFS_LCC, "EPSG:4326", always_xy=True)
X0, Y0 = _TO_LCC.transform(-122.719528, 21.138123)

_xs = X0 + np.arange(NX) * DX
_ys = Y0 + np.arange(NY) * DX
_blon, _blat = _TO_LL.transform(
    np.concatenate([_xs, _xs, np.full(NY, _xs[0]), np.full(NY, _xs[-1])]),
    np.concatenate([np.full(NX, _ys[0]), np.full(NX, _ys[-1]), _ys, _ys]),
)

RRFS_GRID_META = {
    "projection": "lambert_conformal",
    "nx": NX, "ny": NY, "dx": DX, "dy": DX,
    "lat_1": 38.5, "lat_2": 38.5, "lat_0": 38.5, "lon_0": -97.5,
    "earth_radius": 6371229.0,
    "x0": float(X0), "y0": float(Y0),
    "row_order": "south_to_north",
    "bounds": {
        "west": float(_blon.min()), "east": float(_blon.max()),
        "south": float(_blat.min()), "north": float(_blat.max()),
    },
}

# ---------------------------------------------------------------------------
# S3 access. Prototype bucket uses rrfs_a/.../control; switch to
# noaa-rrfs-ops-pds + "rrfs.{date}/{run}" when the operational feed goes live.
# ---------------------------------------------------------------------------
RRFS_S3_BUCKET = "noaa-rrfs-pds"
RRFS_S3_PREFIX_TEMPLATE = "rrfs_a/rrfs_a.{date}/{run}/control"


def rrfs_s3_base(target_date, chosen_run):
    return f"https://{RRFS_S3_BUCKET}.s3.amazonaws.com/{RRFS_S3_PREFIX_TEMPLATE.format(date=target_date, run=chosen_run)}"


def rrfs_grib_name(chosen_run, file_type, step):
    return f"rrfs.t{chosen_run}z.{file_type}.3km.f{step:03d}.conus.grib2"

# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
TIMINGS = {}
_T_LOCK = threading.Lock()


@contextlib.contextmanager
def tm(key):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _T_LOCK:
            t = TIMINGS.setdefault(key, [0.0, 0])
            t[0] += dt
            t[1] += 1


def timed(key):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with tm(key):
                return fn(*a, **k)
        return wrapper
    return deco


def _gzip_to_file(path, raw_bytes):
    with open(path, "wb") as f:
        f.write(gzip.compress(raw_bytes, compresslevel=GZIP_LEVEL))


def _print_timing(pid, phases, total):
    ph = " | ".join(f"{k} {v:.1f}s" for k, v in phases.items())
    work = " | ".join(f"{k} {v[0]:.1f}s/{v[1]}x" for k, v in sorted(TIMINGS.items()))
    print(f"⏱️ [{pid}] total {total:.1f}s || phases: {ph} || thread-summed: {work}", flush=True)


def load_parameter_config(param_key="2t"):
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            params = data.get("parameters", {})
            if param_key in params:
                return params[param_key]
    raise FileNotFoundError(f"Parameter '{param_key}' not found in {CONFIG_FILE_PATH}")


def get_interval_hours(param_config):
    if "interval_hours" in param_config:
        return int(param_config["interval_hours"])
    pid = param_config.get("id", "").lower()
    match = re.search(r'[-_]?(1|3|6|24)h(?:r)?$', pid)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Contours: vectorized extraction + CTV2 binary packing
# ---------------------------------------------------------------------------

def _pack_level(cont_gen, level):
    pts_list, off_list = cont_gen.lines(level)
    pts, off = pts_list[0], off_list[0]
    if pts is None or off is None or len(off) < 2:
        return None

    off = off.astype(np.intp)
    lens = np.diff(off)
    starts = off[:-1]

    x = np.minimum(pts[:, 0], 180.0)
    seam = (np.maximum.reduceat(x, starts) < -179.99) | (np.minimum.reduceat(x, starts) > 179.99)

    q = np.empty((len(pts), 2), dtype=np.int32)
    q[:, 0] = np.rint(x * CONTOUR_COORD_SCALE)
    q[:, 1] = np.rint(pts[:, 1] * CONTOUR_COORD_SCALE)

    keep_line = ~seam
    if not keep_line.all():
        q = q[np.repeat(keep_line, lens)]
        lens = lens[keep_line]
    if len(lens) == 0:
        return None

    n = len(q)
    is_start = np.zeros(n, dtype=bool)
    is_start[np.cumsum(lens) - lens] = True

    if n > 1:
        is_start[1:] |= (np.abs(np.diff(q, axis=0)) > INT16_MAX).any(axis=1)

    seg_starts = np.flatnonzero(is_start)
    seg_lens = np.diff(np.append(seg_starts, n))

    keep_seg = seg_lens >= 2
    if not keep_seg.all():
        q = q[np.repeat(keep_seg, seg_lens)]
        seg_lens = seg_lens[keep_seg]
        seg_starts = np.cumsum(seg_lens) - seg_lens
        if len(seg_lens) == 0:
            return None
        is_start = np.zeros(len(q), dtype=bool)
        is_start[seg_starts] = True

    deltas = np.diff(q, axis=0)
    inner = ~is_start[1:]
    dx = deltas[inner, 0].astype("<i2")
    dy = deltas[inner, 1].astype("<i2")

    return b"".join((
        struct.pack("<I", len(seg_lens)),
        seg_lens.astype("<u4").tobytes(),
        q[seg_starts, 0].astype("<i4").tobytes(),
        q[seg_starts, 1].astype("<i4").tobytes(),
        dx.tobytes(),
        dy.tobytes(),
    ))


def _level_style(level, contours_config):
    name = str(level)
    unit = None
    for c_def in contours_config:
        if c_def.get("dynamic") and c_def.get("unit") in {"dam", "mb", "hPa"}:
            unit = c_def.get("unit")
            name = str(int(level)) if float(level).is_integer() else str(level)
            break

    match = next(
        (c for c in contours_config if c.get("dynamic") or c.get("target") == level),
        {},
    )
    return {
        "name": name,
        "unit": unit or match.get("unit", ""),
        "color": match.get("color", "#000000"),
        "width": match.get("width", 1.6),
        "opacity": match.get("opacity", 0.9),
    }


@timed("contour_math")
def extract_contour_levels(raw_arr, lats, lons, contours_config=None, contour_settings=None):
    if not contours_config:
        return []

    try:
        contour_settings = contour_settings or {}
        stride = max(1, int(contour_settings.get("coarsen_stride", 1)))
        blur_kernel = tuple(contour_settings.get("blur_kernel", (5, 5)))
        blur_sigma = float(contour_settings.get("blur_sigma", 1.2))

        working_arr = raw_arr.astype(np.float32)
        smoothed = cv2.GaussianBlur(working_arr, blur_kernel, blur_sigma)

        cont_gen = contourpy.contour_generator(
            x=lons[::stride, ::stride], y=lats[::stride, ::stride],
            z=smoothed[::stride, ::stride],
            line_type=contourpy.LineType.ChunkCombinedOffset,
        )

        interval = 6.0
        explicit_levels = []

        for c_def in contours_config:
            if c_def.get("dynamic"):
                interval = float(c_def.get("interval", 6.0))
                valid_mask = np.isfinite(working_arr) & (working_arr > 0)
                if not np.any(valid_mask):
                    explicit_levels = []
                    break

                valid_values = working_arr[valid_mask]
                data_min = float(np.nanmin(valid_values))
                data_max = float(np.nanmax(valid_values))
                if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                    start = math.ceil(data_min / interval) * interval
                    end = math.floor(data_max / interval) * interval
                    explicit_levels = [float(v) for v in np.arange(start, end + (interval * 0.5), interval)]
                break
            if "target" in c_def:
                explicit_levels.append(float(c_def["target"]))

        if not explicit_levels:
            return []

        levels = []
        for target_val in sorted(set(explicit_levels)):
            payload = _pack_level(cont_gen, target_val)
            if payload is not None:
                levels.append({"props": _level_style(target_val, contours_config), "payload": payload})

        return levels
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return []


@timed("pack")
def pack_step(levels):
    return {
        "blob": b"".join(l["payload"] for l in levels),
        "props": [l["props"] for l in levels],
    }


def assemble_contour_file(steps):
    parts = [CONTOUR_FORMAT.encode("ascii"), struct.pack("<I", len(steps))]
    for step in sorted(steps):
        entry = steps[step]
        parts.append(struct.pack("<III", int(step), len(entry["props"]), len(entry["blob"])))
        parts.append(entry["blob"])
    return b"".join(parts)


def normalize_array(raw_arr, param_config):
    scaling = param_config.get("scaling", {})
    mode = scaling.get("mode", "linear")

    if mode == "piecewise" and "val_points" in scaling and "byte_points" in scaling:
        multiplier = scaling.get("unit_multiplier", 1.0)
        v = np.nan_to_num(raw_arr, nan=0.0) * multiplier
        val_pts = scaling["val_points"]
        byte_pts = scaling["byte_points"]
        return np.interp(v, val_pts, byte_pts).astype(np.uint8)
    else:
        min_v = scaling.get("min_val", param_config.get("min_val", 0.0))
        max_v = scaling.get("max_val", param_config.get("max_val", 255.0))
        arr = np.nan_to_num(raw_arr, copy=False, nan=min_v)
        np.clip(arr, min_v, max_v, out=arr)
        arr -= min_v
        arr /= (max_v - min_v)
        arr *= 255.0
        return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Native RRFS grid validation
# ---------------------------------------------------------------------------
def prepare_native_grid(raw_arr, lats, lons):
    lons = np.where(lons > 180, lons - 360, lons)
    if raw_arr.shape != (NY, NX) or lats.shape != (NY, NX):
        raise ValueError(f"Unexpected RRFS grid shape {raw_arr.shape}, expected {(NY, NX)}")
    if lats[0, 0] > lats[-1, 0]:
        raw_arr, lats, lons = raw_arr[::-1], lats[::-1], lons[::-1]
    X, Y = _TO_LCC.transform(lons, lats)
    err = max(
        np.abs((X - X0) / DX - np.arange(NX)[None, :]).max(),
        np.abs((Y - Y0) / DX - np.arange(NY)[:, None]).max(),
    )
    if err > 0.05:
        raise ValueError(f"RRFS projection mismatch: {err:.2f} grid cells off")
    return np.ascontiguousarray(raw_arr), lats, lons


# ---------------------------------------------------------------------------
# AWS Open Data HTTP Range Downloader (RRFS naming/path)
# ---------------------------------------------------------------------------
def download_rrfs_idx_slice(target_date, chosen_run, step, search_pattern, file_type, dest_path):
    base_url = rrfs_s3_base(target_date, chosen_run)
    prefix = rrfs_grib_name(chosen_run, file_type, step)
    grib_url = f"{base_url}/{prefix}"
    idx_url = f"{grib_url}.idx"

    req = urllib.request.Request(idx_url, headers={"User-Agent": "BaroClinic-RRFS-Engine"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        idx_lines = resp.read().decode("utf-8").strip().split("\n")

    start_byte = None
    end_byte = None

    for i, line in enumerate(idx_lines):
        if search_pattern in line:
            parts = line.split(":")
            start_byte = int(parts[1])
            if i + 1 < len(idx_lines):
                end_byte = int(idx_lines[i + 1].split(":")[1]) - 1
            break

    if start_byte is None:
        raise ValueError(f"Pattern '{search_pattern}' not found in {prefix}.idx")

    range_header = f"bytes={start_byte}-" if end_byte is None else f"bytes={start_byte}-{end_byte}"
    range_req = urllib.request.Request(grib_url, headers={
        "Range": range_header,
        "User-Agent": "BaroClinic-RRFS-Engine"
    })

    with urllib.request.urlopen(range_req, timeout=30) as r_resp:
        with open(dest_path, "wb") as f_out:
            f_out.write(r_resp.read())

    return dest_path


def _bilinear_latlon(field, la, lo, lat, lon):
    if la[0] > la[-1]:
        la, field = la[::-1], field[::-1]
    field = np.concatenate([field, field[:, :1]], axis=1)
    lo = np.append(lo, lo[0] + 360.0)
    lon = np.mod(lon, 360.0)
    fi = np.interp(lat, la, np.arange(len(la)))
    fj = np.interp(lon, lo, np.arange(len(lo)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(la) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(lo) - 2)
    di, dj = fi - i0, fj - j0
    return (field[i0, j0] * (1 - di) * (1 - dj) + field[i0 + 1, j0] * di * (1 - dj)
            + field[i0, j0 + 1] * (1 - di) * dj + field[i0 + 1, j0 + 1] * di * dj)


def _find_climatology_file():
    for path in CLIMATOLOGY_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Climatology file {CLIMATOLOGY_NAME} not found. Looked in: {CLIMATOLOGY_CANDIDATES}")


def climatology_on_native_grid(level_hpa, month, lats, lons):
    key = (int(level_hpa), int(month))
    if key not in CLIMATOLOGY_CACHE:
        with xr.open_dataset(_find_climatology_file(), decode_times=False) as ds:
            n_time = ds["hgt"].sizes["time"]
            if n_time != 12:
                raise ValueError(f"Climatology has {n_time} time steps, expected 12 monthly means")
            t_idx = int(np.argsort(ds["time"].values)[int(month) - 1])
            field = ds["hgt"].sel(level=float(level_hpa)).isel(time=t_idx).values.astype(np.float64)
            la, lo = ds["lat"].values.astype(np.float64), ds["lon"].values.astype(np.float64)
        CLIMATOLOGY_CACHE[key] = _bilinear_latlon(field, la, lo, lats, lons).astype(np.float32)
    return CLIMATOLOGY_CACHE[key]


def read_native_field(grib_path, grib_var):
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={'errors': 'ignore'})

    lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
    lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values

    if isinstance(grib_var, list) and len(grib_var) == 2:
        raw_arr_k = np.hypot(np.squeeze(ds[grib_var[0]].values), np.squeeze(ds[grib_var[1]].values))
    else:
        target_var = grib_var if (isinstance(grib_var, str) and grib_var in ds) else list(ds.data_vars)[0]
        raw_arr_k = np.squeeze(ds[target_var].values)

    ds.close()
    return prepare_native_grid(raw_arr_k, lats, lons)


def convert_units(raw_arr, cfg):
    if cfg.get("unit") == "dam" or str(cfg.get("grib_param", "")).lower() in {"z", "gh", "hgt"}:
        return raw_arr / 10.0
    if cfg.get("unit") in {"mb", "hPa"} or str(cfg.get("grib_param", "")).lower() in {"msl", "mslma", "prmsl"}:
        return raw_arr / 100.0
    return raw_arr


@timed("decode_all")
def process_grib_to_array(grib_path, param_config, want_raster=True, target_date=None, valid_time=None, want_raw=False):
    raw_arr, lats, lons = read_native_field(grib_path, param_config["grib_param"])

    if param_config.get("category_mode"):
        return np.nan_to_num(raw_arr, nan=0.0).clip(0, 255).astype(np.uint8), []

    if param_config.get("climatology_level") is not None:
        raw_arr = raw_arr - climatology_on_native_grid(param_config["climatology_level"], valid_time.month, lats, lons)
    else:
        raw_arr = convert_units(raw_arr, param_config)

    contour_levels = extract_contour_levels(
        raw_arr, lats, lons,
        param_config.get("contours", []),
        param_config.get("contours_settings")
    )

    if want_raw:
        return np.nan_to_num(raw_arr, nan=0.0).astype(np.float32), contour_levels

    arr_8bit = normalize_array(raw_arr, param_config) if want_raster else None
    return arr_8bit, contour_levels


def extract_source_contours(src, param_config, target_date, chosen_run, step, model_name):
    file_type = src.get("file_type", "2dfld")
    searches = src.get("idx_searches") or [src["idx_search"]]
    tag = src.get("id", "c")
    fields, lats, lons = [], None, None

    for k, pat in enumerate(searches):
        path = f"{model_name}_{param_config['id']}_c_{tag}_{k}_{step:03d}.grib2"
        try:
            download_rrfs_idx_slice(target_date, chosen_run, step, pat, file_type, path)
            raw, lats, lons = read_native_field(path, src["grib_param"])
            fields.append(raw)
        finally:
            if os.path.exists(path):
                try: os.remove(path)
                except Exception: pass

    raw_arr = fields[0] - fields[1] if len(fields) == 2 else fields[0]
    raw_arr = convert_units(raw_arr, src)

    return extract_contour_levels(
        raw_arr, lats, lons,
        src.get("contours") or param_config.get("contours", []),
        src.get("contours_settings") or param_config.get("contours_settings"),
    )


def fetch_and_process_step(target_date, chosen_run, step, param_config, model_name):
    pid = param_config["id"]
    file_type = param_config.get("file_type", "2dfld")
    grib_file = f"{model_name}_{pid}_{step:03d}.grib2"

    try:
        with tm("download"):
            if "idx_search_u" in param_config and "idx_search_v" in param_config:
                u_file = f"{model_name}_{pid}_u_{step:03d}.grib2"
                v_file = f"{model_name}_{pid}_v_{step:03d}.grib2"
                download_rrfs_idx_slice(target_date, chosen_run, step, param_config["idx_search_u"], file_type, u_file)
                download_rrfs_idx_slice(target_date, chosen_run, step, param_config["idx_search_v"], file_type, v_file)

                with open(grib_file, "wb") as f_out:
                    with open(u_file, "rb") as f_u: f_out.write(f_u.read())
                    with open(v_file, "rb") as f_v: f_out.write(f_v.read())
                
                try: os.remove(u_file); os.remove(v_file)
                except Exception: pass
            else:
                search_pat = param_config.get("idx_search", f":{param_config['grib_param']}:")
                download_rrfs_idx_slice(target_date, chosen_run, step, search_pat, file_type, grib_file)

        if os.path.exists(grib_file):
            contour_sources = param_config.get("contour_sources")
            if not contour_sources and param_config.get("contour_source"):
                contour_sources = [param_config["contour_source"]]

            has_contours = bool(contour_sources)
            raster_config = {**param_config, "contours": []} if has_contours else param_config
            valid_time = datetime.datetime.strptime(f"{target_date}{chosen_run}", "%Y%m%d%H") + datetime.timedelta(hours=step)
            is_interval = get_interval_hours(param_config) is not None
            frame_arr, contour_levels = process_grib_to_array(
                grib_file, raster_config, target_date=target_date, valid_time=valid_time, want_raw=is_interval
            )
            contour_levels = list(contour_levels)
            ptype_arr = None

            try: os.remove(grib_file)
            except Exception: pass

            for src in (contour_sources or []):
                try:
                    contour_levels.extend(extract_source_contours(src, param_config, target_date, chosen_run, step, model_name))
                except Exception as e:
                    print(f"  ⚠️ [{pid}] Contours '{src.get('id', src.get('grib_param'))}' failed F{step:03d}: {e}")

            ptype_sources = param_config.get("ptype_sources")
            if ptype_sources:
                ptype_masks = {}
                for p_name, p_search in ptype_sources.items():
                    pt_file = f"{model_name}_{pid}_{p_name}_{step:03d}.grib2"
                    try:
                        download_rrfs_idx_slice(target_date, chosen_run, step, p_search, "2dfld", pt_file)
                        if os.path.exists(pt_file):
                            mask, _ = process_grib_to_array(pt_file, {"grib_param": "", "category_mode": True})
                            ptype_masks[p_name] = mask > 0.5
                    except Exception: pass
                    finally:
                        if os.path.exists(pt_file):
                            try: os.remove(pt_file)
                            except Exception: pass

                if ptype_masks:
                    shape = frame_arr.shape
                    ptype_arr = np.zeros(shape, dtype=np.uint8)
                    if "rain" in ptype_masks: ptype_arr[ptype_masks["rain"]] = 1
                    if "frzr" in ptype_masks: ptype_arr[ptype_masks["frzr"]] = 3
                    if "snow" in ptype_masks: ptype_arr[ptype_masks["snow"]] = 5
                    if "sleet" in ptype_masks: ptype_arr[ptype_masks["sleet"]] = 7

            packed = pack_step(contour_levels)
            print(f"  ⚡ [{pid}] Processed RRFS F{step:03d}")
            return step, frame_arr, packed, ptype_arr

    except Exception as e:
        print(f"  ❌ [{pid}] Error processing F{step:03d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass

    return step, None, None, None


@timed("stack")
def build_volume_chunks(frame_arrays, steps_written, model_name, param_config, target_date, chosen_run, ptype_arrays=None):
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape
    frames_per_volume = 10

    chunks = []
    patterns = param_config["filename_patterns"]
    pattern_key = "volume" if "volume" in patterns else "spritesheet"

    for chunk_idx, i in enumerate(range(0, len(frame_arrays), frames_per_volume)):
        chunk_frames = frame_arrays[i:i + frames_per_volume]
        chunk_steps = steps_written[i:i + frames_per_volume]
        chunk_ptype = ptype_arrays[i:i + frames_per_volume] if ptype_arrays else None

        volume_arr = np.stack(chunk_frames, axis=0).astype(np.uint8)

        volume_filename = patterns[pattern_key].format(
            model=model_name,
            param=param_config["id"],
            date=target_date,
            run=chosen_run,
            chunk_idx=chunk_idx
        )

        chunk_manifest = {
            "file": volume_filename,
            "forecast_steps": chunk_steps,
            "frame_count": len(chunk_frames)
        }
        chunk_entry = {
            "array": volume_arr,
            "manifest_data": chunk_manifest
        }
        if chunk_ptype and all(frame is not None for frame in chunk_ptype):
            ptype_filename = volume_filename.replace("_volume_", "_ptype_volume_")
            chunk_entry["ptype_array"] = np.stack(chunk_ptype, axis=0).astype(np.uint8)
            chunk_manifest["ptype_file"] = ptype_filename
        chunks.append(chunk_entry)

    return chunks, frame_w, frame_h


def upload_to_huggingface(folder_path, repo_id=DEFAULT_HF_REPO):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set in environment. Skipping Hugging Face upload.")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"RRFS update: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z"
        )
        print(f"  ✅ Uploaded to Hugging Face", flush=True)
        return True
    except Exception as e:
        print(f"  ❌ Failed to upload to Hugging Face: {e}", flush=True)
        return False


def prune_old_huggingface_runs(repo_id=DEFAULT_HF_REPO, days_to_keep=2):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return

    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_to_keep)
    cutoff_int = int(cutoff_date.strftime("%Y%m%d"))

    print(f"\n🧹 Checking for RRFS runs older than {cutoff_date.strftime('%Y-%m-%d')} (>{days_to_keep} days)...")

    try:
        from huggingface_hub import HfApi, CommitOperationDelete
        api = HfApi(token=hf_token)
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        files_to_delete = []

        for fname in all_files:
            match = re.search(r'_(\d{8})_', fname)
            if match:
                file_date_int = int(match.group(1))
                if file_date_int < cutoff_int:
                    files_to_delete.append(fname)

        if files_to_delete:
            print(f"  🗑️ Deleting {len(files_to_delete)} expired RRFS files...")
            batch_size = 200
            for i in range(0, len(files_to_delete), batch_size):
                batch = files_to_delete[i:i + batch_size]
                operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Prune RRFS runs older than {days_to_keep} days ({i + len(batch)}/{len(files_to_delete)})"
                )
            print("  ✅ Pruning complete!")
        else:
            print("  ✨ No expired RRFS files found.")

    except Exception as e:
        print(f"  ⚠️ Could not prune old Hugging Face files: {e}")


def rrfs_step_ready(target_date, run_hour, step):
    """True once BOTH the 2dfld and prslev files of this forecast hour are on S3."""
    base = rrfs_s3_base(target_date, run_hour)
    for file_type in ("2dfld", "prslev"):
        url = f"{base}/{rrfs_grib_name(run_hour, file_type, step)}.idx"
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "BaroClinic-RRFS-Engine"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status != 200:
                    return False
        except Exception:
            return False
    return True


def find_latest_started_run(now_utc, max_hours_back, extended_only=False):
    for hours_back in range(0, max_hours_back + 1):
        t_cand = now_utc - datetime.timedelta(hours=hours_back)
        cand_run = t_cand.strftime("%H")
        if extended_only and cand_run not in EXTENDED_RUNS:
            continue
        cand_date = t_cand.strftime("%Y%m%d")
        if rrfs_step_ready(cand_date, cand_run, 0):
            return cand_date, cand_run
    return None


_NEWER_RUN_SEEN = {"yes": False}


def newer_run_started(target_date, chosen_run):
    if _NEWER_RUN_SEEN["yes"]:
        return True
    t = datetime.datetime.strptime(f"{target_date}{chosen_run}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    t += datetime.timedelta(hours=1)
    while t <= now:
        if rrfs_step_ready(t.strftime("%Y%m%d"), t.strftime("%H"), 0):
            _NEWER_RUN_SEEN["yes"] = True
            return True
        t += datetime.timedelta(hours=1)
    return False


def choose_run(custom_run, custom_date):
    if custom_run and custom_date:
        return str(custom_date), str(custom_run).zfill(2)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    pick = None
    if PREFER_EXTENDED_RUNS:
        pick = find_latest_started_run(now_utc, EXTENDED_LOOKBACK_HOURS, extended_only=True)
    if pick is None:
        pick = find_latest_started_run(now_utc, NORMAL_LOOKBACK_HOURS)
    if pick is None:
        t_cand = now_utc - datetime.timedelta(hours=2)
        pick = (t_cand.strftime("%Y%m%d"), t_cand.strftime("%H"))
    return pick


def new_param_state():
    return {
        "results": {},
        "contours": {},
        "ptype": {},
        "attempts": {},
        "published": {},
        "staged": {},
        "dirty": False,
    }


def write_param_outputs(state, param_config, stage_dir, target_date, chosen_run, init_time_iso, expected_frames, final, latest_allowed=True):
    patterns = param_config["filename_patterns"]
    pid = param_config["id"]
    sorted_steps = sorted(state["results"])
    if not sorted_steps:
        return

    interval_h = get_interval_hours(param_config)
    if interval_h:
        frame_arrays = []
        for s in sorted_steps:
            target_prev = s - interval_h
            if target_prev <= 0:
                prev_arr = state["results"].get(0, 0.0)
            else:
                candidates = [st for st in sorted_steps if st <= target_prev]
                prev_s = max(candidates) if candidates else None
                prev_arr = state["results"][prev_s] if prev_s is not None else 0.0

            diff = np.maximum(0.0, state["results"][s] - prev_arr)
            frame_arrays.append(normalize_array(diff, param_config))
    else:
        frame_arrays = [state["results"][s] for s in sorted_steps]

    ptype_arrays = [state["ptype"].get(s) for s in sorted_steps]
    publish_latest = latest_allowed and (final or len(sorted_steps) >= LATEST_MIN_FRAMES)

    contour_steps = state["contours"]
    if contour_steps and "run_contours" in patterns:
        compressed_contours = gzip.compress(assemble_contour_file(contour_steps), compresslevel=6)
        if len(compressed_contours) >= 100:
            master_contours = {
                "model": MODEL_NAME, "parameter": pid, "run": f"{chosen_run}z", "date": target_date,
                "steps": {
                    str(step): {
                        "type": "FeatureCollection",
                        "features": [
                            {"type": "Feature",
                             "geometry": {"type": "MultiLineString", "coordinates": []},
                             "properties": props}
                            for props in contour_steps[step]["props"]
                        ],
                    }
                    for step in sorted(contour_steps)
                },
            }
            run_contour_filename = patterns["run_contours"].format(
                model=MODEL_NAME, param=pid, date=target_date, run=chosen_run.lower())
            run_binary_filename = run_contour_filename.replace(".json", ".bin.gz")
            master_contours["binary"] = {"format": CONTOUR_FORMAT, "scale": CONTOUR_COORD_SCALE, "file": run_binary_filename}

            with open(os.path.join(stage_dir, run_contour_filename), "w") as f:
                json.dump(master_contours, f, separators=(",", ":"))
            with open(os.path.join(stage_dir, run_binary_filename), "wb") as f:
                f.write(compressed_contours)
            if publish_latest and "latest_contours" in patterns:
                latest_contour_filename = patterns["latest_contours"].format(model=MODEL_NAME, param=pid)
                with open(os.path.join(stage_dir, latest_contour_filename), "w") as f:
                    json.dump(master_contours, f, separators=(",", ":"))

    chunks, frame_w, frame_h = build_volume_chunks(
        frame_arrays, sorted_steps, model_name=MODEL_NAME, param_config=param_config,
        target_date=target_date, chosen_run=chosen_run,
        ptype_arrays=ptype_arrays if state["ptype"] else None,
    )
    manifest_chunks = []
    for chunk in chunks:
        md = chunk["manifest_data"]
        filename = md["file"]
        manifest_chunks.append(md)
        signature = tuple(md["forecast_steps"])
        if state["published"].get(filename) == signature:
            continue

        filepath = os.path.join(stage_dir, filename)
        if filename.endswith(".bin"):
            _gzip_to_file(filepath, chunk["array"].tobytes())
            ptype_filename = md.get("ptype_file")
            if ptype_filename and "ptype_array" in chunk:
                _gzip_to_file(os.path.join(stage_dir, ptype_filename), chunk["ptype_array"].tobytes())
        elif filename.endswith(".webp"):
            cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_WEBP_QUALITY), 101])
        else:
            cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
        state["staged"][filename] = signature

    manifest = {
        "model": MODEL_NAME,
        "parameter": pid,
        "name": param_config.get("name", pid),
        "unit": param_config.get("unit", ""),
        "scaling": param_config.get("scaling", {}),
        "run": f"{chosen_run}z",
        "date": target_date,
        "init_time": init_time_iso,
        "type": "volume_chunked",
        "total_frames": len(sorted_steps),
        "expected_frames": expected_frames,
        "complete": bool(final),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "grid": RRFS_GRID_META,
        "temp_min_k": param_config.get("min_val", param_config.get("scaling", {}).get("min_val", 0.0)),
        "temp_max_k": param_config.get("max_val", param_config.get("scaling", {}).get("max_val", 255.0)),
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
    }
    names = [patterns["run_manifest"].format(model=MODEL_NAME, param=pid, date=target_date, run=chosen_run.lower())]
    if publish_latest and "latest_manifest" in patterns:
        names.append(patterns["latest_manifest"].format(model=MODEL_NAME, param=pid))
    names.append("manifest.json")
    for name in names:
        with open(os.path.join(stage_dir, name), "w") as f:
            json.dump(manifest, f, indent=2)


def publish_batch(states, configs, stage_dir, target_date, chosen_run, init_time_iso, expected_frames, final):
    os.makedirs(stage_dir, exist_ok=True)
    latest_allowed = not newer_run_started(target_date, chosen_run)
    for key, state in states.items():
        if state["dirty"] or final:
            try:
                write_param_outputs(state, configs[key], stage_dir, target_date, chosen_run, init_time_iso, expected_frames, final, latest_allowed)
            except Exception as e:
                print(f"  ❌ [{key}] could not write outputs: {e}")
            state["dirty"] = False

    ok = upload_to_huggingface(stage_dir, repo_id=DEFAULT_HF_REPO)
    if ok:
        for state in states.values():
            state["published"].update(state["staged"])
            state["staged"] = {}
        shutil.rmtree(stage_dir, ignore_errors=True)
    return ok


def run_live_pipeline(param_keys, custom_run=None, custom_date=None):
    target_date, chosen_run = choose_run(custom_run, custom_date)
    all_steps = forecast_steps_for_run(chosen_run)
    init_time_iso = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}T{chosen_run}:00:00Z"
    configs = {k: load_parameter_config(k) for k in param_keys}
    states = {k: new_param_state() for k in param_keys}

    print(f"🌍 LIVE RRFS run {chosen_run}z on {target_date} | {len(all_steps)} forecast hours | params: {param_keys}", flush=True)

    for f in glob.glob(f"{MODEL_NAME}_*_*.grib2"):
        try: os.remove(f)
        except Exception: pass
    shutil.rmtree(STAGE_DIR, ignore_errors=True)

    pending = list(all_steps)
    last_progress = time.time()
    last_upload_ok = True
    t_run = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(MAX_CONCURRENT_WORKERS, 8)) as pool:
        while pending:
            ready = []
            for s in pending:
                if len(ready) >= MAX_STEPS_PER_BATCH or not rrfs_step_ready(target_date, chosen_run, s):
                    break
                ready.append(s)

            if not ready:
                if time.time() - last_progress > STALL_TIMEOUT_MINUTES * 60:
                    print(f"⚠️ No new forecast hour for {STALL_TIMEOUT_MINUTES} min. Stopping with {len(all_steps) - len(pending)}/{len(all_steps)} hours done.", flush=True)
                    break
                time.sleep(POLL_SECONDS)
                continue

            last_progress = time.time()
            t_batch = time.perf_counter()
            jobs = {}
            for s in ready:
                for k in param_keys:
                    st = states[k]
                    if s not in st["results"] and st["attempts"].get(s, 0) < MAX_STEP_ATTEMPTS:
                        fut = pool.submit(fetch_and_process_step, target_date, chosen_run, s, configs[k], MODEL_NAME)
                        jobs[fut] = (k, s)

            for fut in concurrent.futures.as_completed(jobs):
                k, s = jobs[fut]
                st = states[k]
                try:
                    _, arr, packed, ptype_arr = fut.result()
                except Exception as e:
                    print(f"  ❌ [{k}] F{s:03d} crashed: {e}", flush=True)
                    arr = packed = ptype_arr = None
                if arr is None:
                    st["attempts"][s] = st["attempts"].get(s, 0) + 1
                    continue
                st["results"][s] = arr
                if packed and packed["props"]:
                    st["contours"][s] = packed
                if ptype_arr is not None:
                    st["ptype"][s] = ptype_arr
                st["dirty"] = True

            unresolved = False
            for s in ready:
                done = all(s in states[k]["results"] or states[k]["attempts"].get(s, 0) >= MAX_STEP_ATTEMPTS for k in param_keys)
                if done:
                    pending.remove(s)
                    for k in param_keys:
                        if s not in states[k]["results"]:
                            print(f"  ⚠️ [{k}] giving up on F{s:03d}", flush=True)
                else:
                    unresolved = True

            final = not pending
            if any(st["dirty"] for st in states.values()) or final:
                last_upload_ok = publish_batch(states, configs, STAGE_DIR, target_date, chosen_run, init_time_iso, len(all_steps), final)

            print(f"⏱️ Batch F{ready[0]:03d}-F{ready[-1]:03d} took {time.perf_counter() - t_batch:.1f}s | "
                  f"{len(all_steps) - len(pending)}/{len(all_steps)} hours done", flush=True)
            if unresolved:
                time.sleep(POLL_SECONDS)

    total_frames = sum(len(st["results"]) for st in states.values())
    if total_frames == 0:
        print("❌ No frames were processed.", flush=True)
        sys.exit(1)

    if not last_upload_ok or pending:
        for st in states.values():
            st["dirty"] = True
        publish_batch(states, configs, STAGE_DIR, target_date, chosen_run, init_time_iso, len(all_steps), final=not pending)

    print(f"\n🎉 Run {chosen_run}z {target_date} finished in {time.perf_counter() - t_run:.0f}s", flush=True)
    prune_old_huggingface_runs(repo_id=DEFAULT_HF_REPO, days_to_keep=2)


if __name__ == "__main__":
    target_params = []
    custom_run = None
    custom_date = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--run" and i + 1 < len(args):
            custom_run = args[i + 1]
            i += 2
        elif args[i] == "--date" and i + 1 < len(args):
            custom_date = args[i + 1]
            i += 2
        else:
            target_params.append(args[i])
            i += 1

    if not target_params:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                target_params = list(data.get("parameters", {}).keys())
        else:
            target_params = ["2t"]

    print(f"🚀 Launching RRFS live pipeline for parameters: {target_params}", flush=True)
    run_live_pipeline(target_params, custom_run, custom_date)
