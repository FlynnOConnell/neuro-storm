#!/usr/bin/env -S GIT_LFS_SKIP_SMUDGE=1 uv run --quiet
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "masknmf",
#     "opencv-python-headless",  # masknmf imports cv2 but does not declare it
#     "zarr>=3",
#     "tifffile>=2024.1.30",
#     "numpy>=1.26",
# ]
#
# [tool.uv]
# # masknmf HEAD is written against fastplotlib's unreleased `ndwidget` branch
# # (it uses fpl.NDWidget) but still pins fastplotlib==0.6.1. Override that pin
# # with the exact commit so masknmf imports. GIT_LFS_SKIP_SMUDGE (see shebang)
# # avoids a missing-LFS-object failure when uv clones fastplotlib the first time.
# override-dependencies = [
#     "fastplotlib[imgui,notebook] @ git+https://github.com/fastplotlib/fastplotlib@a80884a3b14324407b323b5b6d2b3c80d27ba218",
# ]
#
# [tool.uv.sources]
# masknmf = { path = "/home/flynn/repos/masknmf-toolbox", editable = true }
# ///
"""Load an LBM movie, run it through the masknmf pipeline, and view the result.

This is a self-contained PEP 723 script: ``uv run`` builds an ephemeral,
per-script virtualenv from the inline metadata block above (pulling masknmf
straight from the local source tree), so it needs neither this repo's venv nor
masknmf's own venv on your ``PATH``.

masknmf HEAD needs fastplotlib's unreleased ``ndwidget`` branch, so the metadata
overrides masknmf's ``fastplotlib==0.6.1`` pin with an exact commit. The shebang
sets ``GIT_LFS_SKIP_SMUDGE=1`` for that clone; if you invoke this via ``uv run``
directly, prefix the same var on the first build:

    GIT_LFS_SKIP_SMUDGE=1 uv run neuro_storm/cli/masknmf_gui.py --print-params

Pipeline order is masknmf's real order: motion correction -> PMD compression
-> demixing. Parameters are grouped like mbo_utilities' suite2p settings panel
(one dataclass per stage, printed as a grouped panel before the run).

Run standalone (first invocation builds the venv; later runs are cached)::

    uv run neuro_storm/cli/masknmf_gui.py                          # no args: info + how to pick data
    uv run neuro_storm/cli/masknmf_gui.py --pick                   # choose a movie via file dialog
    uv run neuro_storm/cli/masknmf_gui.py --data <movie> --plane 7        # view the movie
    uv run neuro_storm/cli/masknmf_gui.py --data <movie> --plane 7 --run  # run the pipeline + view
    uv run neuro_storm/cli/masknmf_gui.py --print-params           # parameters only, no data/deps

The default action for a dataset is to VIEW it; the pipeline only runs with --run.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

# masknmf and its GUI/torch stack come from the on-the-fly venv (see the inline
# metadata above). They are imported lazily inside the functions that need them
# via ``_require_masknmf`` so that the pure parameter panel below stays usable
# even when the heavy stack is absent or (as with a WIP masknmf checkout) broken.

# Default dataset: the assembled/deinterleaved LBM volume; --plane picks a z-plane.
DEFAULT_DATA = "~/lbm_data/raw/volume/tp00001-00119_zplane01-14_stack.zarr"
DEFAULT_PLANE = 7

# Pipeline artifacts written by TwoPhotonCalciumPipeline.run() (RegistrationArray.export
# refuses to overwrite, so these are cleared before each run).
_ARTIFACTS = ("motion_correction.hdf5", "compression.hdf5", "demixing_results.hdf5")


# =============================================================================
# Parameter panel  (grouped like mbo_utilities' suite2p settings: one dataclass
# per pipeline stage; every field maps 1:1 onto a verified masknmf config field).
# =============================================================================


@dataclass
class GeneralParams:
    """Cross-cutting execution knobs (not algorithm parameters)."""

    device: str = "auto"  # "auto" | "cuda" | "cpu"  (this box has no CUDA -> cpu)
    frame_rate: float = 17.07  # Hz; pipeline.run() requires it (LBM volume rate)
    frame_batch_size: int = 300
    load_into_ram: bool = False
    outdir: str = "masknmf_out"


@dataclass
class MotionCorrectionParams:
    """Rigid moco by default; ``enabled=False`` passes the literal "skip"."""

    enabled: bool = True
    piecewise: bool = False
    # RigidMotionCorrectionConfig.max_shifts
    max_shifts: tuple[int, int] = (15, 15)
    # PiecewiseRigidMotionCorrectionConfig (used only when piecewise=True)
    num_blocks: tuple[int, int] = (15, 15)
    overlaps: tuple[int, int] = (5, 5)
    max_rigid_shifts: tuple[int, int] = (15, 15)
    max_deviation_rigid: tuple[int, int] = (2, 2)


@dataclass
class CompressionParams:
    """PMD compression; ``denoise=True`` selects CompressDenoiseConfig."""

    denoise: bool = True
    block_sizes: tuple[int, int] = (20, 20)
    frame_range: int | None = None
    max_components: int = 20
    sim_conf: int = 5
    max_consecutive_failures: int = 1
    spatial_avg_factor: int = 1
    temporal_avg_factor: int = 1
    compute_normalizer: bool = True
    # CompressDenoiseConfig-only (ignored when denoise=False):
    noise_variance_quantile: float = 0.3
    num_epochs: int = 10


@dataclass
class DemixingParams:
    """An optional custom multi-pass demixing schedule.

    When ``custom_schedule`` is False (default) the pipeline builds its own
    tuned multi-pass schedule; the SuperpixelInit/NMF knobs below are only
    applied when ``custom_schedule`` is True (see ``build_multipass_config``).
    """

    custom_schedule: bool = False
    # SuperpixelInitConfig
    mad_correlation_threshold: float = 0.8
    min_peak_distance: int = 3
    patch_size: tuple[int, int] = (40, 40)
    # NMFConfig
    maxiter: int = 40
    merge_threshold: float = 0.8
    ring_radius: int = 10


@dataclass
class GuiParams:
    """Viewer options for the masknmf fastplotlib GUIs."""

    show: bool = True
    pmd_widget: bool = False  # PMDWidget is imgui/ipywidgets-based (best in Jupyter)
    v_range: tuple[float, float] = (-1.0, 1.0)


@dataclass
class PipelineParams:
    """Top-level grouped parameter panel (the "Run" button is :func:`run`)."""

    general: GeneralParams = field(default_factory=GeneralParams)
    motion: MotionCorrectionParams = field(default_factory=MotionCorrectionParams)
    compression: CompressionParams = field(default_factory=CompressionParams)
    demixing: DemixingParams = field(default_factory=DemixingParams)
    gui: GuiParams = field(default_factory=GuiParams)

    def print_panel(self) -> None:
        """Print the grouped settings, suite2p-style, marking changed fields."""
        print("=" * 66)
        print(" masknmf pipeline settings")
        print("=" * 66)
        for group in fields(self):
            section = getattr(self, group.name)
            print(f"\n[{group.name}]")
            defaults = type(section)()
            for f in fields(section):
                value = getattr(section, f.name)
                changed = value != getattr(defaults, f.name)
                mark = " *" if changed else "  "
                print(f"  {mark} {f.name:<26} {value}")
        print("\n( * = changed from default )\n")


# =============================================================================
# masknmf config construction  (symbol names verified against masknmf source)
# =============================================================================


def _require_masknmf():
    """Import and return masknmf, with a clear message if it cannot load."""
    try:
        import masknmf
    except Exception as exc:  # a WIP checkout can fail with ImportError, AttributeError, ...
        raise SystemExit(
            f"could not import masknmf: {type(exc).__name__}: {exc}\n"
            "This script builds its own venv from the local masknmf source. If the "
            "import fails, the masknmf checkout itself is the problem (e.g. an "
            "undeclared dependency, or code that needs an unreleased fastplotlib). "
            "Run with --print-params to use the parameter panel without masknmf."
        ) from exc
    return masknmf


def _resolve_device(device: str) -> str:
    """Resolve "auto" to a concrete torch device string for the GUIs."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_motion_config(p: MotionCorrectionParams):
    masknmf = _require_masknmf()
    if not p.enabled:
        return "skip"  # the pipeline bypasses moco on the literal string "skip"
    if p.piecewise:
        return masknmf.PiecewiseRigidMotionCorrectionConfig(
            num_blocks=p.num_blocks,
            overlaps=p.overlaps,
            max_rigid_shifts=p.max_rigid_shifts,
            max_deviation_rigid=p.max_deviation_rigid,
        )
    return masknmf.RigidMotionCorrectionConfig(max_shifts=p.max_shifts)


def build_compress_config(p: CompressionParams):
    masknmf = _require_masknmf()
    common = dict(
        block_sizes=p.block_sizes,
        frame_range=p.frame_range,
        max_components=p.max_components,
        sim_conf=p.sim_conf,
        max_consecutive_failures=p.max_consecutive_failures,
        spatial_avg_factor=p.spatial_avg_factor,
        temporal_avg_factor=p.temporal_avg_factor,
        compute_normalizer=p.compute_normalizer,
    )
    if p.denoise:
        return masknmf.CompressDenoiseConfig(
            noise_variance_quantile=p.noise_variance_quantile,
            num_epochs=p.num_epochs,
            **common,
        )
    return masknmf.CompressConfig(**common)


def build_multipass_config(p: DemixingParams):
    """Build a single-pass masknmf demixing schedule from the exposed knobs.

    Returns ``None`` unless ``custom_schedule`` is set, in which case the
    pipeline uses its own (tuned, multi-pass) default schedule.
    """
    if not p.custom_schedule:
        return None
    masknmf = _require_masknmf()
    init = masknmf.SuperpixelInitConfig(
        mad_correlation_threshold=p.mad_correlation_threshold,
        min_peak_distance=p.min_peak_distance,
        patch_size=p.patch_size,
    )
    nmf = masknmf.NMFConfig(
        maxiter=p.maxiter,
        merge_threshold=p.merge_threshold,
        ring_radius=p.ring_radius,
    )
    single = masknmf.SinglepassDemixingConfig(InitConfig=init, NMFConfig=nmf)
    return masknmf.MultipassDemixingConfig(DemixingConfigs=[single])


def build_pipeline(params: PipelineParams) -> "masknmf.TwoPhotonCalciumPipeline":
    """Construct a TwoPhotonCalciumPipeline from the grouped parameters."""
    masknmf = _require_masknmf()
    g = params.general
    outdir = Path(g.outdir)
    demix = build_multipass_config(params.demixing)

    # spatial_highpass_config is left None: masknmf then applies its own
    # SpatialHighpassConfig() default (filter_sigma=4.0) internally. This also
    # dodges a bug in masknmf HEAD, where run() binds a local spatial_highpass_config
    # only in the `is None` branch yet uses it unconditionally, so passing any config
    # raises UnboundLocalError during demixing.
    return masknmf.TwoPhotonCalciumPipeline(
        motion_correct_config=build_motion_config(params.motion),
        compress_config=build_compress_config(params.compression),
        spatial_highpass_config=None,
        filtered_demixing_config=demix,
        unfiltered_demixing_config=demix,
        outpath_motion_correction=str(outdir / "motion_correction.hdf5"),
        outpath_compression=str(outdir / "compression.hdf5"),
        outpath_demixing=str(outdir / "demixing_results.hdf5"),
        load_into_ram=g.load_into_ram,
        frame_batch_size=g.frame_batch_size,
        device=g.device,
    )


# =============================================================================
# Data loading  (3D TIFF -> lazy TiffArray; 4D TIFF / .zarr -> extract a plane)
# =============================================================================


def load_movie(path: str | Path, plane: int | None = None):
    """Return a ``(frames, H, W)`` movie for masknmf from ``path``.

    Supported inputs:
      * 3D multipage TIFF            -> ``masknmf.TiffArray`` (lazy)
      * 4D ScanImage TIFF (T,Z,Y,X)  -> ``plane`` -> ``(T,Y,X)`` float32 ndarray
      * .zarr volume (T,Z,Y,X)       -> ``plane`` -> ``(T,Y,X)`` float32 ndarray

    A ``plane`` index is required for 4D sources (the raw LBM data is 4D).
    """
    masknmf = _require_masknmf()
    import tifffile

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No such dataset: {path}")

    if path.suffix == ".zarr" or (path.is_dir() and (path / "zarr.json").exists()):
        import zarr

        node = zarr.open(str(path), mode="r")
        # OME multiscale groups hold the array under "0"; a plain store is the
        # array itself. Distinguish by whether the node is array-like.
        if hasattr(node, "shape"):
            arr = node
        elif "0" in node:
            arr = node["0"]
        else:
            raise ValueError(f"zarr at {path} has no array at its root or member '0'")
        if arr.ndim == 4:
            if plane is None:
                raise ValueError(f"4D zarr {tuple(arr.shape)} needs --plane to pick a z-plane")
            if plane >= arr.shape[1]:
                raise ValueError(f"--plane {plane} out of range (Z={arr.shape[1]})")
            return np.asarray(arr[:, plane]).astype("float32")
        return np.asarray(arr[:]).astype("float32")

    if path.suffix.lower() in (".h5", ".hdf5"):
        raise ValueError("HDF5 input needs a dataset field: use masknmf.Hdf5Array(path, field=...)")

    with tifffile.TiffFile(str(path)) as tf:
        shape = tf.series[0].shape
    if len(shape) == 4:
        if plane is None:
            raise ValueError(f"4D TIFF {shape} needs --plane to pick a z-plane")
        if plane >= shape[1]:
            raise ValueError(f"--plane {plane} out of range (Z={shape[1]})")
        # Don't read the whole (T,Z,Y,X) stack into RAM just to keep one plane:
        # memory-map and slice, falling back to a full read only if the file
        # isn't memory-mappable (e.g. compressed).
        try:
            stack = tifffile.memmap(str(path))
        except (ValueError, MemoryError):
            stack = tifffile.imread(str(path))
        return np.asarray(stack[:, plane], dtype="float32")
    if len(shape) == 3:
        return masknmf.TiffArray(str(path))  # lazy; the pipeline accepts LazyFrameLoader
    raise ValueError(f"Unsupported TIFF shape {shape}; expected 3D or 4D")


# =============================================================================
# GUIs  (fastplotlib; must run inside a desktop event loop)
# =============================================================================


def gui_available() -> tuple[bool, str]:
    """Can we open an interactive fastplotlib window here?"""
    offscreen = os.environ.get("WGPU_FORCE_OFFSCREEN", "").lower()
    if offscreen and offscreen not in ("0", "false", "no"):
        return False, "WGPU_FORCE_OFFSCREEN set (offscreen only, no interactive window)"
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
            and not os.environ.get("WAYLAND_DISPLAY"):
        return False, "no DISPLAY/WAYLAND_DISPLAY (headless)"
    return True, "ok"


def view_raw(movie, params: PipelineParams) -> None:
    """Open masknmf's fastplotlib ImageWidget on the raw movie, then block."""
    ok, why = gui_available()
    if not (params.gui.show and ok):
        print(f"[gui] skipping raw viewer: {'gui disabled' if not params.gui.show else why}")
        return
    _require_masknmf()
    import fastplotlib as fpl

    iw = fpl.ImageWidget(data=movie, names=["raw"])
    iw.cmap = "gray"
    iw.show()
    fpl.loop.run()  # blocks until the window is closed


def view_results(results, outdir: Path, movie, params: PipelineParams) -> None:
    """Show the masknmf result GUIs (make_demixing_video, optional PMDWidget)."""
    ok, why = gui_available()
    if not (params.gui.show and ok):
        print(f"[gui] skipping result viewers: {'gui disabled' if not params.gui.show else why}")
        return
    masknmf = _require_masknmf()
    import fastplotlib as fpl

    device = _resolve_device(params.general.device)

    # Keep a reference to every widget until the event loop runs, otherwise the
    # canvas can be garbage-collected before it renders.
    widgets = []

    # Primary, desktop-friendly viewer: 2x3 demixing video (top-level export).
    demixing_video = masknmf.make_demixing_video(
        results, device=device, v_range=list(params.gui.v_range), show_histogram=False
    )
    demixing_video.show()
    widgets.append(demixing_video)

    # Optional PMD diagnostic. It is imgui/ipywidgets-based and really wants a
    # Jupyter kernel, so it is opt-in and guarded.
    if params.gui.pmd_widget:
        try:
            moco_path = outdir / "motion_correction.hdf5"
            comparison = (
                masknmf.RegistrationArray.from_hdf5(str(moco_path))
                if moco_path.exists()
                else movie
            )
            pmd_stack = masknmf.PMDArray.from_hdf5(str(outdir / "compression.hdf5"))
            pmd_widget = masknmf.PMDWidget(
                comparison,
                pmd_stack,
                frame_batch_size=params.general.frame_batch_size,
                device=device,
            )
            pmd_widget.show()
            widgets.append(pmd_widget)
        except Exception as exc:  # pragma: no cover - GUI/env dependent
            print(f"[gui] PMDWidget unavailable (try Jupyter): {exc!r}")

    fpl.loop.run()  # one loop serves every window opened above; keeps `widgets` alive


# =============================================================================
# Orchestration
# =============================================================================


def _prepare_outdir(outdir: Path) -> None:
    """Create ``outdir`` and clear stale artifacts (export refuses to overwrite)."""
    outdir.mkdir(parents=True, exist_ok=True)
    for name in _ARTIFACTS:
        artifact = outdir / name
        if artifact.exists():
            print(f"[outdir] removing existing {artifact}")
            artifact.unlink()


def run(params: PipelineParams, data_path: str | Path, plane: int | None,
        run_pipeline: bool):
    """Load the movie, then VIEW it (default) or run the pipeline (``--run``)."""
    movie = load_movie(data_path, plane=plane)
    print(f"[data] {data_path} -> shape={getattr(movie, 'shape', '?')} "
          f"type={type(movie).__name__}")

    if not run_pipeline:
        print("[view] showing the raw movie in masknmf's viewer "
              "(add --run to process it through the pipeline)")
        view_raw(movie, params)
        return None

    params.print_panel()

    # The compression/demixing baseline detrenders reflect-pad by a rolling
    # window of up to ~40s (int(40 * frame_rate) frames); a clip shorter than
    # that fails deep in torch. Warn early with the concrete fix.
    n_frames = int(getattr(movie, "shape", (0,))[0])
    window = int(40 * params.general.frame_rate)
    if n_frames and n_frames <= window:
        print(f"[warn] {n_frames} frames is short for the ~40s baseline window "
              f"({window} frames at {params.general.frame_rate} Hz); compression/"
              f"demixing may fail. Use more frames or a lower --frame-rate.")

    outdir = Path(params.general.outdir)
    _prepare_outdir(outdir)

    pipe = build_pipeline(params)
    print(f"[pipeline] motion correction -> PMD compression -> demixing "
          f"(device={_resolve_device(params.general.device)})...")
    # run() requires frame_rate (rolling-baseline windows in demixing).
    # remove_intermediates=False keeps motion_correction.hdf5 + compression.hdf5 (the
    # artifacts reported below, and what --pmd-widget reloads); masknmf deletes them by default.
    results = pipe.run(movie, params.general.frame_rate, remove_intermediates=False)
    print(f"[pipeline] done; artifacts in {outdir}/")

    view_results(results, outdir, movie, params)
    return results


def print_info() -> None:
    """The no-dataset screen: what the tool does and how to give it data."""
    print(__doc__.strip().splitlines()[0])
    print("\nGive it a dataset:")
    print("  --data PATH    3D TIFF, 4D ScanImage TIFF, or .zarr volume")
    print("  --pick         choose a TIFF from a native file dialog")
    print("  --plane N      z-plane index for 4D / volume sources")
    print("\nBy default it VIEWS the movie in masknmf's GUI; add --run to push it")
    print("through the pipeline (motion-correct -> PMD compress -> demix).")
    print("\nExamples:")
    print("  masknmf_gui.py --data MOVIE --plane 7          # view the movie")
    print("  masknmf_gui.py --data MOVIE --plane 7 --run    # run the pipeline + view results")
    print("  masknmf_gui.py --pick                          # pick a file, then view")
    print("  masknmf_gui.py --print-params                  # show parameters (no data needed)")
    print("  masknmf_gui.py --help                          # every flag")
    print(f"\nExample dataset on this machine:\n  {DEFAULT_DATA} --plane {DEFAULT_PLANE}")
    print("\nNotes: demixing needs a CUDA GPU; that example clip (119 frames) is short for the")
    print("pipeline's baseline windows -- for a full run use more frames or --no-denoise --frame-rate 2.5.")


def pick_dataset() -> str | None:
    """Open a native file dialog to choose a TIFF movie (needs a desktop + zenity/kdialog)."""
    try:
        from imgui_bundle import portable_file_dialogs as pfd
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[pick] file dialog unavailable ({exc}); pass --data PATH instead.")
        return None
    try:
        selection = pfd.open_file(
            "Select a movie (TIFF); for a .zarr volume use --data",
            filters=["TIFF", "*.tif *.tiff", "All files", "*"],
        ).result()
    except Exception as exc:  # pragma: no cover - dialog backend dependent
        print(f"[pick] file dialog failed ({exc}); pass --data PATH instead.")
        return None
    if not selection:
        print("[pick] no file chosen.")
        return None
    return selection[0]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run an LBM movie through masknmf and view it in masknmf's GUIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=None,
                   help="movie: 3D TIFF, 4D ScanImage TIFF, or .zarr volume "
                        "(omit to show info; --pick for a file dialog)")
    p.add_argument("--pick", action="store_true",
                   help="choose the dataset from a native file dialog")
    p.add_argument("--plane", type=int, default=None,
                   help="z-plane index (required for 4D / volume sources)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--frame-rate", type=float, default=17.07, help="acquisition rate (Hz)")
    p.add_argument("--outdir", default="masknmf_out")
    p.add_argument("--frame-batch-size", type=int, default=300)
    p.add_argument("--load-into-ram", action="store_true")
    p.add_argument("--no-motion-correction", action="store_true")
    p.add_argument("--piecewise", action="store_true", help="piecewise-rigid moco")
    p.add_argument("--no-denoise", action="store_true", help="CompressConfig instead of denoise")
    p.add_argument("--max-components", type=int, default=20)
    p.add_argument("--custom-demixing", action="store_true",
                   help="use the exposed superpixel/NMF knobs instead of pipeline defaults")
    p.add_argument("--run", action="store_true",
                   help="run the full pipeline (moco -> PMD -> demix); default is view-only")
    p.add_argument("--no-gui", dest="gui", action="store_false", help="never open windows")
    p.add_argument("--pmd-widget", action="store_true",
                   help="with --run, also open the PMD diagnostic GUI")
    p.add_argument("--print-params", action="store_true",
                   help="print the resolved parameter panel and exit (no data, no run)")
    return p


def params_from_args(args: argparse.Namespace) -> PipelineParams:
    return PipelineParams(
        general=GeneralParams(
            device=args.device,
            frame_rate=args.frame_rate,
            frame_batch_size=args.frame_batch_size,
            load_into_ram=args.load_into_ram,
            outdir=args.outdir,
        ),
        motion=MotionCorrectionParams(
            enabled=not args.no_motion_correction,
            piecewise=args.piecewise,
        ),
        compression=CompressionParams(
            denoise=not args.no_denoise,
            max_components=args.max_components,
        ),
        demixing=DemixingParams(custom_schedule=args.custom_demixing),
        gui=GuiParams(show=args.gui, pmd_widget=args.pmd_widget),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    params = params_from_args(args)

    if args.print_params:
        params.print_panel()
        # Also show the concrete masknmf config objects the panel maps to. These
        # need masknmf; if it cannot load, the panel above is still useful.
        try:
            print("motion  :", build_motion_config(params.motion))
            print("compress:", build_compress_config(params.compression))
            print("demixing:", build_multipass_config(params.demixing))
        except SystemExit as exc:
            print(f"\n[configs] masknmf config objects unavailable:\n{exc}")
        return 0

    data = args.data
    if data is None and args.pick:
        data = pick_dataset()
    if data is None:
        print_info()  # no dataset given: show info instead of running anything heavy
        return 0

    run(params, data, plane=args.plane, run_pipeline=args.run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
