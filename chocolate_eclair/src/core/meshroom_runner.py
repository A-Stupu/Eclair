"""Tools to execute Meshroom photogrammetry pipelines from Python."""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


class MeshroomRunError(RuntimeError):
    """Raised when Meshroom cannot be started."""


def detect_default_meshroom_executable(base_dir: Path | None = None) -> Path | None:
    """Return the Meshroom executable shipped within the chocolate_eclair folder if present."""

    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    root = root.resolve()
    chocolate_root = root / "chocolate_eclair"

    if platform.system() == "Windows":
        search_dirs = [
            chocolate_root / "meshroom_windows",
            chocolate_root,
            root / "meshroom-windows",
            root / "meshroom_windows",
        ]
        executables = ["meshroom_batch.exe", "meshroom_photogrammetry.exe", "Meshroom.exe"]
    else:
        search_dirs = [
            chocolate_root / "meshroom_linux",
            chocolate_root,
            root / "meshroom",
        ]
        executables = ["meshroom_batch", "meshroom_photogrammetry", "Meshroom"]

    for directory in search_dirs:
        for exe in executables:
            candidate = directory / exe
            if candidate.exists():
                return candidate
    return None


def find_meshroom_candidates(root: Path | None = None) -> Sequence[Path]:
    """Return likely Meshroom executables for the current platform."""

    candidates: list[Path] = []
    default_candidate = detect_default_meshroom_executable(root)
    if default_candidate:
        candidates.append(default_candidate)

    search_roots = [root] if root else []

    if platform.system() == "Windows":
        search_roots.extend(
            [
                Path("C:/Program Files/Meshroom"),
                Path("C:/Program Files/meshroom"),
                Path.cwd() / "meshroom-windows",
                Path.cwd() / "meshroom_windows",
            ]
        )
        executables = ["meshroom_batch.exe", "meshroom_photogrammetry.exe", "Meshroom.exe"]
    else:
        search_roots.extend([
            Path.home() / "Meshroom-2023.3.0",
            Path("/opt/meshroom"),
            Path.cwd() / "meshroom",
        ])
        executables = ["meshroom_batch", "meshroom_photogrammetry", "Meshroom"]

    for base in search_roots:
        if not base:
            continue
        for exe in executables:
            candidate = base / exe
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def has_cuda_support() -> bool:
    """Best-effort check for CUDA-capable NVIDIA hardware on the host."""

    if shutil.which("nvidia-smi"):
        return True

    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path and Path(cuda_path).exists():
        return True

    if platform.system() == "Linux":
        possible_paths = [
            Path("/usr/local/cuda"),
            Path("/usr/local/cuda/bin/nvidia-smi"),
        ]
        if any(path.exists() for path in possible_paths):
            return True

    return False


def get_pipeline_steps(cuda_enabled: Optional[bool] = None) -> List[str]:
    """Return the ordered list of Meshroom nodes for the current hardware."""

    if cuda_enabled is None:
        cuda_enabled = has_cuda_support()

    if cuda_enabled:
        return [
            "FeatureExtraction",
            "ImageMatching",
            "FeatureMatching",
            "StructureFromMotion",
            "PrepareDenseScene",
            "DepthMap",
            "DepthMapFilter",
            "Meshing",
            "MeshFiltering",
            "Texturing",
        ]

    return [
        "FeatureExtraction",
        "ImageMatching",
        "FeatureMatching",
        "StructureFromMotion",
        "Meshing",
        "PrepareDenseScene",
        "Texturing",
    ]


def _stage_images(image_paths: Iterable[Path | str], workspace_dir: Path) -> Path:
    """Copy source images into a dedicated folder for Meshroom."""

    staging_dir = workspace_dir / "input"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths):
        src = Path(image_path)
        if not src.is_file():
            raise MeshroomRunError(f"Image file not found: {src}")

        destination = staging_dir / src.name
        if destination.exists():
            destination = staging_dir / f"{index}_{src.name}"
        shutil.copy2(src, destination)

    return staging_dir


def run_meshroom_background(
    meshroom_executable: Path | str,
    image_paths: Iterable[Path | str],
    workspace_dir: Path | str,
    log_file: Path | str | None = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    project_name: Optional[str] = None,
) -> subprocess.Popen:
    """Start Meshroom in background and return the running process."""

    image_paths = list(image_paths)
    if not image_paths:
        raise MeshroomRunError("No images provided to Meshroom.")

    executable = Path(meshroom_executable)
    if not executable.exists():
        raise MeshroomRunError(f"Meshroom executable not found: {executable}")

    lower_name = executable.name.lower()
    if lower_name in {"meshroom.exe", "meshroom"}:
        preferred = [
            "meshroom_batch.exe",
            "meshroom_batch",
            "meshroom_photogrammetry.exe",
            "meshroom_photogrammetry",
        ]
        alt_candidates = []
        for alt_name in preferred:
            alt_candidates.extend(
                [
                    executable.with_name(alt_name),
                    executable.parent / alt_name,
                    executable.parent / "bin" / alt_name,
                ]
            )
        for alt in alt_candidates:
            if alt.exists():
                logger.info("Switching to Meshroom CLI executable: %s", alt)
                executable = alt
                break

    workspace_root = Path(workspace_dir)
    workspace_root.mkdir(parents=True, exist_ok=True)
    project_dir = workspace_root / project_name if project_name else workspace_root
    project_dir.mkdir(parents=True, exist_ok=True)

    staged_input = _stage_images(image_paths, project_dir)

    cuda_enabled = has_cuda_support()
    pipeline_steps = get_pipeline_steps(cuda_enabled)
    pipeline_label = "CUDA" if cuda_enabled else "CPU"
    pipeline_header = f"Meshroom pipeline plan ({pipeline_label} path):"

    cache_dir = project_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(executable),
        "--input",
        str(staged_input),
        "--output",
        str(project_dir),
        "--cache",
        str(cache_dir),
    ]
    if extra_args:
        command.extend(extra_args)

    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)

    popen_kwargs: dict[str, object] = {
        "cwd": str(project_dir),
        "env": env_vars,
    }

    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    else:
        popen_kwargs["start_new_session"] = True

    log_handle = None
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        popen_kwargs["stdout"] = log_handle
        popen_kwargs["stderr"] = subprocess.STDOUT
    else:
        popen_kwargs["stdout"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = subprocess.DEVNULL

    def _log_message(message: str) -> None:
        logger.info(message)
        if log_handle:
            log_handle.write(message + "\n")
            log_handle.flush()

    _log_message("=== Eclair Meshroom run ===")
    _log_message(f"Working directory: {project_dir}")
    if project_name:
        _log_message(f"Project name: {project_name}")
    _log_message(pipeline_header)
    for index, step in enumerate(pipeline_steps, start=1):
        _log_message(f"  Step {index}: {step}")
    _log_message("Launching Meshroom...")

    try:
        process = subprocess.Popen(command, **popen_kwargs)  # type: ignore[arg-type]
    except OSError as exc:  # pragma: no cover - depends on runtime environment
        if log_handle:
            log_handle.close()
        raise MeshroomRunError(f"Unable to launch Meshroom: {exc}") from exc

    immediate_exit_code = process.poll()
    if immediate_exit_code is not None:
        if log_handle:
            log_handle.close()
        log_reference = str(log_file) if log_file else "N/A"
        message = (
            f"Meshroom exited immediately with code {immediate_exit_code}. "
            f"Check the log file at {log_reference} for details."
        )
        raise MeshroomRunError(message)

    if log_handle:
        setattr(process, "log_file_handle", log_handle)
    setattr(process, "pipeline_steps", pipeline_steps)
    setattr(process, "project_dir", project_dir)
    setattr(process, "output_dir", project_dir)
    setattr(process, "staged_input", staged_input)
    setattr(process, "pipeline_cuda", cuda_enabled)

    return process
