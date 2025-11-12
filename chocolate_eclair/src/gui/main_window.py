"""PyQt main window for Eclair GUI."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from chocolate_eclair.src.core.photo_checker import analyze_photo
from chocolate_eclair.src.core.meshroom_runner import (
    MeshroomRunError,
    detect_default_meshroom_executable,
    get_pipeline_steps,
    has_cuda_support,
    run_meshroom_background,
)

from ..core.capture import capture_and_analyze
from ..hardware.arduino_controller import make_controller


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dng"}
WORKSPACE_DIR = Path(__file__).resolve().parents[4] / "workspace"


class _AnalysisSignals(QObject):
    finished = pyqtSignal(str, dict, str)


class _AnalysisWorker(QRunnable):
    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.image_path = image_path
        self.signals = _AnalysisSignals()

    def run(self) -> None:  # type: ignore[override]
        try:
            metrics = analyze_photo(self.image_path)
            self.signals.finished.emit(str(self.image_path), metrics, "")
        except Exception as exc:  # pragma: no cover - defensive path
            logging.exception("Failed to analyze %s", self.image_path)
            self.signals.finished.emit(str(self.image_path), {}, str(exc))


class _CaptureSignals(QObject):
    finished = pyqtSignal(dict, str)


class _CaptureWorker(QRunnable):
    def __init__(
        self,
        save_dir: Path,
        count: int,
        step_deg: float,
        camera_index: int = 0,
        arduino=None,
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.count = count
        self.step_deg = step_deg
        self.camera_index = camera_index
        self.arduino = arduino
        self.signals = _CaptureSignals()

    def run(self) -> None:  # type: ignore[override]
        for i in range(self.count):
            filename = f"capture_{i:03d}.jpg"
            target = self.save_dir / filename
            try:
                info = capture_and_analyze(
                    target,
                    camera_index=self.camera_index,
                    arduino=self.arduino,
                    advance_degrees=self.step_deg,
                )
                info["index"] = i
                self.signals.finished.emit(info, "")
            except Exception as exc:  # pragma: no cover - environment dependent
                logging.exception("Capture failed at index %d", i)
                self.signals.finished.emit({}, str(exc))
                return
        self.signals.finished.emit({"complete": True, "total": self.count}, "")


class DropListWidget(QListWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(self.ExtendedSelection)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        paths: List[str] = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if local_path:
                paths.append(local_path)

        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Eclair Photogrammetry Helper")
        self.resize(900, 600)

        self.thread_pool = QThreadPool.globalInstance()
        self.meshroom_executable: Optional[Path] = None

        self.drop_list = DropListWidget()
        self.drop_list.files_dropped.connect(self._handle_dropped_files)

        self.status_label = QLabel('Drag images here or use "Add Images".')
        self.status_label.setWordWrap(True)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        button_bar = QHBoxLayout()
        add_btn = QPushButton("Add Images")
        add_btn.clicked.connect(self._prompt_for_files)
        button_bar.addWidget(add_btn)

        upload_all_btn = QPushButton("Upload All from Folder")
        upload_all_btn.clicked.connect(self._upload_all_from_folder)
        button_bar.addWidget(upload_all_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_items)
        button_bar.addWidget(remove_btn)

        self.connect_turntable_btn = QPushButton("Connect Turntable")
        self.connect_turntable_btn.setCheckable(True)
        self.connect_turntable_btn.toggled.connect(self._toggle_arduino_connection)
        button_bar.addWidget(self.connect_turntable_btn)

        capture_btn = QPushButton("Capture Sequence")
        capture_btn.clicked.connect(self._capture_sequence)
        button_bar.addWidget(capture_btn)

        clear_btn = QPushButton("Clear List")
        clear_btn.clicked.connect(self.drop_list.clear)
        button_bar.addWidget(clear_btn)

        run_btn = QPushButton("Start Meshroom")
        run_btn.clicked.connect(self._run_meshroom)
        button_bar.addWidget(run_btn)

        self.toggle_log_btn = QPushButton("Hide Log")
        self.toggle_log_btn.setCheckable(True)
        self.toggle_log_btn.toggled.connect(self._toggle_log_visibility)
        button_bar.addWidget(self.toggle_log_btn)

        button_bar.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.drop_list)
        layout.addLayout(button_bar)
        self.log_label = QLabel("Application Log")
        layout.addWidget(self.log_label)
        layout.addWidget(self.log_output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO)

        try:
            self.arduino = make_controller()
        except Exception:
            self.arduino = None

        base_dir = Path(__file__).resolve().parents[3]
        self.meshroom_executable = detect_default_meshroom_executable(base_dir)
        if self.meshroom_executable:
            self._log(f"Meshroom executable detected: {self.meshroom_executable}")
        else:
            self._log(
                "Meshroom executable not found. Expected under chocolate_eclair/meshroom_windows or chocolate_eclair/meshroom_linux."
            )

        self.meshroom_process = None
        self.meshroom_timer = QTimer(self)
        self.meshroom_timer.setInterval(2000)
        self.meshroom_timer.timeout.connect(self._check_meshroom_process)

    # ---- UI helpers -------------------------------------------------
    def _prompt_for_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.dng)",
        )
        if files:
            self._log(f"Selected {len(files)} file(s) from dialog.")
            self._enqueue_files(files)

    def _handle_dropped_files(self, paths: List[str]) -> None:
        self._log(f"Dropped {len(paths)} file(s) onto the list.")
        self._enqueue_files(paths)

    def _enqueue_files(self, paths: Iterable[str]) -> None:
        valid_files = []
        for candidate in paths:
            path = Path(candidate)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if not self._contains_path(path):
                    valid_files.append(path)
            else:
                self._log(f"Skipped unsupported file: {path}")

        if not valid_files:
            return

        for path in valid_files:
            item = QListWidgetItem(str(path))
            item.setData(Qt.UserRole, str(path))
            item.setToolTip("Analysis in progress...")
            self.drop_list.addItem(item)
            self._run_analysis(path, item)

        self._log(f"Queued {len(valid_files)} image(s) for analysis.")
        self.status_label.setText("Analyzing images...")

    def _toggle_log_visibility(self, hidden: bool) -> None:
        visible = not hidden
        self.log_label.setVisible(visible)
        self.log_output.setVisible(visible)
        self.toggle_log_btn.setText("Show Log" if hidden else "Hide Log")

    def _toggle_arduino_connection(self, checked: bool) -> None:
        if checked:
            self._log("Connecting to turntable...")
            if self.arduino and self.arduino.connect():
                self.connect_turntable_btn.setText("Disconnect Turntable")
                self._log("Turntable connected (or simulated).")
            else:
                self._log("Failed to connect to turntable; running simulated controller.")
                self.connect_turntable_btn.setChecked(True)
                self.connect_turntable_btn.setText("Disconnect Turntable")
        else:
            self._log("Disconnecting turntable...")
            if self.arduino:
                try:
                    self.arduino.disconnect()
                except Exception:
                    pass
            self.connect_turntable_btn.setText("Connect Turntable")

    def _capture_sequence(self) -> None:
        if not self.arduino:
            try:
                self.arduino = make_controller()
            except Exception:
                self.arduino = None

        count, ok = QInputDialog.getInt(
            self,
            "Capture count",
            "Number of photos to capture:",
            value=12,
            min=1,
            max=360,
        )
        if not ok:
            return
        step_deg, ok2 = QInputDialog.getInt(
            self,
            "Step degrees",
            "Degrees to step between captures:",
            value=10,
            min=1,
            max=360,
        )
        if not ok2:
            return

        project_dir = WORKSPACE_DIR / self._suggest_project_name()
        input_dir = project_dir / "input"
        project_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(exist_ok=True)

        worker = _CaptureWorker(
            save_dir=input_dir,
            count=count,
            step_deg=float(step_deg),
            arduino=self.arduino,
        )
        worker.signals.finished.connect(self._on_capture_finished)
        self.thread_pool.start(worker)
        self._log(
            f"Started capture sequence: {count} images, {step_deg}° step. Saving to {input_dir}"
        )
        self._log(f"Project folder prepared at {project_dir}")

    def _on_capture_finished(self, info: dict, error: str) -> None:
        if error:
            self._log(f"Capture error: {error}")
            return

        if info.get("complete"):
            total = info.get("total", 0)
            self._log(f"Capture sequence finished. Total images: {total}")
            self.status_label.setText("Capture sequence complete.")
            return

        path = info.get("path")
        ok = info.get("ok", False)
        if path and ok:
            self._enqueue_files([str(path)])
            self._log(f"Captured and accepted: {path}")
        elif path and not ok:
            self._log(
                f"Captured but marked as needing attention: {path}; turntable returned to previous position"
            )
            QMessageBox.warning(
                self,
                "Retake Photo",
                "The captured photo needs attention. Please retake the shot before continuing.",
            )

    def _upload_all_from_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder with images",
            str(WORKSPACE_DIR),
        )
        if not folder:
            return
        p = Path(folder)
        files = [
            str(p / name)
            for name in sorted(p.iterdir())
            if name.is_file() and name.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if files:
            self._enqueue_files(files)
            self._log(f"Uploaded {len(files)} images from {folder}")
        else:
            self._log(f"No supported images found in {folder}")

    def _remove_selected_items(self) -> None:
        selected_items = self.drop_list.selectedItems()
        if not selected_items:
            self.status_label.setText("Select one or more images to remove.")
            return

        for item in selected_items:
            row = self.drop_list.row(item)
            removed_path = item.data(Qt.UserRole) or item.text()
            self.drop_list.takeItem(row)
            self._log(f"Removed image from queue: {removed_path}")

        if self.drop_list.count() == 0:
            self.status_label.setText('Drag images here or use "Add Images".')
        else:
            self.status_label.setText("Ready.")

    def _suggest_project_name(self) -> str:
        return datetime.now().strftime("project_%Y%m%d_%H%M%S")

    @staticmethod
    def _sanitize_project_name(name: str) -> str:
        sanitized = "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_"})
        return sanitized.strip("-_")

    def _log_pipeline_plan(self, cuda_enabled: bool) -> None:
        steps = get_pipeline_steps(cuda_enabled)
        label = "CUDA" if cuda_enabled else "CPU"
        self._log(f"Meshroom pipeline plan ({label}):")
        for index, step in enumerate(steps, start=1):
            self._log(f"  Step {index}: {step}")

    def _contains_path(self, path: Path) -> bool:
        for index in range(self.drop_list.count()):
            existing = self.drop_list.item(index)
            if Path(existing.data(Qt.UserRole)).resolve() == path.resolve():
                return True
        return False

    def _run_analysis(self, path: Path, item: QListWidgetItem) -> None:
        worker = _AnalysisWorker(path)
        worker.signals.finished.connect(lambda p, metrics, err: self._update_item(item, p, metrics, err))
        self.thread_pool.start(worker)

    def _update_item(self, item: QListWidgetItem, path: str, metrics: dict, error: str) -> None:
        if error:
            item.setText(f"{path} — ERROR: {error}")
            item.setToolTip(error)
            item.setForeground(Qt.red)
            self._log(f"Error analyzing {path}: {error}")
        else:
            blur_val = metrics.get("blur")
            brightness_val = metrics.get("brightness")
            resolution = metrics.get("resolution", "n/a")

            blur_text = f"{float(blur_val):.2f}" if isinstance(blur_val, (int, float)) else "n/a"
            brightness_text = (
                f"{float(brightness_val):.2f}" if isinstance(brightness_val, (int, float)) else "n/a"
            )

            status = str(metrics.get("status", "Unknown"))
            issues = metrics.get("issues", [])
            status_text = "OK" if status.upper() == "OK" else "Needs attention"

            issue_lines = (
                "All metrics within recommended ranges."
                if not issues
                else "Issues:\n- " + "\n- ".join(str(issue) for issue in issues)
            )

            tooltip = (
                f"Status: {status_text}\n"
                f"Blur: {blur_text}\n"
                f"Brightness: {brightness_text}\n"
                f"Resolution: {resolution}\n"
                f"{issue_lines}"
            )

            item.setText(
                f"{path} — Status: {status_text}; Blur: {blur_text}; "
                f"Brightness: {brightness_text}; Resolution: {resolution}"
            )
            item.setToolTip(tooltip)

            if status_text == "OK":
                item.setForeground(Qt.darkGreen)
            else:
                item.setForeground(Qt.darkYellow)

            self._log(f"Analysis complete for {path}: {metrics}")

        self.status_label.setText("Ready.")

    def _ensure_meshroom_executable(self) -> Optional[Path]:
        existing = self.meshroom_executable
        if existing and existing.exists():
            return existing

        base_dir = Path(__file__).resolve().parents[3]
        detected = detect_default_meshroom_executable(base_dir)
        if detected and detected != existing:
            self._log(f"Meshroom executable detected: {detected}")
        elif not detected and existing:
            self._log(
                "Meshroom executable could not be found under chocolate_eclair/meshroom_windows or chocolate_eclair/meshroom_linux."
            )

        self.meshroom_executable = detected
        return detected

    def _run_meshroom(self) -> None:
        if self.drop_list.count() == 0:
            QMessageBox.warning(self, "No images", "Add at least one image before starting Meshroom.")
            return

        meshroom_executable = self._ensure_meshroom_executable()
        if not meshroom_executable:
            QMessageBox.warning(
                self,
                "Meshroom not found",
                "Meshroom executable not found. Place it under chocolate_eclair/meshroom_windows (Windows) or chocolate_eclair/meshroom_linux (Linux).",
            )
            return

        self._log(f"Preparing Meshroom run with {self.drop_list.count()} image(s).")

        default_name = self._suggest_project_name()
        project_name, accepted = QInputDialog.getText(
            self,
            "Project Name",
            "Enter a project name for the Meshroom output:",
            text=default_name,
        )
        if not accepted:
            self._log("Meshroom run cancelled by user.")
            return

        sanitized_name = self._sanitize_project_name(project_name)
        if not sanitized_name:
            sanitized_name = self._sanitize_project_name(default_name)
            self._log(f"Using fallback project name: {sanitized_name}")
        elif project_name.strip() != sanitized_name:
            self._log(f"Project name sanitized to: {sanitized_name}")

        image_paths = [Path(self.drop_list.item(i).data(Qt.UserRole)) for i in range(self.drop_list.count())]
        selected_workspace = QFileDialog.getExistingDirectory(
            self,
            "Select output base folder",
            str(WORKSPACE_DIR),
        )
        if selected_workspace:
            workspace = Path(selected_workspace)
            self._log(f"Selected output base folder: {workspace}")
        else:
            workspace = WORKSPACE_DIR
            self._log("Using default workspace for output.")

        workspace.mkdir(parents=True, exist_ok=True)

        project_dir = workspace / sanitized_name
        project_dir.mkdir(parents=True, exist_ok=True)
        input_dir = project_dir / "input"
        input_dir.mkdir(exist_ok=True)
        log_path = project_dir / "meshroom.log"

        cuda_enabled = has_cuda_support()
        self._log_pipeline_plan(cuda_enabled)
        self._log(f"Project directory: {project_dir}")
        self._log("Meshroom outputs will be written inside the project folder.")

        try:
            process = run_meshroom_background(
                meshroom_executable=meshroom_executable,
                image_paths=image_paths,
                workspace_dir=workspace,
                log_file=log_path,
                project_name=sanitized_name,
            )
        except MeshroomRunError as exc:
            QMessageBox.critical(self, "Meshroom error", str(exc))
            self._log(f"Error starting Meshroom: {exc}")
            return

        mode_label = "CUDA" if getattr(process, "pipeline_cuda", False) else "CPU"
        self._log(
            f"Meshroom started (PID {process.pid}) in {mode_label} mode. Log output: {log_path}"  # type: ignore[attr-defined]
        )
        if self.meshroom_process:
            self.meshroom_timer.stop()
            self._log("Cancelling monitoring of previous Meshroom process.")
        self.meshroom_process = process
        self.meshroom_timer.start()
        self.status_label.setText("Meshroom processing in progress...")

    def _log(self, message: str) -> None:
        logging.info(message)
        self.log_output.append(message)

    def _check_meshroom_process(self) -> None:
        process = self.meshroom_process
        if not process:
            self.meshroom_timer.stop()
            return

        exit_code = process.poll()
        if exit_code is None:
            return

        self.meshroom_timer.stop()
        if hasattr(process, "log_file_handle"):
            try:
                process.log_file_handle.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            delattr(process, "log_file_handle")

        if exit_code == 0:
            self._on_meshroom_success(process)
        else:
            self._handle_meshroom_failure(process, exit_code)

        self.meshroom_process = None

    def _on_meshroom_success(self, process) -> None:
        output_path = Path(getattr(process, "output_dir", WORKSPACE_DIR))
        staged_input = getattr(process, "staged_input", None)
        if staged_input:
            self._log(f"Input folder available at {staged_input}.")

        ignore_names = {"input", "cache", "meshroom.log", "meshroomcache"}
        relevant_outputs = [
            p
            for p in output_path.iterdir()
            if p.name.lower() not in ignore_names
        ] if output_path.exists() else []

        if not relevant_outputs:
            self._log(
                f"Meshroom finished successfully, but no additional outputs were detected in {output_path}."
            )
        else:
            self._log(f"Meshroom finished successfully. Outputs stored in {output_path}.")

        log_path = output_path / "meshroom.log"
        if log_path.exists():
            self._log(f"Meshroom log available at {log_path}.")

        self.status_label.setText("Meshroom processing finished successfully.")

    def _handle_meshroom_failure(self, process, exit_code: int) -> None:
        output_path = Path(getattr(process, "output_dir", WORKSPACE_DIR))
        log_path = output_path / "meshroom.log"
        self._log(f"Meshroom finished with exit code {exit_code}. Check {log_path} for details.")
        staged_input = getattr(process, "staged_input", None)
        if staged_input:
            self._log(f"Input folder available at {staged_input}.")
        self.status_label.setText("Meshroom processing finished with errors.")
