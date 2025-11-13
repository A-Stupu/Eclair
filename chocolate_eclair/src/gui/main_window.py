"""PyQt main window for Eclair GUI."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import cv2

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QComboBox,
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
from ..hardware.turntable_controller import TurntableController, TurntableError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dng"}
WORKSPACE_DIR = Path(__file__).resolve().parents[4] / "workspace"


def _preferred_camera_backend() -> int:
    if sys.platform.startswith("win"):
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    if sys.platform.startswith("linux"):
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    if sys.platform == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return cv2.CAP_ANY


def _try_open_camera(index: int, backend: Optional[int]) -> Optional[cv2.VideoCapture]:
    try:
        if backend is None or backend == cv2.CAP_ANY:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
    except Exception:
        return None

    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None
    return cap


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


class CameraSelectionDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        cameras: List[tuple[int, str]],
        current_index: Optional[int],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Camera")
        self.setModal(True)
        self._cameras = cameras
        self.selected_camera_index: Optional[int] = current_index
        self._capture: Optional[cv2.VideoCapture] = None

        layout = QVBoxLayout()

        self.selector = QComboBox()
        for index, label in cameras:
            self.selector.addItem(label, index)
        if current_index is not None:
            combo_index = self.selector.findData(current_index)
            if combo_index >= 0:
                self.selector.setCurrentIndex(combo_index)
        self.selector.currentIndexChanged.connect(self._on_camera_changed)
        layout.addWidget(self.selector)

        inputs_text = "\n".join(f"{idx}: {label}" for idx, label in cameras) or "No inputs detected."
        self.inputs_label = QLabel(f"Detected inputs:\n{inputs_text}")
        self.inputs_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.inputs_label)

        self.preview_label = QLabel("Preview not available")
        self.preview_label.setFixedSize(360, 270)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_preview)

        if cameras:
            initial_index = self.selector.currentData()
            if initial_index is None:
                initial_index = cameras[0][0]
            self._start_preview(int(initial_index))

    def _start_preview(self, index: int) -> None:
        self._stop_preview()

        cap = self._open_capture(index)
        if not cap or not cap.isOpened():
            self.preview_label.setText("Unable to open camera preview")
            self.info_label.setText("")
            return

        self._capture = cap
        self.selected_camera_index = index
        self.preview_label.setText("Starting preview...")
        self._timer.start(1000 // 24)

    def _stop_preview(self, close_capture: bool = True) -> None:
        self._timer.stop()
        if close_capture and self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def _update_preview(self) -> None:
        if not self._capture:
            return
        ret, frame = self._capture.read()
        if not ret or frame is None:
            return

        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            frame_rgb.data,
            width,
            height,
            frame_rgb.strides[0],
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)
        self.info_label.setText(f"Resolution: {width}x{height}")

    def _on_camera_changed(self, combo_index: int) -> None:
        index_data = self.selector.itemData(combo_index)
        if index_data is None:
            return
        self._start_preview(int(index_data))

    def _on_accept(self) -> None:
        if self.selector.count() == 0:
            QMessageBox.warning(self, "No cameras", "No cameras available to select.")
            return
        index_data = self.selector.currentData()
        if index_data is None:
            QMessageBox.warning(self, "Selection error", "Unable to determine selected camera.")
            return
        self.selected_camera_index = int(index_data)
        self._stop_preview(close_capture=True)
        self.accept()

    def reject(self) -> None:  # type: ignore[override]
        self._stop_preview(close_capture=True)
        super().reject()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_preview(close_capture=True)
        super().closeEvent(event)

    @staticmethod
    def _open_capture(index: int) -> Optional[cv2.VideoCapture]:
        backend = _preferred_camera_backend()
        cap = _try_open_camera(index, backend)
        if not cap and backend != cv2.CAP_ANY:
            cap = _try_open_camera(index, cv2.CAP_ANY)
        return cap

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

        self.capture_sequence_btn = QPushButton("Capture Sequence")
        self.capture_sequence_btn.clicked.connect(self._capture_sequence)
        self.capture_sequence_btn.setEnabled(False)
        button_bar.addWidget(self.capture_sequence_btn)

        self.select_camera_btn = QPushButton("Select Camera")
        self.select_camera_btn.clicked.connect(self._select_camera)
        button_bar.addWidget(self.select_camera_btn)

        self.capture_photo_btn = QPushButton("Capture Photo")
        self.capture_photo_btn.clicked.connect(self._capture_photo)
        self.capture_photo_btn.setEnabled(False)
        button_bar.addWidget(self.capture_photo_btn)

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

        self.turntable = None  # type: Optional[TurntableController]
        self.camera_index = None  # type: Optional[int]
        self.capture_project_dir = None  # type: Optional[Path]
        self.capture_step_degrees = 20.0
        self.camera_capture = None  # type: Optional[cv2.VideoCapture]
        self.available_cameras = []  # type: List[tuple[int, str]]
        self._refresh_turntable_availability()

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
        controller = self.turntable
        if not controller:
            self._log("Turntable controller unavailable.")
            self._update_turntable_ui("Arduino not detected. Manual capture only.")
            return

        if checked:
            self._log("Connecting to turntable...")
            if not controller.is_available and not controller.refresh_port():
                self._log("No Arduino turntable detected.")
                QMessageBox.warning(
                    self,
                    "No Arduino detected",
                    "Unable to detect an Arduino turntable. Connect it and try again.",
                )
                self._update_turntable_ui("Arduino not detected. Manual capture only.")
                return

            try:
                controller.connect()
            except TurntableError as exc:
                self._log(f"Turntable connection failed: {exc}")
                QMessageBox.warning(
                    self,
                    "Turntable error",
                    f"Failed to connect to the Arduino turntable.\n{exc}",
                )
                self._update_turntable_ui("Arduino not detected. Manual capture only.")
                return

            self._log("Turntable connected.")
            self._update_turntable_ui("Turntable connected. Ready for capture sequence.")
        else:
            self._log("Disconnecting turntable...")
            try:
                controller.disconnect()
            except Exception:
                pass
            self._update_turntable_ui("Turntable disconnected.")

    def _refresh_turntable_availability(self) -> None:
        try:
            controller = TurntableController()
        except TurntableError as exc:
            self.turntable = None
            self._log(f"Turntable controller not available: {exc}")
            self._update_turntable_ui("Arduino not detected. Manual capture only.")
            return

        self.turntable = controller
        if controller.is_available:
            self._log("Arduino turntable detected. Use 'Connect Turntable' to enable rotation.")
            self._update_turntable_ui()
        else:
            self._log("No Arduino turntable detected.")
            self._update_turntable_ui("Arduino not detected. Manual capture only.")

    def _update_turntable_ui(self, status_message: Optional[str] = None) -> None:
        controller = self.turntable
        if controller and controller.is_connected:
            text = "Disconnect Turntable"
            capture_enabled = True
            checked = True
            enabled = True
        elif controller and controller.is_available:
            text = "Connect Turntable"
            capture_enabled = False
            checked = False
            enabled = True
        else:
            text = "Arduino Not Detected"
            capture_enabled = False
            checked = False
            enabled = False

        self.connect_turntable_btn.blockSignals(True)
        self.connect_turntable_btn.setChecked(checked)
        self.connect_turntable_btn.blockSignals(False)
        self.connect_turntable_btn.setEnabled(enabled)
        self.connect_turntable_btn.setText(text)
        self.capture_sequence_btn.setEnabled(capture_enabled)

        if status_message:
            self.status_label.setText(status_message)

    def _ensure_capture_project_dir(self) -> Path:
        if self.capture_project_dir and self.capture_project_dir.exists():
            return self.capture_project_dir

        project_dir = WORKSPACE_DIR / self._suggest_project_name()
        input_dir = project_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        self.capture_project_dir = project_dir
        self._log(f"Capture project folder prepared: {project_dir}")
        return project_dir

    def _scan_available_cameras(self, max_index: int = 6) -> List[tuple[int, str]]:
        cameras: List[tuple[int, str]] = []
        for index in range(max_index):
            cap = self._open_camera_capture(index)
            if not cap:
                continue
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            label = f"Camera {index}"
            if width and height:
                label = f"{label} ({width}x{height})"
            cameras.append((index, label))
            cap.release()

        self.available_cameras = cameras
        return cameras

    @staticmethod
    def _open_camera_capture(index: int) -> Optional[cv2.VideoCapture]:
        backend = _preferred_camera_backend()
        cap = _try_open_camera(index, backend)
        if not cap and backend != cv2.CAP_ANY:
            cap = _try_open_camera(index, cv2.CAP_ANY)
        return cap

    def _select_camera(self) -> bool:
        cameras = self._scan_available_cameras()
        if not cameras:
            QMessageBox.warning(self, "No cameras", "No webcams detected. Connect one and try again.")
            return False

        dialog = CameraSelectionDialog(self, cameras, self.camera_index)
        result = dialog.exec_()
        selected_idx = dialog.selected_camera_index if dialog.selected_camera_index is not None else None

        if result != QDialog.Accepted or selected_idx is None:
            return False

        capture = self._open_camera_capture(selected_idx)
        if not capture:
            QMessageBox.warning(self, "Camera error", "Unable to open the selected camera.")
            return False

        if self.camera_capture and self.camera_capture.isOpened():
            try:
                self.camera_capture.release()
            except Exception:
                pass

        self.camera_capture = capture
        self.camera_index = selected_idx
        self.capture_photo_btn.setEnabled(True)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        res_text = f" ({width}x{height})" if width and height else ""
        self.status_label.setText(f"Camera {selected_idx}{res_text} selected. Ready to capture.")
        self._log(f"Camera index set to {selected_idx}{res_text}")
        return True

    def _capture_photo(self) -> None:
        if self.camera_index is None and not self._select_camera():
            return

        if self.camera_capture is None or not self.camera_capture.isOpened():
            if self.camera_capture:
                try:
                    self.camera_capture.release()
                except Exception:
                    pass
            self.camera_capture = self._open_camera_capture(int(self.camera_index)) if self.camera_index is not None else None
            if self.camera_capture is None or not self.camera_capture.isOpened():
                QMessageBox.warning(self, "Camera error", "Unable to access the selected camera.")
                return

        project_dir = self._ensure_capture_project_dir()
        save_dir = project_dir / "input"
        filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f.jpg")
        save_path = save_dir / filename

        controller = self.turntable if (self.turntable and self.turntable.is_connected) else None
        advance = self.capture_step_degrees if controller else 0.0

        try:
            info = capture_and_analyze(
                save_path,
                camera_index=int(self.camera_index),
                arduino=controller,
                advance_degrees=advance,
                capture=self.camera_capture,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            logging.exception("Manual capture failed")
            QMessageBox.critical(self, "Capture failed", str(exc))
            self._log(f"Capture failed: {exc}")
            return

        metrics = info.get("metrics", {})
        path = info.get("path")
        ok = info.get("ok", False)

        if ok and path:
            blur_val = float(metrics.get("blur", 0.0))
            brightness_val = float(metrics.get("brightness", 0.0))
            self._enqueue_files([str(path)])
            self._log(
                f"Captured photo {path} — blur {blur_val:.2f} brightness {brightness_val:.2f}"
            )
            self.status_label.setText("Photo captured and queued for analysis.")
        else:
            issues = metrics.get("issues", []) if isinstance(metrics, dict) else []
            issue_text = "\n".join(str(issue) for issue in issues) or "Please retake the photo."
            QMessageBox.warning(self, "Photo needs attention", issue_text)
            self._log(f"Photo requires attention: {issue_text}")

    def _capture_sequence(self) -> None:
        if not self.turntable or not self.turntable.is_connected:
            QMessageBox.warning(
                self,
                "Turntable not connected",
                "Connect the Arduino turntable before starting the capture sequence.",
            )
            self._update_turntable_ui()
            return

        if self.camera_index is None and not self._select_camera():
            return

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
            camera_index=int(self.camera_index) if self.camera_index is not None else 0,
            arduino=self.turntable,
        )
        self.capture_step_degrees = float(step_deg)
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

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.camera_capture and self.camera_capture.isOpened():
            try:
                self.camera_capture.release()
            except Exception:
                pass
        self.camera_capture = None
        super().closeEvent(event)
