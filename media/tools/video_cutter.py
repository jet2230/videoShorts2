#!/usr/bin/env python3
"""
Video Cutter GUI - A tool to cut videos into clips using a visual timeline
"""

import sys
import os
import cv2
from moviepy import VideoFileClip
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QSlider,
                              QTreeWidget, QTreeWidgetItem, QFileDialog,
                              QMessageBox, QFrame, QSplitter, QCheckBox, QDialog,
                              QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QPainter, QColor, QPen, QBrush


class TimelineWidget(QWidget):
    """Custom timeline widget with visual selection and draggable edges"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(80)
        self.setStyleSheet("background-color: #2c3e50;")

        self.duration = 0
        self.current_position = 0
        self.selection_start = None
        self.selection_end = None
        self.clips = []
        self.fps = 30

        self.dragging = False
        self.drag_mode = None  # 'left', 'right', 'move', or None
        self.drag_start_x = 0
        self.drag_start_time = 0
        self.edge_threshold = 10  # pixels from edge to detect

    def set_duration(self, duration):
        self.duration = duration
        self.update()

    def set_position(self, position):
        self.current_position = position
        self.update()

    def set_selection(self, start, end):
        self.selection_start = start
        self.selection_end = end
        self.update()

    def set_clips(self, clips):
        self.clips = clips
        self.update()

    def set_fps(self, fps):
        self.fps = fps

    def get_cursor_position(self, event):
        """Determine cursor position relative to selection edges"""
        if self.selection_start is None or self.selection_end is None or self.duration == 0:
            return None

        x = event.position().x()
        width = self.width()
        start_x = (self.selection_start / self.duration) * width
        end_x = (self.selection_end / self.duration) * width

        # Check if near left edge
        if abs(x - start_x) < self.edge_threshold:
            return 'left'
        # Check if near right edge
        elif abs(x - end_x) < self.edge_threshold:
            return 'right'
        # Check if inside selection
        elif start_x < x < end_x:
            return 'inside'
        else:
            return None

    def update_cursor(self, position):
        """Update cursor based on position"""
        if position == 'left' or position == 'right':
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif position == 'inside':
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Draw background
        painter.fillRect(0, 0, width, height, QColor("#34495e"))

        if self.duration == 0:
            painter.setPen(QColor("#7f8c8d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a video to see timeline")
            return

        # Draw selection if exists
        if self.selection_start is not None and self.selection_end is not None:
            start_x = (self.selection_start / self.duration) * width
            end_x = (self.selection_end / self.duration) * width
            selection_width = end_x - start_x

            # Draw selection rectangle (semi-transparent blue)
            selection_color = QColor("#3498db")
            selection_color.setAlpha(120)
            painter.setBrush(QBrush(selection_color))
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawRect(QRectF(start_x, 5, selection_width, height - 10))

            # Draw left edge handle
            painter.setBrush(QBrush(QColor("#2980b9")))
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawRect(QRectF(start_x - 3, 5, 6, height - 10))

            # Draw right edge handle
            painter.setBrush(QBrush(QColor("#2980b9")))
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawRect(QRectF(end_x - 3, 5, 6, height - 10))

            # Draw time labels on selection edges
            painter.setPen(QColor("white"))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            # Format time
            def format_time_s(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 100)
                if hours > 0:
                    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:02d}"
                else:
                    return f"{minutes:02d}:{secs:02d}.{millis:02d}"

            # Start time label
            start_label = format_time_s(self.selection_start)
            start_rect = painter.fontMetrics().boundingRect(start_label)
            painter.drawText(int(start_x) - start_rect.width() // 2, height - 15, start_label)

            # End time label
            end_label = format_time_s(self.selection_end)
            end_rect = painter.fontMetrics().boundingRect(end_label)
            painter.drawText(int(end_x) - end_rect.width() // 2, height - 15, end_label)

            # Duration label in center
            duration = self.selection_end - self.selection_start
            center_x = (start_x + end_x) / 2
            duration_label = f"Duration: {format_time_s(duration)}"
            duration_rect = painter.fontMetrics().boundingRect(duration_label)
            painter.drawText(int(center_x) - duration_rect.width() // 2, 20, duration_label)

        # Draw clips
        colors = [QColor("#e74c3c"), QColor("#2ecc71"), QColor("#f39c12"),
                  QColor("#9b59b6"), QColor("#1abc9c")]
        for i, (start, end) in enumerate(self.clips):
            start_x = (start / self.duration) * width
            end_x = (end / self.duration) * width
            color = colors[i % len(colors)]

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("white"), 1))
            painter.drawRect(QRectF(start_x, 10, end_x - start_x, height - 20))

        # Draw playhead
        if self.duration > 0:
            playhead_x = (self.current_position / self.duration) * width
            painter.setPen(QPen(QColor("yellow"), 2))
            painter.drawLine(QPointF(playhead_x, 0), QPointF(playhead_x, height))

            # Draw triangle at top of playhead
            painter.setBrush(QBrush(QColor("yellow")))
            painter.setPen(QPen(QColor("yellow"), 2))
            triangle = [
                QPointF(playhead_x - 5, 0),
                QPointF(playhead_x + 5, 0),
                QPointF(playhead_x, 8)
            ]
            painter.drawPolygon(*triangle)

    def mouseMoveEvent(self, event):
        # Update cursor based on position
        if not self.dragging:
            if self.duration == 0:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                return
            cursor_pos = self.get_cursor_position(event)
            self.update_cursor(cursor_pos)
            return

        # Handle dragging
        if self.selection_start is None or self.duration == 0:
            return

        x = event.position().x()
        width = self.width()
        drag_time = (x / width) * self.duration
        drag_time = max(0, min(drag_time, self.duration))

        if self.drag_mode == 'left':
            self.selection_start = min(drag_time, self.selection_end - 0.01)
        elif self.drag_mode == 'right':
            self.selection_end = max(drag_time, self.selection_start + 0.01)
        elif self.drag_mode == 'move':
            # Move the entire selection
            delta = drag_time - self.drag_start_time
            new_start = self.drag_start_time_start + delta
            new_end = self.drag_start_time_end + delta

            # Keep selection within bounds
            if new_start < 0:
                new_start = 0
                new_end = self.selection_end - self.selection_start
            elif new_end > self.duration:
                new_end = self.duration
                new_start = self.duration - (self.selection_end - self.selection_start)

            self.selection_start = new_start
            self.selection_end = new_end
        elif self.drag_mode == 'new':
            # Creating new selection - adjust end time
            self.selection_end = max(0, min(drag_time, self.duration))

        self.selection_changed.emit(self.selection_start, self.selection_end)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Don't allow interaction if no video is loaded
            if self.duration == 0:
                return

            cursor_pos = self.get_cursor_position(event)

            if cursor_pos == 'left':
                self.dragging = True
                self.drag_mode = 'left'
            elif cursor_pos == 'right':
                self.dragging = True
                self.drag_mode = 'right'
            elif cursor_pos == 'inside':
                self.dragging = True
                self.drag_mode = 'move'
                self.drag_start_time = (event.position().x() / self.width()) * self.duration
                self.drag_start_time_start = self.selection_start
                self.drag_start_time_end = self.selection_end
            else:
                # Create new selection - 5 seconds starting from click position
                self.dragging = True
                self.drag_mode = 'new'
                click_time = (event.position().x() / self.width()) * self.duration
                self.selection_start = max(0, min(click_time, self.duration))
                # Create 5-second clip, but don't exceed video duration
                self.selection_end = min(self.selection_start + 5.0, self.duration)

            if self.dragging:
                self.selection_changed.emit(self.selection_start, self.selection_end)
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            drag_was_new = self.drag_mode == 'new'
            self.drag_mode = None
            self.update_cursor(None)

            # Ensure start < end and minimum 5 seconds for new selections
            if self.selection_start and self.selection_end:
                if self.selection_start > self.selection_end:
                    self.selection_start, self.selection_end = self.selection_end, self.selection_start

                # For new clicks (not drags), ensure 5-second minimum
                if drag_was_new and (self.selection_end - self.selection_start) < 5.0:
                    self.selection_end = min(self.selection_start + 5.0, self.duration)

                self.selection_changed.emit(self.selection_start, self.selection_end)

    # Signal for selection changes
    selection_changed = pyqtSignal(float, float)


class ExportOptionsDialog(QDialog):
    """Dialog for export options"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Remove audio checkbox
        self.remove_audio_checkbox = QCheckBox("Remove audio from exported clips")
        self.remove_audio_checkbox.setToolTip("Export videos without audio track")
        layout.addWidget(self.remove_audio_checkbox)

        # Info label
        info_label = QLabel("Clips will be exported as MP4 files with H.264 codec")
        info_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(info_label)

        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_remove_audio(self):
        return self.remove_audio_checkbox.isChecked()


class ExportThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, video_path, clips, output_dir, remove_audio=False):
        super().__init__()
        self.video_path = video_path
        self.clips = clips
        self.output_dir = output_dir
        self.remove_audio = remove_audio

    def run(self):
        try:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]

            for i, (start, end) in enumerate(self.clips):
                output_path = os.path.join(self.output_dir, f"{base_name}_clip_{i+1}.mp4")
                self.progress.emit(f"Exporting clip {i+1}/{len(self.clips)}...")

                # Load video with audio disabled initially to avoid stream issues
                video_clip = VideoFileClip(self.video_path, audio=False)
                subclip = video_clip.subclipped(start, end)

                # Only add audio if not removing it
                if not self.remove_audio:
                    # Try to load audio separately
                    try:
                        audio_clip = VideoFileClip(self.video_path).audio
                        if audio_clip:
                            audio_subclip = audio_clip.subclipped(start, end)
                            final_clip = subclip.with_audio(audio_subclip)
                        else:
                            final_clip = subclip
                    except:
                        final_clip = subclip
                else:
                    final_clip = subclip

                if self.remove_audio:
                    # Export without audio
                    final_clip.write_videofile(
                        output_path,
                        codec="libx264",
                        preset="medium",
                        ffmpeg_params=["-crf", "23", "-avoid_negative_ts", "make_zero", "-an"]
                    )
                else:
                    # Export with audio
                    final_clip.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        preset="medium",
                        ffmpeg_params=["-crf", "23", "-avoid_negative_ts", "make_zero"]
                    )

                # Close clips to free memory
                final_clip.close()
                video_clip.close()

            self.finished.emit(True, f"Exported {len(self.clips)} clip(s) to:\n{self.output_dir}")
        except Exception as e:
            import traceback
            error_details = f"Failed to export clips:\n{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_details)


class VideoCutter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Cutter")
        self.setGeometry(100, 100, 1200, 700)
        self.setAcceptDrops(True)

        # Video properties
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.duration = 0
        self.current_frame = 0

        # Clips storage
        self.clips = []

        # Selection state
        self.selection_start = None
        self.selection_end = None

        # Frame update throttling
        self.pending_frame_update = None
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_pending_frame_update)

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Top toolbar
        toolbar = QHBoxLayout()

        self.btn_open = QPushButton("Open Video")
        self.btn_open.clicked.connect(self.open_video)
        toolbar.addWidget(self.btn_open)

        toolbar.addWidget(QLabel("Start:"))
        self.start_label = QLabel("00:00:00.000")
        self.start_label.setStyleSheet("color: green; font-weight: bold;")
        toolbar.addWidget(self.start_label)

        toolbar.addWidget(QLabel("  End:"))
        self.end_label = QLabel("00:00:00.000")
        self.end_label.setStyleSheet("color: red; font-weight: bold;")
        toolbar.addWidget(self.end_label)

        self.btn_add_clip = QPushButton("Add Clip")
        self.btn_add_clip.clicked.connect(self.add_clip)
        toolbar.addWidget(self.btn_add_clip)

        self.btn_export = QPushButton("Export Clips")
        self.btn_export.clicked.connect(self.export_clips)
        toolbar.addWidget(self.btn_export)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Video preview frame - dual preview
        preview_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.start_preview = self.create_preview_panel("Start Position")
        self.end_preview = self.create_preview_panel("End Position")

        preview_splitter.addWidget(self.start_preview['container'])
        preview_splitter.addWidget(self.end_preview['container'])
        layout.addWidget(preview_splitter)

        # Timeline frame
        timeline_frame = QFrame()
        timeline_frame.setFrameShape(QFrame.Shape.StyledPanel)
        timeline_layout = QVBoxLayout(timeline_frame)

        # Time display
        self.time_label = QLabel("Time: 00:00:00 / 00:00:00")
        timeline_layout.addWidget(self.time_label)

        # Visual timeline widget
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.selection_changed.connect(self.on_selection_changed)
        timeline_layout.addWidget(self.timeline_widget)

        # Timeline slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.on_slider_change)
        timeline_layout.addWidget(self.slider)

        # Navigation controls
        nav_layout = QHBoxLayout()

        self.btn_skip_back_5 = QPushButton("◀◀")
        self.btn_skip_back_5.clicked.connect(lambda: self.skip_frames(-5))
        nav_layout.addWidget(self.btn_skip_back_5)

        self.btn_skip_back_1 = QPushButton("◀")
        self.btn_skip_back_1.clicked.connect(lambda: self.skip_frames(-1))
        nav_layout.addWidget(self.btn_skip_back_1)

        # Set trim start button
        self.btn_set_start = QPushButton("Set Start")
        self.btn_set_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        self.btn_set_start.clicked.connect(self.set_trim_start)
        nav_layout.addWidget(self.btn_set_start)

        # Set trim end button
        self.btn_set_end = QPushButton("Set End")
        self.btn_set_end.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 5px;")
        self.btn_set_end.clicked.connect(self.set_trim_end)
        nav_layout.addWidget(self.btn_set_end)

        self.btn_skip_fwd_1 = QPushButton("▶")
        self.btn_skip_fwd_1.clicked.connect(lambda: self.skip_frames(1))
        nav_layout.addWidget(self.btn_skip_fwd_1)

        self.btn_skip_fwd_5 = QPushButton("▶▶")
        self.btn_skip_fwd_5.clicked.connect(lambda: self.skip_frames(5))
        nav_layout.addWidget(self.btn_skip_fwd_5)

        nav_layout.addStretch()
        timeline_layout.addLayout(nav_layout)

        layout.addWidget(timeline_frame)

        # Clips list
        self.clips_tree = QTreeWidget()
        self.clips_tree.setHeaderLabels(["Start Time", "End Time", "Duration"])
        self.clips_tree.setColumnWidth(0, 150)
        self.clips_tree.setColumnWidth(1, 150)
        self.clips_tree.setColumnWidth(2, 150)
        layout.addWidget(self.clips_tree)

        # Remove clip button
        btn_remove = QPushButton("Remove Selected Clip")
        btn_remove.clicked.connect(self.remove_selected_clip)
        layout.addWidget(btn_remove)

    def create_preview_panel(self, title):
        """Create a preview panel with label"""
        container = QFrame()
        container.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(container)

        label_title = QLabel(title)
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(label_title)

        label_image = QLabel("No selection")
        label_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_image.setMinimumSize(300, 200)
        label_image.setStyleSheet("background-color: #2c3e50; color: white;")
        layout.addWidget(label_image)

        return {'container': container, 'label': label_image}

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg')
            if file_path.lower().endswith(video_extensions):
                self.video_path = file_path
                self.load_video()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        if path:
            self.video_path = path
            self.load_video()

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.current_frame = 0

        # Use higher resolution for smoother seeking
        slider_max = max(1000, int(self.duration * 100))  # At least 1000 steps
        self.slider.setRange(0, slider_max)
        self.clips = []
        self.clips_tree.clear()

        # Update timeline widget
        self.timeline_widget.set_duration(self.duration)
        self.timeline_widget.set_fps(self.fps)
        self.timeline_widget.set_clips([])
        self.timeline_widget.set_selection(None, None)

        self.start_preview['label'].setPixmap(QPixmap())
        self.start_preview['label'].setText("No selection")
        self.end_preview['label'].setPixmap(QPixmap())
        self.end_preview['label'].setText("No selection")

        self.display_frame_now(0)
        self.update_time_label_fast(0)

    def display_frame(self, frame_num, label_widget=None):
        """Display frame (uses throttling for slider, immediate for button clicks)"""
        if label_widget:
            # Preview frames - always immediate
            self.display_frame_now(frame_num, label_widget)
        else:
            # Main preview - use throttled version for slider
            self.pending_frame_update = (frame_num, None)
            self.update_timer.start(16)

    def display_frame_now(self, frame_num, label_widget=None):
        """Display frame immediately without throttling"""
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            max_h, max_w = 300, 450

            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            frame = cv2.resize(frame, (new_w, new_h))

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            if label_widget:
                label_widget.setPixmap(pixmap)
                label_widget.setText("")
            else:
                self.start_preview['label'].setPixmap(pixmap)
                self.start_preview['label'].setText("")

            self.current_frame = frame_num

    def on_slider_change(self, value):
        if not self.cap:
            return
        slider_max = self.slider.maximum()
        time_seconds = (value / slider_max) * self.duration

        # Update time label and timeline immediately (these are fast)
        self.timeline_widget.set_position(time_seconds)
        self.update_time_label_fast(time_seconds)

        # Throttle frame decoding (this is slow)
        frame_num = int(time_seconds * self.fps)
        self.pending_frame_update = (frame_num, None)
        self.update_timer.start(16)  # Update at ~60fps max (16ms delay)

    def update_time_label_fast(self, current_time):
        """Fast time label update without frame access"""
        if self.fps > 0:
            self.time_label.setText(f"Time: {self.format_time(current_time)} / {self.format_time(self.duration)}")

    def process_pending_frame_update(self):
        """Process the pending frame update"""
        if self.pending_frame_update is not None:
            frame_num, label_widget = self.pending_frame_update
            self.pending_frame_update = None
            self.display_frame_now(frame_num, label_widget)

    def on_selection_changed(self, start, end):
        """Handle selection change from timeline widget"""
        self.selection_start = start
        self.selection_end = end
        self.start_label.setText(self.format_time(start))
        self.end_label.setText(self.format_time(end))

        # Update preview panels immediately
        start_frame = int(start * self.fps)
        end_frame = int(end * self.fps)
        self.display_frame_now(start_frame, self.start_preview['label'])
        self.display_frame_now(end_frame, self.end_preview['label'])

    def skip_frames(self, frames):
        """Skip by a specific number of frames"""
        if not self.cap:
            return
        new_frame = self.current_frame + frames
        new_frame = max(0, min(new_frame, self.total_frames - 1))
        time_seconds = new_frame / self.fps

        # For button clicks, update immediately
        self.display_frame_now(new_frame)
        slider_value = int((time_seconds / self.duration) * self.slider.maximum())
        self.slider.setValue(slider_value)
        self.timeline_widget.set_position(time_seconds)
        self.update_time_label_fast(time_seconds)

    def set_trim_start(self):
        """Set the trim start point at current frame"""
        if not self.cap:
            return
        current_time = self.current_frame / self.fps
        if self.selection_end is None:
            self.selection_end = current_time
        self.selection_start = current_time

        # Ensure start < end
        if self.selection_start > self.selection_end:
            self.selection_start, self.selection_end = self.selection_end, self.selection_start

        # Update UI
        self.start_label.setText(self.format_time(self.selection_start))
        self.timeline_widget.set_selection(self.selection_start, self.selection_end)
        self.display_frame_now(self.current_frame, self.start_preview['label'])

    def set_trim_end(self):
        """Set the trim end point at current frame"""
        if not self.cap:
            return
        current_time = self.current_frame / self.fps
        if self.selection_start is None:
            self.selection_start = current_time
        self.selection_end = current_time

        # Ensure start < end
        if self.selection_start > self.selection_end:
            self.selection_start, self.selection_end = self.selection_end, self.selection_start

        # Update UI
        self.end_label.setText(self.format_time(self.selection_end))
        self.timeline_widget.set_selection(self.selection_start, self.selection_end)
        self.display_frame_now(self.current_frame, self.end_preview['label'])

    def skip(self, seconds):
        """Skip by seconds (kept for compatibility)"""
        if not self.cap:
            return
        self.skip_frames(int(seconds * self.fps))

    def update_time_label(self):
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        self.time_label.setText(f"Time: {self.format_time(current_time)} / {self.format_time(self.duration)}")

    def add_clip(self):
        if self.selection_start is None or self.selection_end is None:
            QMessageBox.warning(self, "Warning", "Please select a portion of the video first by clicking on the timeline")
            return

        if abs(self.selection_end - self.selection_start) < 1.0:
            QMessageBox.warning(self, "Warning", "Clip is too short (minimum 1 second)\nAdjust the selection using the draggable edges on the timeline")
            return

        self.clips.append((self.selection_start, self.selection_end))

        duration = self.selection_end - self.selection_start
        item = QTreeWidgetItem([
            self.format_time(self.selection_start),
            self.format_time(self.selection_end),
            self.format_time(duration)
        ])
        self.clips_tree.addTopLevelItem(item)

        # Update timeline widget to show clips
        self.timeline_widget.set_clips(self.clips)

        # Reset selection
        self.selection_start = None
        self.selection_end = None
        self.start_label.setText("00:00:00.000")
        self.end_label.setText("00:00:00.000")
        self.timeline_widget.set_selection(None, None)
        self.start_preview['label'].setPixmap(QPixmap())
        self.start_preview['label'].setText("No selection")
        self.end_preview['label'].setPixmap(QPixmap())
        self.end_preview['label'].setText("No selection")

    def remove_selected_clip(self):
        selected = self.clips_tree.selectedItems()
        if not selected:
            return
        index = self.clips_tree.indexOfTopLevelItem(selected[0])
        self.clips_tree.takeTopLevelItem(index)
        del self.clips[index]
        self.timeline_widget.set_clips(self.clips)

    def export_clips(self):
        if not self.clips:
            QMessageBox.warning(self, "Warning", "No clips to export")
            return

        # Show export options dialog
        dialog = ExportOptionsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        remove_audio = dialog.get_remove_audio()
        self.export_thread = ExportThread(self.video_path, self.clips, output_dir, remove_audio)
        self.export_thread.finished.connect(self.on_export_finished)
        self.export_thread.start()

    def on_export_finished(self, success, message):
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

    @staticmethod
    def format_time(seconds):
        """Format time as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    app = QApplication(sys.argv)
    window = VideoCutter()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
