#!/usr/bin/env python3
"""
YouTube Shorts Creator - PyQt6 GUI
Main application window for creating YouTube Shorts from long-form videos.
"""

import os
import sys
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QTextEdit, QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
    QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QTabWidget,
    QSplitter, QFrame, QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QAction

# Import the core functionality
from shorts_creator import YouTubeShortsCreator, load_settings
from ai_theme_generator import AIThemeGenerator


# =============================================================================
# Worker Thread for Background Processing
# =============================================================================

class ProcessWorker(QThread):
    """Worker thread for video processing to keep UI responsive."""

    # Signals for communicating with the main thread
    progress = pyqtSignal(int, str)  # progress_percent, status_message
    log = pyqtSignal(str)  # log message
    finished = pyqtSignal(bool, str)  # success, message
    video_processed = pyqtSignal(dict)  # video_info dict

    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode
        self.kwargs = kwargs
        self.creator = None

    def run(self):
        """Run the processing task in background thread."""
        try:
            if self.mode == 'download':
                self.process_download()
            elif self.mode == 'local':
                self.process_local()
            elif self.mode == 'subtitles':
                self.generate_subtitles()
            elif self.mode == 'themes':
                self.generate_themes()
            elif self.mode == 'create_shorts':
                self.create_shorts()
            else:
                self.finished.emit(False, f"Unknown mode: {self.mode}")
        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            self.finished.emit(False, str(e))

    def log_output(self, *args, **kwargs):
        """Emit log signal. Accepts print-like arguments."""
        # Join all positional arguments (like print does)
        message = ' '.join(str(arg) for arg in args)
        if message:  # Only emit if there's actual content
            self.log.emit(message)

    def process_download(self):
        """Process a YouTube URL download."""
        url = self.kwargs['url']
        model = self.kwargs.get('model', 'small')

        self.progress.emit(5, "Initializing...")
        self.creator = YouTubeShortsCreator(base_dir=self.kwargs.get('output_dir', 'videos'))

        self.progress.emit(10, "Downloading video...")
        self.log.emit(f"Downloading from: {url}")

        # Monkey patch print to capture output
        import builtins
        original_print = builtins.print
        builtins.print = self.log_output

        try:
            video_info = self.creator.download_video(url)
            self.progress.emit(30, "Creating info file...")
            self.creator.create_video_info(video_info)

            self.progress.emit(40, "Generating subtitles...")
            self.creator.generate_subtitles(video_info, model_size=model)

            self.progress.emit(70, "Generating themes...")
            # Initialize AI generator
            ai_generator = None
            try:
                ai_generator = AIThemeGenerator()
                if ai_generator.is_available():
                    self.log.emit(f"Using AI: {ai_generator.model}")
                else:
                    ai_generator = None
            except Exception:
                pass

            self.creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model)

            self.progress.emit(100, "Complete!")
            self.video_processed.emit(video_info)
            self.finished.emit(True, f"Processing complete!\nFolder: {video_info['folder']}")

        finally:
            builtins.print = original_print

    def process_local(self):
        """Process a local video file."""
        file_path = self.kwargs['file_path']
        model = self.kwargs.get('model', 'small')

        self.progress.emit(5, "Initializing...")
        self.creator = YouTubeShortsCreator(base_dir=self.kwargs.get('output_dir', 'videos'))

        self.progress.emit(10, "Processing local video...")
        self.log.emit(f"Processing: {file_path}")

        import builtins
        original_print = builtins.print
        builtins.print = self.log_output

        try:
            video_info = self.creator.process_local_video(file_path)
            self.progress.emit(30, "Creating info file...")
            self.creator.create_video_info(video_info)

            self.progress.emit(40, "Generating subtitles...")
            self.creator.generate_subtitles(video_info, model_size=model)

            self.progress.emit(70, "Generating themes...")
            ai_generator = None
            try:
                ai_generator = AIThemeGenerator()
                if ai_generator.is_available():
                    self.log.emit(f"Using AI: {ai_generator.model}")
                else:
                    ai_generator = None
            except Exception:
                pass

            self.creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model)

            self.progress.emit(100, "Complete!")
            self.video_processed.emit(video_info)
            self.finished.emit(True, f"Processing complete!\nFolder: {video_info['folder']}")

        finally:
            builtins.print = original_print

    def generate_subtitles(self):
        """Generate subtitles for existing video."""
        folder = self.kwargs['folder']
        model = self.kwargs.get('model', 'small')

        self.creator = YouTubeShortsCreator()

        video_info = self.kwargs.get('video_info', {})

        self.progress.emit(0, "Starting subtitle generation...")
        import builtins
        original_print = builtins.print
        builtins.print = self.log_output

        # Start a timer to increment progress gradually
        self.progress_timer = QTimer()
        self.progress_value = 0
        self.progress_max = 90

        def update_progress():
            self.progress_value += 2
            if self.progress_value > self.progress_max:
                self.progress_value = self.progress_max
            self.progress.emit(self.progress_value, "Generating subtitles...")

        self.progress_timer.timeout.connect(update_progress)
        self.progress_timer.start(500)  # Update every 500ms

        try:
            self.creator.generate_subtitles(video_info, model_size=model)
            self.progress_timer.stop()
            self.progress.emit(100, "Complete!")
            self.finished.emit(True, "Subtitles generated successfully!")
        except Exception as e:
            self.progress_timer.stop()
            raise
        finally:
            builtins.print = original_print

    def generate_themes(self):
        """Generate themes for existing video."""
        folder = self.kwargs['folder']
        model = self.kwargs.get('model', 'small')
        video_info = self.kwargs.get('video_info', {})

        self.creator = YouTubeShortsCreator()

        self.progress.emit(0, "Starting theme generation...")

        import builtins
        original_print = builtins.print
        builtins.print = self.log_output

        # Start a timer to increment progress gradually
        self.progress_timer = QTimer()
        self.progress_value = 0
        self.progress_max = 90

        def update_progress():
            self.progress_value += 2
            if self.progress_value > self.progress_max:
                self.progress_value = self.progress_max
            self.progress.emit(self.progress_value, "Analyzing and generating themes...")

        self.progress_timer.timeout.connect(update_progress)
        self.progress_timer.start(500)  # Update every 500ms

        try:
            ai_generator = None
            try:
                ai_generator = AIThemeGenerator()
                if ai_generator.is_available():
                    self.log.emit(f"Using AI: {ai_generator.model}")
                else:
                    ai_generator = None
            except Exception:
                pass

            self.creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model)
            self.progress_timer.stop()
            self.progress.emit(100, "Complete!")
            self.finished.emit(True, "Themes generated successfully!")
        except Exception as e:
            self.progress_timer.stop()
            raise
        finally:
            builtins.print = original_print

    def create_shorts(self):
        """Create short video clips."""
        folder_number = self.kwargs['folder_number']
        theme_numbers = self.kwargs.get('theme_numbers', 'all')

        self.creator = YouTubeShortsCreator()

        self.progress.emit(5, "Creating shorts...")
        import builtins
        original_print = builtins.print
        builtins.print = self.log_output

        try:
            # Find the video folder
            folder = self.creator.get_video_folder_by_number(folder_number)
            if not folder:
                self.finished.emit(False, f"Video folder '{folder_number}_' not found")
                return

            # Parse themes file
            themes_file = folder / 'themes.md'
            if not themes_file.exists():
                self.finished.emit(False, "themes.md not found. Process video first.")
                return

            themes = self.creator.parse_themes_file(themes_file)

            # Determine which themes to create
            if theme_numbers == 'all':
                selected_themes = themes
            else:
                requested_nums = [int(n.strip()) for n in theme_numbers.split(',')]
                selected_themes = [t for t in themes if t['number'] in requested_nums]

            # Get video and SRT paths
            video_path = self.creator.get_video_file(folder)
            srt_path = None
            for ext in ['*.srt']:
                srt_files = list(folder.glob(ext))
                if srt_files:
                    srt_path = srt_files[0]
                    break

            if not video_path or not srt_path:
                self.finished.emit(False, "Video or subtitle file not found")
                return

            # Create output directory
            shorts_dir = folder / 'shorts'
            shorts_dir.mkdir(exist_ok=True)

            # Create each short
            total = len(selected_themes)
            for i, theme in enumerate(selected_themes):
                progress_pct = int(5 + (i / total) * 95)
                self.progress.emit(progress_pct, f"Creating theme {theme['number']}...")
                self.log.emit(f"Theme {theme['number']}: {theme['title']}")

                result = self.creator.create_short(video_path, theme, shorts_dir, srt_path)

            self.progress.emit(100, "Complete!")
            self.finished.emit(True, f"Created {len(selected_themes)} shorts in:\n{shorts_dir}")

        finally:
            builtins.print = original_print


# =============================================================================
# Main GUI Window
# =============================================================================

class YouTubeShortsGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_video_info = None
        self.current_folder = None

        self.settings = load_settings()
        self.creator = YouTubeShortsCreator()

        self.init_ui()
        self.load_video_folders()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("YouTube Shorts Creator")
        self.setMinimumSize(1000, 700)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("YouTube Shorts Creator")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 2px solid #2196F3; border-top: 2px solid #2196F3; }
            QTabBar::tab {
                background: #333;
                color: white;
                padding: 10px 20px;
                border: 1px solid #444;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                border: 2px solid #2196F3;
                border-bottom: 2px solid #2196F3;
            }
            QTabBar::tab:hover:!selected {
                background: #555;
            }
        """)
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Process Video
        self.create_process_tab()

        # Tab 2: Create Shorts
        self.create_shorts_tab()

        # Tab 3: Settings
        self.create_settings_tab()

        # Status bar at bottom
        self.create_status_area(main_layout)

        # Apply dark theme styling
        self.apply_dark_theme()

    def create_process_tab(self):
        """Create the Process Video tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Input section
        input_group = QGroupBox("Video Input")
        input_layout = QGridLayout()

        # URL input
        input_layout.addWidget(QLabel("YouTube URL:"), 0, 0)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://youtube.com/watch?v=...")
        input_layout.addWidget(self.url_input, 0, 1)

        # Browse button
        self.browse_btn = QPushButton("Browse Local File")
        self.browse_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.browse_btn.clicked.connect(self.browse_file)
        input_layout.addWidget(self.browse_btn, 1, 0)

        # File path display
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("No file selected")
        self.file_path_input.setReadOnly(True)
        input_layout.addWidget(self.file_path_input, 1, 1)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 12px 24px; border-radius: 4px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.process_btn.clicked.connect(self.process_video)
        layout.addWidget(self.process_btn)

        layout.addStretch()

        self.tab_widget.addTab(tab, "Process Video")

    def create_shorts_tab(self):
        """Create the Create Shorts tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Video folder selection
        folder_group = QGroupBox("Select Video Folder")
        folder_layout = QHBoxLayout()

        folder_layout.addWidget(QLabel("Video Folder:"))
        self.folder_combo = QComboBox()
        self.folder_combo.setStyleSheet("QComboBox { background-color: #2196F3; color: white; padding: 6px; border-radius: 4px; } QComboBox::drop-down { border: none; } QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid white; margin-right: 10px; } QComboBox:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.folder_combo.setMinimumWidth(400)
        self.folder_combo.currentTextChanged.connect(self.on_folder_changed)
        folder_layout.addWidget(self.folder_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.refresh_btn.clicked.connect(self.load_video_folders)
        folder_layout.addWidget(self.refresh_btn)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Theme selection
        themes_group = QGroupBox("Themes")
        themes_layout = QVBoxLayout()

        # Header with select all/none and regenerate options
        header_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_themes)
        self.select_all_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        header_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_none_themes)
        self.select_none_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        header_layout.addWidget(self.select_none_btn)

        header_layout.addStretch()

        # Regenerate buttons
        self.regenerate_subtitles_btn = QPushButton("Regenerate Subtitles")
        self.regenerate_subtitles_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #F57C00; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.regenerate_subtitles_btn.clicked.connect(self.regenerate_subtitles)
        header_layout.addWidget(self.regenerate_subtitles_btn)

        self.regenerate_themes_btn = QPushButton("Regenerate Themes")
        self.regenerate_themes_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #F57C00; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.regenerate_themes_btn.clicked.connect(self.regenerate_themes)
        header_layout.addWidget(self.regenerate_themes_btn)

        themes_layout.addLayout(header_layout)

        # Theme list
        self.themes_list = QListWidget()
        self.themes_list.setMinimumHeight(300)
        themes_layout.addWidget(self.themes_list)

        themes_group.setLayout(themes_layout)
        layout.addWidget(themes_group)

        # Create shorts button
        self.create_shorts_btn = QPushButton("Create Selected Shorts")
        self.create_shorts_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 12px 24px; border-radius: 4px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.create_shorts_btn.setMinimumHeight(50)
        self.create_shorts_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.create_shorts_btn.clicked.connect(self.create_shorts)
        layout.addWidget(self.create_shorts_btn)

        layout.addStretch()

        self.tab_widget.addTab(tab, "Create Shorts")

    def create_settings_tab(self):
        """Create the Settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Whisper settings
        whisper_group = QGroupBox("Whisper Settings")
        whisper_layout = QFormLayout()

        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.setStyleSheet("QComboBox { background-color: #2196F3; color: white; padding: 6px; border-radius: 4px; } QComboBox::drop-down { border: none; } QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid white; margin-right: 10px; } QComboBox:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.whisper_model_combo.setCurrentText(self.settings.get('whisper', 'model'))
        whisper_layout.addRow("Model:", self.whisper_model_combo)

        self.whisper_language_input = QLineEdit()
        self.whisper_language_input.setText(self.settings.get('whisper', 'language'))
        whisper_layout.addRow("Language:", self.whisper_language_input)

        whisper_group.setLayout(whisper_layout)
        scroll_layout.addWidget(whisper_group)

        # Video settings
        video_group = QGroupBox("Video Settings")
        video_layout = QFormLayout()

        self.resolution_width_spin = QSpinBox()
        self.resolution_width_spin.setRange(480, 3840)
        self.resolution_width_spin.setValue(int(self.settings.get('video', 'resolution_width')))
        video_layout.addRow("Width:", self.resolution_width_spin)

        self.resolution_height_spin = QSpinBox()
        self.resolution_height_spin.setRange(480, 3840)
        self.resolution_height_spin.setValue(int(self.settings.get('video', 'resolution_height')))
        video_layout.addRow("Height:", self.resolution_height_spin)

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(int(self.settings.get('video', 'crf')))
        video_layout.addRow("CRF Quality:", self.crf_spin)

        video_group.setLayout(video_layout)
        scroll_layout.addWidget(video_group)

        # Subtitle settings
        subtitle_group = QGroupBox("Subtitle Settings")
        subtitle_layout = QFormLayout()

        self.font_name_input = QLineEdit()
        self.font_name_input.setText(self.settings.get('subtitle', 'font_name'))
        subtitle_layout.addRow("Font:", self.font_name_input)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 72)
        self.font_size_spin.setValue(int(self.settings.get('subtitle', 'font_size')))
        subtitle_layout.addRow("Font Size:", self.font_size_spin)

        self.margin_v_spin = QSpinBox()
        self.margin_v_spin.setRange(0, 100)
        self.margin_v_spin.setValue(int(self.settings.get('subtitle', 'margin_v')))
        subtitle_layout.addRow("Bottom Margin:", self.margin_v_spin)

        subtitle_group.setLayout(subtitle_layout)
        scroll_layout.addWidget(subtitle_group)

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Save button
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #1976D2; } QPushButton:disabled { background-color: #444444; color: #666666; border: 1px solid #555555; }")
        self.save_settings_btn.clicked.connect(self.save_settings)
        save_layout.addWidget(self.save_settings_btn)
        layout.addLayout(save_layout)

        self.tab_widget.addTab(tab, "Settings")

    def create_status_area(self, parent_layout):
        """Create the status area at the bottom."""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setMinimumHeight(150)
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))
        status_layout.addWidget(self.log_output)

        status_group.setLayout(status_layout)
        parent_layout.addWidget(status_group)

    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        app = QApplication.instance()
        app.setStyle("Fusion")

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        app.setPalette(palette)

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def browse_file(self):
        """Open file browser to select video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.mkv *.webm *.avi *.mov);;All Files (*)"
        )
        if file_path:
            self.file_path_input.setText(file_path)
            self.log_output.append(f"Selected file: {file_path}")

    def set_processing_state(self, is_processing):
        """Enable or disable UI elements during processing."""
        # Grey out visual effect for all widgets except status area
        if is_processing:
            # Apply greyed-out stylesheet to main window
            self.setStyleSheet("""
                QMainWindow { opacity: 0.95; }
                QWidget:disabled {
                    color: #666666;
                    background-color: #333333;
                }
                QPushButton:disabled {
                    background-color: #444444;
                    color: #666666;
                    border: 1px solid #555555;
                }
                QLineEdit:disabled, QTextEdit:disabled {
                    background-color: #2a2a2a;
                    color: #666666;
                }
                QComboBox:disabled {
                    background-color: #444444;
                    color: #666666;
                }
                QCheckBox:disabled {
                    color: #666666;
                }
            """)
        else:
            # Remove greyed-out stylesheet
            self.setStyleSheet("")

        # Disable/enable tab switching
        self.tab_widget.setEnabled(not is_processing)

        # Process Video tab elements
        self.url_input.setEnabled(not is_processing)
        self.browse_btn.setEnabled(not is_processing)
        self.process_btn.setEnabled(not is_processing)

        # Create Shorts tab elements
        self.folder_combo.setEnabled(not is_processing)
        self.refresh_btn.setEnabled(not is_processing)
        self.select_all_btn.setEnabled(not is_processing)
        self.select_none_btn.setEnabled(not is_processing)
        self.regenerate_subtitles_btn.setEnabled(not is_processing)
        self.regenerate_themes_btn.setEnabled(not is_processing)
        self.create_shorts_btn.setEnabled(not is_processing)

        # Disable theme list items during processing
        for i in range(self.themes_list.count()):
            item = self.themes_list.item(i)
            checkbox = self.themes_list.itemWidget(item)
            checkbox.setEnabled(not is_processing)

        # Settings tab
        self.save_settings_btn.setEnabled(not is_processing)

    def update_theme_selection_state(self):
        """Update button states based on how many themes are selected."""
        has_selection = False
        for i in range(self.themes_list.count()):
            item = self.themes_list.item(i)
            checkbox = self.themes_list.itemWidget(item)
            if checkbox.isChecked():
                has_selection = True
                break

        # Enable/disable Create Selected Shorts button based on selection
        self.create_shorts_btn.setEnabled(has_selection)

    def process_video(self):
        """Process the video (download or local)."""
        # Check input source
        url = self.url_input.text().strip()
        file_path = self.file_path_input.text().strip()

        if not url and not file_path:
            QMessageBox.warning(self, "No Input", "Please enter a YouTube URL or browse for a local file.")
            return

        if url and file_path:
            QMessageBox.warning(self, "Multiple Inputs", "Please use either URL or local file, not both.")
            return

        # Clear log
        self.log_output.clear()

        # Disable all UI elements
        self.set_processing_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        model = self.settings.get('whisper', 'model')

        # Start worker thread
        if url:
            self.worker = ProcessWorker('download', url=url, model=model,
                                        output_dir=self.settings.get('video', 'output_dir'))
            self.log_output.append(f"Starting download from: {url}")
        else:
            self.worker = ProcessWorker('local', file_path=file_path, model=model,
                                        output_dir=self.settings.get('video', 'output_dir'))
            self.log_output.append(f"Processing local file: {file_path}")

        # Connect signals
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.video_processed.connect(self.on_video_processed)

        # Start thread
        self.worker.start()

    def on_progress(self, percent: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setValue(percent)
        self.log_output.append(message)

    def on_log(self, message: str):
        """Append log message."""
        self.log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
        self.set_processing_state(False)
        self.progress_bar.setVisible(False)

        if success:
            QMessageBox.information(self, "Success", message)
            self.load_video_folders()
        else:
            QMessageBox.critical(self, "Error", message)

    def on_video_processed(self, video_info: dict):
        """Store video info for later use."""
        self.current_video_info = video_info
        self.current_folder = video_info['folder']

    def load_video_folders(self):
        """Load available video folders into combo box."""
        self.folder_combo.clear()

        base_dir = Path(self.settings.get('video', 'output_dir'))
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)

        folders = []
        for item in base_dir.iterdir():
            if item.is_dir() and re.match(r'^\d{3}_', item.name):
                # Check if it has video files
                has_video = (any(item.glob("*.mp4")) or
                            any(item.glob("*.mkv")) or
                            any(item.glob("*.webm")))
                if has_video:
                    folders.append(item)

        # Sort by folder number
        folders.sort(key=lambda x: int(x.name.split('_')[0]))

        for folder in folders:
            self.folder_combo.addItem(folder.name, str(folder))

        if folders:
            self.on_folder_changed(self.folder_combo.currentText())

    def on_folder_changed(self, folder_name: str):
        """Handle folder selection change."""
        if not folder_name:
            return

        # Clear themes list
        self.themes_list.clear()

        # Get folder path
        base_dir = Path(self.settings.get('video', 'output_dir'))
        folder = base_dir / folder_name

        # Check which themes already have shorts created
        shorts_dir = folder / 'shorts'
        existing_theme_numbers = set()

        if shorts_dir.exists():
            # Parse existing short filenames to extract theme numbers
            # Format: theme_XXX_title.mp4
            for short_file in shorts_dir.glob('theme_*.mp4'):
                # Extract theme number from filename
                match = re.match(r'theme_(\d+)_', short_file.name)
                if match:
                    existing_theme_numbers.add(int(match.group(1)))

        # Look for themes.md file
        themes_file = folder / 'themes.md'
        if not themes_file.exists():
            self.log_output.append(f"No themes.md found in {folder_name}")

            # Offer to generate themes
            reply = QMessageBox.question(
                self,
                "No Themes Found",
                f"No themes.md file found in {folder_name}.\n\nWould you like to generate themes now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.generate_themes_for_folder(folder)
            return

        # Parse themes file
        try:
            themes = self.creator.parse_themes_file(themes_file)

            for theme in themes:
                item = QListWidgetItem()

                # Check if this theme has already been processed
                if theme['number'] in existing_theme_numbers:
                    label_text = f"Theme {theme['number']}: {theme['title']} (already processed)"
                    checkbox = QCheckBox(label_text)
                    checkbox.setChecked(False)
                    # Use stylesheet for green bold text
                    checkbox.setStyleSheet("QCheckBox { color: #00CC00; font-weight: bold; }")
                else:
                    label_text = f"Theme {theme['number']}: {theme['title']}"
                    checkbox = QCheckBox(label_text)
                    checkbox.setChecked(True)

                # Store theme data in checkbox
                checkbox.setProperty('theme_number', theme['number'])
                checkbox.setProperty('theme_title', theme['title'])

                # Connect state change to update button state
                checkbox.stateChanged.connect(self.update_theme_selection_state)

                # Add to list
                self.themes_list.addItem(item)
                self.themes_list.setItemWidget(item, checkbox)

            # Update button state based on initial selections
            self.update_theme_selection_state()

            # Log summary with debug info
            self.log_output.append(f"DEBUG: Found {len(existing_theme_numbers)} processed themes: {existing_theme_numbers}")
            if existing_theme_numbers:
                self.log_output.append(f"Loaded {len(themes)} themes from {folder_name} ({len(existing_theme_numbers)} already processed)")
            else:
                self.log_output.append(f"Loaded {len(themes)} themes from {folder_name}")

        except Exception as e:
            self.log_output.append(f"Error loading themes: {str(e)}")

    def generate_themes_for_folder(self, folder: Path):
        """Generate themes for an existing video folder."""
        # Check if subtitle file exists
        video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
        if not video_files:
            QMessageBox.warning(self, "No Video", f"No video file found in {folder.name}")
            return

        video_path = video_files[0]
        srt_file = folder / f"{video_path.stem}.srt"

        if not srt_file.exists():
            # Offer to generate subtitles first
            reply = QMessageBox.question(
                self,
                "No Subtitles Found",
                f"No subtitle file found. Subtitles are required for theme generation.\n\nWould you like to generate subtitles first?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Generate subtitles first
                video_info = {
                    'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
                    'url': 'Existing video',
                    'folder': str(folder),
                    'folder_number': folder.name.split('_')[0],
                    'video_path': str(video_path),
                    'is_local': True
                }

                self.log_output.clear()
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)

                # First generate subtitles
                self.worker = ProcessWorker('subtitles', folder=str(folder), video_info=video_info,
                                            model=self.settings.get('whisper', 'model'))
                self.worker.progress.connect(self.on_progress)
                self.worker.log.connect(self.on_log)
                self.worker.finished.connect(lambda success, msg: self.on_subtitles_finished(success, msg, folder))
                self.worker.start()
            else:
                self.log_output.append("Cannot generate themes without subtitles.")
            return

        # Subtitles exist, generate themes directly
        video_info = {
            'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
            'url': 'Existing video',
            'folder': str(folder),
            'folder_number': folder.name.split('_')[0],
            'video_path': str(video_path),
            'is_local': True
        }

        self.log_output.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ProcessWorker('themes', folder=str(folder), video_info=video_info,
                                    model=self.settings.get('whisper', 'model'))
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(lambda success, msg: self.on_themes_finished(success, msg, folder.name))
        self.worker.start()

    def on_subtitles_finished(self, success: bool, message: str, folder: Path):
        """Handle subtitle generation completion, then generate themes."""
        if success:
            # Now generate themes
            video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
            if video_files:
                video_info = {
                    'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
                    'url': 'Existing video',
                    'folder': str(folder),
                    'folder_number': folder.name.split('_')[0],
                    'video_path': str(video_files[0]),
                    'is_local': True
                }

                self.worker = ProcessWorker('themes', folder=str(folder), video_info=video_info,
                                            model=self.settings.get('whisper', 'model'))
                self.worker.progress.connect(self.on_progress)
                self.worker.log.connect(self.on_log)
                self.worker.finished.connect(lambda success, msg: self.on_themes_finished(success, msg, folder.name))
                self.worker.start()
        else:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to generate subtitles: {message}")

    def on_themes_finished(self, success: bool, message: str, folder_name: str):
        """Handle theme generation completion."""
        self.set_processing_state(False)
        self.progress_bar.setVisible(False)

        if success:
            QMessageBox.information(self, "Success", f"Themes generated successfully for {folder_name}!")
            # Reload the folder to show themes
            self.on_folder_changed(folder_name)
        else:
            QMessageBox.critical(self, "Error", f"Failed to generate themes: {message}")

    def select_all_themes(self):
        """Select all themes in the list."""
        for i in range(self.themes_list.count()):
            item = self.themes_list.item(i)
            checkbox = self.themes_list.itemWidget(item)
            checkbox.setChecked(True)
        self.update_theme_selection_state()

    def select_none_themes(self):
        """Deselect all themes in the list."""
        for i in range(self.themes_list.count()):
            item = self.themes_list.item(i)
            checkbox = self.themes_list.itemWidget(item)
            checkbox.setChecked(False)
        self.update_theme_selection_state()

    def regenerate_subtitles(self):
        """Regenerate subtitles for the selected folder."""
        folder_name = self.folder_combo.currentText()
        if not folder_name:
            QMessageBox.warning(self, "No Folder", "Please select a video folder first.")
            return

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Regenerate Subtitles",
            f"This will regenerate subtitles for {folder_name}.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Get folder path
        base_dir = Path(self.settings.get('video', 'output_dir'))
        folder = base_dir / folder_name

        # Find video file
        video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
        if not video_files:
            QMessageBox.warning(self, "No Video", "No video file found in this folder.")
            return

        video_path = video_files[0]

        # Prepare video info
        video_info = {
            'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
            'url': 'Existing video',
            'folder': str(folder),
            'folder_number': folder.name.split('_')[0],
            'video_path': str(video_path),
            'is_local': True
        }

        # Clear log and start processing
        self.log_output.clear()
        self.set_processing_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ProcessWorker('subtitles', folder=str(folder), video_info=video_info,
                                    model=self.settings.get('whisper', 'model'))
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(lambda success, msg: self.on_regenerate_subtitles_finished(success, msg, folder))
        self.worker.start()

    def on_regenerate_subtitles_finished(self, success: bool, message: str, folder: Path):
        """Handle subtitle regeneration completion."""
        if success:
            QMessageBox.information(self, "Success", f"Subtitles regenerated successfully for {folder.name}!")
        else:
            self.set_processing_state(False)
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to regenerate subtitles: {message}")

    def regenerate_themes(self):
        """Regenerate themes for the selected folder."""
        folder_name = self.folder_combo.currentText()
        if not folder_name:
            QMessageBox.warning(self, "No Folder", "Please select a video folder first.")
            return

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Regenerate Themes",
            f"This will regenerate themes for {folder_name}.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Get folder path
        base_dir = Path(self.settings.get('video', 'output_dir'))
        folder = base_dir / folder_name

        # Check if subtitle file exists
        video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
        if not video_files:
            QMessageBox.warning(self, "No Video", "No video file found in this folder.")
            return

        video_path = video_files[0]
        srt_file = folder / f"{video_path.stem}.srt"

        if not srt_file.exists():
            QMessageBox.warning(
                self,
                "No Subtitles",
                "No subtitle file found. Please generate subtitles first."
            )
            return

        # Prepare video info
        video_info = {
            'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
            'url': 'Existing video',
            'folder': str(folder),
            'folder_number': folder.name.split('_')[0],
            'video_path': str(video_path),
            'is_local': True
        }

        # Clear log and start processing
        self.log_output.clear()
        self.set_processing_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ProcessWorker('themes', folder=str(folder), video_info=video_info,
                                    model=self.settings.get('whisper', 'model'))
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(lambda success, msg: self.on_themes_finished(success, msg, folder_name))
        self.worker.start()

    def create_shorts(self):
        """Create shorts from selected themes."""
        folder_name = self.folder_combo.currentText()
        if not folder_name:
            QMessageBox.warning(self, "No Folder", "Please select a video folder first.")
            return

        # Get selected theme numbers
        selected_themes = []
        for i in range(self.themes_list.count()):
            item = self.themes_list.item(i)
            checkbox = self.themes_list.itemWidget(item)
            if checkbox.isChecked():
                theme_num = checkbox.property('theme_number')
                selected_themes.append(str(theme_num))

        if not selected_themes:
            QMessageBox.warning(self, "No Selection", "Please select at least one theme.")
            return

        # Extract folder number
        folder_number = folder_name.split('_')[0]

        # Clear log
        self.log_output.clear()

        # Disable all UI elements
        self.set_processing_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start worker thread
        theme_numbers = ','.join(selected_themes)
        self.worker = ProcessWorker('create_shorts', folder_number=folder_number,
                                    theme_numbers=theme_numbers)

        # Connect signals
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(self.on_shorts_finished)

        # Start thread
        self.log_output.append(f"Creating {len(selected_themes)} short(s) from folder {folder_name}...")
        self.worker.start()

    def on_shorts_finished(self, success: bool, message: str):
        """Handle shorts creation completion."""
        self.set_processing_state(False)
        self.progress_bar.setVisible(False)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

    def save_settings(self):
        """Save settings to settings.ini."""
        import configparser

        config = configparser.ConfigParser()

        # Whisper settings
        config['whisper'] = {
            'model': self.whisper_model_combo.currentText(),
            'language': self.whisper_language_input.text(),
            'task': self.settings.get('whisper', 'task')
        }

        # Video settings
        config['video'] = {
            'output_dir': self.settings.get('video', 'output_dir'),
            'aspect_ratio': self.settings.get('video', 'aspect_ratio'),
            'resolution_width': str(self.resolution_width_spin.value()),
            'resolution_height': str(self.resolution_height_spin.value()),
            'codec': self.settings.get('video', 'codec'),
            'preset': self.settings.get('video', 'preset'),
            'crf': str(self.crf_spin.value())
        }

        # Subtitle settings
        config['subtitle'] = {
            'font_name': self.font_name_input.text(),
            'font_size': str(self.font_size_spin.value()),
            'primary_colour': self.settings.get('subtitle', 'primary_colour'),
            'back_colour': self.settings.get('subtitle', 'back_colour'),
            'outline_colour': self.settings.get('subtitle', 'outline_colour'),
            'alignment': self.settings.get('subtitle', 'alignment'),
            'margin_v': str(self.margin_v_spin.value())
        }

        # Theme and folder settings (unchanged)
        config['theme'] = dict(self.settings['theme'])
        config['folder'] = dict(self.settings['folder'])

        # Save
        with open('settings.ini', 'w') as f:
            config.write(f)

        # Update working settings
        self.settings = load_settings()

        QMessageBox.information(self, "Settings Saved", "Settings have been saved to settings.ini")


# FormLayout helper
class QFormLayout(QGridLayout):
    """Simple form layout for settings."""

    def __init__(self):
        super().__init__()
        self.setColumnStretch(1, 1)
        self.setVerticalSpacing(10)

    def addRow(self, label, widget):
        """Add a row with label and widget."""
        row = self.rowCount()
        self.addWidget(QLabel(label), row, 0)
        self.addWidget(widget, row, 1)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Shorts Creator")

    window = YouTubeShortsGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
