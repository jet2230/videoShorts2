#!/usr/bin/env python3
"""
GUI Showcase - Displays different GUI framework styles
Run this file to see examples of each GUI framework available.
"""

import sys
import subprocess

def show_menu():
    """Display the menu of GUI options."""
    print("=" * 70)
    print(" " * 15 + "GUI Framework Showcase")
    print("=" * 70)
    print()
    print("Choose a GUI framework to preview:\n")
    print("  [1] Tkinter         (Built-in, dated look)")
    print("  [2] CustomTkinter   (Modern, rounded, RECOMMENDED)")
    print("  [3] PyQt6           (Professional, native)")
    print("  [4] Dear PyGui      (Fast, GPU-accelerated)")
    print("  [5] Kivy           (Cross-platform, mobile)")
    print("  [6] Flet            (Flutter-like, modern)")
    print("  [7] Gradio          (Web-based, AI/ML focused)")
    print("  [8] Streamlit       (Web-based, data apps)")
    print()
    print("  [0] Exit")
    print()
    print("=" * 70)

def create_tkinter_example():
    """Create Tkinter example."""
    code = '''#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class YouTubeShortsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Shorts Creator - Tkinter")
        self.root.geometry("600x400")

        # Title
        title = tk.Label(root, text="YouTube Shorts Creator", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Input frame
        input_frame = tk.Frame(root)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Video URL/File:").pack(side=tk.LEFT, padx=5)
        self.entry = tk.Entry(input_frame, width=40)
        self.entry.pack(side=tk.LEFT)

        tk.Button(input_frame, text="Browse", command=self.browse).pack(side=tk.LEFT, padx=5)

        # Options frame
        opts_frame = tk.Frame(root)
        opts_frame.pack(pady=10)

        tk.Label(opts_frame, text="Whisper Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="small")
        models = ["tiny", "base", "small", "medium", "large"]
        ttk.OptionMenu(opts_frame, self.model_var, *models).pack(side=tk.LEFT)

        # Buttons frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="Process Video", command=self.process,
                 bg="#4CAF50", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Create Shorts", command=self.create_shorts,
                 bg="#2196F3", fg="white", width=15).pack(side=tk.LEFT, padx=5)

        # Progress
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(pady=20, fill=tk.X, padx=50)

        # Status
        self.status = tk.Label(root, text="Ready", fg="green")
        self.status.pack(side=tk.BOTTOM, pady=10)

    def browse(self):
        file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.webm")])
        if file:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, file)

    def process(self):
        self.status.config(text="Processing video...", fg="blue")
        self.progress.start(10)
        self.root.after(3000, lambda: [self.progress.stop(),
                                         self.status.config(text="✓ Video processed!", fg="green")])

    def create_shorts(self):
        self.status.config(text="Creating shorts...", fg="blue")
        self.progress.start(10)
        self.root.after(3000, lambda: [self.progress.stop(),
                                         self.status.config(text="✓ Shorts created!", fg="green")])

if __name__ == "__main__":
    root = tk.Tk()
    app = YouTubeShortsGUI(root)
    root.mainloop()
'''
    filename = "showcase_tkinter.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def create_customtkinter_example():
    """Create CustomTkinter example."""
    code = '''#!/usr/bin/env python3
import customtkinter as tk

class YouTubeShortsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Shorts Creator - CustomTkinter")
        self.root.geometry("700x500")

        # Set appearance
        tk.set_appearance_mode("dark")  # dark, light, system
        tk.set_default_color_theme("blue")  # blue, green, dark-blue

        # Title
        title = tk.CTkLabel(root, text="YouTube Shorts Creator",
                            font=tk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)

        # Input frame
        input_frame = tk.CTkFrame(root)
        input_frame.pack(pady=10, padx=20, fill="x")

        tk.CTkLabel(input_frame, text="Video URL/File:", font=tk.CTkFont(size=14)).pack(
            side=tk.LEFT, padx=10, pady=10)
        self.entry = tk.CTkEntry(input_frame, width=50, placeholder_text="Enter URL or file path...")
        self.entry.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill="x")
        tk.CTkButton(input_frame, text="Browse", command=self.browse, width=80).pack(
            side=tk.LEFT, padx=10, pady=10)

        # Options frame
        opts_frame = tk.CTkFrame(root)
        opts_frame.pack(pady=10, padx=20, fill="x")

        tk.CTkLabel(opts_frame, text="Whisper Model:", font=tk.CTkFont(size=14)).pack(
            side=tk.LEFT, padx=10, pady=10)
        self.model_var = tk.StringVar(value="small")
        models = ["tiny", "base", "small", "medium", "large"]
        self.model_menu = tk.CTkOptionMenu(opts_frame, variable=self.model_var, values=models)
        self.model_menu.pack(side=tk.LEFT, padx=10, pady=10)

        # Theme switch
        self.theme_var = tk.StringVar(value="dark")
        theme_frame = tk.CTkFrame(root)
        theme_frame.pack(pady=10, padx=20, fill="x")
        tk.CTkLabel(theme_frame, text="Theme:", font=tk.CTkFont(size=14)).pack(
            side=tk.LEFT, padx=10, pady=10)
        tk.CTkOptionMenu(theme_frame, variable=self.theme_var,
                        values=["dark", "light", "system"],
                        command=self.change_theme).pack(side=tk.LEFT, padx=10, pady=10)

        # Buttons frame
        btn_frame = tk.CTkFrame(root)
        btn_frame.pack(pady=20, padx=20, fill="x")

        tk.CTkButton(btn_frame, text="Process Video", command=self.process,
                    fg_color="#2CC985", hover_color="#23916D", height=40,
                    font=tk.CTkFont(size=16, weight="bold")).pack(side=tk.LEFT, padx=10, expand=True, fill="x")
        tk.CTkButton(btn_frame, text="Create Shorts", command=self.create_shorts,
                    fg_color="#3B8ED0", hover_color="#3672A9", height=40,
                    font=tk.CTkFont(size=16, weight="bold")).pack(side=tk.LEFT, padx=10, expand=True, fill="x")

        # Progress bar
        self.progress = tk.CTkProgressBar(root, width=400)
        self.progress.pack(pady=20, padx=50)

        # Status
        self.status = tk.CTkLabel(root, text="✓ Ready", font=tk.CTkFont(size=14), text_color="green")
        self.status.pack(side=tk.BOTTOM, pady=20)

    def change_theme(self, choice):
        tk.set_appearance_mode(choice)

    def browse(self):
        from tkinter import filedialog
        file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.webm")])
        if file:
            self.entry.delete(0, "end")
            self.entry.insert(0, file)

    def process(self):
        self.status.configure(text="⏳ Processing video...", text_color="yellow")
        self.progress.set(0)
        self.animate_progress()

    def create_shorts(self):
        self.status.configure(text="⏳ Creating shorts...", text_color="yellow")
        self.progress.set(0)
        self.animate_progress()

    def animate_progress(self):
        if self.progress.get() < 1.0:
            self.progress.set(self.progress.get() + 0.1)
            self.root.after(200, self.animate_progress)
        else:
            self.status.configure(text="✓ Complete!", text_color="green")

if __name__ == "__main__":
    root = tk.CTk()
    app = YouTubeShortsGUI(root)
    root.mainloop()
'''
    filename = "showcase_customtkinter.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def create_pyqt6_example():
    """Create PyQt6 example."""
    code = '''#!/usr/bin/env python3
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QComboBox, QProgressBar, QFileDialog, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

class WorkerThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        for i in range(101):
            self.progress.emit(i)
            self.msleep(30)
        self.finished.emit()

class YouTubeShortsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Shorts Creator - PyQt6")
        self.setGeometry(100, 100, 700, 400)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("YouTube Shorts Creator")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Input frame
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)
        label = QLabel("Video URL/File:")
        label.setMinimumWidth(120)
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("Enter URL or file path...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse)
        browse_btn.setMaximumWidth(100)
        input_layout.addWidget(label)
        input_layout.addWidget(self.entry)
        input_layout.addWidget(browse_btn)
        layout.addWidget(input_frame)

        # Options frame
        opts_frame = QFrame()
        opts_layout = QHBoxLayout(opts_frame)
        model_label = QLabel("Whisper Model:")
        model_label.setMinimumWidth(120)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("small")
        self.model_combo.setMaximumWidth(150)
        opts_layout.addWidget(model_label)
        opts_layout.addWidget(self.model_combo)
        opts_layout.addStretch()
        layout.addWidget(opts_frame)

        # Buttons frame
        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        process_btn = QPushButton("Process Video")
        process_btn.setMinimumHeight(50)
        process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        process_btn.clicked.connect(self.process)
        shorts_btn = QPushButton("Create Shorts")
        shorts_btn.setMinimumHeight(50)
        shorts_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        shorts_btn.clicked.connect(self.create_shorts)
        btn_layout.addWidget(process_btn)
        btn_layout.addWidget(shorts_btn)
        layout.addWidget(btn_frame)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3b8ed0;
            }
        """)
        layout.addWidget(self.progress)

        # Status
        self.status = QLabel("✓ Ready")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("color: green; font-size: 14px;")
        layout.addWidget(self.status)

        layout.addStretch()

    def browse(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                               "Video Files (*.mp4 *.mkv *.webm)")
        if file:
            self.entry.setText(file)

    def process(self):
        self.status.setText("⏳ Processing video...")
        self.status.setStyleSheet("color: blue; font-size: 14px;")
        self.worker = WorkerThread()
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(lambda: self.status.setText("✓ Video processed!"))
        self.worker.start()

    def create_shorts(self):
        self.status.setText("⏳ Creating shorts...")
        self.status.setStyleSheet("color: blue; font-size: 14px;")
        self.worker = WorkerThread()
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(lambda: self.status.setText("✓ Shorts created!"))
        self.worker.start()

if __name__ == "__main__":
    app = QApplication([])
    window = YouTubeShortsGUI()
    window.show()
    app.exec()
'''
    filename = "showcase_pyqt6.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def create_dearpygui_example():
    """Create Dear PyGui example."""
    code = '''#!/usr/bin/env python3
import dearpygui.dearpygui as dpg

dpg.create_context()

# Main window
dpg.create_viewport(title='YouTube Shorts Creator - Dear PyGui', width=800, height=600)

with dpg.window(label="YouTube Shorts Creator", tag="main_window"):
    dpg.add_spacer(height=20)

    # Title
    dpg.add_text("YouTube Shorts Creator", color=(100, 200, 100), indent=20, wrap=800)
    dpg.add_separator()
    dpg.add_spacer(height=20)

    # Input section
    dpg.add_text("Video URL/File:", indent=20)
    url_input = dpg.add_input_text(hint="Enter URL or file path...", width=600, indent=20, tag="url_input")

    with dpg.group(horizontal=True, indent=20):
        dpg.add_button(label="Browse", width=150, callback=lambda: dpg.set_value("status_text", "File browser opened"))
        dpg.add_button(label="Test", width=150, callback=lambda: dpg.set_value("status_text", "✓ URL validated!"))

    dpg.add_spacer(height=20)

    # Options section
    dpg.add_separator()
    dpg.add_spacer(height=20)

    dpg.add_text("Options:", indent=20)

    # Whisper model selection
    dpg.add_text("Whisper Model:", indent=40)
    models = ["tiny", "base", "small", "medium", "large"]
    dpg.add_combo(models, default_value="small", width=200, indent=40, tag="model_combo")

    # Progress section
    dpg.add_spacer(height=20)
    dpg.add_separator()
    dpg.add_spacer(height=20)

    # Action buttons
    with dpg.group(horizontal=True, indent=20):
        dpg.add_button(label="Process Video", width=200, callback=lambda: process_video(dpg),
                     tag="process_btn")
        dpg.add_button(label="Create Shorts", width=200, callback=lambda: create_shorts(dpg),
                     tag="shorts_btn")

    # Progress bar
    dpg.add_spacer(height=20)
    progress = dpg.add_progress_bar(width=600, indent=20, overlay="Processing...", tag="progress")

    # Status
    dpg.add_spacer(height=20)
    status = dpg.add_text("✓ Ready", color=(0, 255, 0), indent=20, tag="status_text")

def process_video(dpg):
    dpg.set_value("status_text", "⏳ Processing video...")
    dpg.set_value("status_text", [100, 200, 0])
    import time
    for i in range(101):
        dpg.set_value("progress", i/100)
        time.sleep(0.03)
    dpg.set_value("status_text", [0, 255, 0])
    dpg.set_value("status_text", "✓ Video processed!")

def create_shorts(dpg):
    dpg.set_value("status_text", "⏳ Creating shorts...")
    dpg.set_value("status_text", [100, 200, 0])
    import time
    for i in range(101):
        dpg.set_value("progress", i/100)
        time.sleep(0.03)
    dpg.set_value("status_text", [0, 255, 0])
    dpg.set_value("status_text", "✓ Shorts created!")

dpg.set_primary_window(window="main_window")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
'''
    filename = "showcase_dearpygui.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def create_flet_example():
    """Create Flet example."""
    code = '''#!/usr/bin/env python3
import flet as ft

def main(page: ft.Page):
    page.title = "YouTube Shorts Creator - Flet"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.colors.BLUE_GREY_900
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding = 30

    # Title
    title = ft.Text("YouTube Shorts Creator",
                    size=32,
                    weight=ft.FontWeight.BOLD,
                    color=ft.colors.BLUE_400)

    # Input field
    url_input = ft.TextField(
        label="Video URL/File",
        hint_text="Enter URL or file path...",
        width=600,
        text_size=16,
        color=ft.colors.BLUE_GREY_100,
        focused_color=ft.colors.BLUE_400,
    )

    # Browse button
    def browse_file(e):
        def pick_file_result(result: ft.FilePickerResultEvent):
            if result.path:
                url_input.value = result.path
                page.update()

        pick_file_dialog = ft.FilePicker()
        pick_file_dialog.on_result = pick_file_result

        page.overlay.append(pick_file_dialog)
        pick_file_dialog.save()

    browse_btn = ft.ElevatedButton(
        "Browse",
        icon=ft.icons.FOLDER_OPEN,
        bgcolor=ft.colors.BLUE_600,
        color=ft.colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
        on_click=browse_file
    )

    # Model selection
    model_dropdown = ft.Dropdown(
        label="Whisper Model",
        options=["tiny", "base", "small", "medium", "large"],
        value="small",
        width=200,
        color=ft.colors.BLUE_GREY_100,
        focused_color=ft.colors.BLUE_400,
    )

    # Progress bar
    progress = ft.ProgressBar(
        width=600,
        bar_height=20,
        bgcolor=ft.colors.BLUE_GREY_800,
        color=ft.colors.BLUE_400,
        visible=False
    )

    # Status text
    status = ft.Text(
        "✓ Ready",
        size=16,
        color=ft.colors.GREEN_400,
    )

    # Process button
    def process_video(e):
        progress.visible = True
        status.value = "⏳ Processing video..."
        status.color = ft.colors.YELLOW_400
        page.update()

        import time
        for i in range(101):
            progress.value = i / 100
            time.sleep(0.02)
            page.update()

        progress.visible = False
        status.value = "✓ Video processed!"
        status.color = ft.colors.GREEN_400
        page.update()

    # Create shorts button
    def create_shorts(e):
        progress.visible = True
        status.value = "⏳ Creating shorts..."
        status.color = ft.colors.YELLOW_400
        page.update()

        import time
        for i in range(101):
            progress.value = i / 100
            time.sleep(0.02)
            page.update()

        progress.visible = False
        status.value = "✓ Shorts created!"
        status.color = ft.colors.GREEN_400
        page.update()

    process_btn = ft.ElevatedButton(
        "Process Video",
        icon=ft.icons.PLAY_ARROW,
        bgcolor=ft.colors.GREEN_600,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
        on_click=process_video
    )

    shorts_btn = ft.ElevatedButton(
        "Create Shorts",
        icon=ft.icons.VIDEO_COLLECTION,
        bgcolor=ft.colors.BLUE_600,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
        on_click=create_shorts
    )

    # Layout
    page.add(
        ft.Column(
            [
                ft.Row([title], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([url_input, browse_btn], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([model_dropdown], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([process_btn, shorts_btn], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([progress], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([status], alignment=ft.MainAxisAlignment.CENTER),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
'''
    filename = "showcase_flet.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def create_gradio_example():
    """Create Gradio example."""
    code = '''#!/usr/bin/env python3
import gradio as gr
import time

def process_video(url, model="small"):
    """Process video and generate status."""
    progress = gr.Progress()

    for i in range(101):
        time.sleep(0.03)
        progress(i, desc="Processing video...")

    return f"✓ Video processed successfully!\\n\\nModel: {model}\\nURL: {url}"

def create_shorts(theme_numbers):
    """Create shorts for specified themes."""
    progress = gr.Progress()

    for i in range(101):
        time.sleep(0.03)
        progress(i, desc="Creating shorts...")

    return f"✓ Created {len(theme_numbers.split(',')) if theme_numbers else 1} shorts!"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 YouTube Shorts Creator

        Upload or paste a video URL to create YouTube Shorts automatically!
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="📁 Video URL or File Path",
                placeholder="https://youtube.com/watch?v=xxx or /path/to/video.mp4"
            )

            model_choice = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                value="small",
                label="🤖 Whisper Model"
            )

            with gr.Row():
                process_btn = gr.Button("Process Video", variant="primary", size="lg")
                shorts_btn = gr.Button("Create Shorts", variant="secondary", size="lg")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="📊 Status Output",
                lines=10,
                interactive=False
            )

    # Examples
    gr.Examples(
        examples=[
            ["https://youtube.com/watch?v=dQw4w9WgXcQ", "small"],
            ["/path/to/local/video.mp4", "base"],
        ],
        inputs=[url_input, model_choice]
    )

    # Info section
    gr.Markdown(
        """
        ---
        ### 📝 Instructions:
        1. Enter a YouTube URL or local file path
        2. Select Whisper model (recommended: small)
        3. Click "Process Video" to generate subtitles and themes
        4. Click "Create Shorts" to generate video clips

        ### ⚙️ Features:
        - 🎯 Automatic subtitle generation (Whisper)
        - 🤖 AI-powered theme identification (Llama 3)
        - 📱 9:16 vertical format for YouTube Shorts
        - 🎨 Burned-in subtitles
        """
    )

    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[url_input, model_choice],
        outputs=output
    )

    shorts_btn.click(
        fn=create_shorts,
        inputs=gr.Textbox(value="1,2,3", label="Theme Numbers"),
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=False)
'''
    filename = "showcase_gradio.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def create_streamlit_example():
    """Create Streamlit example."""
    code = '''#!/usr/bin/env python3
import streamlit as st
import time

# Page config
st.set_page_config(
    page_title="YouTube Shorts Creator",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 YouTube Shorts Creator")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large"],
        index=2
    )

    st.markdown("---")
    st.markdown("### 📝 Info")
    st.info("""
    This app automatically:
    - Downloads videos
    - Generates subtitles
    - Identifies themes
    - Creates YouTube Shorts
    """)

    st.markdown("### 💡 Tips")
    st.success("""
    - Use 'small' model for best accuracy
    - Ensure you have Ollama running
    - Videos are saved to 'videos/' folder
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    url = st.text_input(
        "📁 Video URL or File Path",
        placeholder="https://youtube.com/watch?v=xxx or /path/to/video.mp4"
    )

with col2:
    if st.button("Browse"):
        st.info("File browser opened")

st.markdown("---")

# Action buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🎥 Process Video", type="primary"):
        with st.spinner("Processing video..."):
            progress_bar = st.progress(0)
            for i in range(101):
                time.sleep(0.02)
                progress_bar.progress(i)
        st.success("✓ Video processed successfully!")

with col2:
    if st.button("✂️ Create Shorts", type="secondary"):
        with st.spinner("Creating shorts..."):
            progress_bar = st.progress(0)
            for i in range(101):
                time.sleep(0.02)
                progress_bar.progress(i)
        st.success("✓ Shorts created successfully!")

# Output section
st.markdown("---")
st.header("📊 Status")
st.info("✓ Ready to process videos")

# Features section
st.markdown("---")
st.markdown("### 🎯 Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Videos Processed", "0")

with col2:
    st.metric("Shorts Created", "0")

with col3:
    st.metric("Themes Identified", "0")

st.markdown("---")
st.markdown("### 📖 Instructions")
st.markdown("""
1. **Enter URL** - Paste a YouTube URL or browse for a local file
2. **Select Model** - Choose Whisper model in sidebar (small recommended)
3. **Process Video** - Click to generate subtitles and themes
4. **Create Shorts** - Generate 9:16 vertical video clips
""")
'''
    filename = "showcase_streamlit.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def show_installation_info(choice):
    """Show installation info for selected framework."""
    info = {
        "1": """
=== Tkinter ===
✅ NO INSTALLATION NEEDED - Built into Python!

To run:
  python showcase_tkinter.py

Pros: Simple, built-in
Cons: Dated look
""",
        "2": """
=== CustomTkinter ===
📦 Installation:
  pip install customtkinter

To run:
  python showcase_customtkinter.py

Pros: Modern, rounded widgets, dark mode, RECOMMENDED
Cons: Requires installation
""",
        "3": """
=== PyQt6 ===
📦 Installation:
  pip install PyQt6

To run:
  python showcase_pyqt6.py

Pros: Professional, native OS look, industry standard
Cons: Larger library, steeper learning curve
""",
        "4": """
=== Dear PyGui ===
📦 Installation:
  pip install dearpygui

To run:
  python showcase_dearpygui.py

Pros: Fast, GPU-accelerated, modern
Cons: Newer, different API style
""",
        "5": """
=== Flet ===
📦 Installation:
  pip install flet

To run:
  python showcase_flet.py

Pros: Flutter-like, modern, easy to learn
Cons: Newer framework
""",
        "6": """
=== Gradio ===
📦 Installation:
  pip install gradio

To run:
  python showcase_gradio.py

Pros: Web-based, great for AI/ML apps, shareable
Cons: Less customizable
""",
        "7": """
=== Streamlit ===
📦 Installation:
  pip install streamlit

To run:
  python showcase_streamlit.py

Pros: Very fast development, web-based
Cons: Limited layout control
"""
    }
    return info.get(choice, "Invalid choice")

def run_example(filename):
    """Run the selected example."""
    try:
        subprocess.run([sys.executable, filename], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Error running {filename}: {e}")
        print("\\n💡 Make sure the framework is installed first!")
    except KeyboardInterrupt:
        print("\\n\\n✓ Example stopped by user")

def main():
    """Main menu for GUI showcase."""
    while True:
        show_menu()
        choice = input("Enter your choice (0-7): ").strip()

        if choice == "0":
            print("\\n✓ Goodbye!")
            break
        elif choice in ["1", "2", "3", "4", "5", "6", "7"]:
            print(f"\\n{'='*70}")
            print(show_installation_info(choice))
            print(f"{'='*70}")
            print("\\n📝 Creating example file...")

            filename = {
                "1": create_tkinter_example(),
                "2": create_customtkinter_example(),
                "3": create_pyqt6_example(),
                "4": create_dearpygui_example(),
                "5": create_flet_example(),
                "6": create_gradio_example(),
                "7": create_streamlit_example(),
            }[choice]

            print(f"✓ Created: {filename}")
            print()

            run = input("Run the example now? (y/n): ").strip().lower()
            if run == "y":
                print("\\n🚀 Launching GUI...")
                print("   (Close the window to return to menu)")
                print(f"{'='*70}\\n")
                run_example(filename)
                print(f"{'='*70}\\n")
        else:
            print("\\n❌ Invalid choice. Please try again.\\n")

if __name__ == "__main__":
    main()
