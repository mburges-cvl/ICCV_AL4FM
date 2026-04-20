import os
import requests
import json
import types
import time
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QDialogButtonBox,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QProgressBar,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QGroupBox,
    QApplication,
    QLabel,
    QWizard,
    QWizardPage,
)
from PyQt5.QtCore import QTimer, Qt
import rasterio
from PIL import Image
from utils.dataset_sync_dialogs import DatasetSyncDialog
import numpy as np

import shutil

class LoginPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Login")
        self.setSubTitle("Enter your login information")

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Create fields with default values
        self.name_edit = QLineEdit("vsz")
        self.port_edit = QLineEdit("8000")
        self.pps_edit = QLineEdit("16")
        self.imgs_to_smpl_edit = QLineEdit("10")
        # self.dummy_checkbox = QCheckBox("Random Mode")
        # self.debug_checkbox = QCheckBox("Debug")

        # Register fields to store values between pages
        self.registerField(
            "username", self.name_edit
        )  # Remove the asterisk for required fields
        self.registerField("port", self.port_edit)
        self.registerField("pps", self.pps_edit)
        self.registerField("imgs_to_smpl", self.imgs_to_smpl_edit)
        # self.registerField("dummy", self.dummy_checkbox)
        # self.registerField("debug", self.debug_checkbox)

        form_layout.addRow("Name:", self.name_edit)
        form_layout.addRow("Port:", self.port_edit)
        form_layout.addRow("Gridsize:", self.pps_edit)
        form_layout.addRow("Images to sample:", self.imgs_to_smpl_edit)

        # form_layout.addRow("", self.dummy_checkbox)
        # form_layout.addRow("", self.debug_checkbox)
        layout.addLayout(form_layout)

    def isComplete(self):
        """This method is called to determine if Next should be enabled"""
        # Check if the required fields have values
        return bool(self.name_edit.text().strip()) and bool(
            self.port_edit.text().strip()
            and bool(self.pps_edit.text().strip())
            and bool(self.imgs_to_smpl_edit.text().strip())
        )

    def initializePage(self):
        """Called when the page is shown"""
        super().initializePage()
        # Connect text changed signals to completeChanged
        self.name_edit.textChanged.connect(self.completeChanged)
        self.port_edit.textChanged.connect(self.completeChanged)


class ObjectSelectionPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Object Selection")
        self.setSubTitle("Select an object to annotate or create a new one")

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Add object selection dropdown
        self.object_combo = QComboBox()
        self.object_combo.setEditable(False)
        self.object_combo.addItem("Loading objects...")

        # Add new object option
        self.new_object_checkbox = QCheckBox("Create New Object")
        self.new_object_checkbox.stateChanged.connect(self.toggle_new_object)
        self.new_object_edit = QLineEdit()
        self.new_object_edit.setPlaceholderText("New object name")
        self.new_object_edit.setEnabled(False)

        # Register fields for wizard
        self.registerField("object_combo", self.object_combo, "currentText")
        self.registerField("new_object_checkbox", self.new_object_checkbox)
        self.registerField("new_object_name", self.new_object_edit)

        form_layout.addRow("Object:", self.object_combo)
        form_layout.addRow("", self.new_object_checkbox)
        form_layout.addRow("New Object:", self.new_object_edit)
        layout.addLayout(form_layout)

        # Status label for server connection
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Initialize the timer for fetching objects after page becomes visible
        self.fetch_timer = None

    def initializePage(self):
        """Called when the page is shown"""
        # Get the port from the previous page
        port = self.field("port")

        # Set up a timer to fetch objects after a short delay
        self.fetch_timer = QTimer(self)
        self.fetch_timer.timeout.connect(lambda: self.fetch_available_objects(port))
        self.fetch_timer.setSingleShot(True)
        self.fetch_timer.start(100)  # Start after 100ms

    def toggle_new_object(self, state):
        """Enable/disable new object input based on checkbox state"""
        self.new_object_edit.setEnabled(state == Qt.Checked)
        self.object_combo.setEnabled(state != Qt.Checked)

    def fetch_available_objects(self, port):
        """Fetch available objects from server"""
        self.status_label.setText("Connecting to server...")
        QApplication.processEvents()

        try:
            response = requests.get(
                f"http://localhost:{port}/get_available_objects", timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                objects = data.get("objects", [])

                self.object_combo.clear()
                if not objects:
                    self.object_combo.addItem("No objects found")
                else:
                    for obj in objects:
                        self.object_combo.addItem(obj)
                self.status_label.setText("Connected to server successfully")
            else:
                self.object_combo.clear()
                self.object_combo.addItem("Error connecting to server")
                self.status_label.setText(
                    f"Error: Server returned status {response.status_code}"
                )
        except Exception as e:
            self.object_combo.clear()
            self.object_combo.addItem(f"Error: {str(e)}")
            self.status_label.setText(f"Error: Could not connect to server ({str(e)})")

    def validatePage(self):
        """Validate the page before proceeding"""
        if self.field("new_object_checkbox"):
            # If creating a new object, make sure a name is provided
            if not self.field("new_object_name"):
                QMessageBox.warning(
                    self, "Error", "Please enter a name for the new object."
                )
                return False
        else:
            # If using existing object, make sure selection is valid
            if self.object_combo.currentText() in [
                "Loading objects...",
                "No objects found",
            ] or self.object_combo.currentText().startswith("Error:"):
                QMessageBox.warning(
                    self, "Error", "Please select a valid object or create a new one."
                )
                return False

        return True


class DatasetSelectionPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Dataset Selection")
        self.setSubTitle("Select a dataset or create a new one from a TIF file")

        self.dataset_path = None
        self.dataset_name = None

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Dataset selection options
        self.existing_dataset_radio = QRadioButton("Use Existing Dataset")
        self.folder_dataset_radio = QRadioButton("Create New Dataset from Folder of Images")
        self.new_dataset_radio = QRadioButton("Create New Dataset from TIF")
        self.existing_dataset_radio.setChecked(True)

        # Dataset button group
        dataset_group = QButtonGroup(self)
        dataset_group.addButton(self.existing_dataset_radio)
        dataset_group.addButton(self.folder_dataset_radio)
        dataset_group.addButton(self.new_dataset_radio)
        dataset_group.buttonClicked.connect(self.toggle_dataset_selection)

        # Existing dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.setEditable(False)
        self.dataset_combo.addItem("Loading datasets...")

        # TIF selection for new dataset
        self.tif_path_edit = QLineEdit()
        self.tif_path_edit.setReadOnly(True)
        self.tif_path_edit.setEnabled(False)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_tif)
        self.browse_button.setEnabled(False)

        # TIF conversion settings
        settings_group = QGroupBox("Tile Settings")
        settings_layout = QFormLayout()
        self.overlap = QLineEdit("10")
        settings_layout.addRow("Overlap (%):", self.overlap)
        settings_group.setLayout(settings_layout)
        settings_group.setEnabled(False)
        self.settings_group = settings_group

        # Register fields
        self.registerField("use_existing_dataset", self.existing_dataset_radio)
        self.registerField("create_new_folder_dataset", self.folder_dataset_radio)
        self.registerField("create_new_dataset", self.new_dataset_radio)
        self.registerField("selected_dataset", self.dataset_combo, "currentText")
        self.registerField("tif_path", self.tif_path_edit)
        self.registerField("overlap", self.overlap)

        # Folder selection for new dataset
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True)
        self.folder_path_edit.setEnabled(False)

        self.folder_browse_button = QPushButton("Browse Folder...")
        self.folder_browse_button.clicked.connect(self.browse_folder)
        self.folder_browse_button.setEnabled(False)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(self.folder_browse_button)

        self.folder_dataset_name_edit = QLineEdit()
        self.folder_dataset_name_edit.setPlaceholderText("Enter dataset name")
        self.folder_dataset_name_edit.setEnabled(False)

        self.registerField("folder_dataset_name", self.folder_dataset_name_edit)

        # TIF file layout
        tif_layout = QHBoxLayout()
        tif_layout.addWidget(self.tif_path_edit)
        tif_layout.addWidget(self.browse_button)

        # Add widgets to the form layout in the desired order:
        # 1. Existing dataset option
        form_layout.addRow("", self.existing_dataset_radio)
        form_layout.addRow("Dataset:", self.dataset_combo)
        # 2. Folder option
        form_layout.addRow("", self.folder_dataset_radio)
        form_layout.addRow("Selected Image Folder:", folder_layout)
        form_layout.addRow("Dataset Name:", self.folder_dataset_name_edit)
        # 3. TIF option
        form_layout.addRow("", self.new_dataset_radio)
        form_layout.addRow("Selected TIF File:", tif_layout)

        layout.addLayout(form_layout)
        layout.addWidget(settings_group)

        # Add progress display section
        self.counter_label = QLabel("Processed: 0 / 0 (Estimated time remaining: --)")
        self.counter_label.setVisible(False)
        self.progress_label = QLabel("Processing TIF...")
        self.progress_label.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.counter_label)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Initialize timer for fetching datasets
        self.fetch_timer = None

    def initializePage(self):
        """Called when the page is shown"""
        # Get the port from the first page
        port = self.field("port")

        # Set up a timer to fetch datasets after a short delay
        self.fetch_timer = QTimer(self)
        self.fetch_timer.timeout.connect(lambda: self.fetch_available_datasets(port))
        self.fetch_timer.setSingleShot(True)
        self.fetch_timer.start(100)  # Start after 100ms

    def toggle_dataset_selection(self, button):
        if button == self.new_dataset_radio:
            self.dataset_combo.setEnabled(False)
            self.tif_path_edit.setEnabled(True)
            self.browse_button.setEnabled(True)
            self.settings_group.setEnabled(True)
            self.folder_path_edit.setEnabled(False)
            self.folder_browse_button.setEnabled(False)
            self.folder_dataset_name_edit.setEnabled(False)
        elif button == self.folder_dataset_radio:
            self.dataset_combo.setEnabled(False)
            self.tif_path_edit.setEnabled(False)
            self.browse_button.setEnabled(False)
            self.settings_group.setEnabled(False)
            self.folder_path_edit.setEnabled(True)
            self.folder_browse_button.setEnabled(True)
            self.folder_dataset_name_edit.setEnabled(True)
        else:  # existing dataset radio
            self.dataset_combo.setEnabled(True)
            self.tif_path_edit.setEnabled(False)
            self.browse_button.setEnabled(False)
            self.settings_group.setEnabled(False)
            self.folder_path_edit.setEnabled(False)
            self.folder_browse_button.setEnabled(False)
            self.folder_dataset_name_edit.setEnabled(False)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder of Images", "")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            # Optionally set dataset name from folder name
            self.dataset_name = os.path.basename(folder_path.rstrip(os.sep))

    def browse_tif(self):
        """Open file dialog to select a TIF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select TIF File", "", "TIF Files (*.tif *.tiff)"
        )
        if file_path:
            self.tif_path_edit.setText(file_path)
            # Extract dataset name from file path
            self.dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            # Check if dataset already exists
            if os.path.exists(os.path.join("datasets", self.dataset_name)):
                QMessageBox.warning(
                    self,
                    "Dataset Exists",
                    f"A dataset named '{self.dataset_name}' already exists. "
                    "Please rename your TIF file or choose a different file.",
                )
                self.tif_path_edit.setText("")
                self.dataset_name = None

    def fetch_available_datasets(self, port):
        """Fetch available datasets from the server"""
        self.status_label.setText("Loading available datasets...")
        QApplication.processEvents()

        try:
            # Get datasets from server
            response = requests.get(
                f"http://localhost:{port}/get_available_datasets", timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                datasets = data.get("datasets", [])

                self.dataset_combo.clear()
                if not datasets:
                    self.dataset_combo.addItem("No datasets found")
                else:
                    for dataset in datasets:
                        self.dataset_combo.addItem(dataset)
                self.status_label.setText("Datasets loaded successfully")
            else:
                # Fallback: check local datasets directory
                self.dataset_combo.clear()
                self.status_label.setText("Server error, checking local datasets...")
                if os.path.exists("datasets"):
                    datasets = [
                        d
                        for d in os.listdir("datasets")
                        if os.path.isdir(os.path.join("datasets", d))
                    ]
                    if datasets:
                        for dataset in datasets:
                            self.dataset_combo.addItem(dataset)
                        self.status_label.setText(
                            "Loaded datasets from local directory"
                        )
                    else:
                        self.dataset_combo.addItem("No datasets found")
                        self.status_label.setText("No datasets found locally")
                else:
                    self.dataset_combo.addItem("No datasets found")
                    self.status_label.setText("No datasets directory found")
        except Exception as e:
            self.dataset_combo.clear()
            self.dataset_combo.addItem(f"Error: {str(e)}")
            self.status_label.setText(f"Error loading datasets: {str(e)}")

    def validatePage(self):
        if self.field("create_new_dataset"):
            # Validating new dataset creation from TIF file (existing code)
            if not self.field("tif_path"):
                QMessageBox.warning(self, "Error", "Please select a TIF file for the new dataset.")
                return False

            try:
                overlap = int(self.field("overlap"))
                if overlap < 0:
                    raise ValueError("Overlap must be a positive integer.")
            except ValueError:
                QMessageBox.warning(self, "Error", "Overlap must be a positive integer.")
                return False

            result = self.convert_tif_to_dataset()
            if not result:
                return False  # Conversion failed or was canceled

        elif self.field("create_new_folder_dataset"):
            if not self.folder_path_edit.text():
                QMessageBox.warning(self, "Error", "Please select a folder of images for the new dataset.")
                return False

            if not self.field("folder_dataset_name").strip():
                QMessageBox.warning(self, "Error", "Please enter a dataset name for the new folder import.")
                return False

            self.dataset_name = self.field("folder_dataset_name").strip()
            result = self.convert_folder_to_dataset()
            if not result:
                return False

        else:
            # Verify existing dataset (existing code)
            dataset_name = self.field("selected_dataset")
            if dataset_name == "No datasets found" or dataset_name.startswith("Error:"):
                QMessageBox.warning(self, "Error", "Please select a valid dataset or create a new one.")
                return False

            dataset_path = os.path.join("datasets", dataset_name)
            if not os.path.exists(dataset_path):
                QMessageBox.warning(self, "Error", f"Dataset '{dataset_name}' not found in local 'datasets' directory.")
                return False

            self.dataset_path = dataset_path
            self.dataset_name = dataset_name

        self.wizard().setProperty("dataset_path", self.dataset_path)
        self.wizard().setProperty("dataset_name", self.dataset_name)
        return True
    
    def convert_folder_to_dataset(self):
        folder_path = self.folder_path_edit.text()
        output_folder = os.path.join("datasets", self.dataset_name)

        if os.path.exists(output_folder):
            response = QMessageBox.question(
                self,
                "Dataset Exists",
                f"A dataset named '{self.dataset_name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response == QMessageBox.No:
                return False

        os.makedirs(output_folder, exist_ok=True)
        images_output_folder = os.path.join(output_folder, "images")
        os.makedirs(images_output_folder, exist_ok=True)

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "positive"},
            ],
        }

        # Process image files (supporting common image extensions)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        total_images = len(image_files)
        if total_images == 0:
            QMessageBox.warning(self, "Error", "No valid image files found in the selected folder.")
            return False

        processed_count = 0
        image_id = 1

        self.counter_label.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        start_time = time.time()
        last_update_time = start_time

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            tile_filename = f"{self.dataset_name}_{image_id}.tif"
            tile_filepath = os.path.join(images_output_folder, tile_filename)
            try:
                if img_file.lower().endswith((".tif", ".tiff")):
                    # Use rasterio for (possibly multiband) TIFF images
                    with rasterio.open(img_path) as src:
                        data = src.read()  # shape: (bands, height, width)
                        bands, height, width = data.shape

                        print(f"[DEBUG] Processing {img_file}: {bands} bands, {width}x{height} pixels")

                        if width != 1024 or height != 1024:
                            scale = 1024 / max(width, height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            # Read resized data using out_shape (resampling happens automatically)
                            data_resized = src.read(
                                out_shape=(src.count, new_height, new_width),
                                resampling=rasterio.enums.Resampling.bilinear
                            )
                            # Create new padded array with zeros
                            new_data = np.zeros((src.count, 1024, 1024), dtype=data_resized.dtype)
                            pad_x = (1024 - new_width) // 2
                            pad_y = (1024 - new_height) // 2
                            new_data[:, pad_y:pad_y+new_height, pad_x:pad_x+new_width] = data_resized
                            
                            new_meta = src.meta.copy()
                            new_meta.update({"height": 1024, "width": 1024})
                            with rasterio.open(tile_filepath, "w", **new_meta) as dst:
                                dst.write(new_data)                            
                        else:
                            # If already 1024x1024, copy directly
                            print(f"[DEBUG] Copying {img_file} without resizing")
                            shutil.copy(img_path, tile_filepath)                            

                else:
                    # Use PIL for other image formats
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        width, height = img.size
                        scale = 1024 / max(width, height)
                        new_size = (int(width * scale), int(height * scale))
                        resized_img = img.resize(new_size, Image.ANTIALIAS)

                        # Create a new 1024x1024 image and paste the resized image centered
                        new_img = Image.new("RGB", (1024, 1024), (0, 0, 0))
                        paste_x = (1024 - new_size[0]) // 2
                        paste_y = (1024 - new_size[1]) // 2
                        new_img.paste(resized_img, (paste_x, paste_y))
                        new_img.save(tile_filepath)
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": tile_filename,
                    "width": 1024,
                    "height": 1024,
                    "img_file": img_file
                })
                image_id += 1
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
            
            processed_count += 1
            progress_percentage = (processed_count / total_images) * 100
            self.progress_bar.setValue(int(progress_percentage))
            current_time = time.time()
            if processed_count == 1 or processed_count % 5 == 0 or (current_time - last_update_time) > 2:
                elapsed_time = current_time - start_time
                if processed_count > 1:
                    images_per_second = processed_count / elapsed_time
                    remaining_images = total_images - processed_count
                    estimated_remaining_time = (remaining_images / images_per_second) if images_per_second > 0 else 0
                    mins = int(estimated_remaining_time // 60)
                    secs = int(estimated_remaining_time % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                    self.counter_label.setText(f"Processed: {processed_count} / {total_images} (Estimated time remaining: {time_str})")
                last_update_time = current_time
            QApplication.processEvents()

        with open(os.path.join(output_folder, "annotations.json"), "w") as f:
            json.dump(coco_data, f, indent=4)

        self.dataset_path = output_folder
        QMessageBox.information(self, "Conversion Complete", f"Folder images have been successfully converted to dataset '{self.dataset_name}'.")
        self.counter_label.setVisible(False)
        self.progress_label.setVisible(False)
        self.progress_bar.setVisible(False)
        return True

    def convert_tif_to_dataset(self):
        """Convert selected TIF to a dataset with tiles"""
        image_path = self.field("tif_path")
        self.dataset_name = os.path.splitext(os.path.basename(image_path))[0]
        output_folder = os.path.join("datasets", self.dataset_name)

        # Check if dataset already exists
        if os.path.exists(output_folder):
            response = QMessageBox.question(
                self,
                "Dataset Exists",
                f"A dataset named '{self.dataset_name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response == QMessageBox.No:
                return False

        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)

        # Get tile settings
        # tile_w, tile_h = 1024, 1024
        tile_w, tile_h = 256, 256
        overlap_w = int(tile_w * int(self.field("overlap"))) // 100
        overlap_h = int(tile_h * int(self.field("overlap"))) // 100
        step_x = tile_w - overlap_w
        step_y = tile_h - overlap_h

        # Prepare COCO-style structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "positive"},
            ],
        }

        # Show progress bar and labels
        self.counter_label.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Disable wizard navigation during conversion
        self.wizard().button(QWizard.BackButton).setEnabled(False)
        self.wizard().button(QWizard.NextButton).setEnabled(False)
        self.wizard().button(QWizard.CancelButton).setEnabled(False)
        QApplication.processEvents()

        try:
            # Check if TIF
            if image_path.lower().endswith((".tif", ".tiff")):
                with rasterio.open(image_path) as src:
                    width, height = src.width, src.height
                    meta = src.meta.copy()
                    image_id = 1

                    # Calculate total tiles for progress bar
                    total_tiles = ((height - 1) // step_y + 1) * (
                        (width - 1) // step_x + 1
                    )
                    processed_tiles = 0

                    # Setup timing
                    start_time = time.time()
                    last_update_time = start_time

                    for y in range(0, height, step_y):
                        for x in range(0, width, step_x):
                            # Update progress
                            processed_tiles += 1
                            progress_percentage = (processed_tiles / total_tiles) * 100
                            self.progress_bar.setValue(int(progress_percentage))

                            # Update time estimate
                            current_time = time.time()
                            if (
                                processed_tiles == 1
                                or processed_tiles % 5 == 0
                                or (current_time - last_update_time) > 2
                            ):
                                elapsed_time = current_time - start_time
                                if processed_tiles > 1:  # Avoid division by zero
                                    tiles_per_second = processed_tiles / elapsed_time
                                    remaining_tiles = total_tiles - processed_tiles
                                    estimated_remaining_time = (
                                        remaining_tiles / tiles_per_second
                                        if tiles_per_second > 0
                                        else 0
                                    )

                                    # Format time as minutes:seconds
                                    mins = int(estimated_remaining_time // 60)
                                    secs = int(estimated_remaining_time % 60)
                                    time_str = f"{mins:02d}:{secs:02d}"

                                    self.counter_label.setText(
                                        f"Processed: {processed_tiles} / {total_tiles} (Estimated time remaining: {time_str})"
                                    )
                                    last_update_time = current_time

                            QApplication.processEvents()

                            # Process tile
                            window = rasterio.windows.Window(x, y, tile_w, tile_h)
                            tile_data = src.read(window=window)

                            # Skip incomplete tiles
                            if (
                                tile_data.shape[1] < tile_h
                                or tile_data.shape[2] < tile_w
                            ):
                                continue

                            new_meta = meta.copy()
                            new_meta.update(
                                {
                                    "height": tile_data.shape[1],
                                    "width": tile_data.shape[2],
                                    "transform": rasterio.windows.transform(
                                        window, src.transform
                                    ),
                                }
                            )

                            tile_filename = f"{self.dataset_name}_{image_id}.tif"
                            tile_filepath = os.path.join(
                                output_folder, "images", tile_filename
                            )

                            with rasterio.open(tile_filepath, "w", **new_meta) as dst:
                                dst.write(tile_data)

                            coco_data["images"].append(
                                {
                                    "id": image_id,
                                    "file_name": tile_filename,
                                    "width": tile_data.shape[2],
                                    "height": tile_data.shape[1],
                                }
                            )

                            image_id += 1
            else:
                raise ValueError("Only TIF files are supported at the moment.")

            # Save COCO-style JSON
            with open(os.path.join(output_folder, "annotations.json"), "w") as f:
                json.dump(coco_data, f, indent=4)

            self.dataset_path = output_folder
            QMessageBox.information(
                self,
                "Conversion Complete",
                f"TIF file has been successfully converted to dataset '{self.dataset_name}'.",
            )
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to convert TIF: {str(e)}")
            return False
        finally:
            # Hide progress elements and re-enable UI
            self.counter_label.setVisible(False)
            self.progress_label.setVisible(False)
            self.progress_bar.setVisible(False)
            self.wizard().button(QWizard.BackButton).setEnabled(True)
            self.wizard().button(QWizard.NextButton).setEnabled(True)
            self.wizard().button(QWizard.CancelButton).setEnabled(True)


class SetupWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annotation Setup Wizard")

        # Add pages
        self.addPage(LoginPage())
        self.addPage(ObjectSelectionPage())
        self.addPage(DatasetSelectionPage())

        # Set wizard style
        self.setWizardStyle(QWizard.ModernStyle)

        # Store properties for dataset path and name
        self.setProperty("dataset_path", None)
        self.setProperty("dataset_name", None)

        # Resize to reasonable dimensions
        self.resize(700, 500)

    def getCompletedValues(self):
        """Get all user-selected values from the wizard"""
        # Get values from registered fields
        username = self.field("username")
        port = self.field("port")
        pps = self.field("pps")
        imgs_to_smpl = self.field("imgs_to_smpl")
        # dummy = self.field("dummy")
        # debug = self.field("debug")

        # Determine object selection
        if self.field("new_object_checkbox"):
            selected_object = self.field("new_object_name")
            is_new_object = True
        else:
            selected_object = self.field("object_combo")
            is_new_object = False

        # Get dataset info
        dataset_name = self.property("dataset_name")
        dataset_path = self.property("dataset_path")

        return {
            "username": username,
            "port": port,
            # "dummy": dummy,
            # "debug": debug,
            "object": selected_object,
            "is_new_object": is_new_object,
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "pps": pps,
            "imgs_to_smpl": imgs_to_smpl,
        }


def run_startup_wizard():
    """Run the setup wizard and handle the final dataset sync step"""
    # Create and run the wizard
    wizard = SetupWizard()
    if wizard.exec_() != QWizard.Accepted:
        return None

    # Get all values from the wizard
    values = wizard.getCompletedValues()

    # Step 3: Check if dataset needs syncing
    dataset_path = values["dataset_path"]
    if dataset_path and os.path.exists(dataset_path):
        # Pass object information to the sync dialog
        sync_dialog = DatasetSyncDialog(
            dataset_path,
            values["port"],
            object_name=values["object"],
            is_new_object=values["is_new_object"],
            pps=values["pps"],
        )
        if sync_dialog.exec_() != QDialog.Accepted:
            return None

    # Create and return a namespace object with all settings
    return types.SimpleNamespace(
        username=values["username"],
        port=values["port"],
        dummy=False,
        debug=False,
        object=values["object"],
        is_new_object=values["is_new_object"],
        dataset=values["dataset"],
        dataset_path=values["dataset_path"],
        pps=values["pps"],
        imgs_to_smpl=values["imgs_to_smpl"],
    )
