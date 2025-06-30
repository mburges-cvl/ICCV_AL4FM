import json
import os
import sys
import time

import numpy as np
import requests
from PIL import Image, ImageEnhance

# Set the maximum number of image pixels for PIL
Image.MAX_IMAGE_PIXELS = None

try:
    import rasterio
except ImportError:
    rasterio = None

from PyQt5.QtCore import Qt, QRectF, QThread, QTimer
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QPen,
    QColor,
    QKeySequence,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGraphicsScene,
    QProgressBar,
    QSlider,
    QShortcut,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QGraphicsPixmapItem,
    QButtonGroup,
    QRadioButton,
    QLabel,
    QSizePolicy,
    QMessageBox,
    QGraphicsView,
    QSplitter,
    QDialog,
    QFormLayout,
    QSpinBox,
    QCheckBox,
    QDialogButtonBox,
)

from utils.startup_dialogs import run_startup_wizard
from utils.interactive_bounding_box import (
    MapPlotCanvas,
    InteractiveBox,
    CombinedButtonItem,
)
from utils.task_worker import TaskWorker
from utils.synced_graphicsview import SyncedGraphicsView


class MainWindow(QMainWindow):
    """Main application window for managing image annotations in an active learning workflow."""

    def __init__(self, args):
        """Initialize the main window with the provided arguments.
        Args:
            args: Command-line arguments containing configuration settings.
        """
        super().__init__()
        self.init_data(args)
        self.init_ui(args)
        self.load_initial_dataset()
        self.show()

    def init_data(self, args):
        """Initialize data and settings for the main window."""
        # Basic settings
        self.port = args.port
        self.username = args.username
        self.resize(1920, 1080)

        # Image management and annotations
        self.all_images_list = []
        self.active_images = []
        self.current_image_index = 0
        self.current_image_id = None
        self.image_annotations = {}
        self.image_undone_annotations = {}
        self.next_box_id = 0

        # Dataset parameters
        self.dataset_path = args.dataset_path
        self.object_name = args.object
        self.is_new_object = args.is_new_object

        # Image enhancement parameters
        self.contrast_value = 1.0
        self.saturation_value = 1.0
        self.brightness_value = 1.0

    def init_ui(self, args):
        """Initialize the user interface components of the main window."""

        # Set up the central widget and overall layout.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Build and add the three main sections.
        main_layout.addWidget(self.create_top_layout())
        main_layout.addWidget(self.create_middle_layout())
        main_layout.addWidget(self.create_bottom_layout())

        self.setWindowTitle(f"Active Learning - {args.object} - {args.dataset}")
        self.setup_shortcuts()

    def create_top_layout(self):
        """Create the top layout with navigation buttons, save button, and mode selection."""
        top_widget = QWidget()
        top_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(top_widget)

        # Navigation buttons and counter.
        self.prev_button = QPushButton("< Prev")
        self.prev_button.clicked.connect(self.load_previous_image)
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(self.load_next_image)
        self.counter_label = QLabel("0/0")
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.counter_label)

        # Save and mode selection.
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_annotations)
        self.pan_button = QRadioButton("Pan")
        self.pan_button.setChecked(True)
        self.draw_box_button = QRadioButton("Draw Box")
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.setExclusive(True)
        self.mode_button_group.addButton(self.pan_button)
        self.mode_button_group.addButton(self.draw_box_button)
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
        layout.addWidget(self.save_button)
        layout.addWidget(self.pan_button)
        layout.addWidget(self.draw_box_button)

        # Run button.
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.annotate_smart)
        layout.addWidget(self.run_button)

        # Confidence slider.
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.valueChanged.connect(
            self.on_confidence_slider_value_changed
        )
        self.confidence_timer = QTimer(self)
        self.confidence_timer.setSingleShot(True)
        self.confidence_timer.setInterval(50)
        self.confidence_timer.timeout.connect(self.do_confidence_filter)
        layout.addWidget(self.confidence_slider)

        self._last_confidence_slider_value = 0

        # Toggle annotations.
        self.hide_annotations_button = QPushButton("Hide anns")
        self.hide_annotations_button.setCheckable(True)
        self.hide_annotations_button.toggled.connect(self.toggle_annotations_visibility)
        layout.addWidget(self.hide_annotations_button)

        # Enhancement buttons.
        self.increase_contrast_button = QPushButton("Cont +")
        self.decrease_contrast_button = QPushButton("Cont -")
        self.increase_saturation_button = QPushButton("Sat +")
        self.decrease_saturation_button = QPushButton("Sat -")
        self.increase_brightness_button = QPushButton("Bright +")
        self.decrease_brightness_button = QPushButton("Bright -")
        self.reset_effects_button = QPushButton("Reset VE")
        self.increase_contrast_button.clicked.connect(
            lambda: self.update_effects("contrast", 1.1)
        )
        self.decrease_contrast_button.clicked.connect(
            lambda: self.update_effects("contrast", 0.9)
        )
        self.increase_saturation_button.clicked.connect(
            lambda: self.update_effects("saturation", 1.1)
        )
        self.decrease_saturation_button.clicked.connect(
            lambda: self.update_effects("saturation", 0.9)
        )
        self.increase_brightness_button.clicked.connect(
            lambda: self.update_effects("brightness", 1.1)
        )
        self.decrease_brightness_button.clicked.connect(
            lambda: self.update_effects("brightness", 0.9)
        )
        self.reset_effects_button.clicked.connect(self.reset_effects)
        layout.addWidget(self.increase_contrast_button)
        layout.addWidget(self.decrease_contrast_button)
        layout.addWidget(self.increase_saturation_button)
        layout.addWidget(self.decrease_saturation_button)
        layout.addWidget(self.increase_brightness_button)
        layout.addWidget(self.decrease_brightness_button)
        layout.addWidget(self.reset_effects_button)

        return top_widget

    def create_middle_layout(self):
        """Create the middle layout with synchronized image views."""
        images_widget = QWidget()
        images_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create three synchronized graphics views.
        self.view1 = self.setup_graphics_view("sample_image/waffle_homes.jpg")
        self.view2 = self.setup_graphics_view("sample_image/waffle_homes.jpg")
        self.view3 = self.setup_graphics_view("sample_image/waffle_homes.jpg")

        # Connect transform changes.
        self.view1.transformChanged.connect(self.sync_transforms_from_view1)
        self.view2.transformChanged.connect(self.sync_transforms_from_view2)
        self.view3.transformChanged.connect(self.sync_transforms_from_view3)

        # Use a splitter to lay out the views.
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view1)
        splitter.addWidget(self.view2)
        splitter.addWidget(self.view3)

        layout = QVBoxLayout(images_widget)
        layout.addWidget(splitter)
        return images_widget

    def create_bottom_layout(self):
        """Create the bottom layout with information and map plots."""
        info_widget = QWidget()
        info_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(info_widget)

        self.map_plot_ap50 = MapPlotCanvas(width=2, height=1, dpi=100)
        self.map_plot_ap75 = MapPlotCanvas(width=2, height=1, dpi=100)
        self.ap50_label = QLabel("AP50:")
        self.ap75_label = QLabel("AP75:")
        self.bbox_label = QLabel("Boxes: --")

        layout.addWidget(self.ap50_label)
        layout.addWidget(self.map_plot_ap50)
        layout.addWidget(self.ap75_label)
        layout.addWidget(self.map_plot_ap75)
        layout.addWidget(self.bbox_label)

        return info_widget

    def setup_graphics_view(self, image_path):
        """Helper method to initialize a graphics view with a sample image."""
        view = SyncedGraphicsView()
        view.setAlignment(Qt.AlignCenter)
        scene = QGraphicsScene()
        view.setScene(scene)
        pixmap = QPixmap(image_path)
        scene.addItem(QGraphicsPixmapItem(pixmap))
        view.boxDrawn.connect(lambda rect: self.add_bounding_box(rect))
        return view

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for various actions."""
        # Navigation shortcuts.
        self.shortcut_next = QShortcut(QKeySequence("n"), self)
        self.shortcut_next.activated.connect(self.load_next_image)
        self.shortcut_prev = QShortcut(QKeySequence("p"), self)
        self.shortcut_prev.activated.connect(self.load_previous_image)
        # Toggle drawing/panning.
        self.shortcut_toggle_mode = QShortcut(QKeySequence("b"), self)
        self.shortcut_toggle_mode.activated.connect(self.toggle_pan_draw_mode)
        # Toggle annotations visibility.
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_space.activated.connect(self.hide_annotations_button.toggle)

    def load_initial_dataset(self):
        """Load the initial dataset if a path is provided."""
        if self.dataset_path and os.path.exists(self.dataset_path):
            self.load_dataset_from_path(self.dataset_path)
        self.set_new_results(0.0, 0.0)

    def set_new_results(self, mAP50, mAP75):
        """
        Update the map plots and labels with new results.
        Args:
            mAP50: Mean Average Precision at IoU=0.50.
            mAP75: Mean Average Precision at IoU=0.75.
        """

        self.map_plot_ap50.update_plot(mAP50)
        self.map_plot_ap75.update_plot(mAP50)

        self.ap50_label.setText(f"AP50: {mAP50:.2f}")
        self.ap75_label.setText(f"AP75: {mAP75:.2f}")

        self._last_map_50 = mAP50
        self._last_map_75 = mAP75

    def toggle_pan_draw_mode(self):
        """
        Toggles between panning and drawing modes by switching the checked radio button
        and explicitly calling the mode change handler.
        """
        if self.pan_button.isChecked():
            self.draw_box_button.setChecked(True)
            self.on_mode_changed(self.draw_box_button)  # explicitly update the mode
        else:
            self.pan_button.setChecked(True)
            self.on_mode_changed(self.pan_button)  # explicitly update the mode

    def on_mode_changed(self, button):
        """
        Handles the mode change event when the user selects a different mode (pan or draw box).
        Args:
            button: The button that was clicked (either pan or draw box).
        """

        if button == self.draw_box_button:
            self.toggle_draw_box()
        elif button == self.pan_button:
            self.toggle_pan()

    def toggle_draw_box(self):
        """
        Toggles the drawing box mode on or off for all views.
        When enabled, the user can draw bounding boxes on the image.
        When disabled, the user can pan the image.
        """

        enable = self.draw_box_button.isChecked()
        for v in (self.view1, self.view2, self.view3):
            drag_mode = (
                QGraphicsView.RubberBandDrag if enable else QGraphicsView.ScrollHandDrag
            )
            v.setDragMode(drag_mode)
            v.setDrawingBoxMode(enable)

    def toggle_pan(self):
        """
        Toggles the panning mode on or off for all views.
        When enabled, the user can pan the image.
        When disabled, the user can draw bounding boxes on the image.
        """

        for v in (self.view1, self.view2, self.view3):
            v.setDragMode(QGraphicsView.ScrollHandDrag)
            v.setDrawingBoxMode(False)

    def sync_transforms_from_view1(self):
        """
        Synchronize the transforms of view2 and view3 with view1.
        This ensures that all three views display the same image region and zoom level.
        """
        t = self.view1.transform()
        self.view2.blockSignals(True)
        self.view3.blockSignals(True)
        self.view2.setTransform(t)
        self.view3.setTransform(t)
        self.view2.horizontalScrollBar().setValue(
            self.view1.horizontalScrollBar().value()
        )
        self.view2.verticalScrollBar().setValue(self.view1.verticalScrollBar().value())
        self.view3.horizontalScrollBar().setValue(
            self.view1.horizontalScrollBar().value()
        )
        self.view3.verticalScrollBar().setValue(self.view1.verticalScrollBar().value())
        self.view2.blockSignals(False)
        self.view3.blockSignals(False)

    def sync_transforms_from_view2(self):
        """
        Synchronize the transforms of view1 and view3 with view2.
        This ensures that all three views display the same image region and zoom level.
        """
        t = self.view2.transform()
        self.view1.blockSignals(True)
        self.view3.blockSignals(True)
        self.view1.setTransform(t)
        self.view3.setTransform(t)
        self.view1.horizontalScrollBar().setValue(
            self.view2.horizontalScrollBar().value()
        )
        self.view1.verticalScrollBar().setValue(self.view2.verticalScrollBar().value())
        self.view3.horizontalScrollBar().setValue(
            self.view2.horizontalScrollBar().value()
        )
        self.view3.verticalScrollBar().setValue(self.view2.verticalScrollBar().value())
        self.view1.blockSignals(False)
        self.view3.blockSignals(False)

    def sync_transforms_from_view3(self):
        """
        Synchronize the transforms of view1 and view2 with view3.
        This ensures that all three views display the same image region and zoom level.
        """
        t = self.view3.transform()
        self.view1.blockSignals(True)
        self.view2.blockSignals(True)
        self.view1.setTransform(t)
        self.view2.setTransform(t)
        self.view1.horizontalScrollBar().setValue(
            self.view3.horizontalScrollBar().value()
        )
        self.view1.verticalScrollBar().setValue(self.view3.verticalScrollBar().value())
        self.view2.horizontalScrollBar().setValue(
            self.view3.horizontalScrollBar().value()
        )
        self.view2.verticalScrollBar().setValue(self.view3.verticalScrollBar().value())
        self.view1.blockSignals(False)
        self.view2.blockSignals(False)

    def load_next_image(self):
        """Load the next image in the active images list."""

        if not self.active_images:
            return
        self.log_event(
            "image_switch", direction="next", current_image=self.current_image_id
        )
        self.current_image_index += 1
        if self.current_image_index >= len(self.active_images):
            self.current_image_index = 0
        self.display_current_image()
        QTimer.singleShot(0, self.do_confidence_filter)

    def load_previous_image(self):
        """Load the previous image in the active images list."""
        if not self.active_images:
            return
        self.log_event(
            "image_switch", direction="previous", current_image=self.current_image_id
        )
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.active_images) - 1
        self.display_current_image()
        QTimer.singleShot(0, self.do_confidence_filter)

    def toggle_annotations_visibility(self, checked):
        """
        When the button is checked, hide all annotation boxes;
        when unchecked, show them.
        """
        # Iterate over all views
        for view in (self.view1, self.view2, self.view3):
            # Loop through all items in the scene.
            for item in view.scene().items():
                # If the item is an InteractiveBox (i.e. an annotation box)
                if isinstance(item, InteractiveBox):
                    item.setVisible(not checked)  # hide if checked, show if not
        # Update the button text accordingly
        if checked:
            self.hide_annotations_button.setText("Show anns")
        else:
            self.hide_annotations_button.setText("Hide anns")
            QTimer.singleShot(0, self.do_confidence_filter)

    def on_confidence_slider_value_changed(self, value):
        """
        Handle the confidence slider value change event.
        This method is called when the user adjusts the confidence slider.
        Args:
            value: The new value of the confidence slider.
        """

        # Save the latest value and restart the timer.
        self._last_confidence_slider_value = value
        self.confidence_timer.start()

    def do_confidence_filter(self):
        """
        Apply the confidence filter to the predicted boxes based on the slider value.
        This method is called when the confidence slider value changes.
        It filters the predicted boxes in all views based on the selected confidence threshold.
        """

        # Use the stored slider value.
        self.filter_certain_positive_boxes(self._last_confidence_slider_value)

    def filter_certain_positive_boxes(self, value):
        """
        Filter the predicted boxes based on the confidence threshold set by the slider.
        Args:
            value: The confidence threshold value (0-100) set by the slider.
        """

        # Convert the slider's integer value to a float between 0 and 1.
        threshold = value / 100.0
        # For each view, update the visibility of "certain_positive" prediction boxes.
        for view in (self.view1, self.view2, self.view3):
            for item in view.scene().items():
                if isinstance(item, InteractiveBox):
                    # Check if the box is a predicted 'certain_positive' box.
                    if (
                        item.box_type == "predicted"
                        and item.predicted_class == "certain_positive"
                    ):
                        # If its confidence is below the threshold, hide it.
                        if item.confidence is not None and item.confidence < threshold:
                            item.setVisible(False)
                        else:
                            item.setVisible(True)

    # Add this new method to handle dataset loading from path
    def load_dataset_from_path(self, folder):
        """Load a dataset from a specified path"""
        print(f"Loading dataset from path: {folder}")
        images_path = os.path.join(folder, "images")
        annotation_path = os.path.join(folder, "annotations.json")

        if not os.path.isdir(images_path) or not os.path.isfile(annotation_path):
            QMessageBox.warning(
                self,
                "Error",
                "Dataset folder must contain 'images' subfolder and 'annotations.json'.",
            )
            return

        self.dataset_path = folder

        with open(annotation_path, "r") as f:
            self.coco_data = json.load(f)
        if "images" not in self.coco_data:
            QMessageBox.warning(
                self, "Error", "Annotation file does not contain image information."
            )
            return
        self.all_images_list = sorted(
            self.coco_data["images"], key=lambda img: img["id"]
        )

        self.setEnabled(False)
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Loading Dataset")
        layout = QVBoxLayout(progress_dialog)
        label = QLabel(
            f"Loading dataset '{os.path.basename(folder)}'...", progress_dialog
        )
        layout.addWidget(label)
        progress_bar = QProgressBar(progress_dialog)
        progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(progress_bar)
        progress_dialog.setLayout(layout)
        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Get initial active image ids from server
            response = requests.post(
                f"http://localhost:{self.port}/get_next_active_image_ids",
                json={
                    "folder": folder,
                    "num_images": 10,
                    "username": self.username,
                    "object": self.object_name,
                    "is_new_object": self.is_new_object,
                },
                timeout=600,
            )
            data = response.json()
            if "error" in data:
                QMessageBox.warning(self, "Error", data["error"])
                return
            if "new_training_ids" not in data:
                QMessageBox.warning(
                    self, "Error", "Server response missing active image ids."
                )
                return

            active_ids = data["new_training_ids"]
            self.active_images = [
                img for img in self.all_images_list if img["id"] in active_ids
            ]

            self.image_annotations = {}
            # Process certain_boxes from the server response
            certain_boxes = data.get("certain_boxes", {})
            for key, anns in certain_boxes.items():
                try:
                    img_id = int(key)
                except:
                    img_id = key
                if img_id not in self.image_annotations:
                    self.image_annotations[img_id] = []
                for ann in anns:
                    bbox = ann.get("bbox", [])
                    if len(bbox) == 4:
                        ann["x"] = bbox[0]
                        ann["y"] = bbox[1]
                        ann["width"] = bbox[2]
                        ann["height"] = bbox[3]
                    ann["group_id"] = self.next_box_id
                    self.next_box_id += 1
                    ann["type"] = "predicted"
                    ann["predicted_class"] = "certain_positive"
                    self.image_annotations[img_id].append(ann)

            self.current_image_index = 0
            self.image_undone_annotations = {}
            self.display_current_image()
            QTimer.singleShot(0, self.do_confidence_filter)

        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Failed to retrieve active image ids from server: {e}"
            )
        finally:
            progress_dialog.close()
            self.setEnabled(True)

    def normalize_image_mean_std(self, image, num_std=3):
        """Normalize the image using mean and standard deviation."""
        image = np.transpose(image, (1, 2, 0))
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        std = np.std(image, axis=(0, 1), keepdims=True)
        scale = 255 / (2 * num_std * std)
        normalized_image = (image - (mean - num_std * std)) * scale
        normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
        normalized_image = np.transpose(normalized_image, (2, 0, 1))
        return normalized_image

    def update_effects(self, effect, factor):
        """
        Update the image effects (contrast, saturation, brightness) based on the specified factor.
        Args:
            effect: The type of effect to update ("contrast", "saturation", "brightness").
            factor: The factor by which to adjust the effect (e.g., 1.1 for increase, 0.9 for decrease).
        1.0 means no change.
        """
        if effect == "contrast":
            self.contrast_value *= factor
        elif effect == "saturation":
            self.saturation_value *= factor
        elif effect == "brightness":
            self.brightness_value *= factor
        self.apply_image_effects()

    def reset_effects(self):
        """Reset the image effects (contrast, saturation, brightness) to their default values."""
        self.contrast_value = 1.0
        self.saturation_value = 1.0
        self.brightness_value = 1.0
        self.apply_image_effects()

    def apply_image_effects(self):
        """
        Apply the image effects (contrast, saturation, brightness) to the original images.
        """
        if not hasattr(self, "original_pil_images"):
            return

        updated_pil_images = []
        for i, pil_img in enumerate(self.original_pil_images):
            img = pil_img.copy()

            # Apply contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast_value)
            # Apply saturation (using ImageEnhance.Color)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.saturation_value)
            # Apply brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness_value)

            updated_pil_images.append(img)

        # Update all views
        for i, view in enumerate(
            [self.view1, self.view2, self.view3][: len(updated_pil_images)]
        ):
            pixmap = self.pil_to_qpixmap(updated_pil_images[i])
            view.scene().clear()
            item = QGraphicsPixmapItem(pixmap)
            view.scene().addItem(item)

        # After loading the image, add any stored annotations for this image.
        if self.current_image_id in self.image_annotations:
            for ann in self.image_annotations[self.current_image_id]:
                self._create_box_from_annotation(ann)

    def display_current_image(self):
        """Display the current image in all views."""
        if not self.dataset_path or not self.active_images:
            return
        images_path = os.path.join(self.dataset_path, "images")
        image_info = self.active_images[self.current_image_index]
        self.current_image_id = image_info["id"]
        filename = image_info["file_name"]
        self.image_path = os.path.join(images_path, filename)
        ext = filename.lower().split(".")[-1]

        # Update counter label.
        self.counter_label.setText(
            f"{self.current_image_index + 1}/{len(self.active_images)}"
        )

        # Clear the scenes from all views.
        for v in (self.view1, self.view2, self.view3):
            v.scene().clear()

        # Load the image.
        if ext in ["jpg", "jpeg", "png"]:
            pil_img = Image.open(self.image_path).convert("RGB")
            self.original_pil_images = [pil_img.copy(), pil_img.copy(), pil_img.copy()]
            pixmap = self.pil_to_qpixmap(pil_img)
            item1 = QGraphicsPixmapItem(pixmap)
            self.view1.scene().addItem(item1)
            self.view1.show()
            item2 = QGraphicsPixmapItem(pixmap)
            self.view2.scene().addItem(item2)
            self.view2.show()
            item3 = QGraphicsPixmapItem(pixmap)
            self.view3.scene().addItem(item3)
            self.view3.show()
        elif ext in ["tif", "tiff"] and rasterio is not None:
            with rasterio.open(self.image_path) as src:
                data = src.read()
                data = self.normalize_image_mean_std(data)
            if data.ndim != 3:
                return
            bands_count, height, width = data.shape

            if bands_count == 3:
                print("RGB shape:", data.shape)
                rgb_array = np.moveaxis(data, 0, -1)
                rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
                self.original_pil_images = [
                    Image.fromarray(rgb_array),
                    Image.fromarray(rgb_array),
                    Image.fromarray(rgb_array),
                ]
                pixmap = self.numpy_to_qpixmap(rgb_array)
                item1 = QGraphicsPixmapItem(pixmap)
                self.view1.scene().addItem(item1)
                self.view1.show()
                item2 = QGraphicsPixmapItem(pixmap)
                self.view2.scene().addItem(item2)
                self.view2.show()
                item3 = QGraphicsPixmapItem(pixmap)
                self.view3.scene().addItem(item3)
                self.view3.show()
            elif bands_count == 4:
                print("RGBN shape:", data.shape)
                rgb_array = np.moveaxis(data[:3, :, :], 0, -1)
                nir_array = np.repeat(np.moveaxis(data[3:4, :, :], 0, -1), 3, axis=2)

                self.original_pil_images = [
                    Image.fromarray(rgb_array),
                    Image.fromarray(nir_array),
                ]
                pixmap_rgb = self.numpy_to_qpixmap(rgb_array)
                pixmap_nir = self.numpy_to_qpixmap(nir_array)
                item1 = QGraphicsPixmapItem(pixmap_rgb)
                self.view1.scene().addItem(item1)
                self.view1.show()
                item2 = QGraphicsPixmapItem(pixmap_nir)
                self.view2.scene().addItem(item2)
                self.view2.show()
            elif bands_count == 8:
                print("RGBNA shape:", data.shape)
                rgb_array = np.moveaxis(data[[4, 2, 1], :, :], 0, -1)
                rgb_array_other = np.moveaxis(data[[5, 3, 0], :, :], 0, -1)
                nir_array = np.repeat(np.moveaxis(data[7:8, :, :], 0, -1), 3, axis=2)
                self.original_pil_images = [
                    Image.fromarray(rgb_array),
                    Image.fromarray(nir_array),
                    Image.fromarray(rgb_array_other),
                ]
                pixmap_rgb = self.numpy_to_qpixmap(rgb_array)
                pixmap_nir = self.numpy_to_qpixmap(nir_array)
                pixmap_rgb_other = self.numpy_to_qpixmap(rgb_array_other)
                item1 = QGraphicsPixmapItem(pixmap_rgb)
                self.view1.scene().addItem(item1)
                self.view1.show()
                item2 = QGraphicsPixmapItem(pixmap_nir)
                self.view2.scene().addItem(item2)
                self.view2.show()
                item3 = QGraphicsPixmapItem(pixmap_rgb_other)
                self.view3.scene().addItem(item3)
                self.view3.show()

        self.contrast_value = 1.0
        self.saturation_value = 1.0
        self.brightness_value = 1.0

        self.update_bbox_label()

        # After loading the image, add any stored annotations for this image.
        if self.current_image_id in self.image_annotations:
            for ann in self.image_annotations[self.current_image_id]:
                self._create_box_from_annotation(ann)

    def _create_box_from_annotation(self, ann):
        """Create an interactive box from the given annotation."""
        rect = QRectF(ann["x"], ann["y"], ann["width"], ann["height"])
        box_type = ann.get("type", "positive")
        predicted_class = ann.get("predicted_class", None)
        confidence = ann.get("confidence", None)
        group_id = ann["group_id"]
        for view in (self.view1, self.view2, self.view3):
            ibox = InteractiveBox(
                rect, box_type, self, group_id, predicted_class, confidence
            )
            view.scene().addItem(ibox)

    def pil_to_qpixmap(self, pil_img):
        """Convert a PIL image to a QPixmap."""
        image = QImage(
            pil_img.tobytes("raw", "RGB"),
            pil_img.width,
            pil_img.height,
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(image)

    def numpy_to_qpixmap(self, array, is_gray=False):
        """Convert a NumPy array to a QPixmap."""
        height, width = array.shape[:2]
        if is_gray:
            gray_data = array[:, :, np.newaxis]
            gray_data = np.repeat(gray_data, 3, axis=2)
            gray_data = np.clip(gray_data, 0, 255).astype(np.uint8)
            qimage = QImage(
                gray_data.tobytes(), width, height, 3 * width, QImage.Format_RGB888
            )
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
            qimage = QImage(
                array.tobytes(), width, height, 3 * width, QImage.Format_RGB888
            )
        return QPixmap.fromImage(qimage)

    def update_bbox_label(self):
        """Update the bounding box counter label with the current count of positive annotations."""
        # Count only annotations of type "positive" for the current image.
        count = 0
        if self.current_image_id in self.image_annotations:
            count = sum(
                1
                for ann in self.image_annotations[self.current_image_id]
                if ann.get("type") == "positive"
            )
        self.bbox_label.setText(f"Boxes: {count}")

    def add_bounding_box(self, rect: QRectF):
        """
        Add a bounding box to the current image and update the annotations.
        """
        current_id = self.current_image_id
        group_id = self.next_box_id
        self.next_box_id += 1
        ann = {
            "group_id": group_id,
            "x": int(rect.x()),
            "y": int(rect.y()),
            "width": int(rect.width()),
            "height": int(rect.height()),
            "type": "positive",
        }
        if current_id not in self.image_annotations:
            self.image_annotations[current_id] = []
        self.image_annotations[current_id].append(ann)

        self.log_event("draw_box", image_id=current_id, group_id=group_id, bbox=ann)

        # Create interactive boxes for the current image.
        for view in (self.view1, self.view2, self.view3):
            ibox = InteractiveBox(rect, "positive", self, group_id)
            view.scene().addItem(ibox)
        # Clear undone annotations for this image.
        self.image_undone_annotations[current_id] = []

        # Update the bounding box counter after any change.
        self.update_bbox_label()

    def add_prediction_box(self, rect: QRectF, class_name: str, confidence=0.5):
        """Add a predicted bounding box.

        Args:
            rect (QRectF): Bounding box rectangle.
            class_name (str): Predicted class name.
            confidence (float, optional): Confidence score. Defaults to 0.5.
        """
        current_id = self.current_image_id
        group_id = self.next_box_id
        self.next_box_id += 1
        ann = {
            "group_id": group_id,
            "x": int(rect.x()),
            "y": int(rect.y()),
            "width": int(rect.width()),
            "height": int(rect.height()),
            "type": "predicted",
            "predicted_class": class_name,
            "confidence": confidence,
        }
        if current_id not in self.image_annotations:
            self.image_annotations[current_id] = []
        self.image_annotations[current_id].append(ann)
        for view in (self.view1, self.view2, self.view3):
            ibox = InteractiveBox(
                rect, "predicted", self, group_id, class_name, confidence
            )
            view.scene().addItem(ibox)

    def handle_box_action(self, group_id, action):
        """Handle actions on bounding boxes (delete, accept, reject).

        Args:
            group_id (int): Group ID of the annotation.
            action (str): Action to perform.
        """
        current_id = self.current_image_id
        group_ann = None
        for ann in self.image_annotations.get(current_id, []):
            if ann["group_id"] == group_id:
                group_ann = ann
                break
        if group_ann is None:
            return

        self.log_event(action, image_id=current_id, group_id=group_id)

        if action == "delete":
            # Remove the annotation.
            self.image_annotations[current_id] = [
                ann
                for ann in self.image_annotations[current_id]
                if ann["group_id"] != group_id
            ]
            for view in (self.view1, self.view2, self.view3):
                for item in view.scene().items():
                    if isinstance(item, InteractiveBox) and item.group_id == group_id:
                        view.scene().removeItem(item)
        elif action == "accept" and group_ann.get("type") == "predicted":
            group_ann["type"] = "positive"
            for view in (self.view1, self.view2, self.view3):
                for item in view.scene().items():
                    if isinstance(item, InteractiveBox) and item.group_id == group_id:
                        pen = QPen(QColor("green"))
                        pen.setStyle(Qt.SolidLine)
                        pen.setWidth(2)
                        item.setPen(pen)
                        if item.combined_button_item is not None:
                            scene = item.combined_button_item.scene()
                            if scene:
                                scene.removeItem(item.combined_button_item)
                        item.combined_button_item = CombinedButtonItem(
                            item, actions=[("❌", "delete")], parent=item
                        )
                        item.update_controls()
                        if item.confidence_label is not None:
                            item.confidence_label.setVisible(False)
        elif action == "reject" and group_ann.get("type") == "predicted":
            group_ann["type"] = "background"
            for view in (self.view1, self.view2, self.view3):
                for item in view.scene().items():
                    if isinstance(item, InteractiveBox) and item.group_id == group_id:
                        pen = QPen(QColor("red"))
                        pen.setStyle(Qt.DotLine)
                        pen.setWidth(2)
                        item.setPen(pen)
                        if item.combined_button_item is not None:
                            scene = item.combined_button_item.scene()
                            if scene:
                                scene.removeItem(item.combined_button_item)
                        item.combined_button_item = CombinedButtonItem(
                            item, actions=[("❌", "delete")], parent=item
                        )
                        item.update_controls()
                        if item.confidence_label is not None:
                            item.confidence_label.setVisible(False)
        else:
            print("Unknown action:", action)

        # Update the bounding box counter after any change.
        self.update_bbox_label()

    def annotate_smart(self):
        """Perform active learning annotation using a server-side process."""
        self.log_event("smart_annotation")
        self.setEnabled(False)

        # Open the parameter dialog first.
        training_params = self.getTrainingParameters()
        if training_params is None:
            # If the user cancelled, re-enable the widget and exit.
            self.setEnabled(True)
            return

        payload = {
            "image_annotations": self.image_annotations,
            "epochs": training_params["epochs"],
            "use_sam_masks": training_params["use_sam_masks"],
            "retrain": training_params["retrain"],
        }

        task_results = {}

        def onTaskResult(i, result):
            print("Task", i, "returned result:", result)
            task_results[i] = result

        tasks = [
            {
                "name": "Train Model",
                "url": f"http://localhost:{self.port}/train_model",
                "dependencies": [],
                "payload": payload,
            },
            {
                "name": "Extract Prototype Masks",
                "url": f"http://localhost:{self.port}/extract_prototype_masks",
                "dependencies": [0],
            },
            {
                "name": "Evaluate Model",
                "url": f"http://localhost:{self.port}/evaluate_model",
                "dependencies": [0],
            },
            {
                "name": "Extract Prototype Features",
                "url": f"http://localhost:{self.port}/extract_prototype_features",
                "dependencies": [0],
            },
            {
                "name": "Get New Training IDs",
                "url": f"http://localhost:{self.port}/get_new_training_ids_smart",
                "dependencies": [0, 1, 3],
            },
        ]

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Running Tasks")
        layout = QVBoxLayout(progress_dialog)

        task_timer_labels = []  # This will show elapsed time
        task_icon_labels = []
        for task in tasks:
            hbox = QHBoxLayout()
            task_label = QLabel(task["name"], progress_dialog)
            icon_label = QLabel("", progress_dialog)
            timer_label = QLabel("0.0s", progress_dialog)
            timer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            timer_label.setMinimumWidth(60)
            task_label.setMinimumWidth(200)
            hbox.addWidget(task_label)
            hbox.addStretch(1)
            hbox.addWidget(icon_label)
            hbox.addWidget(timer_label)
            layout.addLayout(hbox)
            task_icon_labels.append(icon_label)
            task_timer_labels.append(timer_label)

        progress_dialog.setLayout(layout)
        progress_dialog.show()
        QApplication.processEvents()

        tasks_started = [False] * len(tasks)
        tasks_finished = [False] * len(tasks)
        threads = [None] * len(tasks)
        task_start_times = {}

        def updateElapsedTime():
            current_time = time.time()
            for i in range(len(tasks)):
                if tasks_started[i] and not tasks_finished[i]:
                    elapsed = current_time - task_start_times.get(i, current_time)
                    task_timer_labels[i].setText(f"{elapsed:.1f}s")

        update_timer = QTimer(progress_dialog)
        update_timer.setInterval(1000)  # 1 second interval
        update_timer.timeout.connect(updateElapsedTime)
        update_timer.start()

        def updateTaskIcon(i, state):
            if state == "waiting":
                icon = "⏹️"
            elif state == "in-progress":
                icon = "🔄"
            elif state == "finished":
                icon = "✅"
            task_icon_labels[i].setText(icon)
            QApplication.processEvents()

        def checkAndStartTasks():
            for i, task in enumerate(tasks):
                if not tasks_started[i]:
                    if all(tasks_finished[j] for j in task["dependencies"]):
                        startTask(i)

        def startTask(i):
            tasks_started[i] = True
            updateTaskIcon(i, "in-progress")
            task_start_times[i] = time.time()

            thread = QThread()
            payload = tasks[i].get("payload")
            worker = TaskWorker(i, tasks[i]["url"], payload)
            worker.moveToThread(thread)
            thread.worker = worker
            worker.started.connect(lambda index=i: print(f"Task {index} started"))
            worker.resultReady.connect(
                lambda index, result: onTaskResult(index, result)
            )
            worker.finished.connect(lambda index=i: onTaskFinished(index))
            thread.started.connect(worker.run)
            thread.start()
            threads[i] = thread

        def onTaskFinished(i):
            tasks_finished[i] = True
            updateTaskIcon(i, "finished")
            checkAndStartTasks()
            if all(tasks_finished):
                update_timer.stop()
                if 4 in task_results:
                    new_data = task_results[4]
                    new_ids = new_data.get("new_training_ids", [])
                    certain_boxes = new_data.get("certain_boxes", {})
                    self.active_images = [
                        img for img in self.all_images_list if img["id"] in new_ids
                    ]
                    for gp_id, (key, anns) in enumerate(certain_boxes.items()):
                        try:
                            img_id = int(key)
                        except:
                            img_id = key
                        if img_id not in self.image_annotations:
                            self.image_annotations[img_id] = []
                        for ann in anns:
                            bbox = ann.get("bbox", [])
                            if len(bbox) == 4:
                                ann["x"] = bbox[0]
                                ann["y"] = bbox[1]
                                ann["width"] = bbox[2]
                                ann["height"] = bbox[3]
                            ann["group_id"] = self.next_box_id
                            self.next_box_id += 1
                            ann["type"] = "predicted"
                            ann["predicted_class"] = "certain_positive"
                            ann["confidence"] = ann.get("confidence", 0.3)
                            self.image_annotations[img_id].append(ann)
                    self.current_image_index = 0
                    self.display_current_image()
                    QTimer.singleShot(0, self.do_confidence_filter)

                progress_dialog.close()
                self.setEnabled(True)
                print("All task results:", task_results)
                ap50 = task_results.get(2, {}).get("AP50_all")
                ap75 = task_results.get(2, {}).get("AP75_all")
                if ap50 is not None and ap75 is not None:
                    # Update the plot: append new values and redraw.
                    self.set_new_results(ap50, ap75)
                else:
                    # Optionally handle the case where mAP is missing.
                    pass
                self.log_event("annotation_finished")
            threads[i].quit()
            threads[i].wait()

        checkAndStartTasks()
        progress_dialog.exec_()

        self.save_annotations()

    def getTrainingParameters(self):
        # Create a new dialog for entering additional parameters.
        dialog = QDialog(self)
        dialog.setWindowTitle("Training Parameters")

        # Use a QFormLayout to neatly arrange the input fields.
        layout = QFormLayout(dialog)

        # Input for Number of Epochs.
        epochsSpinBox = QSpinBox(dialog)
        epochsSpinBox.setMinimum(1)
        epochsSpinBox.setMaximum(1000)
        epochsSpinBox.setValue(25)  # default value
        layout.addRow("Number of Epochs:", epochsSpinBox)

        # Checkbox for using SAM masks.
        samCheckbox = QCheckBox(dialog)
        samCheckbox.setChecked(True)  # default value
        layout.addRow("Use SAM Masks:", samCheckbox)

        retrainCheckbox = QCheckBox(dialog)
        layout.addRow("Train from Scratch:", retrainCheckbox)

        # Dialog buttons (OK and Cancel).
        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog
        )
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)

        # Run the dialog modally.
        if dialog.exec_() == QDialog.Accepted:
            return {
                "epochs": epochsSpinBox.value(),
                "use_sam_masks": samCheckbox.isChecked(),
                "retrain": retrainCheckbox.isChecked(),
            }
        else:
            return None  # User canceled

    def save_annotations(self):
        """Save the current annotations to a file."""
        if not self.dataset_path:
            QMessageBox.warning(self, "Error", "No dataset loaded.")
            return
        annotation_path = os.path.join(
            "logs",
            f"annotations_{self.username}_localhost:{self.port}.json",
        )
        annotations_list = []
        # Iterate over the per-image annotations.
        for image in self.all_images_list:
            image_id = image["id"]
            for ann in self.image_annotations.get(image_id, []):
                # Determine category id (here we assume "positive" maps to 1 and others to 0)
                if ann.get("type") == "positive":
                    annotation_entry = {
                        "id": ann["group_id"],
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [ann["x"], ann["y"], ann["width"], ann["height"]],
                    }
                else:
                    continue
                annotations_list.append(annotation_entry)
        self.coco_data["annotations"] = annotations_list
        with open(annotation_path, "w") as f:
            json.dump(self.coco_data, f, indent=4)
        # QMessageBox.information(self, "Success", "Annotations saved successfully.")

    def log_event(self, event_type, **kwargs):
        """Log user actions and events.

        Args:
            event_type (str): Type of event.
            **kwargs: Additional event details.
        """
        event = {"timestamp": time.time(), "event_type": event_type, **kwargs}
        # Append to a log file as a JSON line
        with open(
            f"logs/user_activity_log_{self.username}_localhost:{self.port}.jsonl",
            "a",
        ) as f:
            f.write(json.dumps(event) + "\n")


def main():

    app = QApplication(sys.argv)
    # Use the new two-step dialog process
    args = run_startup_wizard()

    if args is None:  # User cancelled one of the dialogs
        sys.exit(0)

    window = MainWindow(args)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
