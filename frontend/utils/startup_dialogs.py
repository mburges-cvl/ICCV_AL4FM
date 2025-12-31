"""
Startup Wizard Dialog for the Active Learning Frontend.

This module provides a wizard dialog that collects necessary configuration
parameters before launching the main application window.
"""

import os
from types import SimpleNamespace

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QFormLayout,
    QSpinBox,
    QGroupBox,
)


class StartupWizard(QDialog):
    """Dialog to configure the application before starting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Active Learning - Setup Wizard")
        self.setMinimumWidth(500)
        self.result_args = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Server Configuration Group
        server_group = QGroupBox("Server Configuration")
        server_layout = QFormLayout()

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8005)
        server_layout.addRow("Server Port:", self.port_spin)

        self.username_edit = QLineEdit()
        self.username_edit.setText(os.getenv("USER", "user"))
        server_layout.addRow("Username:", self.username_edit)

        server_group.setLayout(server_layout)
        layout.addWidget(server_group)

        # Dataset Configuration Group
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QFormLayout()

        # Dataset path selection
        path_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Select dataset folder...")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_dataset)
        path_layout.addWidget(self.dataset_path_edit)
        path_layout.addWidget(self.browse_button)
        dataset_layout.addRow("Dataset Path:", path_layout)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # Object Configuration Group
        object_group = QGroupBox("Object Configuration")
        object_layout = QFormLayout()

        self.object_combo = QComboBox()
        self.object_combo.setEditable(True)
        self.object_combo.setPlaceholderText("Enter or select object name...")
        object_layout.addRow("Object Name:", self.object_combo)

        self.new_object_check = QCheckBox("This is a new object (start fresh)")
        self.new_object_check.setChecked(True)
        object_layout.addRow("", self.new_object_check)

        object_group.setLayout(object_layout)
        layout.addWidget(object_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.validate_and_accept)
        self.start_button.setDefault(True)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)

    def browse_dataset(self):
        """Open a file dialog to select the dataset folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly,
        )
        if folder:
            self.dataset_path_edit.setText(folder)
            # Load available objects from server workdir if exists
            self.load_available_objects()

    def load_available_objects(self):
        """Load previously trained objects from server workdir."""
        # Check if server_workdir exists and has trained objects
        workdir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "backend",
            "server_workdir",
        )
        if os.path.exists(workdir):
            self.object_combo.clear()
            for item in os.listdir(workdir):
                item_path = os.path.join(workdir, item)
                if os.path.isdir(item_path):
                    self.object_combo.addItem(item)

    def validate_and_accept(self):
        """Validate inputs and accept the dialog."""
        dataset_path = self.dataset_path_edit.text().strip()
        object_name = self.object_combo.currentText().strip()

        # Validate dataset path
        if not dataset_path:
            QMessageBox.warning(self, "Validation Error", "Please select a dataset path.")
            return

        if not os.path.isdir(dataset_path):
            QMessageBox.warning(
                self, "Validation Error", f"Dataset path does not exist:\n{dataset_path}"
            )
            return

        # Check for required files
        images_path = os.path.join(dataset_path, "images")
        annotations_path = os.path.join(dataset_path, "annotations.json")

        if not os.path.isdir(images_path):
            QMessageBox.warning(
                self,
                "Validation Error",
                "Dataset folder must contain an 'images' subfolder.",
            )
            return

        if not os.path.isfile(annotations_path):
            QMessageBox.warning(
                self,
                "Validation Error",
                "Dataset folder must contain 'annotations.json'.",
            )
            return

        # Validate object name
        if not object_name:
            QMessageBox.warning(self, "Validation Error", "Please enter an object name.")
            return

        # Create result args
        self.result_args = SimpleNamespace(
            port=self.port_spin.value(),
            username=self.username_edit.text().strip(),
            dataset_path=dataset_path,
            dataset=os.path.basename(dataset_path),
            object=object_name,
            is_new_object=self.new_object_check.isChecked(),
        )

        self.accept()


def run_startup_wizard():
    """
    Run the startup wizard dialog and return the configuration.

    Returns:
        SimpleNamespace: Configuration args if accepted, None if cancelled.
    """
    wizard = StartupWizard()
    if wizard.exec_() == QDialog.Accepted:
        return wizard.result_args
    return None
