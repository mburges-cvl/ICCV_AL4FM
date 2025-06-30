from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QTextEdit,
    QPushButton,
    QHBoxLayout,
    QApplication,
)
import requests
import os
import hashlib
import time


class ExtractionWorker(QThread):
    """Thread worker for extracting masks from a dataset."""

    extraction_finished = pyqtSignal()
    extraction_error = pyqtSignal(str)

    def __init__(self, url, timeout=1800):
        """Initialize the extraction worker.

        Args:
            url (str): The endpoint URL for mask extraction.
            timeout (int, optional): Request timeout in seconds. Defaults to 1800.
        """
        super().__init__()
        self.url = url
        self.timeout = timeout

    def run(self):
        """Execute the extraction process."""
        try:
            response = requests.get(self.url, timeout=self.timeout)
            if response.status_code != 200:
                self.extraction_error.emit("Error starting mask extraction")
            else:
                self.extraction_finished.emit()
        except Exception as e:
            self.extraction_error.emit(str(e))


class DatasetSyncDialog(QDialog):
    """Dialog for synchronizing a dataset with a remote server."""

    def __init__(
        self,
        dataset_path,
        port,
        object_name=None,
        is_new_object=False,
        parent=None,
        pps=16,
    ):
        """Initialize the dataset synchronization dialog.

        Args:
            dataset_path (str): Path to the local dataset.
            port (int): Server port for dataset synchronization.
            object_name (str, optional): Name of the object (if applicable).
            is_new_object (bool, optional): Whether the object is newly created. Defaults to False.
            parent (QWidget, optional): Parent widget. Defaults to None.
            pps (int, optional): Pixels between points for mask extraction. Defaults to 16.
        """
        super().__init__(parent)
        self.setWindowTitle("Dataset Synchronization")
        self.dataset_path = dataset_path
        self.port = port
        self.dataset_name = os.path.basename(dataset_path)
        self.object_name = object_name
        self.is_new_object = is_new_object
        self.pps = pps

        layout = QVBoxLayout(self)

        self.status_label = QLabel(
            f"Checking if dataset '{self.dataset_name}' exists on server..."
        )
        layout.addWidget(self.status_label)

        self.counter_label = QLabel("Processed: 0 / 0 (Estimated time remaining: --)")
        self.counter_label.setVisible(False)
        layout.addWidget(self.counter_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setVisible(False)
        layout.addWidget(self.details_text)

        button_layout = QHBoxLayout()
        self.sync_button = QPushButton("Sync Dataset")
        self.sync_button.setEnabled(False)
        self.sync_button.clicked.connect(self.sync_dataset)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.continue_button = QPushButton("Continue")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.accept)

        # self.extract_masks_button = QPushButton("Extract Masks")
        # self.extract_masks_button.setEnabled(False)
        # self.extract_masks_button.clicked.connect(self.extract_masks)

        button_layout.addWidget(self.sync_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.continue_button)
        # button_layout.addWidget(self.extract_masks_button)

        layout.addLayout(button_layout)

        # Set up a timer to start the check after dialog is shown
        QTimer.singleShot(100, self.check_dataset_on_server)

        self.resize(500, 300)

    def check_dataset_on_server(self):
        """Check if dataset exists on server and compare file hashes"""
        try:
            # Request dataset status from server
            response = requests.post(
                f"http://localhost:{self.port}/check_dataset_status",
                json={"dataset_name": self.dataset_name},
                timeout=30,
            )

            if response.status_code != 200:
                self.status_label.setText(
                    f"Error checking dataset on server: {response.text}"
                )
                return

            data = response.json()

            if data.get("exists", False):
                self.status_label.setText(
                    f"Dataset '{self.dataset_name}' exists on server."
                )

                # Compare hashes to see if sync is needed
                missing_files = data.get("missing_files", [])
                different_files = data.get("different_files", [])

                if not missing_files and not different_files:
                    self.status_label.setText(
                        f"Dataset '{self.dataset_name}' is already synchronized."
                    )
                    self.continue_button.setEnabled(True)
                else:
                    self.status_label.setText(
                        f"Dataset '{self.dataset_name}' needs synchronization."
                    )
                    self.details_text.setVisible(True)

                    details = []
                    if missing_files:
                        details.append(
                            f"Missing files on server ({len(missing_files)}):"
                        )
                        for file in missing_files[
                            :10
                        ]:  # Show first 10 to avoid clutter
                            details.append(f"  - {file}")
                        if len(missing_files) > 10:
                            details.append(
                                f"  ... and {len(missing_files) - 10} more files."
                            )

                    if different_files:
                        details.append(
                            f"\nFiles with different content ({len(different_files)}):"
                        )
                        for file in different_files[:10]:
                            details.append(f"  - {file}")
                        if len(different_files) > 10:
                            details.append(
                                f"  ... and {len(different_files) - 10} more files."
                            )

                    self.details_text.setText("\n".join(details))
                    self.sync_button.setEnabled(True)
            else:
                self.status_label.setText(
                    f"Dataset '{self.dataset_name}' does not exist on server."
                )
                self.sync_button.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error checking dataset: {str(e)}")

    def sync_dataset(self):
        """Sync the dataset to the server, uploading either the entire dataset or just the changed files."""
        self.sync_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.counter_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Preparing to sync dataset '{self.dataset_name}'...")

        # First, check if the dataset folder exists on the server
        existence_response = requests.post(
            f"http://localhost:{self.port}/check_dataset_status",
            json={"dataset_name": self.dataset_name},
            timeout=30,
        )
        if existence_response.status_code != 200:
            self.status_label.setText(
                f"Error checking dataset existence: {existence_response.text}"
            )
            self.cancel_button.setEnabled(True)
            return
        existence_data = existence_response.json()
        if not existence_data.get("exists", False):
            self.status_label.setText(
                f"Dataset '{self.dataset_name}' does not exist on server. Uploading all files..."
            )
            files_to_sync = []
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    rel_path = os.path.relpath(
                        os.path.join(root, file), self.dataset_path
                    )
                    files_to_sync.append(rel_path)
        else:
            file_hashes = self.calculate_dataset_hashes()
            status_response = requests.post(
                f"http://localhost:{self.port}/check_dataset_status",
                json={"dataset_name": self.dataset_name, "file_hashes": file_hashes},
                timeout=60,
            )
            if status_response.status_code != 200:
                self.status_label.setText(
                    f"Error checking dataset status: {status_response.text}"
                )
                self.cancel_button.setEnabled(True)
                return
            status_data = status_response.json()
            missing_files = status_data.get("missing_files", [])
            different_files = status_data.get("different_files", [])
            files_to_sync = missing_files + different_files
            if not files_to_sync:
                self.status_label.setText(
                    f"Dataset '{self.dataset_name}' is already in sync."
                )
                self.continue_button.setEnabled(True)
                self.cancel_button.setEnabled(True)
                self.counter_label.setVisible(False)
                self.progress_bar.setVisible(False)
                return

        prepare_response = requests.post(
            f"http://localhost:{self.port}/prepare_dataset_sync",
            json={
                "dataset_name": self.dataset_name,
                "total_files": len(files_to_sync),
                "object_name": self.object_name if self.object_name else None,
                "is_new_object": self.is_new_object,
            },
            timeout=30,
        )
        if prepare_response.status_code != 200:
            self.status_label.setText(f"Error preparing sync: {prepare_response.text}")
            self.cancel_button.setEnabled(True)
            return

        total_files = len(files_to_sync)
        uploaded_files = 0
        start_time = time.time()
        last_update_time = start_time

        self.status_label.setText(f"Syncing {total_files} files to server...")

        for file_path in files_to_sync:
            full_path = os.path.join(self.dataset_path, file_path)
            with open(full_path, "rb") as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
            files = {"file": open(full_path, "rb")}
            data = {
                "dataset_name": self.dataset_name,
                "file_path": file_path,
                "md5_hash": md5_hash,
            }
            upload_response = requests.post(
                f"http://localhost:{self.port}/upload_dataset_file",
                files=files,
                data=data,
                timeout=300,
            )
            if upload_response.status_code != 200:
                self.status_label.setText(
                    f"Error uploading file {file_path}: {upload_response.text}"
                )
                self.cancel_button.setEnabled(True)
                return
            uploaded_files += 1
            progress_percent = (uploaded_files / total_files) * 100
            self.progress_bar.setValue(int(progress_percent))
            current_time = time.time()
            if (
                uploaded_files == 1
                or uploaded_files % 5 == 0
                or (current_time - last_update_time) > 2
            ):
                elapsed_time = current_time - start_time
                if uploaded_files > 1:
                    files_per_second = uploaded_files / elapsed_time
                    remaining_files = total_files - uploaded_files
                    estimated_remaining_time = (
                        remaining_files / files_per_second
                        if files_per_second > 0
                        else 0
                    )
                    mins = int(estimated_remaining_time // 60)
                    secs = int(estimated_remaining_time % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                    self.counter_label.setText(
                        f"Uploaded: {uploaded_files} / {total_files} (Estimated time remaining: {time_str})"
                    )
                    last_update_time = current_time
            QApplication.processEvents()

        finalize_response = requests.post(
            f"http://localhost:{self.port}/finalize_dataset_sync",
            json={"dataset_name": self.dataset_name},
            timeout=30,
        )
        if finalize_response.status_code == 200:
            self.status_label.setText(
                f"Dataset '{self.dataset_name}' successfully synchronized ({uploaded_files} files updated). Starting mask extraction..."
            )
            # Immediately trigger mask extraction instead of enabling a button.
            self.extract_masks()
        else:
            self.status_label.setText(
                f"Error finalizing sync: {finalize_response.text}"
            )

    def calculate_dataset_hashes(self):
        """Calculate MD5 hashes for all files in the dataset"""
        self.status_label.setText(
            f"Calculating file hashes for dataset '{self.dataset_name}'..."
        )
        self.counter_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        file_hashes = {}
        all_files = []

        # Get all files in the dataset
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), self.dataset_path)
                all_files.append(rel_path)

        total_files = len(all_files)
        processed_files = 0

        # Setup timing variables
        start_time = time.time()
        last_update_time = start_time

        for file_path in all_files:
            full_path = os.path.join(self.dataset_path, file_path)

            # Calculate MD5 hash
            try:
                with open(full_path, "rb") as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
                file_hashes[file_path] = md5_hash
            except Exception as e:
                self.details_text.append(f"Error hashing file {file_path}: {str(e)}")

            # Update progress
            processed_files += 1
            progress_percent = (processed_files / total_files) * 100
            self.progress_bar.setValue(int(progress_percent))

            # Update time estimate periodically
            current_time = time.time()
            if (
                processed_files == 1
                or processed_files % 5 == 0
                or (current_time - last_update_time) > 2
            ):
                elapsed_time = current_time - start_time
                if processed_files > 1:  # Avoid division by zero
                    files_per_second = processed_files / elapsed_time
                    remaining_files = total_files - processed_files
                    estimated_remaining_time = (
                        remaining_files / files_per_second
                        if files_per_second > 0
                        else 0
                    )

                    # Format time as minutes:seconds
                    mins = int(estimated_remaining_time // 60)
                    secs = int(estimated_remaining_time % 60)
                    time_str = f"{mins:02d}:{secs:02d}"

                    self.counter_label.setText(
                        f"Processed: {processed_files} / {total_files} (Estimated time remaining: {time_str})"
                    )
                    last_update_time = current_time

            QApplication.processEvents()

        # Hide counter when done
        self.counter_label.setVisible(False)

        return file_hashes

    def check_dataset_on_server(self):
        """Check if dataset exists on server. If it exists, then compare file hashes."""
        try:
            self.status_label.setText(
                f"Checking dataset '{self.dataset_name}' on server..."
            )
            # First, check for existence only
            response = requests.post(
                f"http://localhost:{self.port}/check_dataset_status",
                json={"dataset_name": self.dataset_name},
                timeout=30,
            )
            if response.status_code != 200:
                self.status_label.setText(
                    f"Error checking dataset on server: {response.text}"
                )
                return
            data = response.json()
            if not data.get("exists", False):
                self.status_label.setText(
                    f"Dataset '{self.dataset_name}' does not exist on server."
                )
                self.sync_button.setEnabled(True)
                return

            # Dataset exists: now calculate hashes and compare
            file_hashes = self.calculate_dataset_hashes()
            response = requests.post(
                f"http://localhost:{self.port}/check_dataset_status",
                json={"dataset_name": self.dataset_name, "file_hashes": file_hashes},
                timeout=60,
            )
            if response.status_code != 200:
                self.status_label.setText(
                    f"Error checking dataset on server: {response.text}"
                )
                return
            data = response.json()
            missing_files = data.get("missing_files", [])
            different_files = data.get("different_files", [])
            if not missing_files and not different_files:
                self.status_label.setText(
                    f"Dataset '{self.dataset_name}' is already synchronized."
                )
                self.continue_button.setEnabled(True)
            else:
                self.status_label.setText(
                    f"Dataset '{self.dataset_name}' needs synchronization."
                )
                self.details_text.setVisible(True)
                details = []
                if missing_files:
                    details.append(f"Missing files on server ({len(missing_files)}):")
                    for file in missing_files[:10]:
                        details.append(f"  - {file}")
                    if len(missing_files) > 10:
                        details.append(
                            f"  ... and {len(missing_files) - 10} more files."
                        )
                if different_files:
                    details.append(
                        f"\nFiles with different content ({len(different_files)}):"
                    )
                    for file in different_files[:10]:
                        details.append(f"  - {file}")
                    if len(different_files) > 10:
                        details.append(
                            f"  ... and {len(different_files) - 10} more files."
                        )
                self.details_text.setText("\n".join(details))
                self.sync_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Error checking dataset: {str(e)}")
            self.details_text.setVisible(True)
            self.details_text.setText(f"Exception details:\n{str(e)}")

    def extract_masks(self):
        """Initiate the mask extraction process on the server."""
        # self.extract_masks_button.setEnabled(False)
        self.status_label.setText("Starting mask extraction...")
        self.progress_bar.setValue(0)
        self.counter_label.setVisible(True)
        self.extraction_start_time = time.time()  # record start time

        # Call the extraction endpoint (which schedules the background task)
        try:
            # Use a short timeout because the endpoint now returns immediately.
            response = requests.post(
                f"http://localhost:{self.port}/extract_dense_masks",
                json={"pixels_between_points": int(self.pps)},
                timeout=10,
            )

            if response.status_code != 200:
                self.status_label.setText("Error starting mask extraction")
                return
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            return

        # Start polling the extraction progress every second
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.update_extraction_progress)
        self.poll_timer.start(1000)

    def update_extraction_progress(self):
        """Poll the server for mask extraction progress updates."""
        try:
            response = requests.get(
                f"http://localhost:{self.port}/extraction_progress", timeout=10
            )
            if response.status_code != 200:
                return
            progress_data = response.json()
            total = progress_data.get("total", 0)
            current = progress_data.get("current", 0)
            message = progress_data.get("message", "")

            if total > 0:
                progress_percent = int((current / total) * 100)
                self.progress_bar.setValue(progress_percent)
                elapsed = time.time() - self.extraction_start_time
                if current > 0:
                    estimated_total = (elapsed / current) * total
                    estimated_remaining = estimated_total - elapsed
                    mins = int(estimated_remaining // 60)
                    secs = int(estimated_remaining % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                else:
                    time_str = "--:--"
                self.counter_label.setText(
                    f"Processed: {current} / {total} (Estimated time remaining: {time_str})"
                )
            self.status_label.setText(message)

            if total > 0 and current >= total:
                self.poll_timer.stop()
                self.status_label.setText("Mask extraction complete")
                self.continue_button.setEnabled(True)
        except Exception as e:
            print("Error polling extraction progress:", e)
