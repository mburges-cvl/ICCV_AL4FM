import time
import requests
from PIL import Image

# Set the maximum number of image pixels for PIL
Image.MAX_IMAGE_PIXELS = None

from PyQt5.QtCore import pyqtSignal, QObject


class TaskWorker(QObject):
    """Handles network tasks asynchronously using PyQt signals.

    This worker performs GET or POST requests to a given URL and
    emits signals upon starting, completion, and result retrieval.

    Signals:
        started (int): Emitted when the task starts.
        finished (int): Emitted when the task finishes.
        resultReady (int, object): Emitted with the task index and result data.

    Args:
        task_index (int): The index of the task.
        url (str): The URL to request data from.
        payload (dict, optional): The JSON payload for a POST request. Defaults to None.
    """

    started = pyqtSignal(int)
    finished = pyqtSignal(int)
    resultReady = pyqtSignal(int, object)  # New signal for the result

    def __init__(self, task_index, url, payload=None):
        super().__init__()
        self.task_index = task_index
        self.url = url
        self.payload = payload
        print(f"Task {task_index} created")

    def run(self):
        """Execute the network request and emit signals accordingly.

        Performs a GET or POST request based on whether a payload is provided.
        Emits the `started`, `resultReady`, and `finished` signals.
        """
        print(f"Task {self.task_index} started")
        self.started.emit(self.task_index)
        try:
            if self.payload is not None:
                response = requests.post(self.url, json=self.payload, timeout=1200)
            else:
                response = requests.get(self.url, timeout=1200)
            result = response.json()
            self.resultReady.emit(self.task_index, result)
        except Exception as e:
            print(f"Error in task {self.task_index} ({self.url}):", e)
            self.resultReady.emit(self.task_index, None)
        self.finished.emit(self.task_index)


class LocalTaskWorker(QObject):
    """Simulates a local task with a delay, emitting signals upon execution.

    This worker sleeps for a given duration to simulate a task and
    then emits signals upon starting, completion, and result retrieval.

    Signals:
        started (int): Emitted when the task starts.
        finished (int): Emitted when the task finishes.
        resultReady (int, object): Emitted with the task index and result data.

    Args:
        task_index (int): The index of the task.
        delay (int): The time in seconds to simulate task execution.
        result (object): The result to emit after the task completes.
    """

    started = pyqtSignal(int)
    finished = pyqtSignal(int)
    resultReady = pyqtSignal(int, object)

    def __init__(self, task_index, delay, result):
        super().__init__()
        self.task_index = task_index
        self.delay = delay
        self.result = result

    def run(self):
        """Simulate a local task execution and emit signals accordingly.

        Sleeps for the specified delay duration, then emits the `resultReady` and
        `finished` signals with the provided result.
        """
        self.started.emit(self.task_index)
        time.sleep(self.delay)
        self.resultReady.emit(self.task_index, self.result)
        self.finished.emit(self.task_index)
