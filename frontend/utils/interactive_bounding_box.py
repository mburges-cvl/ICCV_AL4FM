from PIL import Image

# Set the maximum number of image pixels for PIL
Image.MAX_IMAGE_PIXELS = None

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPen,
    QColor,
    QBrush,
    QFont,
    QPainter,
    QPainterPath,
    QFontMetrics,
)
from PyQt5.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsTextItem,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MapPlotCanvas(FigureCanvas):
    """Canvas for plotting Active Learning performance metrics (e.g., mAP).

    This class creates a matplotlib canvas inside a PyQt widget.

    Attributes:
        fig (Figure): Matplotlib figure for plotting.
        ax (AxesSubplot): Main axis for plotting data.
        ap_list (list): Stores the AP50 or AP75 values for plotting.

    Methods:
        update_plot(ap): Updates the plot with a new AP value.
    """

    def __init__(self, parent=None, width=2, height=2, dpi=100):
        """Initialize the plotting canvas.

        Args:
            parent (QWidget, optional): Parent widget.
            width (int, optional): Figure width. Defaults to 2.
            height (int, optional): Figure height. Defaults to 2.
            dpi (int, optional): Resolution. Defaults to 100.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Create two subplots (stacked horizontally)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        # Initialize data lists.
        self.ap_list = []

    def update_plot(self, ap):
        """Updates the plot by appending new AP values.

        Args:
            ap (float): New AP value to be plotted.
        """
        # Append new values.
        self.ap_list.append(ap)

        # Clear and plot on each axis.
        self.ax.clear()
        self.ax.set_axis_off()

        self.ax.plot(self.ap_list, marker="o", markersize=3)

        self.draw()


class CombinedButtonItem(QGraphicsRectItem):
    """
    A combined button bar that can be configured to show either three icons
    (✅, ❌, 🗑️) or a single delete button, with a white background (alpha 0.5)
    in the same location.

    Used in `InteractiveBox` to provide action buttons for predicted boxes.

    Attributes:
        parent_box (InteractiveBox): Parent annotation box.
        _radius (int): Corner radius for rounded rectangle.
        icon_items (list): List of button icons and associated actions.
    """

    def __init__(self, parent_box, actions=None, parent=None):
        """Initialize the button bar.

        Args:
            parent_box (InteractiveBox): The parent bounding box.
            actions (list, optional): List of tuples (icon, action).
            parent (QGraphicsItem, optional): Parent graphics item.
        """
        super().__init__(parent)
        self.parent_box = parent_box
        self._radius = 2

        # White background with 50% opacity.
        semi_white = QColor(255, 255, 255)
        semi_white.setAlphaF(0.5)
        self.setPen(QPen(Qt.NoPen))
        self.setBrush(QBrush(semi_white))

        # Default actions: three icons for predicted boxes.
        if actions is None:
            actions = [("✅", "accept"), ("❌", "delete")]  # , ("❌", "reject")

        self.icon_items = []
        self.font = QFont()
        self.font.setPointSize(8)
        fm = QFontMetrics(self.font)

        x_offset = -7.5
        if len(actions) == 3:
            spacing = [1, 1, 4]
        else:
            spacing = [4] * len(actions)

        line_height = 0
        for icon, action in actions:
            br = fm.boundingRect(icon)
            line_height = max(line_height, br.height())

        for i, (icon, action) in enumerate(actions):
            br = fm.boundingRect(icon)
            txt_item = QGraphicsTextItem(icon, self)
            txt_item.setFont(self.font)
            # Position so that the top edge aligns (adjusting by br.top() and a fraction of line height)
            txt_item.setPos(x_offset, -br.top() - (line_height * 3))
            icon_width = br.width() + spacing[i]
            self.icon_items.append((txt_item, action, x_offset + 4, icon_width))
            x_offset += icon_width

        self.setRect(
            -4,
            -br.top() - (line_height * 2.7),
            x_offset + 4,
            -br.top() - (line_height * -0.4),
        )

    def mousePressEvent(self, event):
        """Handles click events on the button bar.

        Args:
            event (QGraphicsSceneMouseEvent): Mouse event.
        """
        pos = event.pos()
        x_clicked = pos.x()
        for txt_item, action, x_start, w in self.icon_items:
            if x_start <= x_clicked <= (x_start + w):
                self.parent_box.handle_button_click(action)
                event.accept()
                return
        super().mousePressEvent(event)

    def paint(self, painter, option, widget=None):
        """Custom painting function for rounded rectangle appearance.

        Args:
            painter (QPainter): Painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): Associated widget.
        """
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRoundedRect(self.rect(), self._radius, self._radius)

    def shape(self):
        """Defines the clickable area shape.

        Returns:
            QPainterPath: The shape of the button item.
        """
        path = QPainterPath()
        path.addRoundedRect(self.rect(), self._radius, self._radius)
        return path


# ----------------- InteractiveBox -----------------
class InteractiveBox(QGraphicsRectItem):
    """Represents an interactive bounding box with action buttons.

    Attributes:
        main_window (QMainWindow): Reference to the main application window.
        group_id (int): Unique ID for this bounding box.
        box_type (str): Type of box ("positive", "predicted", or "background").
        predicted_class (str, optional): Predicted class label.
        confidence (float, optional): Confidence score.
    """

    def __init__(
        self,
        rect,
        box_type,
        main_window,
        group_id,
        predicted_class=None,
        confidence=None,
    ):
        """Initialize the bounding box with appropriate styling.

        Args:
            rect (QRectF): Bounding box dimensions.
            box_type (str): Type of annotation (e.g., "positive").
            main_window (QMainWindow): Parent application window.
            group_id (int): Unique identifier for the annotation.
            predicted_class (str, optional): Predicted category.
            confidence (float, optional): Confidence score.
        """
        super().__init__(QRectF(0, 0, rect.width(), rect.height()))
        self.setPos(rect.topLeft())

        self.main_window = main_window
        self.group_id = group_id
        self.box_type = box_type  # "positive", "predicted", or "background"
        self.predicted_class = predicted_class
        self.confidence = confidence

        # Set pen style based on type.
        if self.box_type == "positive":
            pen = QPen(QColor("green"))
            pen.setStyle(Qt.SolidLine)
        elif self.box_type == "predicted":
            if predicted_class == "certain_positive":
                pen = QPen(QColor("blue"))
                pen.setStyle(Qt.DotLine)
            elif predicted_class == "background":
                pen = QPen(QColor("red"))
                pen.setStyle(Qt.DotLine)
            else:
                pen = QPen(QColor("black"))
        else:  # for background or others
            pen = QPen(QColor("red"))
            pen.setStyle(Qt.DotLine)
        pen.setWidth(2)
        self.setPen(pen)

        # Add confidence label for predicted boxes.
        self.confidence_label = None
        if self.box_type == "predicted" and confidence is not None:
            self.confidence_label = QGraphicsTextItem("", self)
            self.update_confidence_style()

        # Create a combined button bar.
        # For predicted boxes: use the three-button bar.
        # For positive/background (accepted/rejected) boxes: show a single delete button.
        if self.box_type == "predicted":
            self.combined_button_item = CombinedButtonItem(self, parent=self)
        else:
            self.combined_button_item = CombinedButtonItem(
                self, actions=[("❌", "delete")], parent=self
            )

        self.combined_button_item.setZValue(100)

        self.update_controls()

    def update_confidence_style(self):
        """Update the confidence display color based on confidence score."""
        if self.confidence_label is None:
            return
        conf = self.confidence
        if conf > 0.7:
            text_color = "black"
            bg_color = "green"
        elif conf >= 0.4:
            text_color = "black"
            bg_color = "yellow"
        else:
            text_color = "white"
            bg_color = "red"
        self.confidence_label.setHtml(
            f"<div style='background-color:{bg_color}; color:{text_color}; font-size:8px;'>"
            f"{conf:.2f}</div>"
        )

    def update_controls(self):
        """Adjust the positions of labels and buttons inside the bounding box."""
        local_rect = self.rect()
        if self.confidence_label is not None:
            self.confidence_label.setPos(0, 0)
        if self.combined_button_item is not None:
            cb_rect = self.combined_button_item.rect()
            # Place so that the top-right of the button bar aligns with the box's top-right.
            x = local_rect.width() - cb_rect.width()
            y = 0
            self.combined_button_item.setPos(x, y)

    def handle_button_click(self, action):
        """Handle button click actions.

        Args:
            action (str): Action triggered by the button.
        """
        self.main_window.handle_box_action(self.group_id, action)
