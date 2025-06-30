from PIL import Image

# Set the maximum number of image pixels for PIL
Image.MAX_IMAGE_PIXELS = None

from PyQt5.QtCore import pyqtSignal, Qt, QRectF, QPointF
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem, QGraphicsView


class SyncedGraphicsView(QGraphicsView):
    """A custom QGraphicsView with synchronized transformations, zooming,
    panning, crosshair drawing, and rectangle selection.

    Signals:
        transformChanged: Emitted when the transformation changes (e.g., zoom or pan).
        boxDrawn (QRectF): Emitted when a rectangle is drawn using the box selection mode.
    """

    transformChanged = pyqtSignal()
    boxDrawn = pyqtSignal(QRectF)

    def __init__(self, *args, **kwargs):
        """Initialize the SyncedGraphicsView with custom settings."""
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._ignore_transform_event = False
        self.setScene(QGraphicsScene(self))
        self._is_panning = False
        self._is_drawing_box = False
        self._draw_start = QPointF()
        self._current_rect_item = None
        self.setMouseTracking(True)
        self._drawCrosshair = False
        self._mousePos = None

        self.horizontalScrollBar().valueChanged.connect(self._on_scrollbar_changed)
        self.verticalScrollBar().valueChanged.connect(self._on_scrollbar_changed)

    def drawForeground(self, painter, rect):
        """Draws a crosshair at the current mouse position if enabled.

        Args:
            painter: The QPainter used to render the foreground.
            rect: The rectangular region to be updated.
        """
        super().drawForeground(painter, rect)
        if self._drawCrosshair and self._mousePos is not None:
            painter.save()
            pen = QPen(QColor("pink"))
            pen.setWidth(1)
            painter.setPen(pen)

            p1 = QPointF(self._mousePos.x(), rect.top())
            p2 = QPointF(self._mousePos.x(), rect.bottom())
            painter.drawLine(p1, p2)
            p3 = QPointF(rect.left(), self._mousePos.y())
            p4 = QPointF(rect.right(), self._mousePos.y())
            painter.drawLine(p3, p4)
            painter.restore()

    def wheelEvent(self, event):
        """Handles zooming in and out using the mouse wheel.

        Args:
            event: The QWheelEvent containing wheel movement information.
        """
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor

        old_pos = self.mapToScene(event.pos())
        self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        self.transformChanged.emit()

    def mousePressEvent(self, event):
        """Handles mouse press events for panning and rectangle selection.

        Args:
            event: The QMouseEvent containing mouse press information.
        """
        if event.button() == Qt.LeftButton:
            if self._is_drawing_box:
                self._draw_start = self.mapToScene(event.pos())
                self._current_rect_item = QGraphicsRectItem()
                pen = QPen(QColor("red"), 2, Qt.SolidLine)
                self._current_rect_item.setPen(pen)
                self.scene().addItem(self._current_rect_item)
            else:
                self._is_panning = True
                self._last_mouse_pos = self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def _on_scrollbar_changed(self):
        """Emits the transformChanged signal when a scrollbar value changes."""
        self.transformChanged.emit()

    def mouseMoveEvent(self, event):
        """Handles mouse movement for panning, crosshair drawing, and box selection.

        Args:
            event: The QMouseEvent containing mouse movement information.
        """
        if self._drawCrosshair:
            self._mousePos = self.mapToScene(event.pos())
            self.scene().update()
        if self._is_drawing_box and self._current_rect_item:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start, current_pos).normalized()
            self._current_rect_item.setRect(rect)
        elif self._is_panning and not self._ignore_transform_event:
            new_mouse_pos = self.mapToScene(event.pos())
            delta = new_mouse_pos - self._last_mouse_pos
            self._last_mouse_pos = new_mouse_pos
            self.translate(delta.x(), delta.y())
            self.transformChanged.emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handles mouse release events, finalizing panning or box drawing.

        Args:
            event: The QMouseEvent containing mouse release information.
        """
        if event.button() == Qt.LeftButton:
            if self._is_drawing_box and self._current_rect_item:
                rect = self._current_rect_item.rect()
                if rect.width() > 1 and rect.height() > 1:
                    self.boxDrawn.emit(rect)
                else:
                    self.scene().removeItem(self._current_rect_item)
                self.scene().removeItem(self._current_rect_item)
                self._current_rect_item = None
            self._is_panning = False
        super().mouseReleaseEvent(event)

    def setDrawingBoxMode(self, enabled: bool):
        """Enables or disables the drawing box mode.

        Args:
            enabled: A boolean indicating whether the box mode should be enabled.
        """
        self._drawCrosshair = enabled
        if enabled:
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()
        self._is_drawing_box = enabled

    def leaveEvent(self, event):
        """Handles mouse leaving the view area, clearing crosshair.

        Args:
            event: The QEvent for leaving the widget.
        """
        self._mousePos = None
        self.scene().update()
        super().leaveEvent(event)

    def enterEvent(self, event):
        """Handles mouse entering the view area.

        Args:
            event: The QEvent for entering the widget.
        """
        super().enterEvent(event)
