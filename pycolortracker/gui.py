import decimal
from decimal import Decimal
from typing import Tuple

import cv2
import numpy as np
import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QSettings
from PyQt6.QtGui import (QKeySequence, QImage, QPaintEvent, QPainter, QMouseEvent, QPen,
        QDoubleValidator, QCloseEvent, QPixmap, QPalette, QColor)
from PyQt6.QtWidgets import (QMainWindow, QMenuBar, QApplication, QDialog, QDialogButtonBox,
        QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox, QGridLayout, QWidget,
        QLabel, QComboBox, QSlider)
from qtrangeslider import QRangeSlider

from . import version
from .analyzer import Analyzer, PlotType
from .processor import Processor
from .source import ThreadedSource


class Setting:
        CAMERA_SELECTOR = "camera_selector"
        SCALE = "scale"
        UNIT = "unit"
        DISTANCE = "distance"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("settings.ini", QSettings.Format.IniFormat)
        self.setWindowTitle("pyColorTracker - Objektverfolgung")
        self.init_menu_bar()
        
        self.source = None
        self.processor = None
        self.hue = 0
        self.threshold = 20

        layout = QVBoxLayout()

        self.label_video_result = QLabel()
        layout.addWidget(self.label_video_result)

        layout_threshold = QHBoxLayout()
        label_threshold = QLabel("Schwellwert:")
        layout_threshold.addWidget(label_threshold)
        slider_threshold = QSlider(Qt.Orientation.Horizontal)
        slider_threshold.setRange(0, 255)
        slider_threshold.valueChanged.connect(self.change_threshold)
        label_threshold.setBuddy(slider_threshold)
        layout_threshold.addWidget(slider_threshold)
        layout.addLayout(layout_threshold)

        layout_hue = QHBoxLayout()
        label_hue = QLabel("Farbton:")
        layout_hue.addWidget(label_hue)
        self.slider_hue = QSlider(Qt.Orientation.Horizontal)
        self.slider_hue.setRange(0, 127)
        self.slider_hue.valueChanged.connect(self.change_hue)
        self.slider_hue.setValue(0)
        label_hue.setBuddy(self.slider_hue)
        layout_hue.addWidget(self.slider_hue)
        layout.addLayout(layout_hue)

        layout_controls = QHBoxLayout()
        button_start = QPushButton(qta.icon("fa.play"), "Start")
        button_start.clicked.connect(self.start_processing)
        layout_controls.addWidget(button_start)
        button_stop = QPushButton(qta.icon("fa.stop"), "Stopp")
        button_stop.clicked.connect(self.stop_processing)
        layout_controls.addWidget(button_stop)
        layout.addLayout(layout_controls)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_menu_bar(self) -> None:
        menu_bar = QMenuBar()

        menu_file = menu_bar.addMenu("&Datei")
        menu_file.addAction(qta.icon("fa.video-camera"), "&Quelle...", self.show_select_source, QKeySequence.fromString("Ctrl+L"))
        menu_file.addAction(qta.icon("fa.close"), "&Beenden", self.request_quit, QKeySequence.StandardKey.Quit)

        menu_help = menu_bar.addMenu("&Hilfe")
        menu_help.addAction(qta.icon("fa.info-circle"), "&Über", self.show_about)

        self.setMenuBar(menu_bar)
    
    def closeEvent(self, event: QCloseEvent) -> None:
        if self.source:
            self.source.stop_gracefully()
    
    @pyqtSlot(int)
    def change_threshold(self, value: int) -> None:
        self.threshold = value
        if self.processor:
            self.processor.threshold = self.threshold
    
    @pyqtSlot(int)
    def change_hue(self, value: int) -> None:
        palette = self.slider_hue.palette()
        color = QColor()
        color.setHsv(value * 2, 255, 255, 255)
        palette.setColor(QPalette.ColorRole.Highlight, color)
        palette.setColor(QPalette.ColorRole.Button, color)
        self.slider_hue.setPalette(palette)
        self.hue = self.slider_hue.value()
        if self.processor:
            self.processor.hue = self.hue

    @pyqtSlot()
    def request_quit(self) -> None:
        self.close()

    @pyqtSlot()
    def show_select_source(self) -> None:
        if self.source:
            self.source.stop_gracefully()
        dialog = SourceDialog(self.settings)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            self.unit = dialog.unit
            self.pixels_per_unit = float(dialog.distance / dialog.scale)
            try:
                if dialog.source:
                    self.source = dialog.source.reuse()
                else:
                    self.source = ThreadedSource(dialog.camera_selector)
                self.roi = dialog.roi
                self.processor = Processor()
                self.processor.hue
                self.processor.roi = self.roi
                self.source.callback_process_data = self.processor.callback_process_data
                self.source.callback_process_time = self.processor.callback_process_time
                self.source.callback_process_user_image = self.process_result
                self.source.frame_available_timeout = 100_000000 # 100ms 
                self.source.frame_available.connect(self.result_available)
            except IOError:
                QMessageBox.critical(self, "Fehler", "Die angegebene Videoquelle konnte nicht geöffnet werden.")

    @pyqtSlot(np.ndarray)
    def process_result(self, image: np.ndarray, bbox: Tuple[int, int, int, int]):
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255))
        cv2.rectangle(image, (roi_x1+bbox_x, roi_y1+bbox_y), (roi_x1+bbox_x+bbox_w, roi_y1+bbox_y+bbox_h), (255, 0, 0))
        return image

    @pyqtSlot(QImage)
    def result_available(self, image: QImage):
        self.label_video_result.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot()
    def start_processing(self):
        if self.source and not self.source.isRunning():
            self.source.start()

    @pyqtSlot()
    def stop_processing(self):
        if self.source and self.source.isRunning():
            self.source = self.source.reuse()
            analyzer = Analyzer(self.pixels_per_unit, self.unit, PlotType.VELOCITY, PlotType.ACCELERATION) #TODO use matplotlib pyqt widgets instead of pyplot
            analyzer.plot_data(self.processor.bbox_data)
            self.processor = Processor()
            self.processor.hue = self.hue
            self.processor.threshold = self.threshold
            self.processor.roi = self.roi
            self.source.callback_process_data = self.processor.callback_process_data
            self.source.callback_process_time = self.processor.callback_process_time
            self.source.callback_process_user_image = self.process_result
            self.source.frame_available_timeout = 100_000000 # 100ms
            self.source.frame_available.connect(self.result_available)

    @pyqtSlot()
    def show_about(self):
        AboutDialog().exec()


class SourceDialog(QDialog):
    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.setWindowTitle("Quelle...")
        self.setModal(True)

        self.settings = settings
        self.source = None
        self.camera_selector = self.settings.value(Setting.CAMERA_SELECTOR, "")
        self.last_camera_selector = None
        self.units = {"m": 0, "dm": -1, "cm": -2}
        self.unit = self.settings.value(Setting.UNIT, "cm")
        self.scale = Decimal(self.settings.value(Setting.SCALE, "1"))
        self.distance = Decimal(self.settings.value(Setting.DISTANCE, "1"))

        layout = QVBoxLayout()

        layout_camera_selector = QHBoxLayout()
        self.line_edit_camera = QLineEdit(self.camera_selector)
        self.line_edit_camera.setPlaceholderText("OpenCV-Videoquelle...")
        layout_camera_selector.addWidget(self.line_edit_camera)
        button_preview = QPushButton("Vorschau erstellen")
        button_preview.clicked.connect(self.select_preview)
        layout_camera_selector.addWidget(button_preview)
        layout.addLayout(layout_camera_selector)

        layout_preview = QGridLayout()
        self.source_preview = SourcePreview()
        self.source_preview.setFixedHeight(480)
        self.source_preview.distance_changed.connect(self.distance_changed)
        layout_preview.addWidget(self.source_preview, 0, 0)
        self.roi_slider_x = QRangeSlider()
        self.roi_slider_x.setOrientation(Qt.Orientation.Horizontal)
        self.roi_slider_x.valueChanged.connect(self.update_roi)
        layout_preview.addWidget(self.roi_slider_x, 1, 0)
        self.roi_slider_y = QRangeSlider()
        self.roi_slider_y.setOrientation(Qt.Orientation.Vertical)
        self.roi_slider_y.valueChanged.connect(self.update_roi)
        layout_preview.addWidget(self.roi_slider_y, 0, 1)
        layout.addLayout(layout_preview)

        layout_scale = QHBoxLayout()
        label_scale = QLabel("Markierter Abstand entspricht:")
        layout_scale.addWidget(label_scale)
        self.line_edit_scale = QLineEdit()
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.line_edit_scale.setValidator(validator)
        self.line_edit_scale.textChanged.connect(self.scale_text_changed)
        self.update_scale_text()
        layout_scale.addWidget(self.line_edit_scale)
        combo_box_unit = QComboBox()
        combo_box_unit.addItems(self.units.keys())
        combo_box_unit.setCurrentText(self.unit)
        combo_box_unit.currentTextChanged.connect(self.unit_changed)
        layout_scale.addWidget(combo_box_unit)
        layout.addLayout(layout_scale)

        dialog_button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        @pyqtSlot()
        def reject():
            if self.source:
                self.source.stop_gracefully()
            self.reject()
        dialog_button_box.rejected.connect(reject)
        dialog_button_box.accepted.connect(self.accept)
        layout.addWidget(dialog_button_box)

        self.setLayout(layout)
    
    @pyqtSlot()
    def accept(self) -> None:
        self.settings.setValue(Setting.CAMERA_SELECTOR, self.camera_selector)
        self.settings.setValue(Setting.SCALE, np.format_float_positional(self.scale))
        self.settings.setValue(Setting.UNIT, self.unit)
        self.settings.setValue(Setting.DISTANCE, np.format_float_positional(self.distance))
        self.settings.sync()
        super().accept()
    
    @pyqtSlot()
    def reject(self) -> None:
        if self.source:
            self.source.stop_gracefully()
        super().reject()
    
    def update_scale_text(self) -> None:
        self.line_edit_scale.blockSignals(True)
        self.line_edit_scale.setText(np.format_float_positional((self.scale * Decimal("10")**-self.units[self.unit] * self.distance).normalize()))
        self.line_edit_scale.blockSignals(False)
    
    @pyqtSlot(str)
    def scale_text_changed(self, new_scale: str) -> None:
        try:
            self.scale = (Decimal(new_scale) * Decimal("10")**self.units[self.unit] / self.distance).normalize()
        except decimal.InvalidOperation:
            pass

    @pyqtSlot(str)
    def unit_changed(self, new_unit: str) -> None:
        self.unit = new_unit
        self.update_scale_text()
    
    @pyqtSlot(Decimal)
    def distance_changed(self, new_distance: Decimal) -> None:
        self.scale = self.scale / self.distance * new_distance
        self.distance = new_distance

    @pyqtSlot()
    def update_roi(self) -> None:
        if self.source:
            self.roi = (
                min(self.roi_slider_x.value()),
                self.source.height - max(self.roi_slider_y.value()),
                max(self.roi_slider_x.value()),
                self.source.height - min(self.roi_slider_y.value()),
            )
    
    def process_image_for_roi_preview(self, cv_image: np.ndarray, process_data_result: None) -> np.ndarray:
        grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        result[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] = cv_image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        cv2.rectangle(result, (self.roi[0], self.roi[1]), (self.roi[2]-1, self.roi[3]-1), (0, 0, 255))
        return result

    pyqtSlot()
    def select_preview(self) -> None:
        self.camera_selector = self.line_edit_camera.text()
        if self.last_camera_selector != self.camera_selector:
            self.last_camera_selector = self.camera_selector
            if self.source:
                self.source.stop_gracefully()
            try:
                self.source = ThreadedSource(self.camera_selector)
                self.roi_slider_x.setRange(0, self.source.width)
                self.roi_slider_y.setRange(0, self.source.height)
                self.update_roi()
                self.source.callback_process_user_image = self.process_image_for_roi_preview
                self.source.frame_available.connect(self.source_preview.update_image)
                self.source.start()
                self.camera_selector_changed = False
            except IOError:
                QMessageBox.critical(self, "Fehler", "Die angegebene Videoquelle konnte nicht geöffnet werden.")

class SourcePreview(QWidget):
    distance_changed = pyqtSignal(Decimal)

    def __init__(self) -> None:
        super().__init__()
        self._image = None
        self.a = None
        self.b = None
        self.distance = Decimal("0")
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.a = event.pos()
        if event.button() == Qt.MouseButton.RightButton:
            self.b = event.pos()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.a = event.pos()
        if event.buttons() & Qt.MouseButton.RightButton:
            self.b = event.pos()
        self.update()
    
    @pyqtSlot(QImage)
    def update_image(self, image: QImage) -> None:
        self._image = image
        self.setFixedSize(image.size())
        self.update()
    
    def update_distance(self) -> None:
        if self.a and self.b:
            self.distance = ((self.a.x - self.b.x) ** Decimal("2") + (self.a.y - self.b.y) ** Decimal("2")) ** Decimal("0.5")
            self.distance_changed.emit(self.distance)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        if self._image:
            painter.drawImage(self.rect(), self._image)
            if self.a:
                painter.setPen(QPen(Qt.GlobalColor.green, 1))
                painter.drawEllipse(self.a, 8, 8)
            if self.b:
                painter.setPen(QPen(Qt.GlobalColor.blue, 1))
                painter.drawEllipse(self.b, 8, 8)
            if self.a and self.b:
                painter.setPen(QPen(Qt.GlobalColor.red, 1))
                painter.drawLine(self.a, self.b)


class AboutDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Über")
        self.setModal(True)

        layout = QVBoxLayout()
        label_name = QLabel("pyColorTracker")
        label_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_name)
        label_version = QLabel(f"Version {version.get_version_string(True)}")
        label_version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_version)
        label_info = QLabel("Frühe instabile Version <b>mit Fehlern!</b>")
        label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_info)
        self.setLayout(layout)
