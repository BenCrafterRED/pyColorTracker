import sys

from PyQt6.QtCore import QTranslator, QLibraryInfo, QLocale
from PyQt6.QtWidgets import QApplication

from .gui import MainWindow


app = QApplication(sys.argv)
translator = QTranslator()
translator.load(QLocale("de"), "qt", "_", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))
app.installTranslator(translator)
window = MainWindow()
window.show()
app.exec()
