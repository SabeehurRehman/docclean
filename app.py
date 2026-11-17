"""DocClean – desktop app that removes the background from photos of
documents and makes them sharp and readable."""
from __future__ import annotations

import os
import sys
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal, QSize
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGraphicsEllipseItem, QGraphicsPixmapItem,
    QGraphicsPolygonItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressDialog, QSlider, QSplitter, QStatusBar,
    QToolBar, QToolButton, QVBoxLayout, QWidget,
)

import engine

APP_NAME = "DocClean"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
PREVIEW_MAX = 1400


def resource_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ---------------------------------------------------------------- data
class Page:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self._img = None
        self.corners = None
        self.found = False
        self.rotation = 0

    @property
    def img(self) -> np.ndarray:
        if self._img is None:
            self._img = engine.load_image(self.path)
        return self._img

    def ensure_corners(self):
        if self.corners is None:
            self.corners, self.found = engine.detect_document(self.img)
            try:
                self.rotation = engine.suggest_rotation(self.img, self.corners)
            except Exception:
                self.rotation = 0


# ---------------------------------------------------------------- worker
class WorkerSignals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class PreviewJob(QRunnable):
    def __init__(self, token, page, mode, sharp):
        super().__init__()
        self.token, self.page, self.mode, self.sharp = token, page, mode, sharp
        self.signals = WorkerSignals()

    def run(self):
        try:
            img = self.page.img
            corners = self.page.corners
            h, w = img.shape[:2]
            f = min(1.0, PREVIEW_MAX / max(h, w))
            if f < 1:
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                corners = corners * f
            out = engine.process(img, corners, self.mode, self.sharp, self.page.rotation)
            result = (self.signals.done, (self.token, out))
        except Exception:
            result = (self.signals.failed, (self.token, traceback.format_exc()))
        try:
            result[0].emit(*result[1])
        except RuntimeError:
            pass  # window/page closed while we were working


# ---------------------------------------------------------------- views
class FitView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#2b2d31"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.pix)
        self.placeholder = self.scene().addText("")
        self.placeholder.setDefaultTextColor(QColor("#9aa0a6"))

    def set_message(self, text):
        self.pix.setPixmap(QPixmap())
        self.placeholder.setPlainText(text)
        self.scene().setSceneRect(self.placeholder.boundingRect())
        self.fit()

    def set_image(self, qimg: QImage):
        self.placeholder.setPlainText("")
        self.pix.setPixmap(QPixmap.fromImage(qimg))
        self.scene().setSceneRect(QRectF(self.pix.pixmap().rect()))
        self.fit()

    def fit(self):
        r = self.scene().sceneRect()
        if r.width() > 0:
            self.fitInView(r, Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit()


class Handle(QGraphicsEllipseItem):
    def __init__(self, editor, idx):
        super().__init__(-1, -1, 2, 2)
        self.editor, self.idx = editor, idx
        self.setBrush(QBrush(QColor(0, 160, 255, 160)))
        self.setPen(QPen(QColor("white"), 0))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            r = self.editor.scene().sceneRect()
            value = QPointF(min(max(value.x(), 0), r.width() - 1), min(max(value.y(), 0), r.height() - 1))
        if change == QGraphicsEllipseItem.ItemPositionHasChanged:
            self.editor.update_polygon()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.editor.corners_edited()


class CornerEditor(FitView):
    """Shows the original photo with 4 draggable corners around the page."""
    changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.poly = QGraphicsPolygonItem()
        self.poly.setPen(QPen(QColor(0, 160, 255), 0))
        self.poly.setBrush(QBrush(QColor(0, 160, 255, 40)))
        self.poly.setZValue(5)
        self.scene().addItem(self.poly)
        self.handles = [Handle(self, i) for i in range(4)]
        for h in self.handles:
            self.scene().addItem(h)
        self.show_handles(False)

    def show_handles(self, on):
        self.poly.setVisible(on)
        for h in self.handles:
            h.setVisible(on)

    def load(self, qimg, corners):
        self.set_image(qimg)
        size = max(qimg.width(), qimg.height()) / 45
        for h, (x, y) in zip(self.handles, corners):
            h.setRect(-size / 2, -size / 2, size, size)
            h.setPos(float(x), float(y))
        self.show_handles(True)
        self.update_polygon()

    def clear(self, text):
        self.show_handles(False)
        self.set_message(text)

    def update_polygon(self):
        self.poly.setPolygon(QPolygonF([h.pos() for h in self.handles]))

    def corners(self):
        return np.array([[h.pos().x(), h.pos().y()] for h in self.handles], np.float32)

    def corners_edited(self):
        self.changed.emit(self.corners())


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} – Document Background Remover & Enhancer")
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self.pages: list[Page] = []
        self.pool = QThreadPool.globalInstance()
        self.token = 0
        self.jobs = set()
        self.last_dir = os.path.expanduser("~")

        # Left: page list
        self.list = QListWidget()
        self.list.setIconSize(QSize(72, 72))
        self.list.setMinimumWidth(200)
        self.list.currentRowChanged.connect(self.show_page)

        # Centre / right
        self.editor = CornerEditor()
        self.editor.changed.connect(self.on_corners_changed)
        self.result = FitView()
        self.editor.clear("Drop photos of documents here\nor click “Add images”.")
        self.result.set_message("The cleaned document will appear here.")

        def titled(title, w):
            box = QWidget()
            lay = QVBoxLayout(box)
            lay.setContentsMargins(4, 4, 4, 4)
            lab = QLabel(title)
            lab.setStyleSheet("font-weight:600; padding:2px;")
            lay.addWidget(lab)
            lay.addWidget(w)
            return box

        split = QSplitter()
        split.addWidget(titled("Pages", self.list))
        split.addWidget(titled("Original — drag the blue corners to adjust the crop", self.editor))
        split.addWidget(titled("Result", self.result))
        split.setSizes([220, 590, 590])
        self.setCentralWidget(split)

        self.build_toolbar()
        self.setStatusBar(QStatusBar())
        self.status("Ready")

    # ---- UI
    def build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        def act(text, slot, tip="", key=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            a.setToolTip(tip or text)
            if key:
                a.setShortcut(key)
            tb.addAction(a)
            return a

        act("Add images", self.add_images, "Add photos of documents", "Ctrl+O")
        act("Remove", self.remove_selected, "Remove the selected page", "Delete")
        act("Clear all", self.clear_all)
        tb.addSeparator()

        tb.addWidget(QLabel("  Style: "))
        self.mode = QComboBox()
        self.mode.addItems(engine.MODES)
        self.mode.currentIndexChanged.connect(self.refresh)
        tb.addWidget(self.mode)

        tb.addWidget(QLabel("   Sharpness: "))
        self.sharp = QSlider(Qt.Horizontal)
        self.sharp.setRange(0, 20)
        self.sharp.setValue(10)
        self.sharp.setFixedWidth(120)
        self.sharp.sliderReleased.connect(self.refresh)
        self.sharp.valueChanged.connect(lambda v: self.sharp_lbl.setText(f"{v * 10}%"))
        tb.addWidget(self.sharp)
        self.sharp_lbl = QLabel("100%")
        self.sharp_lbl.setFixedWidth(40)
        tb.addWidget(self.sharp_lbl)
        tb.addSeparator()

        act("Rotate left", lambda: self.rotate(-90), "Rotate this page left", "Ctrl+L")
        act("Rotate right", lambda: self.rotate(90), "Rotate this page right", "Ctrl+R")
        rot_btn = QToolButton()
        rot_btn.setText("All pages ▾")
        rot_btn.setToolTip("Rotate every page at once")
        rot_btn.setPopupMode(QToolButton.InstantPopup)
        rot_menu = QMenu(rot_btn)
        rot_menu.addAction("Rotate all left", lambda: self.rotate_all(-90))
        rot_menu.addAction("Rotate all right", lambda: self.rotate_all(90))
        rot_menu.addAction("Rotate all 180°", lambda: self.rotate_all(180))
        rot_btn.setMenu(rot_menu)
        tb.addWidget(rot_btn)
        act("Auto-detect", self.redetect, "Detect the page edges again")
        act("Use whole image", self.whole_image, "Don't crop – keep the whole photo")
        tb.addSeparator()

        act("Save page…", self.export_current, "Save the current page", "Ctrl+S")
        act("Save all…", self.export_all, "Save every page as images into a folder", "Ctrl+Shift+S")
        act("Save all as PDF…", self.export_pdf, "Combine every page into one PDF", "Ctrl+P")

    def status(self, text):
        self.statusBar().showMessage(text)

    def current(self) -> Page | None:
        r = self.list.currentRow()
        return self.pages[r] if 0 <= r < len(self.pages) else None

    def settings(self):
        return self.mode.currentText(), self.sharp.value() / 10.0

    # ---- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.isdir(p):
                paths += [os.path.join(p, f) for f in sorted(os.listdir(p))]
            else:
                paths.append(p)
        self.add_paths(paths)

    # ---- page management
    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add document photos", self.last_dir,
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All files (*)")
        self.add_paths(paths)

    def add_paths(self, paths):
        paths = [p for p in paths if p.lower().endswith(IMAGE_EXTS)]
        if not paths:
            return
        self.last_dir = os.path.dirname(paths[0])
        first_new = len(self.pages)
        bad = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for p in paths:
                page = Page(p)
                try:
                    page.ensure_corners()
                except Exception:
                    bad.append(os.path.basename(p))
                    continue
                self.pages.append(page)
                thumb = cv2.resize(page.img, (96, int(96 * page.img.shape[0] / page.img.shape[1])))
                item = QListWidgetItem(QIcon(QPixmap.fromImage(to_qimage(thumb))), page.name)
                if not page.found:
                    item.setToolTip("Page edges not found – using whole image. Drag the corners to adjust.")
                self.list.addItem(item)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
        if bad:
            QMessageBox.warning(self, APP_NAME, "Could not open:\n" + "\n".join(bad))
        if len(self.pages) > first_new:
            self.list.setCurrentRow(first_new)
        self.status(f"{len(self.pages)} page(s) loaded")

    def remove_selected(self):
        r = self.list.currentRow()
        if r < 0:
            return
        self.pages.pop(r)
        self.list.takeItem(r)
        if not self.pages:
            self.show_page(-1)

    def clear_all(self):
        self.pages.clear()
        self.list.clear()
        self.show_page(-1)

    # ---- preview
    def show_page(self, row):
        page = self.current()
        if page is None:
            self.editor.clear("Drop photos of documents here\nor click “Add images”.")
            self.result.set_message("The cleaned document will appear here.")
            return
        self.editor.load(to_qimage(page.img), page.corners)
        if not page.found:
            self.status("Couldn't find the page edges automatically – drag the blue corners onto the page corners.")
        self.refresh()

    def on_corners_changed(self, corners):
        page = self.current()
        if page is not None:
            page.corners = corners
            self.refresh()

    def refresh(self):
        page = self.current()
        if page is None:
            return
        self.token += 1
        mode, sharp = self.settings()
        job = PreviewJob(self.token, page, mode, sharp)
        job.setAutoDelete(False)
        self.jobs.add(job)  # keep alive until it reports back
        job.signals.done.connect(lambda t, img, j=job: (self.jobs.discard(j), self.on_preview(t, img)))
        job.signals.failed.connect(lambda t, err, j=job: (self.jobs.discard(j),
                                   self.status("Processing error: " + err.splitlines()[-1])))
        self.status("Processing…")
        self.pool.start(job)

    def on_preview(self, token, img):
        if token != self.token:
            return  # a newer request is pending
        self.result.set_image(to_qimage(img))
        self.status("Preview ready")

    def rotate(self, deg):
        page = self.current()
        if page:
            page.rotation = (page.rotation + deg) % 360
            self.refresh()

    def rotate_all(self, deg):
        for page in self.pages:
            page.rotation = (page.rotation + deg) % 360
        self.refresh()

    def redetect(self):
        page = self.current()
        if page:
            page.corners, page.found = engine.detect_document(page.img)
            self.show_page(self.list.currentRow())

    def whole_image(self):
        page = self.current()
        if page:
            page.corners = engine.full_frame(page.img)
            self.show_page(self.list.currentRow())

    # ---- export
    def render_full(self, page):
        mode, sharp = self.settings()
        return engine.process(page.img, page.corners, mode, sharp, page.rotation)

    def _progress(self, n, label):
        dlg = QProgressDialog(label, "Cancel", 0, n, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        return dlg

    def export_current(self):
        page = self.current()
        if not page:
            return
        base = os.path.splitext(page.name)[0] + "_clean"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save page", os.path.join(self.last_dir, base + ".png"),
            "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            out = self.render_full(page)
            if path.lower().endswith(".pdf"):
                engine.save_pdf(path, [out])
            else:
                if not path.lower().endswith((".png", ".jpg", ".jpeg")):
                    path += ".png"
                engine.save_image(path, out)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Could not save:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.status(f"Saved {path}")

    def export_all(self):
        if not self.pages:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder for the cleaned images", self.last_dir)
        if not folder:
            return
        dlg = self._progress(len(self.pages), "Cleaning pages…")
        for i, page in enumerate(self.pages):
            if dlg.wasCanceled():
                break
            dlg.setValue(i)
            dlg.setLabelText(f"Cleaning {page.name}…")
            QApplication.processEvents()
            out = self.render_full(page)
            name = os.path.splitext(page.name)[0] + "_clean.png"
            engine.save_image(os.path.join(folder, name), out)
        dlg.setValue(len(self.pages))
        self.status(f"Saved {len(self.pages)} page(s) to {folder}")

    def export_pdf(self):
        if not self.pages:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", os.path.join(self.last_dir, "documents.pdf"), "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        dlg = self._progress(len(self.pages) + 1, "Building PDF…")
        outs = []
        for i, page in enumerate(self.pages):
            if dlg.wasCanceled():
                return
            dlg.setValue(i)
            dlg.setLabelText(f"Cleaning {page.name}…")
            QApplication.processEvents()
            outs.append(self.render_full(page))
        dlg.setLabelText("Writing PDF…")
        QApplication.processEvents()
        engine.save_pdf(path, outs)
        dlg.setValue(len(self.pages) + 1)
        self.status(f"Saved {path}")


def selftest(out_dir: str) -> int:
    """Headless check used by the Windows build: model loads, a synthetic
    skewed document is detected, cleaned and saved as PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "selftest.log"), "w", encoding="utf-8")
    try:
        assert engine._get_net() is not None, "AI model missing from build"
        page = np.full((1400, 1000, 3), 245, np.uint8)
        for i in range(25):
            cv2.putText(page, f"Line {i:02d} The quick brown fox 12.50", (60, 80 + 45 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        bg = np.full((2000, 1600, 3), (40, 70, 110), np.uint8)
        src = np.float32([[0, 0], [999, 0], [999, 1399], [0, 1399]])
        dst = np.float32([[300, 250], [1250, 330], [1330, 1700], [200, 1600]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(page, M, (1600, 2000))
        mask = cv2.warpPerspective(np.full((1400, 1000), 255, np.uint8), M, (1600, 2000))
        bg[mask > 0] = warped[mask > 0]
        corners, found = engine.detect_document(bg)
        err = float(np.abs(engine.order_corners(corners) - dst).max())
        log.write(f"found={found} max_corner_error_px={err:.1f}\n")
        assert found and err < 40, "document not detected"
        outs = [engine.process(bg, corners, m) for m in engine.MODES]
        engine.save_image(os.path.join(out_dir, "selftest.png"), outs[0])
        engine.save_pdf(os.path.join(out_dir, "selftest.pdf"), outs)
        app = QApplication.instance() or QApplication(sys.argv[:1])
        MainWindow()  # GUI can be constructed
        log.write("OK\n")
        return 0
    except Exception:
        log.write(traceback.format_exc())
        return 1
    finally:
        log.close()


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2] if len(sys.argv) > 2 else "selftest-out"))
    if sys.platform == "win32":
        try:  # show our own icon in the Windows taskbar
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("docclean.app")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    icon = resource_path("icon.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.add_paths(sys.argv[1:])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
