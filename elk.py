# -*- coding: utf-8 -*-

"""
Labelling tool for images for neural networks.
Author: David Bunk
"""

from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt, \
    QItemSelectionModel, pyqtSignal, pyqtSlot, QThread, QObject
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QSizePolicy, \
    QFileDialog, QListView, QColumnView, QMessageBox, QFrame

from gui import Ui_MainWindow

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import FigureCanvasBase
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector

from scipy.ndimage.morphology import binary_erosion, binary_fill_holes

from shapely.geometry import LineString, MultiPoint, MultiLineString, LinearRing, Point, collection
from shapely.ops import split, nearest_points

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, binary_closing
from skimage.measure import label, regionprops
from skimage.io import imread, imsave
from skimage.draw import polygon

import pickle
import time
import sys, os
import warnings
import numpy as np
from glob import glob
from copy import copy

###################################################### Initialisation

init_class_n = 2
autosave_pause = 90
otsu_min_size = 500
zoom_buffer = 50

###################################################### Window class


class VisionGui(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(VisionGui, self).__init__()
        self.setupUi(self)

        self.currentImage = None
        self.currentObject = None
        self.oldObject = None
        self.currentClass = None
        self.imageParent = None

        self.outline_switch = True
        self.overview_switch = True
        self.pan_switch = False
        self.zoom_switch = False

        self.data = Data()

        self.backend = Backend()
        self.backendthread = QThread()
        self.backend.moveToThread(self.backendthread)
        self.mpl_widget.lasso_drawn.connect(self.backend.get_lasso_area)
        self.mpl_widget.multi_drawn.connect(self.backend.get_multi_area)
        self.mpl_widget.ask_redraw.connect(self.ask_redraw)
        self.backend.new_coords.connect(self.mpl_widget.draw_object)
        self.backend.ask_redraw.connect(self.ask_redraw)
        self.backendthread.start()

        self.savetimer = SaveTimer(pause_duration=autosave_pause)
        self.timingthread = QThread()
        self.savetimer.moveToThread(self.timingthread)
        self.savetimer.savingtime.connect(self.autosave)
        self.timingthread.started.connect(self.savetimer.saving_timer)
        self.timingthread.start()

        self.toolbar = NavigationToolbar2QT(self.mpl_widget, self.toolbarwidget)
        self.toolbarwidget.setVisible(False)
        self.zoombutton.clicked.connect(self.switch_zoom)
        self.panbutton.clicked.connect(self.switch_pan)

        self.mpl_widget.switch_zoom.connect(self.switch_zoom)
        self.mpl_widget.switch_pan.connect(self.switch_pan)

        self.mpl_widget.uparrow.connect(self.move_object_up)
        self.mpl_widget.downarrow.connect(self.move_object_down)
        self.mpl_widget.upclass.connect(self.move_class_up)
        self.mpl_widget.downclass.connect(self.move_class_down)

        self.mpl_widget.switch_outlines.connect(self.toggle_outline)
        self.mpl_widget.switch_overview.connect(self.reset_overview)
        self.mpl_widget.add_subobject.connect(self.add_object)
        self.mpl_widget.delete_object.connect(self.remove_object)

        self.actionOpenImages.triggered.connect(self.get_data)
        self.actionOpenProject.triggered.connect(self.load_project)
        self.actionSaveProject.triggered.connect(self.data.save_project)
        self.actionExportLabels.triggered.connect(self.data.export_labels)

        self.actionAdd_object.triggered.connect(self.add_object)
        self.actionRemove_object.triggered.connect(self.remove_object)
        self.actionRemove_all_objects.triggered.connect(self.remove_all_objects)
        self.actionAdd_class.triggered.connect(self.add_class)
        self.actionRemove_class.triggered.connect(self.remove_class)
        self.actionSwitch_outlines_on_off.triggered.connect(self.toggle_outline)

        self.data.class_n_changed.connect(self.change_class_n)
        self.data.loadingfailed.connect(self.loading_error)

        self.loadimagebutton.clicked.connect(self.get_data)
        self.rmimagebutton.clicked.connect(self.remove_image)

        self.class_list = QListView()
        self.class_list.setModel(self.data.listmodel)
        self.class_list.setFrameShape(QFrame.NoFrame)
        self.class_list.setSelectionRectVisible(False)

        self.objects_list.clicked.connect(self.plot)

        self.class_list.clicked.connect(self.set_class)

        self.object_add.clicked.connect(self.add_object)
        self.object_rm.clicked.connect(self.remove_object)

        self.class_add.clicked.connect(self.add_class)
        self.class_rm.clicked.connect(self.remove_class)

        self.overviewbutton.clicked.connect(self.reset_overview)
        self.outline_toggle.clicked.connect(self.toggle_outline)

        self.objects_list.setModel(self.data.treemodel)
        self.objects_list.setPreviewWidget(self.class_list)
        self.objects_list.enterview.connect(self.plot)

    def get_data(self):
        success = self.data.get_data()
        if success:
            self.objects_list.setModel(self.data.treemodel)
        self.objects_list.setColumnWidths([193, 10, 10])

    def load_project(self):
        success = self.data.load_project()
        if success:
            self.objects_list.setModel(self.data.treemodel)

    def autosave(self):
        self.data.save_project(filename='./label_autosave')

    @pyqtSlot(int)
    def ask_redraw(self, case):
        self.mpl_widget.remove_object()

        box = QMessageBox()
        box.setBaseSize(400, 150)
        box.setIcon(QMessageBox.Warning)
        box.setStandardButtons(QMessageBox.Ok)
        box.setText("Please redraw.")
        box.setWindowTitle("Please redraw.")

        if case == 0:
            box.setInformativeText("Unknown problem.")
        elif case == 1:
            box.setInformativeText("Drawn line intersects itself to much.")
        elif case == 2:
            box.setInformativeText("New line needs to intersect main object at start and end point.")
        elif case == 3:
            box.setInformativeText("Please draw main object first.")
        elif case == 4:
            box.setInformativeText("Drawn line intersects with main object more than two times.")

        box.exec_()

    @pyqtSlot(int)
    def loading_error(self, case):
        box = QMessageBox()
        box.setBaseSize(400, 150)
        box.setIcon(QMessageBox.Critical)
        box.setStandardButtons(QMessageBox.Ok)
        box.setText("Not all images could be loaded.")
        box.setWindowTitle("Loading failed.")

        if case == 0:
            box.setInformativeText('Amount of selected labels does not match amount of loaded images. Cancelling loading!')
        elif case == 1:
            box.setIcon(QMessageBox.Warning)
            box.setInformativeText('Some labels were corrupted or of wrong type. Skipping these image&label pairs.')
        elif case == 2:
            box.setInformativeText('Names of selected labels do not match the loaded images. Please select right labels.')
        elif case == 3:
            box.setInformativeText('Error when loading corresponding labels of imported images. Cancelling loading!')

        box.exec_()

    def move_object_up(self):
        if self.currentImage is None:
            return
        if self.currentObject is None:
            curr_index = self.objects_list.currentIndex().row()
        else:
            curr_index = self.currentObject.row()

        if curr_index == 0:
            return

        tempobj = self.data.treemodel.index(curr_index - 1, 0, self.currentImage)
        self.plot(tempobj)
        self.currentObject = tempobj

        self.objects_list.clearSelection()
        self.objects_list.selectionModel().select(self.currentObject, QItemSelectionModel.Select)

    def move_object_down(self):
        if self.currentImage is None:
            return
        if self.currentObject is None:
            curr_index = self.objects_list.currentIndex().row()
        else:
            curr_index = self.currentObject.row()

        if curr_index == self.data.treemodel.rootItem.child(self.currentImage.row()).childCount() - 1:
            return

        tempobj = self.data.treemodel.index(curr_index + 1, 0, self.currentImage)
        self.plot(tempobj)
        self.currentObject = tempobj

        self.objects_list.clearSelection()
        self.objects_list.selectionModel().select(self.currentObject, QItemSelectionModel.Select)

    def move_class_up(self):
        if self.currentImage is None or self.currentObject is None:
            return

        curr_class = self.data.files[self.currentImage.row()].objects[self.currentObject.row()].classtype
        if curr_class == 0:
            return

        index = self.data.listmodel.index(curr_class - 1, 0, 'rootparent')

        self.set_class(index)
        self.class_list.clearSelection()
        self.class_list.selectionModel().select(index, QItemSelectionModel.Select)

    def move_class_down(self):
        if self.currentImage is None or self.currentObject is None:
            return

        curr_class = self.data.files[self.currentImage.row()].objects[self.currentObject.row()].classtype
        if curr_class == self.data.listmodel.rootItem.childCount() - 1:
            return

        index = self.data.listmodel.index(curr_class + 1, 0, 'rootparent')

        self.set_class(index)
        self.class_list.clearSelection()
        self.class_list.selectionModel().select(index, QItemSelectionModel.Select)

    def add_object(self):
        if self.currentImage is None:
            return

        parent = self.currentImage
        index = self.currentObject
        self.data.treemodel.add_object(parent, index)

        if self.currentObject is None:
            if len(self.data.files[self.currentImage.row()].objects) == 0:
                self.data.files[parent.row()].objects.append(Object(0))
                self.currentObject = self.data.treemodel.index(self.data.treemodel.rootItem.child(
                    self.currentImage.row()).childCount() - 1, 0, self.currentImage)
            else:
                childcount = self.data.treemodel.rootItem.child(self.currentImage.row()).childCount() - 1
                name = self.data.treemodel.rootItem.child(self.currentImage.row()).child(childcount).data(0).split(' ')[-1]
                try:
                    name, suffix = name.split('.')
                except ValueError:
                    pass

                self.data.files[parent.row()].objects.append(Object(name))
                self.currentObject = self.data.treemodel.index(self.data.treemodel.rootItem.child(
                    self.currentImage.row()).childCount() - 1, 0, self.currentImage)
                self.oldObject = None
                self.switch_overview_button()
        else:
            name = self.data.treemodel.rootItem.child(self.currentImage.row()).child(
                self.currentObject.row()).data(0).split(' ')[-1]
            try:
                name, suffix = name.split('.')
            except ValueError:
                suffix = 0

            if self.data.files[self.currentImage.row()].objects[self.currentObject.row()].parent is None:
                parentobject = self.data.files[self.currentImage.row()].objects[self.currentObject.row()]
            else:
                parentobject = self.data.files[self.currentImage.row()].objects[self.currentObject.row()].parent

            self.data.files[parent.row()].objects.insert(index.row() + 1,
                Object(name, suffix=suffix, parent=parentobject,
                       zoom=self.data.files[self.currentImage.row()].objects[self.currentObject.row()].zoom))

            self.currentObject = self.data.treemodel.index(index.row() + 1, 0, self.currentImage)
            self.oldObject = self.currentObject

        self.objects_list.clearSelection()
        self.objects_list.selectionModel().select(self.currentObject, QItemSelectionModel.Select)
        self.mpl_widget.set_object(self.data.files[self.currentImage.row()].objects[self.currentObject.row()])

    def remove_object(self):
        if self.currentObject is None:
            return

        if len(self.data.files[self.currentImage.row()].objects) == 0:
            return

        parent = self.currentImage
        index = self.currentObject
        self.objects_list.clearSelection()
        self.data.treemodel.remove_object(parent, index)

        delparent = self.data.files[parent.row()].objects[index.row()].parent
        self.data.files[parent.row()].objects.pop(index.row())

        if self.currentObject.row() == len(self.data.files[self.currentImage.row()].objects) or delparent is not None:
            if self.currentObject.row() == 0:
                self.currentObject = None
                return
            else:
                offset = -1
        else:
            offset = 0

        self.currentObject = self.data.treemodel.index(self.currentObject.row() + offset, 0, self.currentImage)
        try:
            self.mpl_widget.remove_object()
        except KeyError:
            pass
        self.mpl_widget.set_object(self.data.files[self.currentImage.row()].objects[self.currentObject.row()])

        self.objects_list.clearSelection()
        self.objects_list.selectionModel().select(self.currentObject, QItemSelectionModel.Select)
        self.mpl_widget.set_object(self.data.files[self.currentImage.row()].objects[self.currentObject.row()])

    def remove_all_objects(self):
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setBaseSize(300, 150)
        box.setText("Remove all objects?")
        box.setInformativeText("All labels of current image will be lost.")
        box.setWindowTitle("Remove all objects?")
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        delete = box.exec()

        if delete != QMessageBox.Ok or self.currentImage is None \
                or len(self.data.files[self.currentImage.row()].objects) == 0:
            return

        self.currentObject = None
        self.data.files[self.currentImage.row()].objects = [Object(0)]

        self.data.treemodel.remove_all_objects(self.currentImage)

        self.objects_list.clearSelection()
        self.objects_list.selectionModel().select(self.currentImage, QItemSelectionModel.Select)
        self.mpl_widget.set_image(self.data.files[self.currentImage.row()], self.outline_switch)

    def change_class_n(self, change):
        if change > 0:
            for n in range(abs(change)):
                self.remove_class(internal=True)
        else:
            for n in range(abs(change)):
                self.add_class(internal=True)

    def add_class(self, internal=False):
        if self.currentImage is None and internal is False:
            return

        self.data.set_class_max(True)
        self.data.listmodel.add_class()
        self.mpl_widget.set_colormap(self.data.listmodel.rootItem.childCount())

    def remove_class(self, internal=False):
        if (self.data.listmodel.rootItem.childCount() == 1 or self.currentImage is None) and internal is False:
            return

        self.data.set_class_max(False)
        self.class_list.clearSelection()
        self.data.listmodel.remove_class()
        self.mpl_widget.set_colormap(self.data.listmodel.rootItem.childCount())

    def remove_image(self, force=False):
        if self.currentImage is None:
            return

        delete = None
        if not force:
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setBaseSize(300, 150)
            box.setText("Delete current image?")
            box.setInformativeText("All corresponding labels will be lost!")
            box.setWindowTitle("Delete image?")
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            delete = box.exec()

        if delete == QMessageBox.Ok or force:
            self.data.files.pop(self.currentImage.row())
            self.data.treemodel.remove_image(self.currentImage)

            if self.currentImage.row() == len(self.data.files):
                if self.currentImage.row() == 0:
                    self.mpl_widget.draw_white()
                    return
                else:
                    offset = -1
            else:
                offset = 1

            self.currentImage = self.data.treemodel.index(self.currentImage.row() + offset, 0, self.currentImage.parent())
            self.mpl_widget.set_image(self.data.files[self.currentImage.row()], self.outline_switch)

    def reset_overview(self):
        if self.currentImage is not None:
            if self.overview_switch:
                self.mpl_widget.dezoom()
                self.switch_overview_button()
                if self.currentObject is not None:
                    self.oldObject = self.currentObject
                self.currentObject = None
            else:
                if self.oldObject is not None:
                    if self.data.files[self.currentImage.row()].objects[self.oldObject.row()].zoom is not None:
                        self.mpl_widget.set_object(self.data.files[self.currentImage.row()].objects[self.oldObject.row()])
                        self.mpl_widget.zoom()
                        self.currentObject = self.oldObject
                        self.switch_overview_button()

    def switch_pan(self, signal=None):
        self.toolbar.pan()
        self.pan_switch = not self.pan_switch

        if signal is not None:
            if self.zoom_switch:
                self.switch_zoom()

        if self.pan_switch:
            self.panbutton.setText('Deactivate pan')
        else:
            self.panbutton.setText('Activate pan')

    def switch_zoom(self, signal=None):
        self.toolbar.zoom()
        self.zoom_switch = not self.zoom_switch

        if signal is not None:
            if self.pan_switch:
                self.switch_pan()

        if self.zoom_switch:
            self.zoombutton.setText('Deactivate zoom')
        else:
            self.zoombutton.setText('Activate zoom')

    def switch_overview_button(self):
        self.overview_switch = not self.overview_switch

        if self.currentObject is None or self.currentImage is None:
            self.overviewbutton.setText('Image overview')
            self.overview_switch = True
        else:
            if self.overview_switch:
                self.overviewbutton.setText('Image overview')
            else:
                self.overviewbutton.setText('Zoom to cell')

    def switch_outline_button(self):
        self.outline_switch = not self.outline_switch

        if self.outline_switch:
            self.outline_toggle.setText('Turn outlines off')
        else:
            self.outline_toggle.setText('Turn outlines on')

    def plot(self, model_index):
        self.mpl_widget.setFocus()

        if self.imageParent is None:
            self.imageParent = model_index.parent()

        if model_index.parent() == self.imageParent:
            if model_index is not self.currentImage:
                self.currentImage = model_index
                self.currentObject = None
                try:
                    self.mpl_widget.set_image(self.data.files[model_index.row()], self.outline_switch)
                except ValueError:
                    box = QMessageBox()
                    box.setBaseSize(400, 150)
                    box.setIcon(QMessageBox.Warning)
                    box.setStandardButtons(QMessageBox.Ok)
                    box.setText("Image could be displayed.")
                    box.setInformativeText('Selected image is corrupted or of wrong type. Image was removed from list.')
                    box.setWindowTitle("Invalid image.")
                    box.exec_()

                    self.remove_image(force=True)

        elif model_index is not self.currentObject:
            self.class_list.clearSelection()
            if not self.overview_switch:
                self.switch_overview_button()
            index = self.data.files[self.currentImage.row()].objects[model_index.row()].classtype
            index_object = self.data.listmodel.index(index, 0, 'rootparent')
            self.class_list.selectionModel().select(index_object, QItemSelectionModel.Select)
            self.currentObject = model_index
            self.mpl_widget.set_object(self.data.files[self.currentImage.row()].objects[model_index.row()])

    def set_class(self, class_index):
        self.currentClass = class_index
        self.data.files[self.currentImage.row()].objects[self.currentObject.row()].set_class(class_index.row())
        self.mpl_widget.set_class()

    def toggle_outline(self):
        self.switch_outline_button()
        if self.currentImage is not None:
            self.mpl_widget.set_image(self.data.files[self.currentImage.row()], self.outline_switch)
            if self.currentObject is not None and self.overview_switch:
                self.mpl_widget.zoom()
                self.mpl_widget.set_object(self.data.files[self.currentImage.row()].objects[self.currentObject.row()])

    def closeEvent(self, event):
        box = QMessageBox()
        box.setIcon(QMessageBox.Question)
        box.setBaseSize(300, 150)
        box.setText("Quit program?")
        box.setInformativeText("All unsaved data will be lost!")
        box.setWindowTitle("Quit program?")
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        answer = box.exec()
        if answer == QMessageBox.Ok:
            event.accept()
        else:
            event.ignore()


class Data(QObject):
    """wow"""
    class_n_changed = pyqtSignal(int)
    loadingfailed = pyqtSignal(int)

    def __init__(self):
        super(QObject, self).__init__()

        self.files = []
        self.filenames = []
        self.paths = []
        self.savepath = None
        self.max_class = init_class_n

        self.treemodel = ItemModel(self, 'tree')
        self.listmodel = ItemModel(init_class_n, 'list')

        self.dialog = FileDialog()

    def get_data(self):
        error = False
        success = False
        self.paths = self.dialog.open_multiple_images()
        if self.paths == [] or self.paths is None:
            return False

        box = QMessageBox()
        box.setIcon(QMessageBox.Question)
        box.setBaseSize(400, 150)
        box.setText("Generate or load templates?")
        box.setWindowTitle("Generate or load templates?.")
        box.setInformativeText("Do you want to generate, load or not use presegmented templates for labelling?")
        box.addButton('Generate', QMessageBox.AcceptRole)
        box.addButton('Load', QMessageBox.RejectRole)
        box.addButton("No templates", QMessageBox.ActionRole)
        load_preseg = box.exec_()

        if load_preseg == 1:
            preseg = []
            presegpaths = []
            labels = self.dialog.open_multiple_images()
            if labels == [] or labels is None:
                return False
            if len(self.paths) != len(labels):
                self.loadingfailed.emit(0)
                return
            if len(self.paths) == 1:
                for label in labels:
                    if label.endswith('.npy'):
                        preseg.append(np.load(label))
                        presegpaths.append(label)
                    else:
                        try:
                            preseg.append(imread(label))
                            presegpaths.append(label)
                        except:
                            self.loadingfailed.emit(1)
                            return
            else:
                imagenames, labelnames = [], []
                for n in range(len(self.paths)):
                    imagenames.append(os.path.split(self.paths[n])[-1])
                    labelnames.append(os.path.split(labels[n])[-1])
                if imagenames != labelnames:
                    self.loadingfailed.emit(2)
                    return
                else:
                    for label in labels:
                        if label.endswith('.npy'):
                            preseg.append(np.load(label))
                            presegpaths.append(label)
                        else:
                            try:
                                preseg.append(imread(label))
                                presegpaths.append(label)
                            except:
                                self.loadingfailed.emit(1)
                                return

        elif load_preseg == 2:
            preseg = False

        for n, file in enumerate(self.paths):
            try:
                if load_preseg == 1:
                    sample = Sample(file, presegmentation=preseg[n], presegpath=presegpaths[n])
                elif load_preseg == 2:
                    sample = Sample(file, presegmentation=preseg)
                else:
                    sample = Sample(file)
                success = True
            except:
                error = True
                continue

            self.files.append(sample)

        self.treemodel = ItemModel(self, 'tree')

        if error:
            self.loadingfailed.emit(1)

        return success

    def export_labels(self):
        path = self.dialog.export()

        if path == '':
            return

        box = QMessageBox()
        box.setIcon(QMessageBox.Warning)
        box.setText("Please choose label format.")
        box.setWindowTitle("Please choose label format.")
        box.setInformativeText("Do you want to save the labels as images or binaries?.")
        box.addButton('Image', QMessageBox.AcceptRole)
        box.addButton('Binary', QMessageBox.RejectRole)

        label_is_image = box.exec_()

        if label_is_image == 0 or label_is_image == 1:
            for file in self.files:
                image = file.image
                label_img = np.zeros_like(image, dtype=np.uint8)

                for obj in file.objects:
                    if obj.x is not None and len(obj.x) > 5:
                        x, y = polygon(obj.x, obj.y)
                        if self.max_class == 1:
                            label_img[y, x] = 255
                        else:
                            label_img[y, x] = obj.classtype + 1
                    else:
                        if obj.preseg is not None:
                            if self.max_class == 1:
                                label_img[obj.preseg] = 255
                            else:
                                label_img[obj.preseg] = obj.classtype + 1

                if label_is_image == 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imsave(os.path.join(path, file.name), label_img)
                else:
                    np.save(os.path.join(path, file.name.split('.')[0] + '.npy'), label_img)

    def load_project(self):
        skip = False
        success = False
        filename = self.dialog.open_project()
        if filename == '':
            return False

        with open(filename, "rb") as path:
            projectfile = pickle.load(path)

        preseg_n = 0
        for n, sample in enumerate(projectfile):
            if n == 0:
                preseg_load = sample[1]
                continue
            self.paths.append(sample.path)
            preseg_n += 1

        if 1 in preseg_load:
            preseg = []
            labelfolder = self.dialog.export()
            labels = glob(os.path.join(labelfolder, '*.*'))

            for m, sample in enumerate(projectfile):
                if m == 0:
                    pass
                elif preseg_load[m-1] == 0:
                    preseg.append(0)
                elif preseg_load[m-1] == 2:
                    preseg.append(2)
                else:
                    if os.path.join(labelfolder, sample.outlines_path) not in labels:
                        preseg.append(False)
                        continue
                    else:
                        label = os.path.join(labelfolder, sample.outlines_path)
                    if label.endswith('.npy'):
                        preseg.append(np.load(label))
                    else:
                        try:
                            preseg.append(imread(label))
                        except:
                            preseg.append(False)
                            continue
        else:
            preseg = preseg_load

        for n, sample in enumerate(projectfile):
            if n == 0:
                self.set_class_max(sample[0])
                continue
            try:
                sample.load_image()
            except:
                preseg[n-1] = False
                skip = True
                continue

            if type(preseg[n-1]) not in [int, bool]:
                sample.get_segmentation(presegmentation=preseg[n - 1], import_objects=True)
            elif preseg[n-1] == 2:
                sample.get_segmentation(import_objects=True)
            elif not preseg[n-1]:
                skip = True
                continue

            success = True
            self.files.append(sample)
            self.treemodel = ItemModel(self, 'tree')

        if skip:
            self.loadingfailed.emit(3)

        return success

    def save_project(self, filename=None):
        if type(filename) is not str:
            filename = self.dialog.save_project()

        if filename == '':
            return

        projectfile = []
        preseg = []
        projectfile.append([self.max_class])
        for sample in self.files:
            sample_tmp = copy(sample)
            preseg.append(sample.outlines_gen)
            sample_tmp.image = None
            sample_tmp.outlines = None
            projectfile.append(sample_tmp)

        projectfile[0].append(preseg)

        if not filename.endswith('.pickle'):
            filename = filename + '.pickle'

        with open(filename, "wb") as path:
            pickle.dump(projectfile, path)

    def set_class_max(self, upchange):
        if upchange is True:
            self.max_class += 1
        elif upchange is False:
            self.max_class -= 1
        elif type(upchange) == int:
            change = self.max_class - upchange
            self.max_class = upchange
            self.class_n_changed.emit(change)


class Sample:
    def __init__(self, path, presegmentation=None, presegpath=None):
        self.path = path
        self.name = os.path.split(self.path)[-1]

        self.image = imread(path)
        self.objects = []
        self.outlines_path = None
        self.outlines = None
        self.outlines_gen = 0

        self.get_segmentation(presegmentation, presegpath=presegpath)

    def get_segmentation(self, presegmentation=None, import_objects=False, presegpath=None):
        if presegmentation is not False:
            if presegmentation is None:
                self.outlines_gen = 2
                thresh = threshold_otsu(self.image)
                binary = self.image > thresh
            else:
                self.outlines_gen = 1
                if not import_objects:
                    self.outlines_path = os.path.split(presegpath)[-1]
                binary = presegmentation
                binary[binary > 1] = 1

            if binary.shape[0] != self.image.shape[0] or binary.shape[1] != self.image.shape[1]:
                raise ValueError

            binary = binary_closing(binary)
            binary_filled = binary_fill_holes(binary)
            cleared = clear_border(binary_filled)
            label_img = label(cleared)
            label_img = remove_small_objects(label_img, min_size=otsu_min_size)

            pre_outlines = np.zeros_like(label_img, dtype=np.uint8)
            pre_outlines[label_img > 0] = 1
            pre_outlines = pre_outlines - binary_erosion(pre_outlines)

            rgba_outlines = np.zeros((self.image.shape[0], self.image.shape[1], 4))
            rgba_outlines[pre_outlines == 1] = [1, 0, 0, 1]
            rgba_outlines[pre_outlines == 0] = [0, 0, 0, 0]

            self.outlines = rgba_outlines

            if not import_objects:
                regions = regionprops(label_img)
                for n, region in enumerate(regions):
                    bbox = np.asarray(region.bbox)

                    if bbox[0] - zoom_buffer >= 0:
                        bbox[0] -= zoom_buffer
                    else:
                        bbox[2] += (zoom_buffer - bbox[0])
                        bbox[0] = 0

                    if bbox[2] + zoom_buffer <= self.image.shape[0]:
                        bbox[2] += zoom_buffer
                    else:
                        bbox[0] -= (zoom_buffer -
                                    ((self.image.shape[0] + zoom_buffer) - self.image.shape[0]))
                        bbox[2] = self.image.shape[0]

                    if bbox[1] - zoom_buffer >= 0:
                        bbox[1] -= zoom_buffer
                    else:
                        bbox[3] += (zoom_buffer - bbox[1])
                        bbox[1] = 0

                    if bbox[3] + zoom_buffer <= self.image.shape[1]:
                        bbox[3] += zoom_buffer
                    else:
                        bbox[1] -= (zoom_buffer -
                                    ((self.image.shape[1] + zoom_buffer) - self.image.shape[1]))
                        bbox[3] = self.image.shape[1]
                    new_object = Object(n)
                    new_object.set_preseg(np.where(label_img == region.label))
                    new_object.set_zoom(bbox)
                    new_object.set_centroid(region.centroid)
                    self.objects.append(new_object)
        else:
            self.objects.append(Object(0))

    def load_image(self):
        self.image = imread(self.path)


class Object:
    def __init__(self, number, parent=None, suffix=None, zoom=None):
        if suffix is not None:
            self.name = 'Object ' + str(int(number))
            self.name = self.name + '.' + str(int(suffix)+1)
        else:
            self.name = 'Object ' + str(int(number) + 1)

        self.x = None
        self.y = None
        self.preseg = None

        self.classtype = 0
        self.parent = parent

        self.centroid = None
        self.zoom = zoom

    def set_zoom(self, bbox):
        self.zoom = [bbox[1], bbox[3], bbox[0], bbox[2]]

    def set_class(self, classtype):
        self.classtype = classtype

    def set_centroid(self, centroid):
        self.centroid = centroid

    def set_parent(self, parent):
        self.parent = parent

    def set_preseg(self, preseg):
        self.preseg = preseg


class Backend(QObject):
    new_coords = pyqtSignal(list)
    ask_redraw = pyqtSignal(int)

    def __init__(self):
        super(QObject, self).__init__()

    @pyqtSlot(list)
    def get_lasso_area(self, lasso):
        x, y = [], []

        if len(lasso) < 4:
            return

        l1, l2 = np.array_split(lasso, 2)
        templine1 = LineString(l1)
        templine2 = LineString(l2)

        crosspoint = templine1.intersection(templine2)

        if type(crosspoint) == collection.GeometryCollection:
            if list(crosspoint) == []:
                cell1 = LinearRing(lasso)
                cell1 = LineString(cell1)
                coordseq = list(cell1.coords)
                for tuple in coordseq:
                    x.append(tuple[0])
                    y.append(tuple[1])

                self.new_coords.emit([x, y])
                return
            else:
                self.ask_redraw.emit(1)
                return

        else:
            try:
                crosspoint = self.get_intersection_point(crosspoint)
            except ValueError:
                self.ask_redraw.emit(1)
                return

        templine1_mp = MultiPoint(templine1.coords)
        templine2_mp = MultiPoint(templine2.coords)
        crosspoint1 = nearest_points(templine1_mp, crosspoint)[0]
        crosspoint2 = nearest_points(templine2_mp, crosspoint)[0]

        split1 = split(templine1, crosspoint1)
        split2 = split(templine2, crosspoint2)

        if len(split1) < 2 or len(split2) < 2:
            return

        multi = []
        if split1[0].contains(Point(templine1.coords[-1])) or split1[0].contains(Point(templine2.coords[0])):
            multi.append(split1[0])
            multi.append(split2[1])
        else:
            multi.append(split1[1])
            multi.append(split2[0])

        for m, line in enumerate(multi):
            coordseq = list(line.coords)
            for n, coord in enumerate(coordseq):
                if n == 0 and m == 0:
                    firstcoord = coord
                x.append(coord[0])
                y.append(coord[1])

        x.append(firstcoord[0])
        y.append(firstcoord[1])

        self.new_coords.emit([x, y])

    @pyqtSlot(list)
    def get_multi_area(self, lassolist):
        if len(lassolist[1]) < 4:
            return

        x, y = [], []
        maincell = LineString(lassolist[0])
        newline = LineString(lassolist[1])

        if newline.intersects(maincell):
            crosspoint_temp = maincell.intersection(newline)

            if type(crosspoint_temp) != MultiPoint:
                self.ask_redraw.emit(2)
                return
            else:
                if len(crosspoint_temp) > 2:
                    self.ask_redraw.emit(4)
                    return

            multi = []
            line_mp = MultiPoint(newline.coords)
            cell1_mp = MultiPoint(maincell.coords)
            crosspoint1 = nearest_points(line_mp, crosspoint_temp[0])[0]
            crosspoint1_old = nearest_points(cell1_mp, crosspoint_temp[0])[0]
            split1 = split(newline, crosspoint1)
            split1_old = split(maincell, crosspoint1_old)
            crosspoint2 = nearest_points(line_mp, crosspoint_temp[1])[0]
            crosspoint2_old = nearest_points(cell1_mp, crosspoint_temp[1])[0]

            crosspoints = []
            for crosspoint in [crosspoint1, crosspoint1_old, crosspoint2, crosspoint2_old]:
                try:
                    crosspoints.append(self.get_intersection_point(crosspoint))
                except ValueError:
                    self.ask_redraw.emit(1)
                    return

            crosspoint1, crosspoint1_old, crosspoint2, crosspoint2_old = crosspoints

            if split1[0].distance(crosspoint_temp[1]) < 0.1:
                split2 = split(split1[0], crosspoint2)
            elif split1[1].distance(crosspoint_temp[1]) < 0.1:
                split2 = split(split1[1], crosspoint2)
            else:
                self.ask_redraw.emit(0)

            if split2[0].distance(crosspoint1) < 1 and split2[0].distance(crosspoint2) < 1:
                multi.append(split2[0])
            elif split2[1].distance(crosspoint1) < 1 and split2[1].distance(crosspoint2) < 1:
                multi.append(split2[1])
            else:
                self.ask_redraw.emit(0)

            if split1_old[0].distance(crosspoint2_old) < 0.1:
                split2_old = split(split1_old[0], crosspoint2_old)
            elif split1_old[1].distance(crosspoint2_old) < 0.1:
                split2_old = split(split1_old[1], crosspoint2_old)
            else:
                self.ask_redraw.emit(0)

            if split2_old[0].distance(Point(maincell.coords[0])) < 0.1 or split2_old[0].distance(
                    Point(maincell.coords[-1])) < 0.1:
                multi.append(split2_old[1])
            elif split2_old[1].distance(Point(maincell.coords[0])) < 0.1 or split2_old[1].distance(
                    Point(maincell.coords[-1])) < 0.1:
                multi.append(split2_old[0])
            else:
                self.ask_redraw.emit(0)

            for coord in list(multi[0].coords):
                x.append(coord[0])
                y.append(coord[1])
            coordseq = list(multi[1].coords)
            if Point((x[-1], y[-1])).distance(Point(coordseq[0])) < Point((x[-1], y[-1])).distance(Point(coordseq[-1])):
                for coord in coordseq:
                    x.append(coord[0])
                    y.append(coord[1])
            else:
                coordseq_rev = coordseq[::-1]
                for coord in coordseq_rev:
                    x.append(coord[0])
                    y.append(coord[1])

            self.new_coords.emit([x, y])

        else:
            self.ask_redraw.emit(2)

    def get_intersection_point(self, crosspoints):
        if type(crosspoints) == Point:
            return crosspoints
        elif type(crosspoints) == LineString:
            crosspoints = list(crosspoints.coords)
        elif type(crosspoints) == MultiLineString:
            tmp = []
            for linestring in crosspoints:
                tmp.append(self.get_intersection_point(linestring))
                crosspoints = tmp
        elif type(crosspoints) == MultiPoint:
            tmp = []
            for point in crosspoints:
                tmp.append(list(point.coords)[0])
            crosspoints = tmp
        else:
            raise ValueError

        crosspoints = np.round(crosspoints)
        startx, starty = crosspoints[0]
        for point in crosspoints:
            if abs(point[0] - startx) > 1 or abs(point[1] - starty) > 1:
                raise ValueError
            else:
                x = np.round(np.mean(crosspoints, axis=0)[0])
                y = np.round(np.mean(crosspoints, axis=0)[1])
                return Point([x, y])


class SaveTimer(QObject):
    savingtime = pyqtSignal()

    def __init__(self, pause_duration=600):
        super(QObject, self).__init__()
        self.pause_duration = pause_duration

    @pyqtSlot()
    def saving_timer(self):
        while True:
            time.sleep(self.pause_duration)
            self.savingtime.emit()


class MplCanvas(FigureCanvas):
    uparrow = pyqtSignal()
    downarrow = pyqtSignal()
    upclass = pyqtSignal()
    downclass = pyqtSignal()

    switch_outlines = pyqtSignal()
    switch_overview = pyqtSignal()
    delete_object = pyqtSignal()
    add_subobject = pyqtSignal()
    delete_image = pyqtSignal()

    switch_zoom = pyqtSignal()
    switch_pan = pyqtSignal()

    lasso_drawn = pyqtSignal(list)
    multi_drawn = pyqtSignal(list)
    ask_redraw = pyqtSignal(int)

    def __init__(self, parent=None):
        self.currentSample = None
        self.currentImage = None
        self.currentObject = None
        self.lasso = None
        self.centroid = None
        self.repeatdraw = False
        self.colormap = plt.cm.gist_rainbow(np.linspace(0, 1, init_class_n))

        self.linedict = {}

        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        self.axes.invert_yaxis()
        self.axes.axis('off')

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def onselect(self, verts):
        failure = False
        if self.currentObject.parent is None:
            self.lasso_drawn.emit(verts)
        else:
            mainobject = []
            try:
                x = self.currentObject.parent.x
                y = self.currentObject.parent.y

                for n in range(len(x)):
                    mainobject.append((x[n], y[n]))
                self.multi_drawn.emit([mainobject, verts])
            except TypeError or AttributeError:
                failure = True

        x = []
        y = []
        for vert in verts:
            x.append(vert[0])
            y.append(vert[1])
        self.currentObject.x = x
        self.currentObject.y = y

        self.draw_object([x, y])

        if self.currentObject.zoom is None:
            xmin, xmax = self.axes.get_xlim()
            ymin, ymax = self.axes.get_ylim()
            self.currentObject.zoom = [xmin, xmax, ymin, ymax]

        if failure:
            self.ask_redraw.emit(3)

    def clear(self):
        self.axes.clear()
        self.axes.invert_yaxis()
        self.axes.axis('off')

    @pyqtSlot(list)
    def draw_object(self, coords):
        if self.repeatdraw:
            try:
                self.linedict[self.currentObject].remove()
            except:
                pass
        self.currentObject.x = coords[0]
        self.currentObject.y = coords[1]
        self.linedict[self.currentObject], = self.axes.plot(coords[0], coords[1], c=self.colormap[self.currentObject.classtype])
        self.repeatdraw = True
        self.draw()

    def remove_object(self):
        try:
            self.linedict[self.currentObject].remove()
            self.draw()
        except:
            pass

        self.repeatdraw = False
        self.currentObject.x = None
        self.currentObject.y = None

    def draw_white(self):
        self.clear()
        img = self.axes.imshow(np.zeros((10, 10)), cmap='Greys', dtype=np.uint8)
        self.draw()

    def keyPressEvent(self, event):
        key = self._get_key(event)

        if key is not None:
            if key == 'up' or key == 'w':
                self.uparrow.emit()
            elif key == 'down' or key == 's':
                self.downarrow.emit()
            elif key == ' ' or key == '.' or key == 'r':
                self.switch_outlines.emit()
            elif key == 'f' or key == '/' or key == '-':
                self.switch_overview.emit()
            elif key == 'e':
                self.upclass.emit()
            elif key == 'd':
                self.downclass.emit()
            elif key == 'c':
                self.switch_pan.emit()
            elif key == 'x':
                self.switch_zoom.emit()
            elif key == 'right' or key == 'a':
                self.add_subobject.emit()
            elif key == 'left' or key == 'q':
                self.delete_object.emit()
            else:
                FigureCanvasBase.key_press_event(self, key, guiEvent=event)

    def plot(self, outlines_b):
        self.axes.clear()
        self.axes.invert_yaxis()
        self.axes.axis('off')
        self.axes.axis([0, self.currentImage.shape[1], 0, self.currentImage.shape[0]])

        try:
            img = self.axes.imshow(self.currentImage, cmap='Greys')
        except:
            self.delete_image.emit()
            return

        if self.currentSample.outlines is not None and outlines_b is True:
            self.axes.imshow(self.currentSample.outlines, cmap='Greys', interpolation='nearest')

        self.draw()

    def remove_centroid(self):
        try:
            self.centroid.remove()
        except:
            pass

    def set_colormap(self, class_n):
        self.colormap = plt.cm.rainbow(np.linspace(0, 1, class_n))

    def set_image(self, currentFile, outlines_b):
        self.currentSample = currentFile
        self.currentImage = self.currentSample.image
        self.lasso = None

        self.remove_centroid()

        self.plot(outlines_b=outlines_b)

        for obj in range(len(self.currentSample.objects)):
            if self.currentSample.objects[obj].x is not None:
                self.axes.plot(self.currentSample.objects[obj].x, self.currentSample.objects[obj].y,
                               c=self.colormap[self.currentSample.objects[obj].classtype])
                self.draw()

    def set_object(self, currentObject):
        self.currentObject = currentObject
        self.repeatdraw = False
        self.lasso = LassoSelector(self.axes, self.onselect)

        if self.currentObject is not None:
            if self.currentObject.x is not None:
                self.repeatdraw = True
            else:
                self.repeatdraw = False

            if self.currentObject.zoom is not None:
                self.zoom()

    def set_class(self):
        if self.currentObject.x is not None:
            self.axes.plot(self.currentObject.x, self.currentObject.y, c=self.colormap[self.currentObject.classtype])
            self.draw()

    def dezoom(self):
        self.remove_centroid()
        self.axes.axis([0, self.currentImage.shape[1], 0, self.currentImage.shape[0]])
        self.draw()

    def zoom(self):
        if self.currentObject.zoom is not None:
            self.remove_centroid()
            self.axes.axis([self.currentObject.zoom[0], self.currentObject.zoom[1],
                            self.currentObject.zoom[2], self.currentObject.zoom[3]])
            if self.currentObject.centroid is not None:
                self.centroid = self.axes.scatter(self.currentObject.centroid[1], self.currentObject.centroid[0], s=50, c='b')
            self.draw()


class FileDialog(QWidget):
    def __init__(self):
        super().__init__()
        pass

    def export(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder = QFileDialog.getExistingDirectory(self, "Select export path.", "", options=options)

        return folder

    def open_project(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Pickle Files (*.pickle)", options=options)

        return fileName

    def open_multiple_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select files to load.", "",
                                                "Tif Files (*.tif);;All Files (*)", options=options)
        return files

    def save_project(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getSaveFileName(self, "Select saving path.", "",
                                                  "Pickle Files (*.pickle)", options=options)

        return file


class TreeItem:
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.childItems = []

    def child(self, row):
        return self.childItems[row]

    def childCount(self):
        return len(self.childItems)

    def childNumber(self):
        if self.parentItem != None:
            return self.parentItem.childItems.index(self)
        return 0

    def columnCount(self):
        return len(self.itemData)

    def data(self, column):
        return self.itemData[column]

    def insertChildren(self, position, count, columns):
        if position < 0 or position > len(self.childItems):
            return False

        for row in range(count):
            data = [None for v in range(columns)]
            item = TreeItem(data, self)
            self.childItems.insert(position, item)

        return True

    def insertColumns(self, position, columns):
        if position < 0 or position > len(self.itemData):
            return False

        for column in range(columns):
            self.itemData.insert(position, None)

        for child in self.childItems:
            child.insertColumns(position, columns)

        return True

    def parent(self):
        return self.parentItem

    def removeChildren(self, position, count):
        if position < 0 or position + count > len(self.childItems):
            return False

        for row in range(count):
            self.childItems.pop(position)

        return True

    def removeColumns(self, position, columns):
        if position < 0 or position + columns > len(self.itemData):
            return False

        for column in range(columns):
            self.itemData.pop(position)

        for child in self.childItems:
            child.removeColumns(position, columns)

        return True

    def setData(self, column, value):
        if column < 0 or column >= len(self.itemData):
            return False

        self.itemData[column] = value

        return True


class Treeview(QColumnView):
    enterview = pyqtSignal(QModelIndex)

    def __init__(self, parent):
        QColumnView.__init__(self, parent)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.enterview.emit(self.currentIndex())
        elif event.key() == Qt.Key_F2:
            self.rename()
        elif event.key() == Qt.Key_Delete:
            self.delete()
        elif event.key() == Qt.Key_Backspace:
            self.go_to_parent_directory()
        else:
            QColumnView.keyPressEvent(self, event)


class ItemModel(QAbstractItemModel):
    def __init__(self, data, modeltype, parent=None):
        super(ItemModel, self).__init__(parent)

        self.rootItem = TreeItem(("",))
        self.modeltype = modeltype
        self.setupModelData(data, modeltype, self.rootItem)

    def remove_image(self, index):
        if self.modeltype == 'tree':
            indexObject = self.index(index.row(), 0, 'rootparent')
            self.beginRemoveRows(indexObject, index.row(), index.row())
            self.rootItem.removeChildren(index.row(), 1)
            self.endRemoveRows()
            self.layoutChanged.emit()

    def add_object(self, parent, index):
        if self.modeltype == 'tree':
            if index is None:
                self.rootItem.child(parent.row()).insertChildren(self.rootItem.child(parent.row()).childCount(),
                                                                 1, self.rootItem.columnCount())
                if self.rootItem.child(parent.row()).childCount() > 1:
                    self.rootItem.child(parent.row()).child(self.rootItem.child(parent.row()).childCount() - 1).\
                        setData(0, 'Object ' + self.get_subname(parent, index))
                else:
                    self.rootItem.child(parent.row()).child(self.rootItem.child(parent.row()).childCount() - 1). \
                        setData(0, 'Object 1')
            else:
                name, offset = self.get_subname(parent, index)
                self.rootItem.child(parent.row()).insertChildren(index.row() + 1 + offset, 1, self.rootItem.columnCount())
                self.rootItem.child(parent.row()).child(index.row() + 1 + offset).setData(0, name)

            self.layoutChanged.emit()

    def remove_object(self, parent, index):
        if self.modeltype == 'tree':
            self.beginRemoveRows(parent, index.row(), index.row())
            self.rootItem.child(parent.row()).removeChildren(index.row(), 1)
            self.endRemoveRows()
            self.layoutChanged.emit()

    def remove_all_objects(self, parent):
        if self.modeltype == 'tree':
            self.beginRemoveRows(parent, 0, self.rootItem.child(parent.row()).childCount())
            self.rootItem.child(parent.row()).removeChildren(0, self.rootItem.child(parent.row()).childCount())
            self.endRemoveRows()
            self.layoutChanged.emit()

    def add_class(self):
        if self.modeltype == 'list':
            self.rootItem.insertChildren(self.rootItem.childCount(), 1, self.rootItem.columnCount())
            self.rootItem.child(self.rootItem.childCount() - 1).setData(0, 'Class ' + str(self.rootItem.childCount()))
            self.layoutChanged.emit()

    def remove_class(self):
        if self.modeltype == 'list':
            indexObject = self.index(-1, 0, 'rootparent')
            self.beginRemoveRows(indexObject.parent(), -1, -1)
            self.rootItem.removeChildren(self.rootItem.childCount() - 1, 1)
            self.endInsertRows()
            self.layoutChanged.emit()

    def get_subname(self, parent, index):
        if index is None:
            name = self.rootItem.child(parent.row()).child(self.rootItem.child(parent.row()).childCount() - 2).data(0)
            name = name.split(' ')[-1]
            try:
                name, suffix = name.split('.')
            except ValueError:
                pass
            name = str(int(name) + 1)

            return name
        else:
            name = self.rootItem.child(parent.row()).child(index.row()).data(0)
            n = 0
            while True:
                try:
                    nextname = self.rootItem.child(parent.row()).child(index.row() + n+1).data(0)
                except IndexError:
                    if '.' in name:
                        number, suffix = name.split('.')
                        name = number + '.' + str(int(suffix) + 1)
                        break
                    else:
                        name = name + '.' + str(n + 1)
                        break
                if '.' in name:
                    if not '.' in nextname:
                        name = name.split('.')[0] + '.' + str(int(name.split('.')[-1]) + 1 + n)
                        break
                    else:
                        n += 1
                elif '.' in nextname:
                    n += 1
                else:
                    name = name + '.' + str(n+1)
                    break

            return name, n

    def columnCount(self, parent=QModelIndex()):
        return self.rootItem.columnCount()

    def data(self, index, role):
        try:
            if not index.isValid():
                return None

            if role != Qt.DisplayRole and role != Qt.EditRole:
                return None

            item = self.getItem(index)
            return item.data(index.column())
        except:
            return None

    def flags(self, index):
        if not index.isValid():
            return 0

        return Qt.ItemIsEditable | super(ItemModel, self).flags(index)

    def getItem(self, index):
        if index.isValid():
            item = index.internalPointer()
            if item:
                return item

        return self.rootItem

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.rootItem.data(section)

        return None

    def index(self, row, column, parent=QModelIndex()):
        if parent == 'rootparent':
            parentItem = self.rootItem
        else:
            if not self.hasIndex(row, column, parent):
                return QModelIndex()
            if not parent.isValid():
                parentItem = self.rootItem
            else:
                parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QModelIndex()

    def insertColumns(self, position, columns, parent=QModelIndex()):
        self.beginInsertColumns(parent, position, position + columns - 1)
        success = self.rootItem.insertColumns(position, columns)
        self.endInsertColumns()

        return success

    def insertRows(self, position, rows, parent=QModelIndex()):
        parentItem = self.getItem(parent)
        self.beginInsertRows(parent, position, position + rows - 1)
        success = parentItem.insertChildren(position, rows,
                self.rootItem.columnCount())
        self.endInsertRows()

        return success

    def parent(self, index):
        try:
            if not index.isValid():
                return QModelIndex()

            childItem = self.getItem(index)
            parentItem = childItem.parent()

            if parentItem == self.rootItem:
                return QModelIndex()

            return self.createIndex(parentItem.childNumber(), 0, parentItem)

        except:
            return QModelIndex()

    def removeColumns(self, position, columns, parent=QModelIndex()):
        self.beginRemoveColumns(parent, position, position + columns - 1)
        success = self.rootItem.removeColumns(position, columns)
        self.endRemoveColumns()

        if self.rootItem.columnCount() == 0:
            self.removeRows(0, self.rowCount())

        return success

    def removeRows(self, position, rows, parent=QModelIndex()):
        parentItem = self.getItem(parent)

        self.beginRemoveRows(parent, position, position + rows - 1)
        success = parentItem.removeChildren(position, rows)
        self.endRemoveRows()

        return success

    def rowCount(self, parent=QModelIndex()):
        parentItem = self.getItem(parent)

        return parentItem.childCount()

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole:
            return False

        item = self.getItem(index)
        result = item.setData(index.column(), value)

        if result:
            self.dataChanged.emit(index, index)

        return result

    def setHeaderData(self, section, orientation, value, role=Qt.EditRole):
        if role != Qt.EditRole or orientation != Qt.Horizontal:
            return False

        result = self.rootItem.setData(section, value)
        if result:
            self.headerDataChanged.emit(orientation, section, section)

        return result

    def setupModelData(self, data, modeltype, parent):
        if modeltype == 'list':
            for n in range(data):
                parent.insertChildren(parent.childCount(), 1, self.rootItem.columnCount())
                parent.child(parent.childCount() - 1).setData(0, 'Class ' + str(n + 1))
        elif modeltype == 'tree':
            parents = [parent]
            for m, key in enumerate(data.files):
                parents[0].insertChildren(parents[0].childCount(), 1, self.rootItem.columnCount())
                parents[0].child(parents[0].childCount() - 1).setData(0, key.name)

                parents.append(parents[0].child(parents[0].childCount() - 1))

                for n, obj in enumerate(key.objects):
                    parents[-1].insertChildren(parents[-1].childCount(), 1, self.rootItem.columnCount())
                    parents[-1].child(parents[-1].childCount() - 1).setData(0, obj.name)


###################################################### Functions

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VisionGui()
    window.show()
    app.exec_()
