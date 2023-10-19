import sys
import os
import re
import cv2
import csv
import shutil
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QLabel,
    QHBoxLayout,
    QDesktopWidget,
    QGraphicsDropShadowEffect,
    QProgressBar,
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPixmap, QFontDatabase
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import savgol_filter

# import VirtualIMU functions
sys.path.append("../src/")
sys.path.append("../utils/")
from extract_coordinates import extractWristCoordinates
from augment import augmentMonteCarlo

if not os.path.exists("data"):
    os.mkdir("data")

plt.rcParams["font.size"] = 8
plt.rcParams['axes.facecolor'] = '#F5F5F5'
plt.rcParams['figure.facecolor'] = '#F5F5F5'

shadow_effect = QGraphicsDropShadowEffect()
shadow_effect.setOffset(0, -0.1)
shadow_effect.setBlurRadius(5)

def adaptive_medfilt(x, ws=None, ws_max=None, axis=None, mode=None):
    if isinstance(axis, int): axis = (axis,)
    if isinstance(ws, int): ws = len(axis) * (ws,)
    
    y = np.zeros_like(x)
    x_pad = np.pad(x, [2*[ws[axis.index(ax)]//2 if ax in axis else 0] for ax in range(x.ndim)], mode)
    x_stride = np.lib.stride_tricks.sliding_window_view(x_pad, ws, axis)
    x_min = np.min(x_stride, tuple(x_stride.ndim-1-np.arange(len(axis))))
    x_med = np.median(x_stride, tuple(x_stride.ndim-1-np.arange(len(axis))))
    x_max = np.max(x_stride, tuple(x_stride.ndim-1-np.arange(len(axis))))
    
    mask1 = (x_med - x_min > 0) * (x_med - x_max < 0)
    mask2 = (x - x_min > 0) * (x - x_max < 0)
    y[mask1 * mask2] = x[mask1 * mask2]
    y[mask1 * ~mask2] = x_med[mask1 * ~mask2]
    y[~mask1] = adaptive_medfilt(x, [ws[ax]+2 for ax in range(len(axis))], ws_max, axis, mode)[~mask1] if all([ws[ax]<=ws_max for ax in range(len(axis))]) else x_med[~mask1]

    return y

class MultiPageApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multi-Page App")
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height() - 200)
        self.setStyleSheet("background-color: #ffffff; font-family: manrope;")
        self.setContentsMargins(0,0,0,0)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setContentsMargins(0,0,0,0)
        self.stacked_widget.setStyleSheet("background-color: #ffffff; font-family: manrope;")

        self.page1 = QWidget()
        self.page2 = QWidget()

        self.setup_page1()
        self.setup_page2()

        self.page1.setStyleSheet("background-color: #ffffff; font-family: manrope;")
        self.page2.setStyleSheet("background-color: #ffffff; font-family: manrope;")
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stacked_widget)

        central_widget.setLayout(layout)

    def setup_page1(self):
        QFontDatabase.addApplicationFont("manrope.ttf")
        self.setWindowTitle("VirtualIMU")
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height() - 200)

        self.layout = QVBoxLayout()

        self.augment_files = os.listdir("data")
        self.augment_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

        self.augment_widgets = {}

        self.current_shown_btn = None

        ahha_label = QLabel()
        ahha_label.setText("Virtual IMU")
        ahha_label.setFixedHeight(40)
        ahha_label.setStyleSheet(
            "color: black; border-radius: 0px; font-weight: bold; font-size: 35px; margin-top: 12px; padding: 0px;"
        )

        # 1 for external camera
        self.camera = cv2.VideoCapture(0)

        self.record_video = False
        self.video_writer = None

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(1100, 820)

        self.prepare_overlay = OverlayWidget(self.image_label)
        self.prepare_overlay.show()

        self.record_button = QPushButton("Start Recording", self)
        self.record_button.clicked.connect(self.prepare_record)
        self.record_button.setStyleSheet(
            "QPushButton { background-color: #1b9aaa; border-radius: 10px; margin: 0px; padding: 0px; color: white; font-size: 20px;}"
            "QPushButton:pressed { background-color: #115F6A; }"
        )
        self.record_button.setFixedSize(200, 40)

        self.plot_button = QPushButton("See Data", self)
        self.plot_button.clicked.connect(self.plot_graph)
        self.plot_button.setStyleSheet(
            "QPushButton { background-color: #1b9aaa; border-radius: 10px; margin: 0px; padding: 0px; color: white; font-size: 20px;}"
            "QPushButton:pressed { background-color: #115F6A; }"
        )
        self.plot_button.setFixedSize(200, 40)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(1040)
        self.setStyleSheet("QProgressBar::chunk "
                  "{"
                    "background-color: #cf1d5f;"
                  "}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(35)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.plot_button)
        button_layout.setContentsMargins(0,0,30,0)
        button_layout.setSpacing(12)

        button_widget = QWidget()
        button_widget.setLayout(button_layout)

        record_layout = QVBoxLayout()
        record_layout.addWidget(ahha_label)
        record_layout.addWidget(self.image_label)
        record_layout.addWidget(self.progress_bar)
        record_layout.addWidget(button_widget)

        record_layout.setSpacing(19)
        record_layout.setAlignment(self.image_label, Qt.AlignHCenter)
        record_layout.setAlignment(self.progress_bar, Qt.AlignHCenter)
        record_layout.setAlignment(button_widget, Qt.AlignHCenter)
        record_layout.setContentsMargins(20, 0, 0, 0)

        record_widget = QWidget()
        record_widget.setLayout(record_layout)

        self.page1 = record_widget

    def setup_page2(self):
        QFontDatabase.addApplicationFont("manrope.ttf")
        self.setWindowTitle("Virtual IMU")
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height() - 200)

        self.layout = QVBoxLayout()

        self.augment_files = os.listdir("data")
        self.augment_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

        ahha_label = QLabel()
        ahha_label.setText("Virtual IMU")
        ahha_label.setFixedHeight(40)
        ahha_label.setStyleSheet(
            "color: black; border-radius: 0px; font-weight: bold; font-size: 35px; margin-bottom: 0px; padding: 0px;"
        )

        self.prev_button = QPushButton("Back To Recording", self)
        self.prev_button.clicked.connect(self.prev_page)
        self.prev_button.setGraphicsEffect(shadow_effect)
        self.prev_button.setStyleSheet(
            "QPushButton { background-color: #1b9aaa; border-radius: 10px; margin: 0px; padding: 0px; color: white; font-size: 20px;}"
            "QPushButton:pressed { background-color: #115F6A; }"
        )
        self.prev_button.setFixedSize(200, 40)

        top_layer = QHBoxLayout()
        top_layer.addWidget(ahha_label)
        top_layer.addWidget(self.prev_button)

        top_layer_widget = QWidget()
        top_layer_widget.setLayout(top_layer)

        plotting_layout = QVBoxLayout()

        text_synthesized_imu = QLabel()
        text_synthesized_imu.setText("Mean and Standard Deviation of Synthesized IMU")
        text_synthesized_imu.setStyleSheet("color: black; font-size: 20px; font-weight: bold;")

        self.matplotlib_widget = StdMatplotlibWidget(self)
        self.matplotlib_widget.setFixedHeight(240)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(text_synthesized_imu)
        graph_layout.addWidget(self.matplotlib_widget)
        graph_layout.setSpacing(0)

        graph_widget = QWidget()
        graph_widget.setLayout(graph_layout)
        graph_widget.setStyleSheet(
            "border-radius: 10px; margin: 5px; background-color: #F5F5F5;"
        )

        best_text_synthesized_imu = QLabel()
        best_text_synthesized_imu.setText("Best Synthesized IMU")
        best_text_synthesized_imu.setStyleSheet("color: black; font-size: 20px; font-weight: bold;")

        self.best_matplotlib_widget = MatplotlibWidget(self)
        self.best_matplotlib_widget.setFixedHeight(240)
        
        best_graph_layout = QVBoxLayout()
        best_graph_layout.addWidget(best_text_synthesized_imu)
        best_graph_layout.addWidget(self.best_matplotlib_widget)
        best_graph_layout.setSpacing(0)

        best_graph_widget = QWidget()
        best_graph_widget.setLayout(best_graph_layout)
        best_graph_widget.setStyleSheet(
            "border-radius: 10px; margin: 5px; background-color: #F5F5F5;"
        )
        
        text_real_imu = QLabel()
        text_real_imu.setText("Real IMU")
        text_real_imu.setStyleSheet("color: black; font-size: 20px; font-weight: bold;")

        self.imu_matplotlib_widget = IMUMatplotlibWidget(self)
        self.imu_matplotlib_widget.setFixedHeight(240)

        imu_graph_layout = QVBoxLayout()
        imu_graph_layout.addWidget(text_real_imu)
        imu_graph_layout.addWidget(self.imu_matplotlib_widget)
        imu_graph_layout.setSpacing(0)

        imu_graph_widget = QWidget()
        imu_graph_widget.setLayout(imu_graph_layout)
        imu_graph_widget.setStyleSheet(
            "border-radius: 10px; margin: 5px; background-color: #F5F5F5;"
        )

        plotting_layout.addWidget(top_layer_widget)
        plotting_layout.addWidget(graph_widget)
        plotting_layout.addWidget(best_graph_widget)
        plotting_layout.addWidget(imu_graph_widget)

        plotting_widget = QWidget()
        plotting_widget.setLayout(plotting_layout)

        self.page2 = plotting_widget

    def prev_page(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index > 0:
            self.stacked_widget.setCurrentIndex(current_index - 1)

    def next_page(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(current_index + 1)

    def calculate_error(self, realFile, vidFile):
        y_label, y_IMU_label, time = self.getData(realFile, vidFile)
        count = min(len(y_label[0]), len(y_IMU_label[0]))
        
        new_y_label = []
        new_y_IMU_label = []

        for parameter in y_label:
            new_y_label.append(parameter[:count])

        for parameter in y_IMU_label:
            new_y_IMU_label.append(parameter[:count])

        np_vid = np.array(new_y_label)
        np_real = np.array(new_y_IMU_label)

        error_array = np.absolute(np_vid - np_real)
        error = np.sum(error_array)

        return error

    def calculate_mean(self):
        y_arrays = []
        for vidFile in self.augment_files:
            y_arrays.append(self.getVidData(vidFile))

        sum_acc_xF_left = np.array(y_arrays[0][0])
        sum_acc_yF_left = np.array(y_arrays[0][1])
        sum_acc_zF_left = np.array(y_arrays[0][2])
        sum_gyr_xF_left = np.array(y_arrays[0][3])
        sum_gyr_yF_left = np.array(y_arrays[0][4])
        sum_gyr_zF_left = np.array(y_arrays[0][5])

        for i in range(1, len(y_arrays)):
            sum_acc_xF_left += np.array(y_arrays[i][0])
            sum_acc_yF_left += np.array(y_arrays[i][1])
            sum_acc_zF_left += np.array(y_arrays[i][2])
            sum_gyr_xF_left += np.array(y_arrays[i][3])
            sum_gyr_yF_left += np.array(y_arrays[i][4])
            sum_gyr_zF_left += np.array(y_arrays[i][5])

        mean_acc_xF_left = sum_acc_xF_left / 10
        mean_acc_yF_left = sum_acc_yF_left / 10
        mean_acc_zF_left = sum_acc_zF_left / 10
        mean_gyr_xF_left = sum_gyr_xF_left / 10
        mean_gyr_yF_left = sum_gyr_yF_left / 10
        mean_gyr_zF_left = sum_gyr_zF_left / 10

        return [mean_acc_xF_left, mean_acc_yF_left, mean_acc_zF_left, mean_gyr_xF_left, mean_gyr_yF_left, mean_gyr_zF_left]

    def calculate_std(self):
        y_arrays = []
        for vidFile in self.augment_files:
            y_arrays.append(self.getVidData(vidFile))

        acc_xF_left = []
        acc_yF_left = []
        acc_zF_left = []
        gyr_xF_left = []
        gyr_yF_left = []
        gyr_zF_left = []

        for i in range(0, len(y_arrays)):
            acc_xF_left.append(y_arrays[i][0])
            acc_yF_left.append(y_arrays[i][0])
            acc_zF_left.append(y_arrays[i][0])
            gyr_xF_left.append(y_arrays[i][0])
            gyr_yF_left.append(y_arrays[i][0])
            gyr_zF_left.append(y_arrays[i][0])

        acc_xF_left = np.array(acc_xF_left).T
        acc_yF_left = np.array(acc_yF_left).T
        acc_zF_left = np.array(acc_zF_left).T
        gyr_xF_left = np.array(gyr_xF_left).T
        gyr_yF_left = np.array(gyr_yF_left).T
        gyr_zF_left = np.array(gyr_zF_left).T

        std_acc_xF_left = []
        std_acc_yF_left = []
        std_acc_zF_left = []
        std_gyr_xF_left = []
        std_gyr_yF_left = []
        std_gyr_zF_left = []

        for datapoint in acc_xF_left:
            std_acc_xF_left.append(np.std(datapoint))

        for datapoint in acc_yF_left:
            std_acc_yF_left.append(np.std(datapoint))
        
        for datapoint in acc_zF_left:
            std_acc_zF_left.append(np.std(datapoint))

        for datapoint in gyr_xF_left:
            std_gyr_xF_left.append(np.std(datapoint))

        for datapoint in gyr_xF_left:
            std_gyr_yF_left.append(np.std(datapoint))

        for datapoint in gyr_xF_left:
            std_gyr_zF_left.append(np.std(datapoint))

        return [std_acc_xF_left, std_acc_yF_left, std_acc_zF_left, std_gyr_xF_left, std_gyr_yF_left, std_gyr_zF_left]

    def plot_graph(self, btn):
        
        std_array = self.calculate_std()
        self.calculate_mean()

        min_error_file = None
        min_error = math.inf

        for vidFile in self.augment_files:
            error = self.calculate_error("current.txt", vidFile)
            if error < min_error:
                min_error_file = vidFile
                min_error = error

        filename = "synthAug0VidIMU.csv"
        y_label, y_IMU_label, time = self.getData("current.txt", filename)
        y_label = self.calculate_mean()

        acc_min = min(min(y_label[0]), min(y_IMU_label[0]), min(y_label[1]), min(y_IMU_label[1]), min(y_label[2]), min(y_IMU_label[2]))
        acc_max = max(max(y_label[0]), max(y_IMU_label[0]), max(y_label[1]), max(y_IMU_label[1]), max(y_label[2]), max(y_IMU_label[2]))
        gyro_min = min(min(y_label[3]), min(y_IMU_label[3]), min(y_label[4]), min(y_IMU_label[4]), min(y_label[5]), min(y_IMU_label[5]))
        gyro_max = max(max(y_label[3]), max(y_IMU_label[3]), max(y_label[4]), max(y_IMU_label[4]), max(y_label[5]), max(y_IMU_label[5]))

        limit = [acc_min, acc_max, gyro_min, gyro_max]

        best_y_label, best_y_IMU_label, best_time = self.getData("current.txt", min_error_file)

        best_acc_min = min(min(best_y_label[0]), min(best_y_IMU_label[0]), min(best_y_label[1]), min(best_y_IMU_label[1]), min(best_y_label[2]), min(best_y_IMU_label[2]))
        best_acc_max = max(max(best_y_label[0]), max(best_y_IMU_label[0]), max(best_y_label[1]), max(best_y_IMU_label[1]), max(best_y_label[2]), max(best_y_IMU_label[2]))
        best_gyro_min = min(min(best_y_label[3]), min(best_y_IMU_label[3]), min(best_y_label[4]), min(best_y_IMU_label[4]), min(best_y_label[5]), min(best_y_IMU_label[5]))
        best_gyro_max = max(max(best_y_label[3]), max(best_y_IMU_label[3]), max(best_y_label[4]), max(best_y_IMU_label[4]), max(best_y_label[5]), max(best_y_IMU_label[5]))

        best_limit = [best_acc_min, best_acc_max, best_gyro_min, best_gyro_max]
        
        self.matplotlib_widget.plot(time, y_label, limit, std_array)
        self.best_matplotlib_widget.plot(best_time, best_y_label, best_limit)
        self.imu_matplotlib_widget.plot(time, y_IMU_label, limit)

        current_index = self.stacked_widget.currentIndex()
        if current_index < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(current_index + 1)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            if self.record_video:
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    self.video_writer = cv2.VideoWriter(
                        "recorded_video.mp4",
                        fourcc,
                        60.0,
                        (frame.shape[1], frame.shape[0]),
                    )
                self.video_writer.write(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1080,820))
            image = QImage(
                frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)

    def prepare_record(self):
        self.prepare_timeout = 6

        self.prepare_timer = QTimer(self)
        self.prepare_timer.timeout.connect(self.change_prepare)
        self.prepare_timer.start(1000)

    def change_prepare(self):
        if self.prepare_timeout == 1:
            self.prepare_overlay.set_text("Start")
        else:
            self.prepare_overlay.set_text(str(self.prepare_timeout - 1))
        self.prepare_timeout -= 1

        if self.prepare_timeout < 0:
            self.prepare_timer.stop()
            self.prepare_timeout = 6
            self.prepare_overlay.set_text("")
            self.record()

    def record(self):

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.total_steps = 100
        self.current_step = 0
        self.timer_interval = 90
        self.progress_timer.start(self.timer_interval)

        self.timer.stop()
        if (self.camera.isOpened() == False): 
            print("Error reading video file")
        
        frame_width = int(self.camera.get(3))
        frame_height = int(self.camera.get(4))
        
        size = (frame_width, frame_height)
        
        result = cv2.VideoWriter('recorded_video.mp4', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                30., size)

        start_time = time.time()
        try:
            open("stream.txt", "w").close()
        except IOError:
            open("stream.txt", "w").close()

        while (int(time.time() - start_time) < 10):
            ret, frame = self.camera.read()
        
            if ret == True: 
        
                result.write(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (1080,820))
                image = QImage(
                    frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888
                )

                pixmap = QPixmap.fromImage(image)
                self.image_label.setPixmap(pixmap)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
        
            else:
                break

        try:
            shutil.copyfile("C:\\Users\\Ramita\\Desktop\\virtualIMU\\stream.txt", "current.txt")
        except:
            shutil.copyfile("C:\\Users\\Ramita\\Desktop\\virtualIMU\\stream.txt", "current.txt")

        result.release()
            
        cv2.destroyAllWindows()
        
        print("The video was successfully saved")

        extractWristCoordinates(
            videoFile="recorded_video.mp4", coordinatesFile="recorded_video.csv" 
        )
        df = pd.read_csv("recorded_video.csv")
        df.rename(columns={"frame": "time"}, inplace=True)
        df.iloc[:, 0] /= 30.
            
        df.to_csv("recorded_video_time.csv", index=False)
        augmentMonteCarlo(
            WristsCoordFileName="recorded_video_time.csv", VidIMUdirName="data/", samplingRate=30., tStartUsr=0.2 , tStopUsr=np.inf, totalAug=10
        )
        self.timer.start(35)
        self.plot_graph(None)

    def update_progress_bar(self):
        self.current_step += 1
        self.progress_bar.setValue(self.current_step)

        if self.current_step >= self.total_steps:
            self.progress_timer.stop()
            self.record_button.setText("Start Recording")
            self.progress_bar.setValue(0)

    def getVidData(self, vidFile):
        with open(f"data/{vidFile}", "r") as f:
            time = []
            acc_xF_left = []
            acc_yF_left = []
            acc_zF_left = []
            gyr_xF_left = []
            gyr_yF_left = []
            gyr_zF_left = []

            lines = csv.reader(f, delimiter=",")
            next(lines)

            for row in lines:
                time.append(row[0])
                acc_xF_left.append(float(row[1]))
                acc_yF_left.append(float(row[2]))
                acc_zF_left.append(float(row[3]))
                gyr_xF_left.append(float(row[7]))
                gyr_yF_left.append(float(row[8]))
                gyr_zF_left.append(float(row[9]))

            gyr_xF_left = np.array(gyr_xF_left) / 360 * 2 * math.pi
            gyr_yF_left = np.array(gyr_yF_left) / 360 * 2 * math.pi
            gyr_zF_left = np.array(gyr_zF_left) / 360 * 2 * math.pi

            y_array = [
                acc_xF_left,
                acc_yF_left,
                acc_zF_left,
                gyr_xF_left,
                gyr_yF_left,
                gyr_zF_left,
            ]
            return y_array

    def getData(self, realFile, vidFile):
        with open(realFile, "r") as f:
            acc_IMU_xF_left = []
            acc_IMU_yF_left = []
            acc_IMU_zF_left = []
            gyr_IMU_xF_left = []
            gyr_IMU_yF_left = []
            gyr_IMU_zF_left = []

            lines = csv.reader(f, delimiter=",")
            for row in lines:
                acc_IMU_xF_left.append(float(row[0]))
                acc_IMU_yF_left.append(float(row[1]))
                acc_IMU_zF_left.append(float(row[2]))
                gyr_IMU_xF_left.append(float(row[3]))
                gyr_IMU_yF_left.append(float(row[4]))
                gyr_IMU_zF_left.append(float(row[5]))

            acc_IMU_xF_left = adaptive_medfilt(np.array(acc_IMU_xF_left) * 1, 3, 11, 0, 'constant')
            acc_IMU_yF_left = adaptive_medfilt(np.array(acc_IMU_yF_left), 3, 11, 0, 'constant')
            acc_IMU_zF_left = adaptive_medfilt(np.array(acc_IMU_zF_left) * -1, 3, 11, 0, 'constant')
            gyr_IMU_xF_left = adaptive_medfilt(np.array(gyr_IMU_xF_left)/360. * 2 * math.pi , 3, 11, 0, 'constant')
            gyr_IMU_yF_left = adaptive_medfilt(np.array(gyr_IMU_yF_left)/360. * 2 * math.pi * 1, 3, 11, 0, 'constant')
            gyr_IMU_zF_left = adaptive_medfilt(np.array(gyr_IMU_zF_left)/360. * 2 * math.pi, 3, 11, 0, 'constant')

            acc_IMU_xF_left = savgol_filter(np.array(acc_IMU_xF_left), 31, 3, axis=0, mode='constant')
            acc_IMU_yF_left = savgol_filter(np.array(acc_IMU_yF_left), 31, 3, axis=0, mode='constant')
            acc_IMU_zF_left = savgol_filter(np.array(acc_IMU_zF_left), 31, 3, axis=0, mode='constant')
            gyr_IMU_xF_left = savgol_filter(np.array(gyr_IMU_xF_left), 31, 3, axis=0, mode='constant')
            gyr_IMU_yF_left = savgol_filter(np.array(gyr_IMU_yF_left), 31, 3, axis=0, mode='constant')
            gyr_IMU_zF_left = savgol_filter(np.array(gyr_IMU_zF_left), 31, 3, axis=0, mode='constant')

            y_IMU_array = [acc_IMU_xF_left, acc_IMU_yF_left, acc_IMU_zF_left, gyr_IMU_xF_left, gyr_IMU_yF_left, gyr_IMU_zF_left]

        with open(f"data/{vidFile}", "r") as f:
            time = []
            acc_xF_left = []
            acc_yF_left = []
            acc_zF_left = []
            gyr_xF_left = []
            gyr_yF_left = []
            gyr_zF_left = []

            lines = csv.reader(f, delimiter=",")
            next(lines)

            for row in lines:
                time.append(row[0])
                acc_xF_left.append(float(row[1]))
                acc_yF_left.append(float(row[2]))
                acc_zF_left.append(float(row[3]))
                gyr_xF_left.append(float(row[7]))
                gyr_yF_left.append(float(row[8]))
                gyr_zF_left.append(float(row[9]))

            gyr_xF_left = np.array(gyr_xF_left) / 360 * 2 * math.pi
            gyr_yF_left = np.array(gyr_yF_left) / 360 * 2 * math.pi
            gyr_zF_left = np.array(gyr_zF_left) / 360 * 2 * math.pi

            y_array = [
                acc_xF_left,
                acc_yF_left,
                acc_zF_left,
                gyr_xF_left,
                gyr_yF_left,
                gyr_zF_left,
            ]
    
        return y_array, y_IMU_array, time
    
    def frame_to_time():
        df = pd.read_csv("recorded_video.csv")
        df.rename(columns={"frame": "time"}, inplace=True)
        df.columns

class IMUMatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(IMUMatplotlibWidget, self).__init__(parent)

        self.figure, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.figure.subplots_adjust(
            left=0.15, bottom=0.1, right=0.95, top=0.91, wspace=0.4, hspace=0.2
        )

        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        titles = ["acceleration", "gyroscope"]

        self.subplots = [self.ax[0], self.ax[1]]
        for i, ax in enumerate(self.subplots):
            ax.clear()
            ax.set_title(titles[i], fontsize=12)
            ax.set_xlabel("time")
            ax.set_ylabel(
                "acceleration [m/s]",
            ) if i == 0 else ax.set_ylabel("angular vel. [rad/s]")
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.yaxis.label.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.title.set_color('black')

        self.setLayout(layout)

    def plot(self, original_time, y_array, limit):
        n_data = len(y_array[0])
        period = (float(max(original_time)) - float(min(original_time))) / (n_data - 1)

        time = []
        for i in range(n_data):
            time.append(i * period + float(min(original_time)))

        n = 0
        for t in time:
            if t < 0.2:
                n += 1
        
        time = time[n:]
        for y in y_array:
            y = y[n:]
        

        tick_period = (float(max(original_time)) - float(min(original_time))) / 5
        ticks = [tick_period * i + float(min(original_time)) for i in range(6)]
        y_name = ["X", "Y", "Z", "X", "Y", "Z"]
    
        titles = ["accelerometer", "gyroscope"]
        self.subplots = [self.ax[0], self.ax[1]]

        for i, ax in enumerate(self.subplots):
            ax.clear()
            ax.set_title(titles[i], fontsize=12)
            ax.plot(time, y_array[3 * i], label=y_name[3 * i], linewidth=2)
            ax.plot(time, y_array[3 * i + 1], label=y_name[3 * i + 1], linewidth=2)
            ax.plot(time, y_array[3 * i + 2], label=y_name[3 * i + 2], linewidth=2)
            ax.set_xlabel("time")
            ax.set_ylabel(
                "acceleration [m/s]",
            ) if i == 0 else ax.set_ylabel("angular vel. [rad/s]")
            ax.set_xticks(ticks)
            ax.set_xticklabels([0,2,4,6,8,10])
            ax.set_ylim(limit[2 * i] - 5, limit[2 * i + 1] + 5)
            ax.set_xlim(float(min(original_time)), float(max(original_time)))
            legends = ax.legend()
            for legend in legends.get_texts():
                legend.set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.yaxis.label.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.title.set_color('black')

        self.canvas.draw()

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.figure.subplots_adjust(
            left=0.15, bottom=0.1, right=0.95, top=0.91, wspace=0.4, hspace=0.2
        )
        

        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        titles = ["accelerometer", "gyroscope"]

        self.subplots = [self.ax[0], self.ax[1]]
        for i, ax in enumerate(self.subplots):
            ax.clear()
            ax.set_title(titles[i], fontsize=12)
            ax.set_xlabel("time")
            ax.set_ylabel(
                "acceleration [m/s]",
            ) if i == 0 else ax.set_ylabel("angular vel. [rad/s]")
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.yaxis.label.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.title.set_color('black')

        self.setLayout(layout)

    def plot(self, original_time, y_array, limit):
        y_name = ["X", "Y", "Z", "X", "Y", "Z"]

        time = []
        for i in original_time:
            time.append(float(i))

        tick_period = (float(max(time)) - float(min(time))) / 5
        ticks = [tick_period * i + float(min(time)) for i in range(6)]
        titles = ["accelerometer", "gyroscope"]

        self.subplots = [self.ax[0], self.ax[1]]
        for i, ax in enumerate(self.subplots):
            ax.clear()
            ax.set_title(titles[i], fontsize=12)
            ax.plot(time, y_array[3 * i], label=y_name[3 * i], linewidth=2)
            ax.plot(time, y_array[3 * i + 1], label=y_name[3 * i + 1], linewidth=2)
            ax.plot(time, y_array[3 * i + 2], label=y_name[3 * i + 2], linewidth=2)
            ax.set_xlabel("time")
            ax.set_ylabel(
                "acceleration [m/s]",
            ) if i == 0 else ax.set_ylabel("angular vel. [rad/s]")
            ax.set_xticks(ticks)
            ax.set_xticklabels([0,2,4,6,8,10])
            ax.set_ylim(limit[2 * i] - 5, limit[2 * i + 1] + 5)
            ax.set_xlim(float(min(time)), float(max(time)))
            legends = ax.legend()
            for legend in legends.get_texts():
                legend.set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.yaxis.label.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.title.set_color('black')

        self.canvas.draw()

class StdMatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(StdMatplotlibWidget, self).__init__(parent)

        self.figure, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.figure.subplots_adjust(
            left=0.15, bottom=0.1, right=0.95, top=0.91, wspace=0.4, hspace=0.2
        )
        

        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        titles = ["accelerometer", "gyroscope"]

        self.subplots = [self.ax[0], self.ax[1]]
        for i, ax in enumerate(self.subplots):
            ax.clear()
            ax.set_title(titles[i], fontsize=12)
            ax.set_xlabel("time")
            ax.set_ylabel(
                "acceleration [m/s]",
            ) if i == 0 else ax.set_ylabel("angular vel. [rad/s]")
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.yaxis.label.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.title.set_color('black')

        self.setLayout(layout)

    def plot(self, original_time, y_array, limit, std_array):
        y_name = ["X", "Y", "Z", "X", "Y", "Z"]

        time = []
        for i in original_time:
            time.append(float(i))

        tick_period = (float(max(time)) - float(min(time))) / 5
        ticks = [tick_period * i + float(min(time)) for i in range(6)]
        titles = ["accelerometer", "gyroscope"]

        self.subplots = [self.ax[0], self.ax[1]]
        for i, ax in enumerate(self.subplots):
            ax.clear()
            ax.set_title(titles[i], fontsize=12)
            ax.plot(time, y_array[3 * i], label=y_name[3 * i], linewidth=2)
            ax.fill_between(time, y_array[3 * i] - std_array[3 * i], y_array[3 * i] + std_array[3 * i], alpha=0.4)
            ax.plot(time, y_array[3 * i + 1], label=y_name[3 * i + 1], linewidth=2)
            ax.fill_between(time, y_array[3 * i + 1] - std_array[3 * i + 1], y_array[3 * i + 1] + std_array[3 * i + 1], alpha=0.4)
            ax.plot(time, y_array[3 * i + 2], label=y_name[3 * i + 2], linewidth=2)
            ax.fill_between(time, y_array[3 * i + 2] - std_array[3 * i + 2], y_array[3 * i + 2] + std_array[3 * i + 2], alpha=0.4)
            ax.set_xlabel("time")
            ax.set_ylabel(
                "acceleration [m/s]",
            ) if i == 0 else ax.set_ylabel("angular vel. [rad/s]")
            ax.set_xticks(ticks)
            ax.set_xticklabels([0,2,4,6,8,10])
            ax.set_ylim(limit[2 * i] - 5, limit[2 * i + 1] + 5)
            ax.set_xlim(float(min(time)), float(max(time)))
            legends = ax.legend()
            for legend in legends.get_texts():
                legend.set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.yaxis.label.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.title.set_color('black')

        self.canvas.draw()

class OverlayWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setGeometry(parent.geometry())
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.time_label = QLabel()
        self.time_label.setAttribute(Qt.WA_TranslucentBackground)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #62AB37; font-weight: 30px; font-size: 300px")

        font = QFont()
        font.setPointSize(120)
        self.time_label.setFont(font)

        layout = QVBoxLayout()
        layout.addWidget(self.time_label)

        self.setLayout(layout)

    def set_text(self, text):
        self.time_label.setText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiPageApp()
    window.show()
    sys.exit(app.exec_())
