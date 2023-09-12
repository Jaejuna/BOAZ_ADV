import sys
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

class PointCloudVisualizer(QMainWindow):
    def __init__(self, data_queue):
        super().__init__()

        self.data_queue = data_queue

        # GUI setup
        self.setWindowTitle('Point Cloud Visualizer')
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # Button to start visualization
        self.start_button = QPushButton('Start Visualization')
        self.start_button.clicked.connect(self.start_visualization)
        layout.addWidget(self.start_button)

        # Open3D Visualization Setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='3D Viewer', width=800, height=600, visible=False)

        # Timer to update point cloud
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_point_cloud)

    def start_visualization(self):
        self.vis.run()
        self.timer.start(1000)  # Update every second

    def update_point_cloud(self):
        # Fetch new point cloud data (this is dummy data for the example)
        if not self.data_queue.empty():
            point_cloud = self.data_queue.get()

            self.vis.add_geometry(point_cloud)

            # Update the visualization
            self.vis.poll_events()
            self.vis.update_renderer()

    def closeEvent(self, event):
        self.vis.destroy_window()


def live_visualization(data_queue):
    app = QApplication(sys.argv)
    mainWin = PointCloudVisualizer(data_queue)
    mainWin.show()