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
        
        # Open3D Visualization Setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='3D Viewer', width=800, height=600, visible=True)

        # Flag for the initial point cloud
        self.initial_cloud = True

        # Timer to update point cloud
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_point_cloud)
        self.timer.start(1000)  # Update every second

    def update_point_cloud(self):
        # Fetch new point cloud data (this is dummy data for the example)
        if not self.data_queue.empty():
            data = self.data_queue.get()

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.asarray(data["points"]))
            point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(data["colors"]))

            if self.initial_cloud:
                self.vis.add_geometry(point_cloud)
                self.initial_cloud = False
            else:
                self.vis.update_geometry(point_cloud)

            # Update the visualization
            self.vis.poll_events()
            self.vis.update_renderer()

    def closeEvent(self, event):
        self.vis.destroy_window()

def live_visualization(data_queue):
    app = QApplication(sys.argv)
    mainWin = PointCloudVisualizer(data_queue)
    mainWin.show()
    sys.exit(app.exec_())

def live_visualization_(queue):
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud Viewer", 800, 600)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    while True:
        if not queue.empty():
            data = queue.get()

            if data is None:
                break

            pcd.points = o3d.utility.Vector3dVector(np.asarray(data["points"]))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(data["colors"]))

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

    vis.destroy_window()