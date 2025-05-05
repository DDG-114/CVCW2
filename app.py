import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sys
import os
import argparse
import time

# Ensure the yolov5 path is in the sys.path
yolov5_root = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.append(yolov5_root)

from main import VideoTracker  # Import VideoTracker


# Function to run the tracker in a background thread
def run_tracker_thread(args):
    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()


class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedestrian Tracking with YOLOv5 & DeepSORT")
        self.root.geometry("400x350")

        self.create_widgets()

    def create_widgets(self):
        # File selection section
        self.input_path_label = tk.Label(self.root, text="Input Video Path:")
        self.input_path_label.grid(row=0, column=0, padx=10, pady=5)

        self.input_path_entry = tk.Entry(self.root, width=40)
        self.input_path_entry.grid(row=0, column=1, padx=10, pady=5)
        self.input_path_entry.insert(0, "input.py")  # Default path is input.py

        self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_video)
        self.browse_button.grid(row=0, column=2, padx=10, pady=5)

        # Output folder selection section
        self.output_path_label = tk.Label(self.root, text="Output Folder:")
        self.output_path_label.grid(row=1, column=0, padx=10, pady=5)

        self.output_path_entry = tk.Entry(self.root, width=40)
        self.output_path_entry.grid(row=1, column=1, padx=10, pady=5)
        self.output_path_entry.insert(0, "output/")  # Default output path is output/

        self.browse_output_button = tk.Button(self.root, text="Browse", command=self.browse_output)
        self.browse_output_button.grid(row=1, column=2, padx=10, pady=5)

        # Parameter setup section
        self.frame_interval_label = tk.Label(self.root, text="Frame Interval:")
        self.frame_interval_label.grid(row=2, column=0, padx=10, pady=5)

        self.frame_interval_entry = tk.Entry(self.root, width=40)
        self.frame_interval_entry.grid(row=2, column=1, padx=10, pady=5)
        self.frame_interval_entry.insert(0, "2")  # Default value

        self.conf_thres_label = tk.Label(self.root, text="Confidence Threshold:")
        self.conf_thres_label.grid(row=3, column=0, padx=10, pady=5)

        self.conf_thres_entry = tk.Entry(self.root, width=40)
        self.conf_thres_entry.grid(row=3, column=1, padx=10, pady=5)
        self.conf_thres_entry.insert(0, "0.5")  # Default value

        self.iou_thres_label = tk.Label(self.root, text="IOU Threshold:")
        self.iou_thres_label.grid(row=4, column=0, padx=10, pady=5)

        self.iou_thres_entry = tk.Entry(self.root, width=40)
        self.iou_thres_entry.grid(row=4, column=1, padx=10, pady=5)
        self.iou_thres_entry.insert(0, "0.5")  # Default value

        # Start tracking button
        self.start_button = tk.Button(self.root, text="Start Tracking", command=self.start_tracking)
        self.start_button.grid(row=5, column=0, columnspan=3, pady=20)

        # Stop button
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.grid(row=6, column=0, columnspan=3, pady=5)

    def browse_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if video_path:
            self.input_path_entry.delete(0, tk.END)
            self.input_path_entry.insert(0, video_path)

    def browse_output(self):
        output_folder = filedialog.askdirectory()
        if output_folder:
            self.output_path_entry.delete(0, tk.END)
            self.output_path_entry.insert(0, output_folder)

    def start_tracking(self):
        # Get input parameters
        input_path = self.input_path_entry.get()
        output_path = self.output_path_entry.get()
        frame_interval = int(self.frame_interval_entry.get())
        conf_thres = float(self.conf_thres_entry.get())
        iou_thres = float(self.iou_thres_entry.get())

        # Run tracking in a new thread
        self.stop_button.config(state=tk.NORMAL)  # Enable the stop button
        self.start_button.config(state=tk.DISABLED)  # Disable the start button

        args = argparse.Namespace(
            input_path=input_path,
            save_path=output_path,
            frame_interval=frame_interval,
            fourcc="mp4v",
            device='',
            save_txt="output/predict/",
            display=False,
            display_width=800,
            display_height=600,
            camera=-1,
            weights='weights/best.pt',
            img_size=640,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=[0],
            agnostic_nms=False,
            augment=False,
            config_deepsort='./configs/deep_sort.yaml'
        )

        # Start the tracking thread
        tracking_thread = threading.Thread(target=run_tracker_thread, args=(args,))
        tracking_thread.daemon = True  # Daemon thread will exit when the main thread exits
        tracking_thread.start()

    def stop_tracking(self):
        messagebox.showinfo("Tracking", "Tracking has been stopped.")
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)


if __name__ == '__main__':
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()
