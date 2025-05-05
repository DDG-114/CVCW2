import argparse
import os
import time
import numpy as np
import warnings
import cv2
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path
import sys

# Append YOLOv5 root to path
yolov5_root = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.append(yolov5_root)

from yolov5.utils.general import check_img_size, non_max_suppression, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync  # updated from time_synchronized
from yolov5.utils.augmentations import letterbox  # new location in recent YOLOv5

from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker

cudnn.benchmark = True

# -------------------- Adapted scale_coords --------------------
def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


class VideoTracker:
    def __init__(self, args):
        print('Initialize DeepSORT & YOLOv5')
        self.args = args
        self.img_size = args.img_size
        self.frame_interval = args.frame_interval

        self.device = select_device(args.device)
        self.half = self.device.type != 'cpu'

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        # self.vdo = cv2.VideoCapture(args.cam if args.cam != -1 else 0)
        self.vdo = cv2.VideoCapture()

        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)
        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        weights = torch.load(args.weights, map_location=self.device)
        self.detector = weights['model'].float().fuse().eval()
        self.detector.to(self.device)
        if self.half:
            self.detector.half()

        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

        if self.device.type == 'cpu':
            warnings.warn("Running in CPU mode may be slow!", UserWarning)

    def __enter__(self):
        # if self.args.cam != -1:
        #     ret, frame = self.vdo.read()
        #     assert ret, "Camera error"
        #     self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # else:
        #     assert os.path.isfile(self.args.input_path), "Invalid path"
        #     self.vdo.open(self.args.input_path)
        #     self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     assert self.vdo.isOpened()

        assert os.path.isfile(self.args.input_path), "Invalid path"
        self.vdo.open(self.args.input_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(self.args.save_path, "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))

        if self.args.save_txt:
            os.makedirs(self.args.save_txt, exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.vdo.release()
        self.writer.release()

    def run(self):
        yolo_time, sort_time, avg_fps = [], [], []
        idx_frame = 0
        last_out = None
        t_start = time.time()

        while self.vdo.grab():
            t0 = time.time()
            _, img0 = self.vdo.retrieve()
            if idx_frame % self.args.frame_interval == 0:
                outputs, yt, st = self.image_track(img0)
                last_out = outputs
                yolo_time.append(yt)
                sort_time.append(st)
            else:
                outputs = last_out
            t1 = time.time()
            avg_fps.append(t1 - t0)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                img0 = draw_boxes(img0, bbox_xyxy, identities)
                cv2.putText(img0, f"frame: {idx_frame} fps: {len(avg_fps) / sum(avg_fps):.2f}",
                            (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

            if self.args.display:
                cv2.imshow("test", img0)
                if cv2.waitKey(1) == ord('q'):
                    break

            if self.args.save_path:
                self.writer.write(img0)

            if self.args.save_txt:
                with open(self.args.save_txt + f"{idx_frame:04d}.txt", 'a') as f:
                    for x1, y1, x2, y2, idx in outputs:
                        f.write(f"{x1}\t{y1}\t{x2}\t{y2}\t{idx}\n")

            idx_frame += 1

        print(f"Avg YOLO time: {np.mean(yolo_time):.3f}s, SORT time: {np.mean(sort_time):.3f}s")
        print(f"Total time: {time.time() - t_start:.3f}s, Total frames: {idx_frame}")

    def image_track(self, im0):
        img = letterbox(im0, new_shape=self.img_size, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_sync()
        with torch.no_grad():
            pred = self.detector(img, augment=self.args.augment)[0]
        pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,
                                   classes=self.args.classes, agnostic=self.args.agnostic_nms)
        t2 = time_sync()

        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
            confs = det[:, 4:5].cpu()
            outputs = self.deepsort.update(bbox_xywh, confs, im0)
        else:
            outputs = torch.zeros((0, 5))

        t3 = time.time()
        return outputs, t2 - t1, t3 - t2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='input.mp4')
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--frame_interval', type=int, default=2)
    parser.add_argument('--fourcc', type=str, default='mp4v')
    parser.add_argument('--device', default='')
    parser.add_argument('--save_txt', default='output/predict/')

    parser.add_argument('--display', action='store_true')
    parser.add_argument('--display_width', type=int, default=800)
    parser.add_argument('--display_height', type=int, default=600)
    # parser.add_argument('--camera', dest='cam', type=int, default=-1)

    # parser.add_argument('--weights', type=str, default='yolov5/runs/train/exp9/weights/best.pt')
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.5)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--classes', nargs='+', type=int, default=[0])
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--config_deepsort', type=str, default='./configs/deep_sort.yaml')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()
