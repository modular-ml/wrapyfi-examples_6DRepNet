import time
import math
import re
import sys
import os
import argparse

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from face_detection import RetinaFace
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use('TkAgg')

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR
from wrapyfi_interfaces.templates.orientation import OrientationInterface
from wrapyfi_interfaces.io.video.interface import VideoCapture, VideoCaptureReceiver
from sixdrepnet.model import SixDRepNet
from sixdrepnet import utils


def str_or_int(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        return arg


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help="Camera device id to use [0]."
                             "If none is selected, the default camera by the OS is used. "
                             "If webcam is a string, this equates to the port (topic) name. e.g., /icub/cam/left",
                        type=str_or_int, default="0")
    parser.add_argument("--img_width", type=int, default=320, help="The captured image width")
    parser.add_argument("--img_height", type=int, default=240, help="The captured image height")
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--headless',
                        dest='headless', action="store_true", help='Disable CV2 GUI',
                        default=False)
    parser.add_argument("--video_mware", type=str, choices=MiddlewareCommunicator.get_communicators(),
                        help="Middleware for listening to video stream")
    parser.add_argument("--orientation_port", type=str, default="",
                        help="Port (topic) to publish head orientation")
    parser.add_argument("--orientation_mware", type=str, choices=MiddlewareCommunicator.get_communicators(),
                        help="Middleware to publish head orientation")

    args = parser.parse_args()
    return args


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    cam = args.cam_id
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    print('Loading data.')

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    # cap = VideoCapture(cam)
    if args.video_mware:
        video_device = VideoCaptureReceiver
    else:
        video_device = VideoCapture
    cap = video_device(str(cam), headless=True, img_width=args.img_width, img_height=args.img_height,
                       mware=args.video_mware if args.video_mware is not None else DEFAULT_COMMUNICATOR, multithreading=False)
    
    if args.orientation_mware:
        # Extend when required: Broadcasting multiple faces detected can be achieved by creating a 
        #    separate orientation interface for each face idx, and adding index to port name after the first like 
        #    /control_interface/orientation_out for 1st and /control_interface/orientation_out_2 for 2nd and so on.
        #    This takes place automatically below
        orientation_broadcasters =  [OrientationInterface(
            orientation_port_out=args.orientation_port, mware_out=args.orientation_mware,
            orientation_port_in="")]
    else:
        orientation_broadcasters = []

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if frame is None:
                continue
                
            faces = detector(frame)

            for face_idx, (box, landmarks, score) in enumerate(faces):

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).cuda(gpu)

                c = cv2.waitKey(1)
                if c == 27:
                    break

                start = time.time()
                R_pred = model(img)
                end = time.time()
                print('Head pose estimation: %2f ms' % ((end - start)*1000.))

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu().item()
                y_pred_deg = euler[:, 1].cpu().item()
                r_pred_deg = euler[:, 2].cpu().item()
                
                # broadcast orientation to middleware
                if args.orientation_mware:
                    if face_idx >= len(orientation_broadcasters):
                        # Extend when required: Broadcasting multiple faces detected can be achieved by creating a 
                        #    separate orientation interface for each face idx, and adding index to port name after the first like 
                        #    /control_interface/orientation_out for 1st and /control_interface/orientation_out_2 for 2nd and so on
                        orientation_broadcasters.append(OrientationInterface(orientation_port_out=args.orientation_port + "_" + str(face_idx+1), mware_out=args.orientation_mware, orientation_port_in=""))
                    # TODO (fabawi): _mware and orientation_port should have been updated during instantiation and don't need to be provided again
                    if face_idx == 0:
                        orientation, = orientation_broadcasters[face_idx].transmit_orientation(quaternion=False, order="xyz", pitch=p_pred_deg, roll=r_pred_deg, yaw=y_pred_deg, speed=None, _mware=args.orientation_mware, orientation_port=args.orientation_port)
                    else:
                        orientation, = orientation_broadcasters[face_idx].transmit_orientation(quaternion=False, order="xyz", pitch=p_pred_deg, roll=r_pred_deg, yaw=y_pred_deg, speed=None, _mware=args.orientation_mware, orientation_port=args.orientation_port + "_" + str(face_idx+1))
                if not args.headless:
                    #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                    utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                        x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
            if not args.headless:
                cv2.imshow("Demo", frame)
                cv2.waitKey(5)
            else:
                time.sleep(0.005)
