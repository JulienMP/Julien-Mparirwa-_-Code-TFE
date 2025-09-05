#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import time
import cv2
import torch
from pathlib import Path

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


def create_output_structure(output_dir, video_path):
    """Creates output directory structure based on input path"""
    video_path = Path(video_path)
    video_name = video_path.stem
    
    category = video_path.parent.name
    split = video_path.parent.parent.name if video_path.parent.parent.name in ['train', 'val', 'test'] else 'unknown'
    
    output_path = Path(output_dir) / split / category / video_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path, video_name


class Predictor(object):
    def __init__(self, model, exp, trt_file=None, decoder=None, device=torch.device("cpu"), fp16=False):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
            
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def process_video(predictor, output_path, video_name, video_path, args):
    """Processes a single video file for object tracking"""
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if args.save_result:
        save_video_path = output_path / f"{video_name}_tracked.mp4"
        logger.info(f"video save_path is {save_video_path}")
        vid_writer = cv2.VideoWriter(
            str(save_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
                
            if args.save_result:
                vid_writer.write(online_im)
                frame_path = output_path / f"frame{frame_id:06d}.jpg"
                cv2.imwrite(str(frame_path), online_im)
        else:
            break
        frame_id += 1

    cap.release()
    if args.save_result:
        vid_writer.release()
        
        res_file = output_path / f"{video_name}_tracking.txt"
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def setup_model(exp, args):
    """Sets up and loads the tracking model"""
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(exp.output_dir, args.experiment_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    if args.trt:
        trt_file = osp.join(exp.output_dir, args.experiment_name, "model_trt.pth")
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    return model, trt_file, decoder


def main():
    parser = argparse.ArgumentParser(description="ByteTrack video tracking")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_dir", help="Directory to save tracking results")
    parser.add_argument("--exp-file", "-f", required=True, help="Experiment description file")
    parser.add_argument("--ckpt", "-c", required=True, help="Checkpoint file for evaluation")
    parser.add_argument("--experiment-name", "-expn", type=str, default=None)
    parser.add_argument("--name", "-n", type=str, default=None, help="model name")
    
    parser.add_argument("--device", default="gpu", type=str, help="device to run model (cpu or gpu)")
    parser.add_argument("--conf", default=None, type=float, help="test confidence threshold")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate")
    parser.add_argument("--fp16", action="store_true", help="use mixed precision evaluation")
    parser.add_argument("--fuse", action="store_true", help="fuse conv and bn for testing")
    parser.add_argument("--trt", action="store_true", help="use TensorRT model for testing")
    parser.add_argument("--save-result", action="store_true", help="save inference results")
    
    parser.add_argument("--track-thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track-buffer", type=int, default=30, help="frames to keep lost tracks")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect-ratio-thresh", type=float, default=1.6, help="aspect ratio threshold")
    parser.add_argument("--min-box-area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument("--mot20", action="store_true", help="test mot20")

    args = parser.parse_args()

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    exp = get_exp(args.exp_file, args.name)
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    logger.info("Args: {}".format(args))

    model, trt_file, decoder = setup_model(exp, args)
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    
    output_path, video_name = create_output_structure(args.output_dir, args.video_path)
    
    if args.video_path.endswith('.mkv'):
        process_video(predictor, output_path, video_name, args.video_path, args)


if __name__ == "__main__":
    main()