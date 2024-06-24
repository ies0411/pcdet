import argparse
import glob
import time
from pathlib import Path
import os
import sys
import natsort

import torch
import argparse
import time


import copy
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tracking_modules.model import Spb3DMOT
from tracking_modules.utils import Config
from tracking_modules.nms import nms

# import tracking_modules.evaluation.mailpy as mailpy


from utils import read_calib, bb3d_2_bb2d, velo_to_cam, vel_to_cam_pose
from torch.utils.data import Dataset, DataLoader

# from tracking_modules.evaluation.evaluate_tracking import evaluate

# https://github.com/hailanyi/3D-Multi-Object-Tracker/tree/master
# https://github.com/JonathonLuiten/TrackEval?tab=readme-ov-file
# https://github.com/hailanyi/3D-Detection-Tracking-Viewer


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="/mnt/nas2/users/eslim/tracking/dsvt_kitti_det_7_mix/dsvt_voxel.yaml",
        help="specify the config for demo",
    )
    # "/mnt/nas2/users/eslim/tracking/dsvt_kitti_det_2/dsvt_voxel.yaml"
    # /mnt/nas2/users/eslim/tracking/detector_test2/dsvt_voxel.yaml
    # "./cfgs/kitti_models/pv_rcnn.yaml"
    # "./cfgs/kitti_models/dsvt_voxel.yaml"
    # "/mnt/nas2/users/eslim/tracking/wo_aug/dsvt_voxel_tracking.yaml"

    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/nas3/Data/kitti-processed/object_tracking/training/velodyne/",
        help="specify the point cloud data file or directory",
    )
    # ../sample/lidar
    # /mnt/nas3/Data/kitti-processed/object_tracking/training/velodyne/
    parser.add_argument(
        "--ckpt",
        default="/mnt/nas2/users/eslim/tracking/dsvt_kitti_det_7_mix/ckpt/latest_model.pth",
        type=str,
        help="specify the pretrained model",
    )
    # /mnt/nas2/users/eslim/tracking/dsvt_kitti_det/ckpt/latest_model.pth
    # "/mnt/nas2/users/eslim/tracking/pv_rcnn_8369.pth"
    # "/mnt/nas2/users/eslim/tracking/detector_test2/ckpt/latest_model.pth"
    # /mnt/nas2/users/eslim/tracking/w_aug/ckpt
    # "/mnt/nas2/users/eslim/tracking/pretrained_aug/ckpt/latest_model.pth"
    # /mnt/nas2/users/eslim/result_log/kitti_anchor_r
    # /mnt/nas2/users/eslim/tracking/detector_test2/ckpt/latest_model.pth
    # "/mnt/nas2/users/eslim/tracking/detector_2/ckpt/checkpoint_epoch_80.pth"
    # "/mnt/nas2/users/eslim/tracking/wo_aug/ckpt/latest_model.pth"
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument(
        "--tracking_config",
        type=str,
        default="./tracking_modules/configs/config.yml",
        help="tracking config file path",
    )
    parser.add_argument(
        "--tracking_output_dir",
        default="./tracking_result/",
        type=str,
    )
    # ../sample/calib/
    # "/mnt/nas3/Data/kitti-processed/object_tracking/training/calib"
    parser.add_argument(
        "--calib_dir",
        default="/mnt/nas3/Data/kitti-processed/object_tracking/training/calib",
        type=str,
    )

    parser.add_argument(
        "--eval_dir",
        default="./tracking_modules/evaluation/results/sha_key/data/",
        type=str,
    )
    parser.add_argument(
        "--eval",
        default=True,
        type=bool,
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    tracking_cfg = Config(args.tracking_config)
    return args, cfg, tracking_cfg


class TrackerDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        tracking_seqs,
        training=True,
        root_path=None,
        logger=None,
        ext=".bin",
        args=None,
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = args.data_path
        self.ext = ext
        self.args = args
        self.tracking_seqs = tracking_seqs
        path_list = os.listdir(args.data_path)
        path_list = natsort.natsorted(path_list)
        path_list = [x for x in path_list if int(x) in self.tracking_seqs]
        self.num_file_list = []
        self.lidar_files = []
        for dir_path in path_list:
            files = os.listdir(os.path.join(args.data_path, dir_path))
            files = natsort.natsorted(files)
            self.num_file_list.append(len(files))
            for file in files:
                self.lidar_files.append(os.path.join(self.root_path, dir_path, file))
        self.total_num = len(self.lidar_files)

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):

        scene = self.lidar_files[index].split("/")[-2]
        frame_idx = self.lidar_files[index].split("/")[-1].split(".")[0]
        P2, V2C = read_calib(os.path.join(self.args.calib_dir, scene + ".txt"))
        if self.ext == ".bin":

            max_row = 374  # y
            max_col = 1241  # x

            lidar = np.fromfile(self.lidar_files[index], dtype=np.float32).reshape(
                -1, 4
            )

            mask = lidar[:, 0] > 0
            lidar = lidar[mask]

            lidar_copy = np.zeros(shape=lidar.shape)
            lidar_copy[:, :] = lidar[:, :]
            velo_tocam = V2C
            lidar[:, 3] = 1
            lidar = np.matmul(lidar, velo_tocam.T)
            img_pts = np.matmul(lidar, P2.T)

            velo_tocam = np.mat(velo_tocam).I
            velo_tocam = np.array(velo_tocam)
            normal = velo_tocam
            normal = normal[0:3, 0:4]
            lidar = np.matmul(lidar, normal.T)
            lidar_copy[:, 0:3] = lidar
            x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
            mask = np.logical_and(
                np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row)
            )
            points = lidar_copy[mask]

        elif self.ext == ".npy":
            pass
            # points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        input_dict = {
            "points": points,
            "frame_id": int(frame_idx),
            "scene": int(scene),
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def _detection_postprocessing(pred_dicts, num_objects):
    """
    x,y,z,dx,dy,dz,rot,score

    """
    tracking_info_data = {}
    for idx in range(num_objects):
        tracking_info_data.setdefault(str(idx + 1), [])
    for idx, pred_bbox in enumerate(pred_dicts[0]["pred_boxes"]):
        label = str(pred_dicts[0]["pred_labels"][idx].item())
        pred_bbox = pred_bbox.tolist()
        pred_bbox.append(pred_dicts[0]["pred_scores"][idx].tolist())
        tracking_info_data[label].append(pred_bbox)
    return tracking_info_data


def main():
    args, detection_cfg, tracking_cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info("----------------- Spb3D Tracker -----------------")
    logger.info(f"cuda available : {torch.cuda.is_available()}")
    tracking_dataset = TrackerDataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
        training=False,
        root_path=None,
        ext=args.ext,
        logger=logger,
        tracking_seqs=tracking_cfg.tracking_seqs,
        args=args,
    )
    logger.info(f"Total number of samples: \t{len(tracking_dataset)}")

    model = build_network(
        model_cfg=detection_cfg.MODEL,
        num_class=len(detection_cfg.CLASS_NAMES),
        dataset=tracking_dataset,
    )

    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    tracking_time = time.time()
    ID_start_dict = {}
    for class_name in detection_cfg.CLASS_NAMES:
        ID_start_dict.setdefault(class_name, 1)
    logger.info(f"ID_start_dict : {ID_start_dict}")
    tracking_results_dict = {}
    tracker_dict = {}
    for class_name in detection_cfg.CLASS_NAMES:
        tracker_dict[class_name] = Spb3DMOT(ID_init=ID_start_dict.get(class_name))
    for idx in range(len(detection_cfg.CLASS_NAMES)):
        tracking_results_dict.setdefault(str(int(idx) + 1), {})
    for class_name in detection_cfg.CLASS_NAMES:
        Path(os.path.join(args.tracking_output_dir, class_name)).mkdir(
            parents=True, exist_ok=True
        )
    scene = str(9999).zfill(4)
    with torch.no_grad():
        for idx, data_dict in enumerate(tracking_dataset):
            if data_dict == None:
                break

            if scene != str(int(data_dict["scene"])).zfill(4):
                ID_start_dict = {}
                tracker_dict = {}
                for class_name in detection_cfg.CLASS_NAMES:
                    ID_start_dict.setdefault(class_name, 1)
                    tracker_dict[class_name] = Spb3DMOT(
                        ID_init=ID_start_dict.get(class_name)
                    )

                scene = str(int(data_dict["scene"])).zfill(4)

            data_dict = tracking_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            detection_results_dict = _detection_postprocessing(
                pred_dicts, len(detection_cfg.CLASS_NAMES)
            )

            # TODO : nms
            for label, pred_bboxes in detection_results_dict.items():
                # if (
                #     detection_cfg.CLASS_NAMES[int(label) - 1] == "Cyclist"
                #     or detection_cfg.CLASS_NAMES[int(label) - 1] == "Car"
                # ):
                #     continue
                pred_bboxes = nms(pred_bboxes) if len(pred_bboxes) != 0 else []
                if pred_bboxes is None:
                    pred_bboxes = []
                frame_idx = str(data_dict["frame_id"][0])
                tracker = tracker_dict[detection_cfg.CLASS_NAMES[int(label) - 1]]
                tracking_result, _ = tracker.track(pred_bboxes)
                tracking_result = tracking_result[0].tolist()

                tracking_results_dict[label].setdefault(frame_idx, [])
                tracking_results_dict[label][frame_idx].append(tracking_result)
                P2, V2C = read_calib(os.path.join(args.calib_dir, f"{scene}.txt"))

                if len(tracking_result) == 0:
                    continue
                tracking_results = tracking_result

                for tracking_result in tracking_results:
                    save_path = os.path.join(
                        args.tracking_output_dir,
                        detection_cfg.CLASS_NAMES[int(label) - 1],
                        f"{scene}.txt",
                    )
                    if os.path.exists(save_path):
                        with open(save_path, "a") as f:
                            box = copy.deepcopy(tracking_result)
                            box[:3] = tracking_result[3:6]
                            box[3:6] = tracking_result[:3]
                            box[2] -= box[5] / 2
                            box[6] = -box[6] - np.pi / 2
                            box[:3] = vel_to_cam_pose(box[:3], V2C)[:3]
                            box2d = bb3d_2_bb2d(box, P2)
                            f.write(
                                f"{frame_idx} {str(int(tracking_result[-1]))} {detection_cfg.CLASS_NAMES[int(label) - 1]} -1 -1 -10 {box2d[0][0]:.6f} {box2d[0][1]:.6f} {box2d[0][2]:.6f} {box2d[0][3]:.6f} {box[5]:.6f} {box[4]:.6f} {box[3]:.6f} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[6]:.6f} \n"
                            )
                    else:
                        with open(save_path, "w") as f:
                            print(f"open {save_path}!!")
                            box = copy.deepcopy(tracking_result)
                            box[:3] = tracking_result[3:6]
                            box[3:6] = tracking_result[:3]
                            box[2] -= box[5] / 2
                            box[6] = -box[6] - np.pi / 2
                            box[:3] = vel_to_cam_pose(box[:3], V2C)[:3]
                            box2d = bb3d_2_bb2d(box, P2)

                            f.write(
                                f"{frame_idx} {str(int(tracking_result[-1]))} {detection_cfg.CLASS_NAMES[int(label) - 1]} -1 -1 -10 {box2d[0][0]:.6f} {box2d[0][1]:.6f} {box2d[0][2]:.6f} {box2d[0][3]:.6f} {box[5]:.6f} {box[4]:.6f} {box[3]:.6f} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[6]:.6f} \n"
                            )

    logger.info(f"tracking time : { time.time()-tracking_time}")
    logger.info("=========Tracking Finish =========")


# https://github.com/pratikac/kitti/blob/master/readme.tracking.txt
# frame, track_id, type, truncated, occluded, alpha, bbox(2d-left,top,right,bottom), dimensions(height,width,lennth), location(3d-x,y,z), rotation_y, score
# 0 2 Pedestrian 0 0 -2.523309 (1106.137292 166.576807 1204.470628 323.876144) (1.714062 0.767881 0.972283) (6.301919 1.652419 8.455685) -1.900245
if __name__ == "__main__":
    main()
