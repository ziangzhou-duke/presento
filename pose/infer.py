import matplotlib

matplotlib.use('Agg')
import argparse
import cv2
import os
import numpy as np
import torch
from pose.extract_skeleton import SkeletonExtractor
from pose.config import NUM_FRAMES_PER_SEGMENT
from pose import POSE_ROOT_DIR
from pose.models import BodyFaceEmotionClassifier
from pose.utils import visualize_skeleton_openpose
from pose.datasets import BodyFaceDataset, normalize_skeleton, inv_babyrobot_mapper
from torch.utils.data import DataLoader


class PoseEmotionEstimator:
    def __init__(self):
        self.poses = []
        self.left_hands = []
        self.right_hands = []
        self.emotion = 6
        self.model: BodyFaceEmotionClassifier = torch.load(os.path.join(POSE_ROOT_DIR, '8.ckpt')).cuda()
        self.model.eval()
        self.count = 0

        self.pose_extractor = SkeletonExtractor()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        out_frame = np.zeros_like(frame)

        pose, left_hand, right_hand = self.get_first_person(frame)
        pose, left_hand, right_hand = normalize_skeleton(pose, left_hand, right_hand)

        self.poses.append(pose.ravel())
        self.left_hands.append(left_hand.ravel())
        self.right_hands.append(right_hand.ravel())

        if self.count >= NUM_FRAMES_PER_SEGMENT:
            self.emotion = self.infer_emotion(
                np.vstack(self.poses), np.vstack(self.left_hands), np.vstack(self.right_hands)
            )
            self.poses = []
            self.left_hands = []
            self.right_hands = []

        # draw label
        label_size, base_line = cv2.getTextSize(
            f"{inv_babyrobot_mapper[self.emotion]}", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        startx = label_size[0]
        starty = label_size[1]
        cv2.rectangle(
            out_frame,
            (startx, starty + 1 - label_size[1]),
            (startx + label_size[0], starty + 1 + base_line),
            (223, 128, 255),
            cv2.FILLED,
        )
        cv2.putText(
            out_frame,
            f"{inv_babyrobot_mapper[self.emotion]}",
            (startx, starty + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

        self.count += 1
        return out_frame

    def infer_emotion(self, poses: np.ndarray, left_hands: np.ndarray, right_hands: np.ndarray):
        n = left_hands.shape[0]
        dataset = BodyFaceDataset(
            [0], np.expand_dims(poses, 0), np.expand_dims(left_hands, 0), np.expand_dims(right_hands, 0), [n],
            [0], [0]
        )
        dataset.set_scaler(self.model.scalers)
        dataset.to_tensors()
        dataset.prepad()
        dataloader = DataLoader(
            dataset, shuffle=True, batch_size=32, num_workers=4
        )

        for i, batch in enumerate(dataloader):  # should only have one iter
            body, hand_right, hand_left, length = (
                batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), torch.tensor([n]).cuda()
            )
            out = self.model((body, hand_right, hand_left, length))
            print(out)
            return np.argmax(out.cpu().detach().numpy(), -1)[0]

    def get_first_person(self, image: np.ndarray):
        pose, hands = self.pose_extractor.extract_skeletons(image)

        # pose and hands of the first person
        pose = pose[0]
        left_hand = hands[0][0]
        right_hand = hands[1][0]
        return pose, left_hand, right_hand


def parse_opts():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_opts()

    estimator = PoseEmotionEstimator()

    vidcap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(
        filename='out.mp4', fourcc=fourcc, fps=vidcap.get(cv2.CAP_PROP_FPS),
        frameSize=(
            int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    )

    success, frame = vidcap.read()
    while success:
        frame = estimator(frame)

        out_vid.write(frame)
        success, frame = vidcap.read()

    # visualize_skeleton_openpose(poses[0].reshape(-1, 3), left_hands[0].reshape(-1, 3), right_hands[1].reshape(-1, 3))

    out_vid.release()


if __name__ == '__main__':
    main()
