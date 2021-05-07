import matplotlib

matplotlib.use('Agg')
import argparse
import cv2
import os
import numpy as np
import torch
from pose.extract_skeleton import SkeletonExtractor
from pose.config import NUM_FRAMES_PER_SEGMENT
from pose.models import BodyFaceEmotionClassifier
from pose.utils import visualize_skeleton_openpose
from pose.datasets import BodyFaceDataset, normalize_skeleton, inv_babyrobot_mapper
from torch.utils.data import DataLoader

pose_extractor = SkeletonExtractor()


def parse_opts():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    return args


def get_first_person(image: np.ndarray):
    pose, hands = pose_extractor.extract_skeletons(image)

    # pose and hands of the first person
    pose = pose[0]
    left_hand = hands[0][0]
    right_hand = hands[1][0]
    return pose, left_hand, right_hand


def infer_emotion(model: BodyFaceEmotionClassifier, poses: np.ndarray, left_hands: np.ndarray, right_hands: np.ndarray):
    n = left_hands.shape[0]
    dataset = BodyFaceDataset(
        [0], np.expand_dims(poses, 0), np.expand_dims(left_hands, 0), np.expand_dims(right_hands, 0), [n],
        [0], [0]
    )
    dataset.set_scaler(model.scalers)
    dataset.to_tensors()
    dataset.prepad()
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=32, num_workers=4
    )

    for i, batch in enumerate(dataloader):  # should only have one iter
        body, hand_right, hand_left, length = (
            batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), torch.tensor([n]).cuda()
        )
        out = model((body, hand_right, hand_left, length))
        print(out)
        return np.argmax(out.cpu().detach().numpy(), -1)[0]


def main():
    args = parse_opts()
    model = torch.load(args.model_path).cuda()
    model.eval()

    vidcap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(
        filename='out.mp4', fourcc=fourcc, fps=vidcap.get(cv2.CAP_PROP_FPS),
        frameSize=(
            int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    )

    poses = []
    left_hands = []
    right_hands = []
    success, image = vidcap.read()
    emotion = 6
    count = 0
    while success:
        pose, left_hand, right_hand = get_first_person(image)
        pose, left_hand, right_hand = normalize_skeleton(pose, left_hand, right_hand)

        poses.append(pose.ravel())
        left_hands.append(left_hand.ravel())
        right_hands.append(right_hand.ravel())

        if count >= NUM_FRAMES_PER_SEGMENT:
            emotion = infer_emotion(
                model, np.vstack(poses), np.vstack(left_hands), np.vstack(right_hands)
            )
            poses = []
            left_hands = []
            right_hands = []

        # draw label
        label_size, base_line = cv2.getTextSize(
            f"{inv_babyrobot_mapper[emotion]}", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        startx = label_size[0]
        starty = label_size[1]
        cv2.rectangle(
            image,
            (startx, starty + 1 - label_size[1]),
            (startx + label_size[0], starty + 1 + base_line),
            (223, 128, 255),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            f"{inv_babyrobot_mapper[emotion]}",
            (startx, starty + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )
        out_vid.write(image)

        success, image = vidcap.read()
        count += 1

    # print(infer_emotion(
    #     model, np.vstack(poses), np.vstack(left_hands), np.vstack(right_hands)
    # ))

    # visualize_skeleton_openpose(poses[0].reshape(-1, 3), left_hands[0].reshape(-1, 3), right_hands[1].reshape(-1, 3))

    out_vid.release()


if __name__ == '__main__':
    main()
