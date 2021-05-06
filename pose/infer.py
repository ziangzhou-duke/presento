import matplotlib

matplotlib.use('Agg')
import argparse
import cv2
import os
import numpy as np
import torch
from pose.extract_skeleton import SkeletonExtractor

torch.manual_seed(7)
np.random.seed(7)

pose_extractor = SkeletonExtractor()


def parse_opts():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    return args


def get_first_person(image: np.ndarray):
    pose, hands = pose_extractor.extract_skeletons(image)

    # pose and hands of the first person
    pose = pose[0]
    left_hand = hands[0][0]
    right_hand = hands[1][0]
    return pose, left_hand, right_hand


def main():
    args = parse_opts()
    vidcap = cv2.VideoCapture(args.input)

    success, image = vidcap.read()
    count = 0
    while success:
        pose, left_hand, right_hand = get_first_person(image)
        print(pose)
        success, image = vidcap.read()
        count += 1


if __name__ == '__main__':
    main()
