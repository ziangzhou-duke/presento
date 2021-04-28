import matplotlib

matplotlib.use('Agg')
import argparse
import cv2
import os
import copy
import numpy as np
import torch
from pose import POSE_ROOT_DIR

torch.manual_seed(7)
np.random.seed(7)


# def parse_opts():
#   parser = argparse.ArgumentParser(description='')
#   parser.add_argument('--input', type=str)
#   args = parser.parse_args()
#   return args


def main():
    # args = parse_opts()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()

        # TODO: model inference
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
