import os
import numpy as np
import cv2
import torch
import argparse
from pose.infer import PoseEmotionEstimator
from pose.config import NUM_FRAMES_PER_SEGMENT
from realtime_face.ssd_infer import FaceEmotionEstimator, FER_2013_EMO_DICT
from multimodal.utils import blend_image, draw_annotation


def parse_opts():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_opts()

    # vid = cv2.VideoCapture(0)
    vid = cv2.VideoCapture(args.input)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(
        filename='out.mp4', fourcc=fourcc, fps=vid.get(cv2.CAP_PROP_FPS),
        frameSize=(
            int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    )

    # cv2.namedWindow('disp')
    # cv2.resizeWindow('disp', width=800)

    estimators = [PoseEmotionEstimator(), FaceEmotionEstimator()]
    ensemble_weights = np.asarray([0.53, 0.74])
    softmaxes = [np.zeros(7), np.zeros(7)]
    n_preds = [0, 0]
    count = 0
    label = 'neutral'
    with torch.no_grad():
        while True:
            success, frame = vid.read()
            if frame is None or success is not True:
                break

            out_frame = np.array(frame)

            for i, e in enumerate(estimators):
                annotation, softmax = e(frame)

                if softmax is not None:
                    softmaxes[i] += softmax
                    n_preds[i] += 1

                out_frame = blend_image(out_frame, annotation)

            final = np.zeros(7)
            if count > 0 and count % NUM_FRAMES_PER_SEGMENT == 0:
                for i, s in enumerate(softmaxes):
                    # if n_preds != 0:
                    softmaxes[i] /= n_preds[i]
                    final += ensemble_weights[i] * softmaxes[i]

                pred = np.argmax(final)
                label = FER_2013_EMO_DICT[pred]

                # reset
                softmaxes = [np.zeros(7), np.zeros(7)]
                n_preds = [0, 0]

            draw_annotation(out_frame, 10, 10, label)

            out_vid.write(out_frame)
            count += 1

        out_vid.release()


if __name__ == "__main__":
    main()
