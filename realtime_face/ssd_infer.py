import os
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from realtime_face.models.resmasking import resmasking_dropout1
from realtime_face import FACE_ROOT_DIR


class FaceEmotionEstimator:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(FACE_ROOT_DIR, "deploy.prototxt.txt"),
            os.path.join(FACE_ROOT_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        )
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        # load configs and set random seed
        self.configs = json.load(open(os.path.join(FACE_ROOT_DIR, "configs", "fer2013_config.json")))
        self.image_size = (self.configs["image_size"], self.configs["image_size"])

        self.model = resmasking_dropout1(in_channels=3, num_classes=7)
        self.model.cuda()
        self.state = torch.load(
            os.path.join(FACE_ROOT_DIR, "Z_resmasking_dropout1_rot30_2019Nov30_13.32")
        )
        self.model.load_state_dict(self.state["net"])
        self.model.eval()

        self.count = 0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        out_frame = np.zeros_like(frame)

        # frame = np.fliplr(frame)
        frame = frame.astype(np.uint8)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        faces = self.net.forward()

        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")

            # convert to square images
            center_x, center_y = (start_x + end_x) // 2, (start_y + end_y) // 2
            square_length = ((end_x - start_x) + (end_y - start_y)) // 2 // 2

            square_length *= 1.1

            start_x = int(center_x - square_length)
            start_y = int(center_y - square_length)
            end_x = int(center_x + square_length)
            end_y = int(center_y + square_length)

            cv2.rectangle(
                out_frame, (start_x, start_y), (end_x, end_y), (179, 255, 179), 2
            )

            face = gray[start_y:end_y, start_x:end_x]

            face = ensure_color(face)

            face = cv2.resize(face, self.image_size)
            face = self.transform(face).cuda()
            face = torch.unsqueeze(face, dim=0)

            output = torch.squeeze(self.model(face), 0)
            proba = torch.softmax(output, 0)

            # emo_idx = torch.argmax(proba, dim=0).item()
            emo_proba, emo_idx = torch.max(proba, dim=0)
            emo_idx = emo_idx.item()
            emo_proba = emo_proba.item()

            emo_label = FER_2013_EMO_DICT[emo_idx]

            label_size, base_line = cv2.getTextSize(
                "{}: 000".format(emo_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )

            cv2.rectangle(
                out_frame,
                (end_x, start_y + 1 - label_size[1]),
                (end_x + label_size[0], start_y + 1 + base_line),
                (223, 128, 255),
                cv2.FILLED,
            )
            cv2.putText(
                out_frame,
                "{} {}".format(emo_label, int(emo_proba * 100)),
                (end_x, start_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

        self.count += 1
        return out_frame


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def main():
    # vid = cv2.VideoCapture(0)
    vid = cv2.VideoCapture("test_images/1.mp4")
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

    estimator = FaceEmotionEstimator()
    with torch.no_grad():
        while True:
            success, frame = vid.read()
            if frame is None or success is not True:
                break

            frame = estimator(frame)
            out_vid.write(frame)

            # cv2.imshow("disp", frame)
            # cv2.imshow('disp', np.concatenate((gray ), axis=1))
            # if cv2.waitKey(1) == ord("q"):
            #     break
        # cv2.destroyAllWindows()
        out_vid.release()


if __name__ == "__main__":
    main()
