import json
import os
import torch
import numpy as np
from pose.utils import pad_sequence
import torch.utils.data as data
import pandas as pd

babyrobot_mapper = {
    "Happiness": 0,
    "Sadness": 1,
    "Surprise": 2,
    "Fear": 3,
    "Disgust": 4,
    "Anger": 5,
    "Neutral": 6,
}

inv_babyrobot_mapper = {
    5: "Anger",
    4: "Disgust",
    3: "Fear",
    0: "Happiness",
    1: "Sadness",
    2: "Surprise",
    6: "Neutral"
}


def get_db_splits():
    """Get the splits for each dataset for cross validation"""
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21],
            [22, 23, 24], [25, 26, 27], [28, 29, 30]]


def get_all_db_subjects():
    """Get subjects for each db"""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30]


def get_babyrobot_annotations():
    """ load the annotations of the babyrobot dataset """
    unique_subjects = []
    subject_to_number = {}
    data = []

    subj_idx = 0
    with open("BRED_dataset/annotations.csv") as data_file:
        for x in data_file.readlines()[1:]:
            tokens = x.split(",")
            v = {
                "path": tokens[0].split(".")[0],
                "subject": tokens[0].split("/")[2],
                "emotion": babyrobot_mapper[tokens[1].strip()],  # map emotion to number
                "ann_1_does_emotion_body": tokens[2].strip(),
                "ann_1_does_emotion_face": tokens[3].strip(),
                "ann_2_does_emotion_body": tokens[4].strip(),
                "ann_2_does_emotion_face": tokens[5].strip(),
                "ann_3_does_emotion_body": tokens[6].strip(),
                "ann_3_does_emotion_face": tokens[7].strip(),
            }

            # take as ground truth the majority
            does_emotion = [v['ann_1_does_emotion_face'], v['ann_2_does_emotion_face'], v['ann_3_does_emotion_face']]
            v['does_emotion_face'] = max(set(does_emotion), key=does_emotion.count)
            does_emotion = [v['ann_1_does_emotion_body'], v['ann_2_does_emotion_body'], v['ann_3_does_emotion_body']]
            v['does_emotion_body'] = max(set(does_emotion), key=does_emotion.count)

            data.append(v)

            subject = v['subject']

            if subject not in unique_subjects:
                unique_subjects.append(subject)
                subject_to_number[subject] = subj_idx
                subj_idx += 1

    return data, subject_to_number


def get_babyrobot_data():
    annotations, subject_to_number = get_babyrobot_annotations()

    (
        faces,
        bodies,
        lengths,
        hands_right,
        hands_left,
        Y,
        Y_face,
        Y_body,
        raw_face_paths
    ) = [], [], [], [], [], [], [], [], []
    paths = []

    groups = []

    for video in annotations:
        label = video['emotion']

        # the hierarchical body label is equal to the whole body emotion label if the child did the emotion with the
        # body or neutral otherwise
        label_body = label if video['does_emotion_body'] == "yes" else babyrobot_mapper['Neutral']
        # the hierarchical face label is equal to the whole body emotion label if the child did the emotion with the
        # face or neutral otherwise
        label_face = label if video['does_emotion_face'] == "yes" else babyrobot_mapper['Neutral']

        groups.append(subject_to_number[video['subject']])

        # ========================= Load OpenFace Features ==========================
        # name = video['path'].split("/")[-1]
        csv = os.path.join(video['path'], "openface_output.csv")  # path of csv openface file

        seq = pd.read_csv(csv, delimiter=",")
        seq.columns = seq.columns.str.strip()
        seq = seq.values.astype(np.float32)

        # ========================= Load OpenPose Features ==========================
        json_dir = os.path.join(video['path'] + "/openpose_output/json")

        if not os.path.exists(json_dir):
            print(json_dir)
            raise

        json_list = sorted(os.listdir(json_dir))

        (
            keypoints_array,
            hand_left_keypoints_array,
            hand_right_keypoints_array
        ) = get_keypoints_from_json_list(json_list, json_dir)

        keypoints_array = np.stack(keypoints_array).astype(np.float32)
        hand_right_keypoints_array = np.stack(hand_right_keypoints_array).astype(np.float32)
        hand_left_keypoints_array = np.stack(hand_left_keypoints_array).astype(np.float32)

        hands_right.append(hand_right_keypoints_array)
        hands_left.append(hand_left_keypoints_array)
        bodies.append(keypoints_array)
        faces.append(seq)
        lengths.append(keypoints_array.shape[0])
        Y.append(label)
        Y_face.append(label_face)
        Y_body.append(label_body)
        paths.append(video['path'])

    return faces, bodies, hands_right, hands_left, lengths, Y, Y_face, Y_body, paths, groups


def get_keypoints_from_json_list(json_list, json_dir):
    """
    https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html#autotoc_md33
    """
    keypoints_array, hand_left_keypoints_array, hand_right_keypoints_array = [], [], []

    for json_file in json_list:
        if not json_file.endswith(".json"):
            raise
        js = os.path.join(json_dir, json_file)

        with open(js) as f:
            json_data = json.load(f)

        # ========================= Load OpenPose Features ==========================
        if len(json_data['people']) == 0:  # no person found in current frame
            keypoints = np.zeros(75, dtype=np.float32)
            hand_left_keypoints = np.zeros(63, dtype=np.float32)
            hand_right_keypoints = np.zeros(63, dtype=np.float32)
        else:  # use the first person
            keypoints = np.asarray(json_data['people'][0]['pose_keypoints_2d'], dtype=np.float32)
            hand_left_keypoints = np.asarray(json_data['people'][0]['hand_left_keypoints_2d'], dtype=np.float32)
            hand_right_keypoints = np.asarray(json_data['people'][0]['hand_right_keypoints_2d'], dtype=np.float32)

        keypoints = np.reshape(keypoints, (-1, 3))  # reshape to (num_points * dimension,)
        hand_left_keypoints = np.reshape(hand_left_keypoints, (-1, 3))  # reshape to (num_points * dimension,)
        hand_right_keypoints = np.reshape(hand_right_keypoints, (-1, 3))  # reshape to (num_points * dimension,)

        # ========================= Spatial Normalization ==========================
        # TODO: extract this into a function, and use it on openpose output
        normalize_point_x = keypoints[8, 0]
        normalize_point_y = keypoints[8, 1]

        keypoints[:, 0] -= normalize_point_x
        keypoints[:, 1] -= normalize_point_y

        hand_left_keypoints[:, 0] = hand_left_keypoints[:, 0] - hand_left_keypoints[0, 0]
        hand_left_keypoints[:, 1] = hand_left_keypoints[:, 1] - hand_left_keypoints[0, 1]

        hand_right_keypoints[:, 0] = hand_right_keypoints[:, 0] - hand_right_keypoints[0, 0]
        hand_right_keypoints[:, 1] = hand_right_keypoints[:, 1] - hand_right_keypoints[0, 1]

        keypoints_array.append(np.reshape(keypoints, (-1)))
        hand_left_keypoints_array.append(np.reshape(hand_left_keypoints, (-1)))
        hand_right_keypoints_array.append(np.reshape(hand_right_keypoints, (-1)))

    return keypoints_array, hand_left_keypoints_array, hand_right_keypoints_array


class BodyFaceDataset(data.Dataset):
    def __init__(self, args, data=None, indices=None, subjects=None, phase=None):
        self.args = args
        self.phase = phase

        if data is not None:
            faces, bodies, hands_right, hands_left, lengths, Y, Y_face, Y_body, paths, groups = data

            self.faces = [faces[x] for x in indices]
            self.bodies = [bodies[x] for x in indices]
            self.hands_right = [hands_right[x] for x in indices]
            self.hands_left = [hands_left[x] for x in indices]
            self.lengths = [lengths[x] for x in indices]
            self.Y = [Y[x] for x in indices]
            self.Y_face = [Y_face[x] for x in indices]
            self.Y_body = [Y_body[x] for x in indices]
            self.paths = [paths[x] for x in indices]
            self.groups = [groups[x] for x in indices]

        elif subjects is not None:
            (
                self.faces, self.bodies, self.hands_right, self.hands_left,
                self.lengths, self.Y, self.Y_face,
                self.Y_body, self.paths, self.groups
            ) = get_babyrobot_data()

        self.lengths = []
        for index in range(len(self.bodies)):
            self.lengths.append(self.bodies[index].shape[0])

        self.features = []
        for index in range(len(self.bodies)):
            features_path = self.paths[index] + "/cnn_features"

            features = torch.load(features_path, map_location=lambda storage, loc: storage)
            self.features.append(features)

        self.scaler = None

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.hands_right = [scaler['hands_right'].transform(x) for x in self.hands_right]
        self.hands_left = [scaler['hands_left'].transform(x) for x in self.hands_left]
        self.bodies = [scaler['bodies'].transform(x) for x in self.bodies]
        self.faces = [scaler['faces'].transform(x) for x in self.faces]

    def to_tensors(self):
        self.hands_right = [torch.from_numpy(x).float() for x in self.hands_right]
        self.hands_left = [torch.from_numpy(x).float() for x in self.hands_left]
        self.bodies = [torch.from_numpy(x).float() for x in self.bodies]
        self.faces = [torch.from_numpy(x).float() for x in self.faces]

    def prepad(self):
        """Pre-pad sequences to the max length sequence of each database"""
        max_len = 323
        self.bodies = pad_sequence(self.bodies, batch_first=True, max_len=max_len)
        self.hands_right = pad_sequence(self.hands_right, batch_first=True, max_len=max_len)
        self.hands_left = pad_sequence(self.hands_left, batch_first=True, max_len=max_len)
        self.faces = pad_sequence(self.faces, batch_first=True, max_len=max_len)
        self.features = pad_sequence(self.features, batch_first=True, max_len=max_len)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        body = self.bodies[index]
        hand_right = self.hands_right[index]
        hand_left = self.hands_left[index]
        length = self.lengths[index]

        label_body = self.Y_body[index]

        return {
            "body": body,
            "hand_left": hand_left,
            "hand_right": hand_right,
            "label": self.Y[index],
            "label_body": label_body,
            "length": length,
            "paths": self.paths[index],
        }
