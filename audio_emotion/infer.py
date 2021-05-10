import librosa
import numpy as np
from audio_emotion.audio_demo import infer_audio
from multimodal.config import SECS_PER_SEGMENT


class AudioEmotionEstimator:
    def __init__(self, audio: np.ndarray, sr=16000):
        self.seg_length = librosa.time_to_samples(SECS_PER_SEGMENT, sr=sr)
        self.count = 0
        self.segment_idx = 0
        self.y = audio

    def __call__(self, frame: np.ndarray):
        ret = None
        if self.count > 0 and self.count % self.seg_length:
            segment = self.y[self.segment_idx * self.seg_length: (self.segment_idx + 1) * self.seg_length]
            self.segment_idx += 1
            if segment.shape[0] > 10:
                ret = infer_audio(segment)

        self.count += 1
        return np.zeros_like(frame), ret
