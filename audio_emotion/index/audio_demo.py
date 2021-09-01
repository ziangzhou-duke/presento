import numpy as np
import os
import soundfile as sf
import torch 
from python_speech_features import logfbank, mfcc
from module.model import Gvector
import argparse
from scipy.special import softmax
from numpy import argmax

def parse_args():
    desc="parse model info"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--wav_dir',type=str)
    parser.add_argument('--model_dir', type=str, default='exp/logfbank_channels8_bs128/chkpt/chkpt_050.pth')
    return parser.parse_args()
    
args = parse_args()
audio_dir = args.wav_dir

# config:
mdl_kwargs = {
    "channels": 8, 
    "block": "BasicBlock", 
    "num_blocks": [3,4,6,3], 
    "embd_dim": 512, 
    "drop": 0.5, 
    "n_class": 7
}

fbank_kwargs = {
    "winlen": 0.025, 
    "winstep": 0.01, 
    "nfilt": 64, 
    "nfft": 512, 
    "lowfreq": 0, 
    "highfreq": None, 
    "preemph": 0.97    
}



class SVExtractor():
    def __init__(self, mdl_kwargs, fbank_kwargs, resume, device):
        self.model = self.load_model(mdl_kwargs, resume)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.fbank_kwargs = fbank_kwargs

    def load_model(self, mdl_kwargs, resume):
        model = Gvector(**mdl_kwargs)
        state_dict = torch.load(resume,map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        return model

    def extract_fbank(self, y, sr, cmn=True):
        feat = logfbank(y, sr, **self.fbank_kwargs)
        if cmn:
            feat -= feat.mean(axis=0, keepdims=True)
        return feat.astype('float32')

    def __call__(self, y, sr):
        assert sr == 16000, "Support 16k wave only!"
        if len(y) > sr * 30:
            y = y[:int(sr*30)]  # truncate the maximum length of 30s.
        feat = self.extract_fbank(y, sr, cmn=True)
        feat = torch.from_numpy(feat).unsqueeze(0)
        feat = feat.float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            embd = self.model.extractor(feat)
            rslt = self.model.forward(feat)
        embd = embd.squeeze(0).cpu().numpy()
        rslt = rslt.squeeze(0).cpu().numpy()
        return embd, rslt


if __name__ == "__main__":
    import soundfile
    import librosa
    with open('index/int2label.json','r') as f:
        identi = json.load(f)
    resume = model_dir
    sv_extractor = SVExtractor(mdl_kwargs, fbank_kwargs, resume, device='cpu')
#     y, sr = soundfile.read(audio_dir)
    y, sr = librosa.load(audio_dir, sr=16000)
    embd, result = sv_extractor(y, sr)
    print(identi[str(argmax(softmax(result)))])
    print(embd, result)
