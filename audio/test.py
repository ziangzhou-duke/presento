import numpy as np
from audio.stt import stt
from audio.analyze_text import analyze

if __name__ == '__main__':
    res = []
    model = 'models/deepspeech-0.9.1-models.pbmm'
    scorer = 'models/deepspeech-0.9.1-models.scorer'
    for f in range(1, 6):
        trans = stt(model, f'tests/ted{f}.wav', scorer_path=scorer)
        tmp = analyze(trans)
        print(tmp)
        res.append(list(tmp.values()))
    print(np.mean(np.asarray(res), axis=0))
