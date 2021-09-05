import os
import audiofile as af
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from random import shuffle
from python_speech_features import logfbank, fbank, mfcc

# parse args
def parse_args():
    desc="extract features to h5 struct"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default=None, help="path to the input/output dir")
    parser.add_argument('--output', type=str, default=None, help="path to the output dir")
    parser.add_argument('--limit', type=int, default=99999)
    return parser.parse_args()

def featExtractWriter(wavPath, cmn=True):
    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 256,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 11000,
        "preemph": 0.97
    }
    y, sr = af.read(wavPath)
    featMfcc = mfcc(y, sr, winfunc=np.hamming, **kwargs)
    featLogfbank = logfbank(y, sr, **kwargs)
    if cmn:
        featMfcc -= np.mean(featMfcc, axis=0, keepdims=True)
        featLogfbank -= np.mean(featLogfbank, axis=0, keepdims=True)
    return (featMfcc,featLogfbank)

def train(i):
    args = parse_args()
    dataDir = args.dataset
    out_h5Dir = args.output
    limit = args.limit
    trainAllSegs = []

    allBirds = os.listdir(dataDir)
    train_stats = {bird:len(os.listdir(dataDir+bird)) for bird in allBirds}
    train_stats = {k: v for k, v in sorted(train_stats.items(), key=lambda item: item[1], reverse=True)}
    allBirds = list(train_stats.keys())
    if i == 0:
        allBirds = allBirds[:2]
    elif i == 1:
        allBirds = allBirds[2:4]
    elif i == 2:
        allBirds = allBirds[4:6]
    elif i == 3:
        allBirds = allBirds[6:]
    print('... total %d birds ...'%len(allBirds))

    for bird in allBirds:# init all segs and output subdirs
        train_tmp = [dataDir+bird+'/'+x for x in os.listdir(dataDir+bird)]
        if len(train_tmp) > limit:
            shuffle(train_tmp)
            trainAllSegs += train_tmp[:limit]
        else:
            trainAllSegs += train_tmp

        if not os.path.isdir(out_h5Dir+bird):
            os.mkdir(out_h5Dir+bird)

    alreadyH5 = []
    for bird in allBirds:
        alreadyH5 += os.listdir(out_h5Dir+bird)

    print('Start sorting training h5 files (this may take a while...)')
    for song in tqdm(trainAllSegs, desc="Train featExtract"):
        songName = song.split('/')[-1]
        birdName = song.split('/')[-2]
        if songName+'.h5' in alreadyH5:
            continue
        h5Out = out_h5Dir + birdName + '/' + songName + '.h5'
        try:
            featMfcc, featLogfbank = featExtractWriter(song)
        except:
            continue

        hf = h5py.File(h5Out, 'w')
        hf.create_dataset('mfcc', data=featMfcc)
        hf.create_dataset('logfbank', data=featLogfbank)
    print('finished sorting h5 files!')
    
if __name__ == "__main__":
    from multiprocessing import Process
    worker_count = 4
    worker_pool = []
    args = parse_args()
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    out_h5Dir = args.output.strip('/')+'/'
    if not os.path.isdir(out_h5Dir):#init output dir
        os.mkdir(out_h5Dir)
    for i in range(worker_count):
        p = Process(target=train, args=(i,))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")
