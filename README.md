# STATS402 final project

## Explanation of the audio model training code

Ziang conducted this training framework during his research assistant role on DKU SMIIP lab, thus Ziang and DKU SMIIP
lab own mutual ownership to the training code. Ziang has full ownership of the newly trained model and inference code.
Please contact zz188@duke.edu for further interest and details. Thanks!

## Prerequisite

- Install python libraries in `requirements.txt`
- [Compile and install](https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_installation_0_index.html)
  OpenPose, make sure to pass `-DBUILD_PYTHON=ON` to cmake in order to enable python bindings
- Place the compiled python library `libopenpose.so.*` to a directory, and specify the path to this file
  in `pose/config.py`. For example, `OPENPOSE_INSTALL_DIR = '~/openpose_built'`

## Running

To run the program, use the following command in the project root directory:

```bash
python multimodal/infer.py --input=xxx
```

where `xxx` is the path to the video file you wanna analysis

The output will be `out.mp4`

## Data sources

- https://zenodo.org/record/3233060#.YIPjaaERVPY
- https://www.kaggle.com/uldisvalainis/audio-emotions/download
- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

## Acknowledgement

Some of the model training code and utilities are based on open source projects:

- https://github.com/phamquiluan/ResidualMaskingNetwork
- https://github.com/filby89/body-face-emotion-recognition
- https://github.com/CMU-Perceptual-Computing-Lab/openpose