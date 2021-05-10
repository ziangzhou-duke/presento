# STATS402 final project

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
