import sys
import os
import argparse
from pose.config import OPENPOSE_INSTALL_DIR

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(os.path.join(OPENPOSE_INSTALL_DIR, 'bin', 'python', 'openpose', 'Release'))
        os.environ['PATH'] = os.environ['PATH'] + ';' + \
                             os.path.join(OPENPOSE_INSTALL_DIR, 'x64', 'Release') + ';' + \
                             os.path.join(OPENPOSE_INSTALL_DIR, 'bin') + ';'
        import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
            'script in the right folder?'
        )
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = os.path.join(OPENPOSE_INSTALL_DIR, "models")

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
    opWrapper.configure(params)
    opWrapper.execute()
except Exception as e:
    print(e)
    sys.exit(-1)
