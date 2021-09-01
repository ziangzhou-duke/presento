import os
import sys
from pose.config import OPENPOSE_INSTALL_DIR

__all__ = ['openpose', 'openpose_model_folder']

try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(
        os.path.join(OPENPOSE_INSTALL_DIR, 'bin', 'python', 'openpose', 'Release')
    )
    sys.path.append(
        os.path.join(OPENPOSE_INSTALL_DIR, 'python', 'openpose')
    )
    os.environ['PATH'] = os.environ['PATH'] + ';' + \
                         os.path.join(OPENPOSE_INSTALL_DIR, 'x64', 'Release') + ';' + \
                         os.path.join(OPENPOSE_INSTALL_DIR, 'bin') + ';'
    import pyopenpose

    openpose = pyopenpose

    openpose_model_folder = os.path.join(OPENPOSE_INSTALL_DIR, "models")
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
        'script in the right folder?'
    )
    raise e
