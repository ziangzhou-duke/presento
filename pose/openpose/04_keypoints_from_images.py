import sys
import cv2
import argparse
import time
from pose.openpose.import_libs import openpose, openpose_model_folder

try:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", default="../examples/media/",
        help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.)."
    )
    parser.add_argument("--no_display", default=True, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = openpose_model_folder

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = openpose.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = openpose.get_images_on_directory(args[0].image_dir)
    start = time.time()

    # Process and display images
    for imagePath in imagePaths:
        datum = openpose.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(openpose.VectorDatum([datum]))

        print("Body keypoints: \n" + str(datum.poseKeypoints))

        if not args[0].no_display:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27:
                break

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
