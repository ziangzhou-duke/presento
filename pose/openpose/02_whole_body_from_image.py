import sys
import cv2
import argparse
from pose.openpose.import_libs import openpose, openpose_model_folder

try:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", default="../examples/media/COCO_val2014_000000000241.jpg",
        help="Process an image. Read all standard formats (jpg, png, bmp, etc.)."
    )
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = openpose_model_folder
    params["face"] = True
    params["hand"] = True
    params["num_gpu"] = 2

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

    # Starting OpenPose
    opWrapper = openpose.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = openpose.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(openpose.VectorDatum([datum]))

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    print("Face keypoints: \n" + str(datum.faceKeypoints))
    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
