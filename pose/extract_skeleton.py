from pose.openpose.import_libs import openpose, openpose_model_folder
import numpy as np


class SkeletonExtractor:
    def __init__(self):
        params = dict()
        params["model_folder"] = openpose_model_folder
        # params["face"] = True
        params["hand"] = True
        params['keypoint_scale'] = 3  # scale to [0, 1]
        params["num_gpu"] = 1

        # starting OpenPose
        self.opWrapper = openpose.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def extract_skeletons(self, image: np.ndarray):
        # process image
        datum = openpose.Datum()
        # imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop(openpose.VectorDatum([datum]))

        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # print("Face keypoints: \n" + str(datum.faceKeypoints))
        # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(0)

        return datum.poseKeypoints, datum.handKeypoints
