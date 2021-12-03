import queue
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import time
import numpy as np
import math

POSE = []
POSE.append(mp.solutions.pose.Pose(static_image_mode=False,
                                   smooth_landmarks=False,
                                   enable_segmentation=False,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.8))
POSE.append(mp.solutions.pose.Pose(static_image_mode=False,
                                   smooth_landmarks=False,
                                   enable_segmentation=False,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.8))


class poseDetector():

    def __init__(self, pose_number):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = POSE[pose_number]
        print(f"INFO: Start poseDetector id:{id(self.pose)}")

    def findPose(self, image):
        """
        Findpose and draw
        return :
            image, lmList(33,2), world_landmarks(13,3)
        """
        self.results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        lmList = []
        world_landmarks = []
        # check if pose_marks
        if not self.results.pose_landmarks:
            return image, lmList, world_landmarks
        # pose_landmarks
        self.mp_drawing.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        h, w, c = image.shape
        for lm in self.results.pose_landmarks.landmark:
            lmList.append([lm.x * w, lm.y * h])
        # pose_world_landmarks
        for i, v in enumerate(self.results.pose_world_landmarks.landmark):
            if i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                world_landmarks.append([v.x * 100, v.y * 100, v.z * 100])

        return image, lmList, world_landmarks


def Cal(lmList: list, aligned_depth_frame, color_intrin, Direc):
    """ Deal jointset
        lmList.shape = (33,2)
        JointSet_xy.shape = (13,2)
        JointSet.shape = (13,3)
        return:
            shape = (39)
    """
    JointSet_xy = []
    JointSet = []
    for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        JointSet_xy.append(lmList[i])
    try:
        if len(JointSet_xy) != 0:
            for i in JointSet_xy:
                if Direc == 1:  # right
                    depth = aligned_depth_frame.get_distance(int(i[0] + 480), int(i[1]))
                else:  # left
                    depth = aligned_depth_frame.get_distance(int(i[0]), int(i[1]))
                #############
                #############
                ############# QUESTION???? rs2_deproject_pixel_to_point  What unit? 
                # if depth!= 0.0:
                #     dx ,dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [int(i[0]),int(i[1])], depth)
                #     JointSet.extend([dx*100,-dy*100,-dz*100])
                #############
                #############
                #############
                JointSet.extend([i[0], i[1], depth])

    except Exception as e:
        print(f"ERR: Depth_frame.get_distance() Wrong! [(x,y) = ({i[0]},{i[1]})")
        print(e)
        return list(np.zeros(39).astype(float))

    ######### STILL NEED????
    ####        NORMALIZATION
    ####        ROTATION
    #########
    return JointSet


class frame_queue():

    def __init__(self, frame_data, game_status: queue.Queue):
        self.frame_data = frame_data
        self.game_status = game_status
        # Lidar
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        print("INFO: Lidar Start")

    def get_frame_data(self):
        right_detector = poseDetector(0)
        left_detector = poseDetector(1)
        start = time.time()
        try:
            while True:
                if not self.game_status.empty():
                    status = self.game_status.get()
                else:
                    status = False
                if status is True:
                    break
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue

                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                color_image = np.asanyarray(color_frame.get_data())

                # right
                image1, lmList1, world_lm_1 = left_detector.findPose(color_image[0:540, 480:960, 0:3])
                # left
                image2, lmList2, world_lm_2 = right_detector.findPose(color_image[0:540, 0:480, 0:3])

                JointSet = []
                JointSet_World = []
                if len(world_lm_1) != 0:  # right
                    # JointSet.append(Cal(lmList1, aligned_depth_frame, color_intrin, 0))
                    JointSet_World.append(list(np.array(world_lm_1).reshape(39)))
                else:
                    # JointSet.append(list(np.zeros(39).astype(float)))
                    JointSet_World.append(list(np.zeros(39).astype(float)))

                if len(world_lm_2) != 0:  # left
                    # JointSet.append(Cal(lmList2, aligned_depth_frame, color_intrin, 0))
                    JointSet_World.append(list(np.array(world_lm_2).reshape(39)))
                else:
                    # JointSet.append(list(np.zeros(39).astype(float)))
                    JointSet_World.append(list(np.zeros(39).astype(float)))

                # self.frame_data.put(JointSet)
                self.frame_data.put(JointSet_World)

                # show
                rv = np.hstack([image2, image1])
                cv2.putText(rv, str(int(time.time() - start)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                cv2.imshow('MediaPipe Pose', rv)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        except Exception as e:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("ERR: get_frame_data Wrong")
            print(e)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("INFO: pipeline is over")