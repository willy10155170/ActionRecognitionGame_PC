#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import pyrealsense2 as rs
import time
import numpy as np
import math
import imutils
import winsound
import queue


History_joint1 = [
    [0, -0.009208951145410538, -0.77310711145401, 2.2537500858306885],
    [1, 0.007155116647481918, -0.800692617893219, 2.2775001525878906],
    [2, 0.020479582250118256, -0.8099214434623718, 2.2945001125335693],
    [3, 0.03380057215690613, -0.8110517263412476, 2.2977499961853027],
    [4, -0.03239806741476059, -0.7979946136474609, 2.2790000438690186],
    [5, -0.0455860011279583, -0.7947496771812439, 2.2790000438690186],
    [6, -0.05874336138367653, -0.7910668253898621, 2.277750015258789],
    [7, 0.060051921755075455, -0.7798907160758972, 2.283250093460083],
    [8, -0.06975442916154861, -0.770786464214325, 2.3142499923706055],
    [9, 0.02341325208544731, -0.733429491519928, 2.2502501010894775],
    [10, -0.02541416510939598, -0.7254341840744019, 2.2452502250671387],
    [11, 0.20709067583084106, -0.5538091659545898, 2.254500150680542],
    [12, -0.13951700925827026, -0.5382019281387329, 2.24150013923645],
    [13, 0.3656696677207947, -0.3451029658317566, 2.277750015258789],
    [14, -0.22960935533046722, -0.3287741243839264, 2.3192501068115234],
    [15, 0.46662020683288574, -0.12980696558952332, 2.283750057220459],
    [16, -0.2917855381965637, -0.08647804707288742, 2.2715001106262207],
    [17, 0.5051304697990417, -0.05679616704583168, 2.293750047683716],
    [18, -0.3253891170024872, -0.009990726597607136, 2.274749994277954],
    [19, 0.4641883969306946, -0.0529751181602478, 2.2710001468658447],
    [20, -0.30585965514183044, 0.0033318321220576763, 2.277250051498413],
    [21, 0.43974190950393677, -0.07262998819351196, 2.2632501125335693],
    [22, -0.27046912908554077, -0.01675511710345745, 2.286750078201294],
    [23, 0.14860232174396515, -0.07846488803625107, 2.058000087738037],
    [24, -0.05337410047650337, -0.07824794948101044, 2.0502500534057617],
    [25, 0.16298474371433258, 0.33378151059150696, 2.0965001583099365],
    [26, -0.06955010443925858, 0.31793275475502014, 2.090250015258789],
    [27, 0.17736251652240753, 0.6620067954063416, 1.9720001220703125],
    [28, -0.11580567806959152, 0.654568076133728, 2.0082499980926514],
    [29, 0.16654875874519348, 0.6867295503616333, 1.9155000448226929],
    [30, -0.10808215290307999, 0.6887651681900024, 1.9750001430511475],
    [31, -0.10808215290307999, 0.6887651681900024, 1.9750001430511475],
    [32, -0.10808215290307999, 0.6887651681900024, 1.9750001430511475],
]
History_joint2 = [
    [0, -0.009208951145410538, -0.77310711145401, 2.2537500858306885],
    [1, 0.007155116647481918, -0.800692617893219, 2.2775001525878906],
    [2, 0.020479582250118256, -0.8099214434623718, 2.2945001125335693],
    [3, 0.03380057215690613, -0.8110517263412476, 2.2977499961853027],
    [4, -0.03239806741476059, -0.7979946136474609, 2.2790000438690186],
    [5, -0.0455860011279583, -0.7947496771812439, 2.2790000438690186],
    [6, -0.05874336138367653, -0.7910668253898621, 2.277750015258789],
    [7, 0.060051921755075455, -0.7798907160758972, 2.283250093460083],
    [8, -0.06975442916154861, -0.770786464214325, 2.3142499923706055],
    [9, 0.02341325208544731, -0.733429491519928, 2.2502501010894775],
    [10, -0.02541416510939598, -0.7254341840744019, 2.2452502250671387],
    [11, 0.20709067583084106, -0.5538091659545898, 2.254500150680542],
    [12, -0.13951700925827026, -0.5382019281387329, 2.24150013923645],
    [13, 0.3656696677207947, -0.3451029658317566, 2.277750015258789],
    [14, -0.22960935533046722, -0.3287741243839264, 2.3192501068115234],
    [15, 0.46662020683288574, -0.12980696558952332, 2.283750057220459],
    [16, -0.2917855381965637, -0.08647804707288742, 2.2715001106262207],
    [17, 0.5051304697990417, -0.05679616704583168, 2.293750047683716],
    [18, -0.3253891170024872, -0.009990726597607136, 2.274749994277954],
    [19, 0.4641883969306946, -0.0529751181602478, 2.2710001468658447],
    [20, -0.30585965514183044, 0.0033318321220576763, 2.277250051498413],
    [21, 0.43974190950393677, -0.07262998819351196, 2.2632501125335693],
    [22, -0.27046912908554077, -0.01675511710345745, 2.286750078201294],
    [23, 0.14860232174396515, -0.07846488803625107, 2.058000087738037],
    [24, -0.05337410047650337, -0.07824794948101044, 2.0502500534057617],
    [25, 0.16298474371433258, 0.33378151059150696, 2.0965001583099365],
    [26, -0.06955010443925858, 0.31793275475502014, 2.090250015258789],
    [27, 0.17736251652240753, 0.6620067954063416, 1.9720001220703125],
    [28, -0.11580567806959152, 0.654568076133728, 2.0082499980926514],
    [29, 0.16654875874519348, 0.6867295503616333, 1.9155000448226929],
    [30, -0.10808215290307999, 0.6887651681900024, 1.9750001430511475],
    [31, -0.10808215290307999, 0.6887651681900024, 1.9750001430511475],
    [32, -0.10808215290307999, 0.6887651681900024, 1.9750001430511475],
]


# In[2]:


class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5,
                trackCon = 0.8):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upBody, self.smooth,
                                      self.detectionCon, self.trackCon)


    def findPose(self, image, draw = True):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(image,
                            self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def findPosition(self, image, draw = True):
        lmList = []
        h,w,c = image.shape
        t = h+w
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = lm.x*w, lm.y*h
                lmList.append([id, cx, cy])

        return lmList



# In[3]:


def W_cor_Erfix(World_cor,Direc):
    for i in range(33):
        if(World_cor[i][3] == 0.0):
            if Direc == 1:
                World_cor[i] = History_joint1[i]
            else:
                World_cor[i] = History_joint2[i]
        else:
            if Direc == 1:
                History_joint1[i] = World_cor[i]
            else:
                History_joint2[i] = World_cor[i]
    return World_cor


# In[4]:


def Normalizing(dest,MM,mm,uu):
    uu =  uu / 13
    dest = round(((dest-uu)/(MM-mm)) * 100,2)
    return dest


# In[5]:


def Cor_Normalization(JointSet):

    sum_x = 0
    sum_y = 0
    sum_z = 0
    Max_x = 0
    Min_x = 999
    Max_y = 0
    Min_y = 999

    for i in range(13):
        sum_x += JointSet[i][1]
        sum_y += JointSet[i][2]
        sum_z += JointSet[i][3]
        if JointSet[i][1]> Max_x:
            Max_x = JointSet[i][1]
        if JointSet[i][1] < Min_x:
            Min_x = JointSet[i][1]
        if JointSet[i][2]> Max_y:
            Max_y = JointSet[i][2]
        if JointSet[i][2] < Min_y:
            Min_y = JointSet[i][2]

    for i in range(13):
        JointSet[i][1] = Normalizing(JointSet[i][1],Max_x,Min_x,sum_x)
        JointSet[i][2] = Normalizing(JointSet[i][2],Max_y,Min_y,sum_y)

    return JointSet


# In[6]:


def Cal(image,lmList,aligned_depth_frame,color_intrin,ht,Direc):
    JointSet = []
    JointDepth = []
    World_cor = []
    his_joint = np.zeros((13,4))
    new_position = np.zeros((5, 3))

    hip_x = 0
    hip_y = 0
    hip_z = 0
    try:
        if len(lmList) !=0:
            for i in range(33):
                try:
                    if Direc == 1:
                        depth = aligned_depth_frame.get_distance(int(lmList[i][1]+480),int(lmList[i][2]))
                    else:
                        depth = aligned_depth_frame.get_distance(int(lmList[i][1]),int(lmList[i][2]))
                    dx ,dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [int(lmList[i][1]),int(lmList[i][2])], depth)
                    distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
                except:
                    return []

                World_cor.append([i,dx,dy,dz])

            World_cor = W_cor_Erfix(World_cor,Direc)

#             for fp in World_cor:
#                 fp = str(fp)
#                 fp = fp.strip("[]")
#                 print(fp)
#             print("\n")


            hip_x = (World_cor[23][1]+World_cor[24][1])/2
            hip_y = (World_cor[23][2]+World_cor[24][2])/2
            hip_z = (World_cor[23][3]+World_cor[24][3])/2


            new_position[0,0] = (World_cor[0][1]+World_cor[1][1]+World_cor[2][1]+World_cor[3][1]+World_cor[4][1]+World_cor[5][1]+World_cor[6][1]+World_cor[7][1]+World_cor[8][1]+World_cor[9][1]+World_cor[10][1]) / 11
            new_position[0,1] = (World_cor[0][2]+World_cor[1][2]+World_cor[2][2]+World_cor[3][2]+World_cor[4][2]+World_cor[5][2]+World_cor[6][2]+World_cor[7][2]+World_cor[8][2]+World_cor[9][2]+World_cor[10][2]) / 11
            new_position[0,2] = (World_cor[0][3]+World_cor[1][3]+World_cor[2][3]+World_cor[3][3]+World_cor[4][3]+World_cor[5][3]+World_cor[6][3]+World_cor[7][3]+World_cor[8][3]+World_cor[9][3]+World_cor[10][3]) / 11

            new_position[1,0] = (World_cor[15][1]+World_cor[21][1]+World_cor[19][1]+World_cor[17][1]) / 4
            new_position[1,1] = (World_cor[15][2]+World_cor[21][2]+World_cor[19][2]+World_cor[17][2]) / 4
            new_position[1,2] = (World_cor[15][3]+World_cor[21][3]+World_cor[19][3]+World_cor[17][3]) / 4

            new_position[2,0] = (World_cor[16][1]+World_cor[22][1]+World_cor[20][1]+World_cor[18][1]) / 4
            new_position[2,1] = (World_cor[16][2]+World_cor[22][2]+World_cor[20][2]+World_cor[18][2]) / 4
            new_position[2,2] = (World_cor[16][3]+World_cor[22][3]+World_cor[20][3]+World_cor[18][3]) / 4

            new_position[3,0] = (World_cor[27][1]+World_cor[31][1]+World_cor[29][1]) / 3
            new_position[3,1] = (World_cor[27][2]+World_cor[31][2]+World_cor[29][2]) / 3
            new_position[3,2] = (World_cor[27][3]+World_cor[31][3]+World_cor[29][3]) / 3

            new_position[4,0] = (World_cor[28][1]+World_cor[30][1]+World_cor[32][1]) / 3
            new_position[4,1] = (World_cor[28][2]+World_cor[30][2]+World_cor[32][2]) / 3
            new_position[4,2] = (World_cor[28][3]+World_cor[30][3]+World_cor[32][3]) / 3


            JointSet.append([0,int( (new_position[0,0] - hip_x) * 100),int( (new_position[0,1] - hip_y) * 100),int( (new_position[0,2] - hip_z) * 100) - 10])
            JointSet.append([1,int( (World_cor[11][1] - hip_x) * 100),int( (World_cor[11][2] - hip_y) * 100),int( (World_cor[11][3] - hip_z) * 100) - 10])
            JointSet.append([2,int( (World_cor[12][1] - hip_x) * 100),int( (World_cor[12][2] - hip_y) * 100),int( (World_cor[12][3] - hip_z) * 100) - 10])
            JointSet.append([3,int( (World_cor[13][1] - hip_x) * 100),int( (World_cor[13][2] - hip_y) * 100),int( (World_cor[13][3] - hip_z) * 100) - 10])
            JointSet.append([4,int( (World_cor[14][1] - hip_x) * 100),int( (World_cor[14][2] - hip_y) * 100),int( (World_cor[14][3] - hip_z) * 100) - 10])

            JointSet.append([5,int( (new_position[1,0] - hip_x) * 100),int( (new_position[1,1] - hip_y) * 100),int( (new_position[1,2] - hip_z) * 100) - 10])

            JointSet.append([6,int( (new_position[2,0] - hip_x) * 100),int( (new_position[2,1] - hip_y) * 100),int( (new_position[2,2] - hip_z) * 100) - 10])

            JointSet.append([7,int( (World_cor[23][1] - hip_x) * 100),int( (World_cor[23][2] - hip_y) * 100),int( (World_cor[23][3] - hip_z) * 100)])
            JointSet.append([8,int( (World_cor[24][1] - hip_x) * 100),int( (World_cor[24][2] - hip_y) * 100),int( (World_cor[24][3] - hip_z) * 100)])
            JointSet.append([9,int( (World_cor[25][1] - hip_x) * 100),int( (World_cor[25][2] - hip_y) * 100),int( (World_cor[25][3] - hip_z) * 100)])
            JointSet.append([10,int( (World_cor[26][1] - hip_x) * 100),int( (World_cor[26][2] - hip_y) * 100),int( (World_cor[26][3] - hip_z) * 100)])
            JointSet.append([11,int( (new_position[3,0] - hip_x) * 100),int( (new_position[3,1] - hip_y) * 100),int( (new_position[3,2] - hip_z) * 100)])
            JointSet.append([12,int( (new_position[4,0] - hip_x) * 100),int( (new_position[4,1] - hip_y) * 100),int( (new_position[4,2] - hip_z) * 100)])




            JointSet = Cor_Normalization(JointSet)

            Nj = []
            for i in range(13):
                Nj.append(JointSet[i][1])
                Nj.append(JointSet[i][2])
                Nj.append(JointSet[i][3])

            return Nj
    except Exception as e:
        print(e)
        return []


# In[7]:

def find_camera(found_rgb,device):
    print(device)
    for s in device:
        print(s.get_info(rs.camera_info.name))
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            return True
    return False



class frame_queue():

    def __init__(self, frame_data, game_status):
        self.frame_data = frame_data
        self.game_status = game_status

    def get_frame_data(self):
        # while True:
        #     self.frame_data.put("hi")

        pipeline = rs.pipeline()
        config = rs.config()
    
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    #     found_rgb = find_camera(False,device.sensors)
    
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
        profile = pipeline.start(config)
    
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
    
        clipping_distance_in_meters = 5
        clipping_distance = clipping_distance_in_meters / depth_scale
    
        align_to = rs.stream.color
        align = rs.align(align_to)
    
        right_detector = poseDetector()
        left_detector = poseDetector()
        pFps = 0
        start = time.time()
        end = 0
        #file = open('C:\\Users\\Ray\\Desktop\\NTOU\\Graduation_Project\\BodyData\\Test_Data.txt','w')
        ht = 0
    
        try:
            while True:
    
                if self.game_status.empty() is False:
                    status = self.game_status.get()
                else:
                    status = False
                if status is True:
                    break
    
    
                frames = pipeline.wait_for_frames()
    
                aligned_frames = align.process(frames)
    
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
    
                if not aligned_depth_frame or not color_frame:
                    continue
    
    
                depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
    
                grey_color = 153
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
                #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0),grey_color, color_image)
    
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
                
                bg_right = bg_removed[0:540,0:480,0:3]
                bg_left = bg_removed[0:540,480:960,0:3]
                
                image1 = left_detector.findPose(bg_left)
                image2 = right_detector.findPose(bg_right)
                lmList1 = left_detector.findPosition(bg_left)
                lmList2 = right_detector.findPosition(bg_right)
                
                
#                 image = detector.findPose(bg_removed)
#                 lmList = detector.findPosition(image)
#                 height,width,channel = image.shape
#                 total = height+width

                JointSet = []
                JointSet1 = []
                JointSet2 = []
    
                try:
                    if len(lmList1) !=0:
                        JointSet1 = Cal(bg_left,lmList1,aligned_depth_frame,color_intrin,ht,0)
                    else:
                        JointSet1 = []
                    if len(lmList2) !=0:
                        JointSet2 = Cal(bg_right,lmList2,aligned_depth_frame,color_intrin,ht,1)
                    else:
                        JointSet2 = []
                    end = time.time()
                    ht = end-start
    
                except Exception as e:
                    print(e)
                    pass
                
                
                
                JointSet.append(JointSet1)
                JointSet.append(JointSet2)
                self.frame_data.put(JointSet)

                if ht >= 90:
                    cv2.destroyAllWindows()
                
                ht += 1
                cFps = time.time()
                fps = 1/(cFps-pFps)
                pFps = cFps
    
                rv = np.hstack([image2,image1])

                cv2.putText(rv, str(int(ht)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
                cv2.imshow('MediaPipe Pose', rv)
    
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
    
                if self.game_status.empty() is False:
                    status = self.game_status.get()
                else:
                    status = False

                if status is True:
                    break
    
        finally:
            pipeline.stop()


# class queue_test():
#     def __init__(self, q):
#         self.q = q
#
#     def datatest(self):
#         i = 0
#         while True:
#             i += 1
#             time.sleep(1)
#             self.q.put(rhi())
#             #print(self.q.qsize())

# In[8]:


# if __name__ == "__main__":
#     main()


# In[ ]:




