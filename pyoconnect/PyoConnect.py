# 2015.08.13 17:21:26 UTC
#Embedded file name: PyoConnect.py
"""
        PyoConnect v0.1
        
        Author:
          Fernando Cosentino - fbcosentino@yahoo.com.br
          
        Official source:
          http://www.fernandocosentino.net/pyoconnect
          
        Based on the work of dzhu: https://github.com/dzhu/myo-raw
        
        License:
                Use at will, modify at will. Always keep my name in this file as original author. And that's it.
        
        Steps required (in a clean debian installation) to use this library:
                // permission to ttyACM0 - must restart linux user after this
                sudo usermod -a -G dialout $USER

                // dependencies
                apt-get install python-pip
                pip install pySerial --upgrade
                pip install enum34
                pip install PyUserInput
                apt-get install python-Xlib

                // now reboot   
"""
from __future__ import print_function
import sys
import select
import time
from subprocess import Popen, PIPE
import re
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('GTKAgg')

from keras.models import load_model
import utils.dataprep as dp
from sklearn import preprocessing
modelDir = "../models"

try:
    from pymouse import PyMouse
    pmouse = PyMouse()
except:
    print('PyMouse error: No mouse support')
    pmouse = None

try:
    from pykeyboard import PyKeyboard
    pkeyboard = PyKeyboard()
except:
    print('PyKeyboard error: No keyboard support')
    pkeyboard = None

from common import *
from myo_raw import MyoRaw, Pose, Arm, XDirection

class Myo(MyoRaw):

    def __init__(self, cls, tty = None):
        self.locked = True
        self.use_lock = True
        self.timed = True
        self.lock_time = 5.0
        self.time_to_lock = self.lock_time
        self.last_pose = -1
        self.last_tick = 0
        self.current_box = 0
        self.last_box = 0
        self.box_factor = 0.25
        self.current_arm = 0
        self.current_xdir = 0
        self.current_gyro = None
        self.current_accel = None
        self.current_roll = 0
        self.current_pitch = 0
        self.current_yaw = 0
        self.center_roll = 0
        self.center_pitch = 0
        self.center_yaw = 0
        self.first_rot = 0
        self.current_rot_roll = 0
        self.current_rot_pitch = 0
        self.current_rot_yaw = 0
        self.mov_history = ''
        self.gest_history = ''
        self.act_history = ''
        if pmouse != None:
            self.x_dim, self.y_dim = pmouse.screen_size()
            self.mx = self.x_dim / 2
            self.my = self.y_dim / 2
        self.centered = 0
        MyoRaw.__init__(self, tty)
        self.add_emg_handler(self.emg_handler)
        self.add_arm_handler(self.arm_handler)
        self.add_imu_handler(self.imu_handler)
        self.add_pose_handler(self.pose_handler)
        self.onEMG = None
        self.onPoseEdge = None
        self.onLock = None
        self.onUnlock = None
        self.onPeriodic = None
        self.onWear = None
        self.onUnwear = None
        self.onBoxChange = None

    def tick(self):
        now = time.time()
        if now - self.last_tick >= 0.01:
            if self.onPeriodic != None:
                self.onPeriodic()
            if self.use_lock and self.locked == False and self.timed:
                if self.time_to_lock <= 0:
                    print('Locked')
                    self.locked = True
                    self.vibrate(1)
                    self.time_to_lock = self.lock_time
                    if self.onLock != None:
                        self.onLock()
                else:
                    self.time_to_lock -= 0.01
            self.last_tick = now

    def emg_handler(self, emg, moving):
        if self.onEMG != None:
            self.onEMG(emg, moving)
        self.current_emg = emg

    def arm_handler(self, arm, xdir):
        if arm == Arm(0):
            self.current_arm = 'unknown'
        elif arm == Arm(1):
            self.current_arm = 'right'
        elif arm == Arm(2):
            self.current_arm = 'left'
        if xdir == XDirection(0):
            self.current_xdir = 'unknown'
        elif xdir == XDirection(1):
            self.current_xdir = 'towardWrist'
        elif xdir == XDirection(2):
            self.current_xdir = 'towardElbow'
        if Arm(arm) == 0:
            if self.onUnwear != None:
                self.onUnwear()
        elif self.onWear != None:
            self.onWear(self.current_arm, self.current_xdir)

    def imu_handler(self, quat, acc, gyro):
        q0, q1, q2, q3 = quat
        q0 = q0 / 16384.0
        q1 = q1 / 16384.0
        q2 = q2 / 16384.0
        q3 = q3 / 16384.0
        self.current_quat = q0, q1, q2, q3
        self.current_gyro = gyro
        self.current_accel = acc
        self.current_roll = math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
        self.current_pitch = -math.asin(max(-1.0, min(1.0, 2.0 * (q0 * q2 - q3 * q1))))
        self.current_yaw = -math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        self.current_rot_roll = self.angle_dif(self.current_roll, self.center_roll)
        self.current_rot_yaw = self.angle_dif(self.current_yaw, self.center_yaw)
        self.current_rot_pitch = self.angle_dif(self.current_pitch, self.center_pitch)
        if self.first_rot == 0:
            self.rotSetCenter()
            self.first_rot = 1
        self.current_box = self.getBox()
        if self.current_box != self.last_box:
            self.mov_history = str(self.mov_history[-100:]) + str(self.current_box)
            self.act_history = str(self.act_history[-100:]) + str(self.current_box)
            if self.onBoxChange != None:
                self.onBoxChange(self.last_box, 'off')
                self.onBoxChange(self.current_box, 'on')
            self.last_box = self.current_box

        # code to get xy according to mouse mover
        # self.rotSetCenter()
        # (x,y) = (self.current_rot_yaw*2000, -self.current_rot_pitch*2000)
        # print(x)
        # print(y)

        # print(q0)
        # print(q1)
        # print(q2)
        # print(q3)

    def pose_handler(self, p):
        if p == Pose(0):
            pn = 0
        elif p == Pose(1):
            pn = 1
        elif p == Pose(2):
            pn = 2
        elif p == Pose(3):
            pn = 3
        elif p == Pose(4):
            pn = 4
        elif p == Pose(5):
            pn = 5
        else:
            pn = 6
        if pn != self.last_pose:
            self.gest_history = str(self.gest_history[-100:]) + str(self.PoseToChar(pn))
            self.act_history = str(self.act_history[-100:]) + str(self.PoseToChar(pn))
            if self.locked == False:
                self.time_to_lock = self.lock_time
                if self.onPoseEdge != None:
                    if self.last_pose > -1:
                        self.onPoseEdge(self.PoseToStr(self.last_pose), 'off')
                    self.onPoseEdge(self.PoseToStr(pn), 'on')
            self.last_pose = pn
        if pn == 5 and self.locked and self.use_lock:
            self.locked = False
            self.vibrate(1)
            print('unlock')
            if self.onUnlock != None:
                self.onUnlock()

    def getArm(self):
        return self.current_arm

    def getXDirection(self):
        return self.current_xdir

    def getGyro(self):
        return self.current_gyro

    def getAccel(self):
        return self.current_accel

    def getTimeMilliseconds(self):
        return round(time.time() * 1000)

    def getRoll(self):
        return self.current_roll

    def getPitch(self):
        return self.current_pitch

    def getYaw(self):
        return self.current_yaw

    def setLockingPolicy(self, policy):
        if policy == 'none':
            self.use_lock = False
        elif policy == 'standard':
            self.use_lock = True

    def lock(self):
        self.locked = True
        self.vibrate(1)
        if self.onLock != None:
            self.onLock()

    def unlock(self, unlock_type):
        if unlock_type == 'timed':
            self.vibrate(1)
            self.locked = False
            self.timed = True
        if unlock_type == 'hold':
            self.vibrate(1)
            self.locked = False
            self.timed = False

    def isUnlocked(self):
        if self.locked:
            return False
        else:
            return True

    def notifyUserAction(self):
        self.vibrate(1)

    def keyboard(self, kkey, kedge, kmod):
        if pkeyboard != None:
            tkey = kkey
            if tkey == 'left_arrow':
                tkey = pkeyboard.left_key
            if tkey == 'right_arrow':
                tkey = pkeyboard.right_key
            if tkey == 'up_arrow':
                tkey = pkeyboard.up_key
            if tkey == 'down_arrow':
                tkey = pkeyboard.down_key
            if tkey == 'space':
                pass
            if tkey == 'return':
                tkey = pkeyboard.return_key
            if tkey == 'escape':
                tkey = pkeyboard.escape_key
            if kmod == 'left_shift':
                pkeyboard.press_key(pkeyboard.shift_l_key)
            if kmod == 'right_shift':
                pkeyboard.press_key(pkeyboard.shift_r_key)
            if kmod == 'left_control':
                pkeyboard.press_key(pkeyboard.control_l_key)
            if kmod == 'right_control':
                pkeyboard.press_key(pkeyboard.control_r_key)
            if kmod == 'left_alt':
                pkeyboard.press_key(pkeyboard.alt_l_key)
            if kmod == 'right_alt':
                pkeyboard.press_key(pkeyboard.alt_r_key)
            if kmod == 'left_win':
                pkeyboard.press_key(pkeyboard.super_l_key)
            if kmod == 'right_win':
                pkeyboard.press_key(pkeyboard.super_r_key)
            if kedge == 'down':
                pkeyboard.press_key(tkey)
            elif kedge == 'up':
                pkeyboard.release_key(tkey)
            elif kedge == 'press':
                pkeyboard.tap_key(tkey)
            if kmod == 'left_shift':
                pkeyboard.release_key(pkeyboard.shift_l_key)
            if kmod == 'right_shift':
                pkeyboard.release_key(pkeyboard.shift_r_key)
            if kmod == 'left_control':
                pkeyboard.release_key(pkeyboard.control_l_key)
            if kmod == 'right_control':
                pkeyboard.release_key(pkeyboard.control_r_key)
            if kmod == 'left_alt':
                pkeyboard.release_key(pkeyboard.alt_l_key)
            if kmod == 'right_alt':
                pkeyboard.release_key(pkeyboard.alt_r_key)
            if kmod == 'left_win':
                pkeyboard.release_key(pkeyboard.super_l_key)
            if kmod == 'right_win':
                pkeyboard.release_key(pkeyboard.super_r_key)

    def centerMousePosition(self):
        if pmouse != None:
            x_dim, y_dim = pmouse.screen_size()
            pmouse.move(x_dim / 2, y_dim / 2)

    def mouse(self, button, edge, mod):
        if pmouse != None:
            mpos = pmouse.position()
            if button == 'left':
                mbut = 1
            elif button == 'right':
                mbut = 2
            elif button == 'center':
                mbut = 3
            else:
                mbut = 1
            if edge == 'down':
                pmouse.press(mpos[0], mpos[1], mbut)
            elif edge == 'up':
                pmouse.release(mpos[0], mpos[1], mbut)
            elif edge == 'click':
                pmouse.click(mpos[0], mpos[1], mbut)

    def getPose(self):
        return self.PoseToStr(self.last_pose)

    def getPoseSide(self):
        if self.last_pose == 2 and self.current_arm == 'right' or self.last_pose == 3 and self.current_arm == 'left':
            return 'waveLeft'
        elif self.last_pose == 3 and self.current_arm == 'right' or self.last_pose == 2 and self.current_arm == 'left':
            return 'waveRight'
        else:
            return self.PoseToStr(self.last_pose)

    def isLocked(self):
        return self.locked

    def mouseMove(self, x, y):
        if pmouse != None:
            pmouse.move(x, y)

    def title_contains(self, text):
        window_str = self.get_active_window_title()
        if window_str.find(text) > -1:
            return True
        else:
            return False

    def class_contains(self, text):
        window_str = self.get_active_window_class()
        if window_str.find(text) > -1:
            return True
        else:
            return False

    def rotSetCenter(self):
        self.center_roll = self.current_roll
        self.center_pitch = self.current_pitch
        self.center_yaw = self.current_yaw

    def rotRoll(self):
        return self.current_rot_roll

    def rotPitch(self):
        return self.current_rot_pitch

    def rotYaw(self):
        return self.angle_dif(self.current_yaw, self.center_yaw)

    def getBox(self):
        if self.current_rot_pitch > self.box_factor:
            if self.current_rot_yaw > self.box_factor:
                return 2
            elif self.current_rot_yaw < -self.box_factor:
                return 8
            else:
                return 1
        elif self.current_rot_pitch < -self.box_factor:
            if self.current_rot_yaw > self.box_factor:
                return 4
            elif self.current_rot_yaw < -self.box_factor:
                return 6
            else:
                return 5
        else:
            if self.current_rot_yaw > self.box_factor:
                return 3
            if self.current_rot_yaw < -self.box_factor:
                return 7
            return 0

    def getHBox(self):
        if self.current_rot_yaw > self.box_factor:
            return 1
        elif self.current_rot_yaw < -self.box_factor:
            return -1
        else:
            return 0

    def getVBox(self):
        if self.current_rot_pitch > self.box_factor:
            return 1
        elif self.current_rot_pitch < -self.box_factor:
            return -1
        else:
            return 0

    def clearHistory(self):
        self.mov_history = ''
        self.gest_history = ''
        self.act_history = ''

    def getLastMovements(self, num):
        if num >= 0:
            return self.mov_history[-num:]
        else:
            return self.mov_history

    def getLastGestures(self, num):
        if num >= 0:
            return self.gest_history[-num:]
        else:
            return self.gest_history

    def getLastActions(self, num):
        if num >= 0:
            return self.act_history[-num:]
        else:
            return self.act_history

    def PoseToStr(self, posenum):
        if posenum == 0:
            return 'rest'
        elif posenum == 1:
            return 'fist'
        elif posenum == 2:
            return 'waveIn'
        elif posenum == 3:
            return 'waveOut'
        elif posenum == 4:
            return 'fingersSpread'
        elif posenum == 5:
            return 'doubleTap'
        else:
            return 'unknown'

    def PoseToChar(self, posenum):
        if posenum == 0:
            return 'R'
        elif posenum == 1:
            return 'F'
        elif posenum == 2:
            return 'I'
        elif posenum == 3:
            return 'O'
        elif posenum == 4:
            return 'S'
        elif posenum == 5:
            return 'D'
        else:
            return 'U'

    def limit_angle(self, angle):
        if angle > math.pi:
            return angle - 2.0 * math.pi
        if angle < -2.0 * math.pi:
            return angle + 2.0 * math.pi
        return angle

    def angle_dif(self, angle, ref):
        if ref >= 0:
            if angle >= 0:
                return self.limit_angle(angle - ref)
            elif angle >= ref - math.pi:
                return self.limit_angle(angle - ref)
            else:
                return self.limit_angle(angle + 2.0 * math.pi - ref)
        else:
            if angle <= 0:
                return self.limit_angle(angle - ref)
            if angle <= ref + math.pi:
                return self.limit_angle(angle - ref)
            return self.limit_angle(angle - 2.0 * math.pi - ref)

    def get_active_window_title(self):
        try:
            root = Popen(['xprop', '-root', '_NET_ACTIVE_WINDOW'], stdout=PIPE)
            for line in root.stdout:
                mw = re.search('^_NET_ACTIVE_WINDOW.* ([\\w]+)$', line)
                if mw != None:
                    id_ = mw.group(1)
                    id_w = Popen(['xprop',
                     '-id',
                     id_,
                     'WM_NAME'], stdout=PIPE)
                    break

            if id_w != None:
                for line in id_w.stdout:
                    match = re.match('WM_NAME\\(\\w+\\) = (?P<name>.+)$', line)
                    if match != None:
                        return match.group('name')

            return ''
        except:
            return ''

    def get_active_window_class(self):
        try:
            root = Popen(['xprop', '-root', '_NET_ACTIVE_WINDOW'], stdout=PIPE)
            for line in root.stdout:
                mw = re.search('^_NET_ACTIVE_WINDOW.* ([\\w]+)$', line)
                if mw != None:
                    id_ = mw.group(1)
                    id_w = Popen(['xprop',
                     '-id',
                     id_,
                     'WM_CLASS'], stdout=PIPE)
                    break

            if id_w != None:
                for line in id_w.stdout:
                    match = re.match('WM_CLASS\\(\\w+\\) = (?P<name>.+)$', line)
                    if match != None:
                        return match.group('name')

            return ''
        except:
            return ''


def proc_imu(quat, acc, gyro):
    print(quat)

def preprocess_dxdy(sampleData):
    sampleData = dp.smoothdata(sampleData)
    sampleData = dp.remove_redundantdata(sampleData)
    sampleData = dp.normalizedata1(sampleData)
    sampleData = dp.interpolatedata(sampleData)
    return sampleData

def preprocess_imu(acc, gyr, ori, maxTrainSteps):
    scaler = preprocessing.MaxAbsScaler()
    accList, gyrList, oriList = dp.separateRawData(acc, gyr, ori)

    # scaling data
    accList, gyrList, oriList = dp.scaledatactc(accList, gyrList, oriList, scaler)

    imuData = np.concatenate((accList, gyrList, oriList), axis=0)
    # imuData = imuData.T
    imuDataList,_ = dp.addPadding(np.array([imuData]), maxTrainSteps)
    return np.array(imuDataList)

if __name__ == '__main__':
    takeInput = False

    #loading models
    # dxdyModel = load_model(modelDir + "/" + "BGRU1layerdxdy.h5")
    imuModel = load_model(modelDir + "/" + "BGRU2layerImu.h5")
    maxTrainSteps = imuModel.flattened_layers[0].batch_input_shape[1]

    # plot code starts
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.hold(True)

    background = fig.canvas.copy_from_bbox(ax.bbox) # cache the background
    plt.show(False)
    plt.draw()
    points = ax.plot(0, 0, 'o')[0]
    # plot code ends

    m = Myo(sys.argv[1] if len(sys.argv) >= 2 else None)
    # m.add_imu_handler()
    m.connect()

    sampleData = []
    accData = []
    gyrData = []
    oriData = []
    while True:

        # Controls the start and stop of taking data using  "l" and "p" keys in keyboard
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:

            line = sys.stdin.readline()
            # print("hey" + line + "th")

            if line == "l\n":
                m.rotSetCenter()
                takeInput = True
                sampleData = []
                accData = []
                gyrData = []
                oriData = []


            if line == "p\n":
                takeInput = False
                plt.close()

                # redraw plot
                fig, ax = plt.subplots(1, 1)
                ax.set_aspect('equal')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.hold(True)

                background = fig.canvas.copy_from_bbox(ax.bbox)  # cache the background
                plt.show(False)
                plt.draw()
                points = ax.plot(0, 0, 'o')[0]

                # call classifier here on sampleData

                # calling dxdy model on sampleData
                # sampleData = preprocess_dxdy(sampleData)
                # prediction1 = dxdyModel.predict_classes(np.array([sampleData]),batch_size=1)
                # print(prediction1)

                # calling imu model on sampleData
                imuDataList = preprocess_imu(accData,gyrData,oriData, maxTrainSteps)
                prediction2 = imuModel.predict_classes(imuDataList, batch_size=1)
                print(prediction2)

        m.run()

        if takeInput:
            (x, y) = (-m.current_rot_yaw * 2000, m.current_rot_pitch * 2000)
            sampleData.append((x,y))
            accData.append(m.current_accel)
            gyrData.append(m.current_gyro)
            oriData.append(m.current_quat)

            x = (x + 2000)/(4000)
            y = (y + 2000)/(4000)
            x = 2*x - 1
            y = 2*y - 1

            # mirror image in y axis
            x *= -1
            # mirror image in x axis
            y *= -1

            # plot code starts
            points.set_data(x, y)
            # restore background
            fig.canvas.restore_region(background)

            # redraw just the points
            ax.draw_artist(points)

            # fill in the axes rectangle
            fig.canvas.blit(ax.bbox)
            # plot code ends

            # print(m.current_emg)
            # print(str(x)+ " " + str(y))


#+++ okay decompyling PyoConnect.pyc 
# decompiled 1 files: 1 okay, 0 failed, 0 verify failed
# 2015.08.13 17:21:34 UTC
