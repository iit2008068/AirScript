import sys
import serial
import time
import json
import argparse
import os

import memcache
from urllib import urlopen

def playIR(path):
  ir_serial = serial.Serial("/dev/ttyACM1", 9600, timeout = 1)
  if path and os.path.isfile(path):
    print ("Playing IR with %s ..." % path)
    f = open(path)
    data = json.load(f)
    f.close()
    recNumber = len(data['data'])
    rawX = data['data']

    ir_serial.write("n,%d\r\n" % recNumber)
    ir_serial.readline()

    postScale = data['postscale']
    ir_serial.write("k,%d\r\n" % postScale)
    #time.sleep(1.0)
    msg = ir_serial.readline()
    #print msg

    for n in range(recNumber):
        bank = n / 64
        pos = n % 64
        if (pos == 0):
          ir_serial.write("b,%d\r\n" % bank)

        ir_serial.write("w,%d,%d\n\r" % (pos, rawX[n]))

    ir_serial.write("p\r\n")
    msg = ir_serial.readline()
    print msg
    #ir_serial.close()
  else:
    print "Playing IR..."
    ir_serial.write("p\r\n")
    time.sleep(1.0)
    msg = ir_serial.readline()
    print msg
  ir_serial.close()


filename = '/sys/class/leds/led0/brightness'
def flash(t):
   tf = open(filename, 'w')
   tf.write('1')
   tf.close()
   time.sleep(t)
   tf = open(filename, 'w')
   tf.write('0')
   tf.close()

mc = memcache.Client(['127.0.0.1:11211'], debug=0)

time.sleep(1)

while True:
    pose = mc.get('pose')
    fist_edge = mc.get('fist_edge')
    finger_edge = mc.get('finger_edge')
    wave_edge = mc.get('wave_edge')
    recent = mc.get('recent')
    ttl = mc.get('ttl')
    #print(pose)
    #print edge
    if (pose == 'fist') and (fist_edge == 'on') and (ttl != 'yes'):
        mc.set('ttl', 'yes', 2.9)
        mc.set('sound', 'power', 2.9)
        playIR('led/led-power.json')
        print('POWER '+ str(time.time()))
        flash(0.1)
        sys.stdout.flush()
    elif (pose == 'waveOut') and (wave_edge == 'on') and (ttl != 'yes'):
        mc.set('ttl', 'yes', 2.9)
        mc.set('sound', 'wo', 2.9)
        playIR('led/led-jump7.json')
        print('QUICK '+ str(time.time()))
        flash(0.1)
        sys.stdout.flush()
    elif (pose == 'waveIn') and (wave_edge == 'on') and (ttl != 'yes'):
        mc.set('ttl', 'yes', 2.9)
        mc.set('sound', 'wi', 2.9)
        playIR('led/led-fade7.json')
        print('SLOW '+ str(time.time()))
        flash(0.1)
        sys.stdout.flush()
    time.sleep(0.05)
