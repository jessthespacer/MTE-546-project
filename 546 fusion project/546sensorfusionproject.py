import csv
import numpy as np
import sys
from navpy import dcm2quat
import math
import matplotlib.pyplot as plt

POSANDHEAD = [0,0,0,0,0,0]
simulation = [[],[],[],[],[],[]]
VELOCITY = [0,0,0]
SAMPRATE = 0.02
GRAVITY = 9.81
NUMSENSORS = 2
previouseulerorientation = [None,None]


def euler_to_directionvector(roll,pitch,yaw):
    x = math.cos(yaw)*math.cos(pitch)
    y = math.sin(yaw)*math.cos(pitch)
    z = math.sin(pitch)
    return[x,y,z]

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return [roll_x, pitch_y, yaw_z] # in radians

def converttoseconds(hhmmssinput):
    return int(hhmmssinput[:2])*3600+int(hhmmssinput[3:5])*60+int(hhmmssinput[6:])

def pullorgenerateexperimentaldata(filename,noise,noiseval,write,writefilename):
    store = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if len(row) == 10:
                    if isinstance(row[0], str):
                        row[0]= converttoseconds(row[0])
                    row = [float(i) for i in row]
                    if noise:
                        gennoise = np.random.normal(0,noiseval,10)
                        gennoise[0] = 0
                        row = row + gennoise
                    store.append(row)
                    line_count += 1
        line_count -= 1
        print(f'Processed {line_count} lines.')

        if write:
            with open('rawdatatwo.csv', mode='w', newline='') as employee_file:
                filewriter= csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(['Time', 'Ax', 'Ay','Az','Gx','Gy','Gz','Mx','My','Mz'])
                filewriter.writerows(store)


    return store

def fuseIMU(IMUdatarow,sensor):
    if len(IMUdatarow) == 10:
        accel = IMUdatarow[1:4]
        gyro = IMUdatarow[4:7]
        mag = IMUdatarow[7:10]
        magnorm = np.linalg.norm(mag)
        magunit = [x/magnorm for x in mag]

        accelnorm = np.linalg.norm(accel)
        accelunit = [x/accelnorm for x in accel]

        Dvect = [-1*x for x in accelunit]
        Evect = np.cross(np.array(Dvect), np.array(magunit))
        Enorm = np.linalg.norm(Evect)
        Evect = [x/Enorm for x in Evect]

        Nvect = np.cross(Evect, Dvect)
        Nnorm = np.linalg.norm(Nvect)
        Nvect = [x/Nnorm for x in Nvect]

        # Build the DCM
        Cmat = [Nvect, Evect, Dvect]
        q0,Qvect = dcm2quat(np.array(Cmat))
        Qvect = list(Qvect)
        Qvect.append(q0)
        Eulervect = euler_from_quaternion(Qvect[0],Qvect[1],Qvect[2],Qvect[3])
        headingvect = euler_to_directionvector(Eulervect[0],Eulervect[1],Eulervect[2])

        if not previouseulerorientation[sensor]:
            previouseulerorientation[sensor] = Eulervect
            finalheading = headingvect
        else:
            previouseulerorientation[sensor][0] += gyro[0]*SAMPRATE
            previouseulerorientation[sensor][1] += gyro[1]*SAMPRATE
            previouseulerorientation[sensor][2] += gyro[2]*SAMPRATE
            gyroheadingvect = euler_to_directionvector(previouseulerorientation[sensor][0],previouseulerorientation[sensor][1],previouseulerorientation[sensor][2])
            gyroheadingnorm = np.linalg.norm(gyroheadingvect)
            gyroheadingvect = [x/gyroheadingnorm for x in gyroheadingvect]
            finalheading = [(x + y)/2 for x, y in zip(gyroheadingvect, headingvect)]

        accelunit.insert(0,(accelnorm-GRAVITY))
        finalresult = accelunit+finalheading


        return finalresult


def storeandapplysimulationpoint(accelheading):
    VELOCITY[0] += accelheading[1] * accelheading[0] * SAMPRATE
    VELOCITY[1] += accelheading[2] * accelheading[0] * SAMPRATE
    VELOCITY[2] += accelheading[3] * accelheading[0] * SAMPRATE
    POSANDHEAD[0] += VELOCITY[0]*SAMPRATE
    POSANDHEAD[1] += VELOCITY[1]*SAMPRATE
    POSANDHEAD[2] += VELOCITY[2]*SAMPRATE
    POSANDHEAD[3] = accelheading[4]
    POSANDHEAD[4] = accelheading[5]
    POSANDHEAD[5] = accelheading[6]

    simulation[0].append(POSANDHEAD[0])
    simulation[1].append(POSANDHEAD[1])
    simulation[2].append(POSANDHEAD[2])
    simulation[3].append(POSANDHEAD[3])
    simulation[4].append(POSANDHEAD[4])
    simulation[5].append(POSANDHEAD[5])

def plotsimulation():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(simulation[0], simulation[1], simulation[2], simulation[3], simulation[4], simulation[5], length=0.1, normalize=True)
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]')
    plt.show()

def runsimulation(IMUdataone,IMUdatatwo):
    lenIMUtwo = len(IMUdatatwo)
    lenIMUone = len(IMUdataone)
    for i in range(len(IMUdataone)):
        if i < lenIMUone and i < lenIMUtwo:
            accelheadingone = fuseIMU(IMUdataone[i],0)
            accelheadingtwo = fuseIMU(IMUdatatwo[i],1)
            # @SHAWN ekf fusion here
            storeandapplysimulationpoint(accelheadingone)

    plotsimulation()
    return False

# Used to generate IMU 2 data
# IMUdatatwo = pullorgenerateexperimentaldata('rawdata.csv',True,.1,True,'rawdatatwo.csv')



IMUdataone = pullorgenerateexperimentaldata('rawdata.csv',False,0,False,'')
IMUdatatwo = pullorgenerateexperimentaldata('rawdatatwo.csv',False,0,False,'')
runsimulation(IMUdataone,IMUdatatwo)

