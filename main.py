#!/home/cantab/miniconda3/envs/env2/bin/python2

import sys, time, timeit, threading, pickle
import Queue as queue
import numpy as np
from scipy.misc import imresize
from matplotlib import pyplot as plt
import pygame, cv2, math
import pygame.camera
from pygame.locals import *


# Parameters
target_filename   = "Heart_49x49.png"
CGH_filename      = "CGH_Heart.png"
light_spots       = 29
m_CGH             = 32                                                          # CGH resolusion size
n_CGH             = 32                                                          # uint8
n1_replay         = 347
n2_replay         = 1243

each_pics_token   = 200

frames_combined   = 2
perturb           = 50
t                 = 100.                                                        # Temperture
t_stop            = 5

pixel_block       = 1
gamma             = 1.0                                                         # According to algorithm
bg_noise          = 0                                                           # Measure!
mse_threshold     = 0.002

CCD_precap_time   = 0.0                                                         # minute
CCD_port          = "/dev/video0"
m_CCD             = 896  # 480                                                  # DMD resolusion size
n_CCD             = 1600 # 640                                                  # uint8 > float32
CCD_delay_time    = 0.005                                                       # 30Hz
queue_delay_time  = 0.001

m_DMD   = 1080                                                                  # DMD resolusion size
n_DMD   = 1920                                                                  # uint8
m_shift = int(m_DMD/2 - m_CGH/2)
n_shift = int(n_DMD/2 - n_CGH/2)


# Functions
def DMDupdate(DMD_image0,m_CGH,n_CGH,m_DMD,n_DMD,DMDupdate_Queue):
    DMD_image1 = np.copy(DMD_image0)
    v = np.random.randint(0, n_CGH)
    u = np.random.randint(0, m_CGH)
    if DMD_image1[v,u,0] == 255:
        DMD_image1[v::n_CGH,u::m_CGH,:] = [0,0,0]
    else:
        DMD_image1[v::n_CGH,u::m_CGH,:] = [255,255,255]                         # DMD_image1[v::n_CGH,u::m_CGH,:]=[255,0,0];DMD_image1[v+1::n_CGH,u+1::m_CGH,:]=[255,0,0];DMD_image1[v+2::n_CGH,u+2::m_CGH,:]=[255,0,0]
    DMD_surface = pygame.surfarray.make_surface(DMD_image1)
    screen.blit(DMD_surface, (0,0))
    pygame.display.update()
    DMDupdate_Queue.put(DMD_image1)

def MSE(DMD_image1,tmp,img_target,n,m_CGH,n_CGH,m_regionW,n_regionW,m_regionB,n_regionB,gamma,bg_noise,CCD_delay_time,MSE_Queue):
    for p in range(frames_combined):
        pic_measured = camera.get_image()                                       # Pygame data type
        time.sleep(CCD_delay_time)
        tmp[:,:,p] = pygame.surfarray.array3d(pic_measured)[n1_replay:n2_replay,:,0] # Numpy array type; Red image (:,:,0)
    img_measured = np.mean(tmp, 2)
    img_measured = img_measured.astype(np.float32)
    img_measured = imresize(img_measured, [light_spots,light_spots], interp="bilinear", mode=None)

    # Cost Functions
    mse1 = np.linalg.norm(img_target - img_measured)                            # Judge Methods v0.1
    MSE_Queue.put([mse1, pic_measured, img_measured, DMD_image1])

def DMSE(n,total_iter,mse0,mse1,DMD_image0,DMD_image1,t,DMSE_Queue):
    d   = mse1-mse0
    psa = np.exp(-d/t)
    if abs(d) > mse_threshold:
        if d < 0.0:
            mse0 = np.copy(mse1)
            DMD_image0 = np.copy(DMD_image1)
            print(str(n)+"/"+str(total_iter)+"  O        mse0:"+str(mse0)+"  mse1:"+str(mse1)+"  Psa:"+str(psa)+"  t:"+str(t))   # Check
        else:
            if psa>np.random.random():
                mse0 = np.copy(mse1)
                DMD_image0 = np.copy(DMD_image1)
                print(str(n)+"/"+str(total_iter)+"    -      mse0:"+str(mse0)+"  mse1:"+str(mse1)+"  Psa:"+str(psa)+"  t:"+str(t))   # Check
            else:
                print(str(n)+"/"+str(total_iter)+"      X    mse0:"+str(mse0)+"  mse1:"+str(mse1)+"  Psa:"+str(psa)+"  t:"+str(t))   # Check
    else:
        print(str(n)+"/"+str(total_iter)+"       NG  mse0:"+str(mse0)+"  mse1:"+str(mse1)+"  mse2:"+str(psa)+"  t:"+str(t))   # Check
    DMSE_Queue.put([mse0, DMD_image0, psa])


# Functions
# DMD: Display
pygame.init()

DMD_image0 = cv2.imread("CGHs/" + CGH_filename, 0)                              # Gray Target Image
if np.shape(DMD_image0) == ():
    sys.exit("No CGH Image!!!")
DMD_image0 = np.dstack([DMD_image0,DMD_image0,DMD_image0])
DMD_image0 = DMD_image0.astype(np.float32)

DMD_surface = pygame.surfarray.make_surface(DMD_image0)                         # Pygame Matrix
screen = pygame.display.set_mode((n_DMD,m_DMD), pygame.RESIZABLE)
screen.blit(DMD_surface, (0,0))
pygame.display.update()


# CCD: Measured Image
time.sleep(7)
pygame.camera.init()
if pygame.camera.list_cameras() == []:                                          # Check Camera
    sys.exit("No Camera!!!")
camera = pygame.camera.Camera("/dev/video0",(n_CCD,m_CCD))
camera.start()
pic_initial = camera.get_image()                                                # Pygame data type
time.sleep(0.1)
pygame.image.save(pic_initial,"Results/Initial.png")                            # Check
time.sleep(0.005)                                                               # Check


# Computer: Target Image
img_target = cv2.imread("Targets/" + target_filename, 0)                        # Gray Target Image
if np.shape(img_target) == ():
    sys.exit("No Target Image!!!")
img_target = imresize(img_target, [light_spots,light_spots], interp="lanczos", mode=None)
img_target = np.rot90(img_target, axes=(1,0))
img_target = np.flip(img_target, 1)
img_target = img_target.astype(np.float32)
m_regionW, n_regionW = np.where(img_target>=51.0)
m_regionB, n_regionB = np.where(img_target<51.0)
cv2.imwrite("Results/Target.png", img_target)
tmp = np.zeros([m_CCD,m_CCD,frames_combined], dtype=np.uint8)


# CCD Pre-capture
time0 = timeit.default_timer()
time1 = timeit.default_timer()
while time1-time0 < CCD_precap_time*60:
    camera.get_image()
    time.sleep(CCD_delay_time)
    time1 = timeit.default_timer()


# Total Iterations
N = 0; T = t
while T>t_stop:
    N += 1
    T *= 0.99
print("N: "+str(N)+"; n:"+str(N*perturb))
total_iter = N*perturb*frames_combined


# Main Loop
Loop_exit = False
mse0      = 1E+10
mse_array = np.array([mse0], dtype=np.float32)
psa_array = np.array([], dtype=np.float32)
j = 0; n = 0; s = 0; x = 0
time0 = timeit.default_timer()
while t > t_stop and not Loop_exit:
    i = 0
    while i < perturb and not Loop_exit:
        # Break Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                with open('Results/MSE.pickle', 'wb') as data:
                    pickle.dump([mse_array, psa], data, protocol=pickle.HIGHEST_PROTOCOL)
                Loop_exit = True
                pygame.quit()

        # Update DMD
        DMDupdate_Queue = queue.Queue()
        threadDMD = threading.Thread(target = DMDupdate(DMD_image0,m_CGH,n_CGH,m_DMD,n_DMD,DMDupdate_Queue))
        threadDMD.start()
        threadDMD.join()
        DMD_image1 = DMDupdate_Queue.get()
        time.sleep(queue_delay_time)

        # MSE Calculation
        MSE_Queue = queue.Queue()
        threadMSE = threading.Thread(target = MSE(DMD_image1,tmp,img_target,n,m_CGH,n_CGH,m_regionW,n_regionW,m_regionB,n_regionB,gamma,bg_noise,CCD_delay_time,MSE_Queue))
        threadMSE.start()
        threadMSE.join()
        mse1, pic_measured, img_measured, DMD_image1 = MSE_Queue.get()
        time.sleep(queue_delay_time)

        # Fix the begining MSE issue
        if n == 0:
            mse1 = 1E+7

        # Forward/Backward
        DMSE_Queue = queue.Queue()
        threadDMSE = threading.Thread(target = DMSE(n,total_iter,mse0,mse1,DMD_image0,DMD_image1,t,DMSE_Queue))
        threadDMSE.start()
        threadDMSE.join()
        mse0, DMD_image0, psa = DMSE_Queue.get()
        time.sleep(queue_delay_time)

        # Trace back to the old trace (A new direction for developing an optimisation algorithm)
        if mse0 < mse_array.min():
            DMD_image_best = DMD_image0
            pygame.image.save(pic_measured,"Results/"+str(x)+"_"+str(mse0)+".png")
            cv2.imwrite("Results/MSE_"+str(x)+"_"+str(n)+".png", img_measured)               # Check
            cv2.imwrite("Results/CGH_"+str(x)+"_"+str(n)+".png", DMD_image1[:n_CGH,:m_CGH])  # Check
            x += 1
            s = 0
        else:
            s += 1
            if s == 2*perturb:
                DMD_image0 = DMD_image_best                                     # Update the old solution
                s = 0
                print("Back to Previous Best CGH!")

        mse_array = np.hstack((mse_array, mse0))
        psa_array = np.hstack((psa_array, psa))

        n += 1                                                                  # (j*perturb)+i
        i += 1

    j += 1
    t = t*0.999

    time1 = timeit.default_timer()
    dtime = time1 -time0
    processing = j/float(N)
    time_left  = (1/processing - 1)*dtime/3600.0

    # print(str(processing*100)+"%,  "+str(time_left)+" hour")
    print(str(time_left)+"hour,    "+str(processing*100)+"%")


# Results
# stop = timeit.default_timer()
# print('Computation Time: ' + str(stop-start))
# print('MSE: ' + str(mse_array))

pic_ending = camera.get_image()
time.sleep(0.1)
pygame.image.save(pic_ending,"Results/Reconstructed Image.png")
cv2.imwrite("Results/CGH.png", DMD_image0)

# Save Data
with open('Results/MSE.pickle', 'wb') as data:
    pickle.dump([mse_array, psa], data, protocol=pickle.HIGHEST_PROTOCOL)

# Show Results
plt.figure(0)
plt.plot(mse_array)
plt.show()

# CCD Release
camera.stop()

# DMD Release
pygame.quit()
quit()
