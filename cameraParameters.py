#!/home/cantab/miniconda3/envs/env2/bin/python2

import os

def CCDsetting(GAIN, WB, Exposure, Focus):
    os.system("v4l2-ctl --set-ctrl brightness=128")
    os.system("v4l2-ctl --set-ctrl contrast=128")
    os.system("v4l2-ctl --set-ctrl saturation=128")
    os.system("v4l2-ctl --set-ctrl white_balance_temperature_auto=0")
    os.system("v4l2-ctl --set-ctrl gain=" + str(GAIN))                          # Check
    os.system("v4l2-ctl --set-ctrl power_line_frequency=2")
    os.system("v4l2-ctl --set-ctrl white_balance_temperature=" + str(WB))       # Check
    os.system("v4l2-ctl --set-ctrl sharpness=128")
    os.system("v4l2-ctl --set-ctrl backlight_compensation=0")
    os.system("v4l2-ctl --set-ctrl exposure_auto=0")
    os.system("v4l2-ctl --set-ctrl exposure_absolute=" + str(Exposure))         # Check
    os.system("v4l2-ctl --set-ctrl exposure_auto_priority=0")
    os.system("v4l2-ctl --set-ctrl pan_absolute=0")
    os.system("v4l2-ctl --set-ctrl tilt_absolute=0")
    os.system("v4l2-ctl --set-ctrl focus_absolute=" + str(Focus))               # Check
    os.system("v4l2-ctl --set-ctrl focus_auto=0")
    os.system("v4l2-ctl --set-ctrl zoom_absolute=500")

def CCDgetting():
    print(os.system("v4l2-ctl --get-ctrl brightness"),
          os.system("v4l2-ctl --get-ctrl contrast"),
          os.system("v4l2-ctl --get-ctrl saturation"),
          os.system("v4l2-ctl --get-ctrl white_balance_temperature_auto"),
          os.system("v4l2-ctl --get-ctrl gain"),
          os.system("v4l2-ctl --get-ctrl power_line_frequency"),
          os.system("v4l2-ctl --get-ctrl white_balance_temperature"),
          os.system("v4l2-ctl --get-ctrl sharpness"),
          os.system("v4l2-ctl --get-ctrl backlight_compensation"),
          os.system("v4l2-ctl --get-ctrl exposure_auto"),
          os.system("v4l2-ctl --get-ctrl exposure_absolute"),
          os.system("v4l2-ctl --get-ctrl exposure_auto_priority"),
          os.system("v4l2-ctl --get-ctrl pan_absolute"),
          os.system("v4l2-ctl --get-ctrl tilt_absolute"),
          os.system("v4l2-ctl --get-ctrl focus_absolute"),
          os.system("v4l2-ctl --get-ctrl focus_auto"),
          os.system("v4l2-ctl --get-ctrl zoom_absolute"))


if __name__ == '__main__':
    CCDsetting(GAIN=5, WB=3125, Exposure=777, Focus=45)
    CCDgetting()
