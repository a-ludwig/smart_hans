# How To Setup Smart Hans

[TOC]

## Setting up the Hardware

<img src="/Users/adi/Nextcloud/smart_hans/AP3/Doku/DSC06218.JPG" alt="DSC06218" style="zoom:50%;" />

### PC

- Set up the PC in a place that fits your environment, we suggest you connect an external screen (besides the projection screen) to be able to eventually handle debugging tasks  like restarting Hans

### Audio

- Audio is being sent by the internal audio of the PC. You can connect a 3.5mm Jack to the back of the PC to a speaker of your liking. Should also work when connecting to a Display using HDMI as output source of the PC (not yet tested, but maybe valid for your scenario)
- **While quality of the audio is not a major issue, it is important that it is existent in order to create better feedback for interactors**

### Camera

- Camera tilted 90° to the left
- about 60° Angle to face o person standing in front of it
- Mark the Floor with something like in the picture above
- Distance Face <> Camera about 40-50 cm
- Height Camera  (lens) about 1.70m
- Hansi has to have a size of about 1.70m
- Set the camera up on the **left** of your Projection/screen

#### checking the camera-setup with software	

- in Windows go to Start and search for "camera" or "Kamera"
- Compare your picture to this one concerning angles etc:
  - <img src="/Users/adi/Nextcloud/smart_hans/AP3/Doku/marked camera pos.png" alt="marked camera pos" style="zoom:50%;" />

- **IT IS IMPORTANT THAT THE INTERACTORS CHIN IS IN THE VIEWFIELD**

- Close the camera App! (Otherwise the python script for Hansi will not start)


### Extras

- If possible place some hey in a place people cannot see. It increases immersion for the installation.

## External screen

- In windows change Hansis screen to be your main screen (if there is a second screen) and the layout from horizontal to vertical 
- When using the projection setup:
  - Make sure the Projector is setup in an angle so the Person standing in Front of the camera does not stand in front of Hansi (see the Picture above) and angled vertical
  - <img src="/Users/adi/Nextcloud/smart_hans/AP3/Doku/DSC06215.JPG" alt="DSC06215" style="zoom:50%;" />

### User position

- Mark the users position about 1m to 1,40m away from Hans so that they **amost** get uncomfortable when a horse of their size stands in front of them. Use refence picture at the top to check.

## Running Hansi

### Starting Hansi

- To Start the Installation go to the Desktop and click on the "start_hansi.bat" file. It will take a short moment to initiate, then hansi will appear on the screen. You can now interact with hansi according to the manual (interaktionsanleitung_hans)

### Stopping Hansi

- Single screen:
  - Doubleclick the video to move it around
  - enter the command prompt and press "ctrl+c", then follow the instructions on screen
- Multiscreen:
  - turn on your second screen and locate the command prompt which is running the script
  - enter the command prompt and press "ctrl+c", then follow the instructions on screen

### Configuring some Parameters at Hans Core

- Hansi Runs from main.py locatet in .... .There are a few Parameters you might want to change, but we suggest you change the physical setup accordingly
  - Thresh: default: 35 - Used to determine if hansi will start to get going by checking for face distance. Increase/decrease in increments of 5. the higher, the further away
  - waiting_time: default: 5 - used to determine how many seconds it will take before hansi will switch from idleing into tapping. decrease for impatient interactors, increase for annoyance

## General Stuff

- Hansi was tested throughougly and did run for 12 hours without interruptions so far. 

- We record every "successful" prediction in abstracted Data -> this will be used to try unsupervised learning for hans

  