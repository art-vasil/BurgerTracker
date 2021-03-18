# BurgerTracker

## Overview

This project is to detect and track the burgers on the video stream and estimate the cooking time using EdgeTpu. 
The quantized ssd_mobilenet_v2 model for burger detection is trained and optimized as a tflife model for USB Coral board, 
which is necessary to support real-time processPi.

## Structure
   
- src

    The main source code for burger detection and tracking.
    
- utils

    * The optimized models for TPU
    * The source code for utilization of this project
    
- app

    The main execution file

- requirements

    All the dependencies for running project on PC

- settings

    Several settings for this project
    
## Installation

- Environment

    Ubuntu 18.04, Python 3.6

- Dependency Installation

    Please navigate to this project directory and run the following command in the terminal.
    
    ```
        echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
        sudo apt-get update
        sudo apt-get install libedgetpu1-std
        pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
        pip3 install -r requirements.txt
    ```

## Execution

- Please connect USB Coral Accelerator and Web Camera with your PC via USB 3.0 port.

- Please navigate to this project directory and run the following command in the terminal.

    ```
        python3 app.py
    ```

