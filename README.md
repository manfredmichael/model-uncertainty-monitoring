# hand-sign-detector

American Sign Language Letter Detector using YOLOv4-Tiny for my week 4 project during Nodeflux internship.

![](https://github.com/manfredmichael/hand-sign-detector/blob/main/assets/demo.gif?raw=true)

## How to Run This App

### Via Cloning This Repo
First, clone this repo: `git clone https://github.com/manfredmichael/hand-sign-detector.git`. Next, you need to run the streamlit interface to use this demo. Download this [model](https://drive.google.com/file/d/14PGd7IWgrK0WzL39MsmKpMHRV18J3NmS/view?usp=sharing) and put it in `hand-sign-detector/checkpoints/`


- Install dependenies: `pip install -r requirements.txt`
- Change the working directory: `cd hand-sign-detector`
- Run streamlit app: `streamlit run app.py`


## Technologies & Tools
![](https://github.com/manfredmichael/hand-sign-detector/blob/main/assets/workspace.png?raw=true)

## About the model

[YOLOv4](https://arxiv.org/abs/2004.10934) Tiny was retrained on [American Sign Language Letter Dataset](https://public.roboflow.com/object-detection/american-sign-language-letters) with 1,728 images of 26 classes (A-Z).

# model-uncertainty-monitoring
# model-uncertainty-monitoring
# model-uncertainty-monitoring
