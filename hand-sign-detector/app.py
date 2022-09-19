import cv2
import time
import streamlit as st
from st_btn_select import st_btn_select

from HandSignModel.HandSignModel import HandSignModel 
from utils import draw_bounding_box, preprocessing

model = HandSignModel(onnx_path="checkpoints/yolov4_tiny_1_3_416_416_static.onnx")

def from_camera():
    run = False
    conf = st.sidebar.slider('Confidence: ', 0.0, 1.0, 0.4)
    iou = st.sidebar.slider('IOU Threshold: ', 0.0, 1.0, 0.4)

    selection = st_btn_select(('Run', 'Stop'))
    if selection == 'Run':
        run = True 
    if selection == 'Stop':
        run = False

    if run:
        FRAME_WINDOW = st.image([])


    cap = cv2.VideoCapture(0)

    # used to record the time when we processed last frame
    prev_frame_time = 0
     
    # used to record the time at which we processed current frame
    new_frame_time = 0
    i = 0
    boxes = [[]]
    while run:
        _, frame = cap.read()
        i+=1
        # prep_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prep_frame = preprocessing(frame)

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time
        frame = cv2.rectangle(frame, (0, 0), (30, 30), (0, 102, 255), -1)
        frame = cv2.putText(frame, str(fps), (3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1)

        if i % 5 == 0:
            boxes = model.predict(prep_frame, threshold=conf, iou_threshold=iou)

        frame = draw_bounding_box(frame, boxes)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

def flip():
    i = 0
    run = False

    selection = st_btn_select(('Run', 'Stop'))
    if selection == 'Run':
        run = True 
    if selection == 'Stop':
        run = False

    if run:
        FRAME_WINDOW = st.image([])


    cap = cv2.VideoCapture(0)

    # used to record the time when we processed last frame
    prev_frame_time = 0
     
    # used to record the time at which we processed current frame
    new_frame_time = 0
 
    while run:
        i+=1
        _, frame = cap.read()
        # prep_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i % 2 == 0:
            frame = frame[::, ::-1, ::] 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

def main():
    page_names_to_funcs = {
        # "From Uploaded Picture": from_picture,
        "From Camera": from_camera
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Hand Sign Detector", page_icon=":pencil2:"
    )
    st.title("Hand Sign Detector")
    # st.sidebar.subheader("Configuration")
    main()


