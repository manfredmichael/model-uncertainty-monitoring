from HandSignModel.HandSignModel import HandSignModel 
import cv2
from utils import draw_bounding_box

import time

model = HandSignModel(onnx_path="checkpoints/yolov4_tiny_1_3_416_416_static.onnx")

def test():
    img = cv2.imread("test_images/B19_jpg.rf.69527cc1f34d694cc04e55db80ed9b1a.jpg")
    img = cv2.imread("test_images/A1_jpg.rf.c4ccc21338f79e0f68d89dfc817ddd1f.jpg")
    img = cv2.imread("test_images/D9_jpg.rf.3250856285c1a522ba86b3135b5dd6bc.jpg")
    # img = cv2.imread("test_images/E0_jpg.rf.926e842cd69d98b54aec8c371d61bf8d.jpg")

# # Test Predict
    print(model.predict(img))
    cv2.imshow("frame", img)
    while True:
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

def webcam_inference():
    cap= cv2.VideoCapture(0)

    # used to record the time when we processed last frame
    prev_frame_time = 0
     
    # used to record the time at which we processed current frame
    new_frame_time = 0
 
    while True:
        _, frame = cap.read()
        # prep_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prep_frame = cv2.GaussianBlur(frame,(3,3),0)

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time
        frame = cv2.rectangle(frame, (0, 0), (30, 30), (0, 102, 255), -1)
        frame = cv2.putText(frame, str(fps), (3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1)

        t0 = time.time()
        boxes = model.predict(prep_frame)
        
        print("latency: ", time.time() - t0)

        frame = draw_bounding_box(frame, boxes)
        cv2.imshow('hand sign detector', frame)


     
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

def evaluate():
    import os
    for annotation_filename in os.listdir("test_dataset"):
        filename, extension = os.path.splitext(annotation_filename)
        if extension == ".txt": 
            with open("test_dataset/"+annotation_filename) as f:
                img = cv2.imread(f"test_dataset/{filename}.jpg")
                pred = model.predict(img, threshold=0)
                print(annotation_filename)
                annotation = [[]]
                for box in f.read().split('\n'):
                    box = box.split(' ')
                    x_center = float(box[1])
                    y_center = float(box[2])
                    width = float(box[3])
                    height =  float(box[4])
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    annot_row = [x1, y1, x1+width, y1+height, 1,
                                 model.class_names[int(box[0])],
                                 int(box[0])]
                    annotation[0].append(annot_row)

                height, width, _ = img.shape
                with open(f"test_dataset/evaluations/groundtruths/{filename}.txt", 'w') as f:
                    boxes = []
                    for box in annotation[0]:
                        label = box[5]
                        x1 = str(int(box[0] * width))
                        y1 = str(int(box[1] * height))
                        x2 = str(int(box[2] * width))
                        y2 = str(int(box[3] * height))
                        boxes.append(' '.join([label, x1, y1, x2, y2]))
                    f.write('\n'.join(boxes))

                with open(f"test_dataset/evaluations/detections/{filename}.txt", 'w') as f:
                    boxes = []
                    for box in pred[0]:
                        label = box[5]
                        conf = "{:.8f}".format(box[4])
                        x1 = str(int(box[0] * width))
                        y1 = str(int(box[1] * height))
                        x2 = str(int(box[2] * width))
                        y2 = str(int(box[3] * height))
                        boxes.append(' '.join([label, conf, x1, y1, x2, y2]))
                    f.write('\n'.join(boxes))

                
                # pred_frame = draw_bounding_box(img, pred)
                # annotation_frame = draw_bounding_box(img, annotation)
                # cv2.imshow('annotation', annotation_frame)
                # cv2.imshow('pred', pred_frame)
                # print()
                # k = ''
                # while k != 27:
                #     k = cv2.waitKey(5) & 0xFF
                # cv2.destroyAllWindows()
                        



if __name__ == "__main__":
    # webcam_inference()
    # evaluate()
    test()
