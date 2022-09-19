import cv2
import numpy as np

def draw_bounding_box(img, boxes):
    img = np.array(img)
    for box in boxes[0]:
        width = img.shape[1]
        height = img.shape[0]
        x = int(box[0] * width)
        y = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        w = x2 - x
        h = y2 - y
        confidence = box[4]
        label = box[5]
        img = cv2.rectangle(img,(x,y,w,h),(0,102,255),2)
        img = cv2.rectangle(img, (x, y - 20), (x + w, y), (0, 102, 255), -1)
        img = cv2.putText(img, "{}: {:.3f}".format(label, confidence), (x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255), 1)
    return img

def preprocessing(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # value = 42 #whatever value you want to add
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    # h, s, v = cv2.split(img)
    # v = cv2.add(v,  value)
    # img = cv2.merge((h, s, v))
    return img

