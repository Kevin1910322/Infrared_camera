'''
AIoT Sample Application
Object Detection Network Model : tinyYolo
input data : video stream file
'''

# Import packages
import os
import colorsys
import random

import cv2
import sys
import time
import numpy as np
import logging
import threading
import signal
import serial
import time
import numpy as np
import binascii
import cv2

if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from multiprocessing import Queue

from flask import Flask, render_template, Response
import tensorflow as tf


scale = 0.0039215

print('=== Start Object Detection ===')


'''
- threads 1: Read Stream using OpenCV
- threads 2: Inference using LNE and View Result using Flask Web Server

1. VideoStream Start
2. Frame Read (640, 480, 3)
3. Pre-Process : resize, crop, BGR2RBG, reshape..., if necessary
4. Inference : use LNE
5. Post-Process : interpret result
6. Show Result : use Flask

output          : 1 * 2535 * 85
boxes           : 1 * 2535 * 4
box_confidences : 1 * 2535 * 1
box_class_probs : 1 * 2535 * 80
'''

def handle_predictions(predictions, confidence=0.6, iou_threshold=0.5):
    '''
    output : 1 * 2535 * 85
    boxes : 1 * 2535 * 4
    box_confidences : 1 * 2535 * 1
    box_class_probs: 1 * 2535 * 80
    '''
    boxes = predictions[:, :, :4]
    #print('14r')
    #print(boxes)
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    #print(box_confidences)
    box_class_probs = predictions[:, :, 5:]
    #print(box_class_probs)

    box_scores = box_confidences * box_class_probs # (1, 2535, 80)
    box_classes = np.argmax(box_scores, axis=-1) # get idex of the maximum box_scores (1, 2535)
    box_class_scores = np.max(box_scores, axis=-1) # get the maximum value of box_scores (1, 2535)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes)
        scores = np.concatenate(n_scores)

        return boxes, classes, scores

    else:
        return None, None, None

#None Max Supression
def nms_boxes(boxes, classes, scores, iou_threshold):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        x = b[:, 0]
        y = b[:, 1]
        w = b[:, 2]
        h = b[:, 3]

        areas = w * h
        order = s.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    return nboxes, nclasses, nscores

def generate_colors(class_names):
    hsv_tuples = [((x+0.0) / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def convert_to_original_size(box, size, original_size):
    
    box = box.reshape(2, 2)
    box[0, :] = letter_box_pos_to_original_pos(box[0, :], size, original_size)
    box[1, :] = letter_box_pos_to_original_pos(box[1, :], size, original_size)

    return list(box.reshape(-1))

def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size):
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos

class TinyYoloNet:

    def __init__(self, model_path):
#        print('init 1')
        self.interpreter = tf.lite.Interpreter(model_path = model_path)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()
        self.output_detail = self.interpreter.get_output_details()
#        print('init 2')
        input_shape = self.input_detail[0]['shape']
        self.width = input_shape[1]
        self.height = input_shape[2]

        #read classes file
        self.class_names = {}
        with open('./labels/obj.names') as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]

        # generate color per class name
        self.colors = generate_colors(self.class_names)

        #print('width {}, height {}'.format(input_shape[1], input_shape[2]))
        # width 416, height 416


    def preprocess_img(self, image, fill_value=128):
        h, w, c = image.shape  # original image size (640*480)

        # opencv read image as BGR. covert it to RGB
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # type cast self.height, self.width from numpy.int32 to int
        height_ratio = np.float32(self.height).item() / h
        width_ratio = np.float32(self.width).item() / w

        fit_ratio = min(width_ratio, height_ratio)

        fit_height = int(h * fit_ratio)
        fit_width = int(w * fit_ratio)

        # Keep original image proportion and resize
        fit_image = cv2.resize(image, (fit_width, fit_height))

        # make new array for 3 channels
        if isinstance(fill_value, int):
            fill_value = np.full(3, fill_value, fit_image.dtype)

        #make padding using new array filled with fill_value: 416 * 416
        int_h = np.uint32(self.height).item()
        int_w = np.uint32(self.width).item()
        to_return = np.tile(fill_value, (int_h, int_w, 1))

        # Place the resized image in the center of the padding image.
        pad_top = int(0.5 * (int_h - fit_height))
        pad_left = int(0.5 * (int_w - fit_width))

        to_return[pad_top:pad_top + fit_height, pad_left:pad_left + fit_width] = fit_image

        to_return = to_return.astype(np.float32)
        to_return = np.expand_dims(to_return, axis=0)
        
        # input image scaling
#        to_return = to_return * scale

        return to_return


    def inference(self, img):
        self.interpreter.set_tensor(self.input_detail[0]['index'], img)
        self.interpreter.invoke()
        #print('12r')
        predictions = [self.interpreter.get_tensor(self.output_detail[i]['index']) for i in range(len(self.output_detail))]
        #print(predictions)
        #print('13r')
        boxes, classes, scores = handle_predictions(predictions[0],
                                                    confidence=0.28,
                                                    iou_threshold=0.49)
        #print(boxes)
        #print(classes)
        #print(scores)
        return boxes, classes, scores


    def post_draw(self, img, boxes, classes, scores):
        h, w = img.shape # original image size (640*480)

        for i, c in reversed(list(enumerate(classes))):
            predicted_class = self.class_names[c]
            box = boxes[i]
            score = scores[i]

            int_h = np.uint32(self.height).item()
            int_w = np.uint32(self.width).item()
            detection_size = (int_h, int_w)

            # covert box for original image size    
            box = convert_to_original_size(box, np.array(detection_size),
                                           np.array((w,h)))

            label = '{} {:.2f}'.format(predicted_class, score)
            left, top, right, bottom = box

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
            right = min(w, np.floor(right + 0.5).astype('int32'))          

            #print(label, (left, top), (right, bottom))

            # colors: RGB, opencv: BGR
            cv2.rectangle(img, (left, top), (right, bottom), tuple(reversed(self.colors[c])), 2)

            font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 1
            font_thickness = 1

            label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
            label_rect_left, label_rect_top = int(left - 3), int(top - 3)
            label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])

            cv2.rectangle(img, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),
                          tuple(reversed(self.colors[c])), -1)

            cv2.putText(img, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness,
                        cv2.LINE_AA)

            #cv2.imshow('d', img)
            #cv2.waitKey(1)

        return cv2.imencode('.jpg', img)[1].tobytes()

uart = serial.Serial("COM8", 115200)

sending_1 = [0x02, 0x00, 0x04, 0x00, 0x01, 0x55, 0xaa, 0x03, 0xFA]
sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x5, 0x03, 0x01]
#sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x0a, 0x03, 0x0E]
sending_3 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x00, 0x01, 0x03, 0x06]
sending_4 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x07]

cnt = 0
cnt1 = 0
cnt2 = 0


frame = np.zeros(4800)

tinyyolo = TinyYoloNet('models/es.tflite')
interpreter = tf.lite.Interpreter(model_path='models/es.tflite')
interpreter.allocate_tensors()    
time.sleep(0.1)
print("second command to fly")
uart.write(sending_2)
time.sleep(0.1)
first = 1
image_cnt = 0
passFlag = np.zeros(6)
start_frame = 0
uart.write(sending_4)
begin = 0
check_cnt = 0

uart.write(sending_1)
while True:
    line = uart.read()
    cnt = cnt + 1
    if cnt >= 9:
        cnt = 0
        break

uart.write(sending_4)

while True:
    try:
        #global fvs # FileVideoStream
        line = uart.read()
        cnt1 = cnt1 + 1
        if begin == 0 and cnt1 == 1:
            rawDataHex = binascii.hexlify(line)
            rawDataDecimal = int(rawDataHex, 16)
            if rawDataDecimal == 2:
                begin = 1
            else:
                begin = 0
                cnt1 = 0
                continue
        if begin == 1 and cnt1 == 20:
            for i in range(0, 9600):
                line = uart.read()
                cnt1 = cnt1 + 1
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if first == 1:
                    dec_10 = rawDataDecimal * 10
                    first = 2
                elif first == 2:
                    first = 1
                    dec = rawDataDecimal
                    frame[image_cnt] = dec_10 + dec
                    image_cnt = image_cnt + 1
                    if image_cnt >= 4800:

                        check_cnt = check_cnt + 1
                        
                        x = [110, 210]
                        y = [0, 255]

                        fpit = np.polyfit(x, y, 1)

                        for i in range(0, 4800):
                            frame[i] = fpit[0] * frame[i] + fpit[1]
                        #print(fpit)

                        image = frame.reshape(60, 80)
                        # print(image)
                        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        image = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite('1.png', image)
                        image = cv2.imread('1.png', cv2.IMREAD_COLOR)

                    # AGC Algorithms
                    

                        cv2.imshow("src", image)
                        cv2.waitKey(1)
                        image_cnt = 0

            #print(cnt1)

        if cnt1 == 2 and begin == 1:
            rawDataHex = binascii.hexlify(line)
            rawDataDecimal = int(rawDataHex, 16)
            if rawDataDecimal == 0x25:
                begin = 1
            else:
                begin = 0
                cnt1 = 0
                continue
        if cnt1 == 3 and begin == 1:
            rawDataHex = binascii.hexlify(line)
            rawDataDecimal = int(rawDataHex, 16)
            if rawDataDecimal == 0xA1:
                begin = 1
            else:
                begin = 0
                cnt1 = 0
                continue

        if cnt1 == 9638 and begin == 1:
            begin = 0
            cnt1 = 0
        else:
            continue

        if check_cnt % 5 != 0:
            continue

        frame1 = cv2.resize(image, (480, 480))
        #network_img = cv2.imencode('.jpg', frame)[1].tobytes()
        input_img = tinyyolo.preprocess_img(frame1)
        
        #cv2.imshow('d', frame1)
        #cv2.waitKey(1)

        input_details = interpreter.get_input_details()
        output_detail = interpreter.get_output_details()
        input_shape = input_details[0]['shape']


        
        input_data = np.array(np.random.randint(0,1000, size=input_shape), dtype=np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_img)

        interpreter.invoke()
        
        predictions = [interpreter.get_tensor(output_detail[i]['index']) for i in range(len(output_detail))]
        
        boxes, classes, scores = handle_predictions(predictions[0],
                                                confidence=0.1,
                                                iou_threshold=0.2)

        print(classes)
        if classes[0] == 1:
            print("Face : {}, {}".format(scores[0], boxes[0]))
        if classes[1] == 1:
            print("Person : {}, {}".format(scores[1], boxes[1]))
        
        
                                                

    except Exception as e:
        logging.error(e)


run_lne()


