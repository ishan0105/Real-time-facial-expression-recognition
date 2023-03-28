import cvlib as cv
import cv2
import io
import boto3
from PIL import Image, ImageDraw
import csv
import time
import os
import csv
import datetime
import requests
import json
import matplotlib.pyplot as plt
import numpy as np



def render_emotion(emo_response):
    json_str = json.dumps(emo_response)

    resp = json.loads(json_str)
    print(resp)

    h_bars = ('SURPRISED', 'SAD', 'HAPPY', 'FEAR', 'DISGUSTED', 'CONFUSED', 'CALM', 'ANGRY')

    # resp['FaceDetails'][0]['Emotions']
    emo = resp['FaceDetails'][0]['Emotions']
    print(emo)

    height = []

    SURPRISED = next(d for d in emo if d['Type'] == 'SURPRISED')
    height.append(int(SURPRISED['Confidence']))

    SAD = next(d for d in emo if d['Type'] == 'SAD')
    height.append(int(SAD['Confidence']))

    HAPPY = next(d for d in emo if d['Type'] == 'HAPPY')
    height.append(int(HAPPY['Confidence']))

    FEAR = next(d for d in emo if d['Type'] == 'FEAR')
    height.append(int(FEAR['Confidence']))

    DISGUSTED = next(d for d in emo if d['Type'] == 'DISGUSTED')
    height.append(int(DISGUSTED['Confidence']))

    CONFUSED = next(d for d in emo if d['Type'] == 'CONFUSED')
    height.append(int(CONFUSED['Confidence']))

    CALM = next(d for d in emo if d['Type'] == 'CALM')
    height.append(int(CALM['Confidence']))

    ANGRY = next(d for d in emo if d['Type'] == 'ANGRY')
    height.append(int(ANGRY['Confidence']))

    # try:
    #     UNKNOWN = next(d for d in emo if d['Type'] == 'UNKNOWN')
    #     height.append(int(UNKNOWN['Confidence']))
    # except Exception as err:
    #     UNKNOWN = 0
    #     print("Error")
    #     print(str(err))

    # print(int(SURPRISED['Confidence']))

    # plt.figure(figsize=(10, 5))
    # bars = ('A', 'B', 'C', 'D', 'E')
    y_pos = np.arange(len(h_bars))

    # Create bars
    plt.bar(y_pos, height)

    # Create names on the x-axis
    plt.xticks(y_pos, h_bars)

    # Show graphic
    plt.pause(0.05)
    plt.clf()
    # plt.show(block=False)
    # plt.close()

    return

def search_images(source_image):
    ## Read image in binary form
    # bucket = 'salil-bucket'
    collectionId = 'DemoJam'
    # collectionId = 'EmpList'
    fileName = source_image
    threshold = 90
    maxFaces = 4
    with open('credentials.csv', 'r') as input:
        next(input)
        reader = csv.reader(input)
        for line in reader:
            access_key_id = line[2]
            secret_access_key = line[3]

    ## Call AWS Rekognition Service
    client = boto3.client('rekognition',
                          region_name='us-west-2',
                          aws_access_key_id=access_key_id,
                          aws_secret_access_key=secret_access_key)
    with open(fileName, 'rb') as source_image1:
        source_bytes = source_image1.read()

    # response = client.search_faces_by_image(
    #     SourceImage={
    #         'Bytes': source_bytes
    #     }
    # )
    try:
        response = client.detect_faces(Image={'Bytes': source_bytes},
                                       Attributes=['ALL'],
                                       )

        print(response)
        return response
    # response['FaceDetails'][0]['Emotions'][0]['Type']
    except Exception as err:
        print("Error")
        print(str(err))


webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print(webcam)
    print("Could not open webcam")
    exit()
frame_skip = 5
plt.figure(figsize=(10, 5))

# loop through frames
while webcam.isOpened():
    try:
        # read frame from webcam
        webcam.set(cv2.CAP_PROP_POS_FRAMES, frame_skip)
        frame_skip = frame_skip + 35
        status, frame = webcam.read()
        frame = cv2.resize(frame, (0, 0), fx=.5, fy=.5)
        if not status:
            print("Could not read frame")
            exit()

        crop_img = frame
        cv2.imwrite("fire.jpg", crop_img)
        source_image = "fire.jpg"

        # Call AWS Search Image API
        response = search_images(source_image)
        render_emotion(response)

        cv2.imshow("Real-time face detection", frame)

        # press "q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as err:
        print("Error1")
        print(str(err))
        cv2.imshow("Real-time face detection", frame)

        continue

    plt.show(block=False)