import cv2
import numpy as np
import datetime
import sys, os
import requests
import firebase_admin
import torch
import json
import yolov5.utils.torch_utils
import socket

from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
from uuid import uuid4
from flask import Flask,request

app = Flask(__name__)
    
#load model
model_name='/home/ksb/yolov5/coopgo/coopgo_model/weights/best.pt'
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_name, force_reload = True)

#firebase initialization
PROJECT_ID = "coopgo-b58bd"
cred = credentials.Certificate("/home/ksb/coopgo-b58bd-firebase-adminsdk-q4m0h-913cfe20db.json")
default_app = firebase_admin.initialize_app(cred,{'storageBucket':f"{PROJECT_ID}.appspot.com"})
db = firestore.client()
bucket = storage.bucket()
print("firebase setup end")

#capture
threshold_move = 50
diff_compare = 10 
cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
picture_directory = "/home/ksb/webcam/" 

print("capture setup end")

def savePhoto(frame,filename,img_path):
    img_path = picture_directory+filename
    cv2.imwrite(img_path, frame)

def runModel(img_path, objects) : 
    objects = []
    print("Run Model")
    results = model(img_path)
    detections = results.pandas().xyxy[0]

    for _, detection in detections.iterrows():
        cls_name = detection['name']
        if cls_name not in objects:
            objects[cls_name] = 0
        objects[cls_name]+= 1
        
    if objects :                          #if any object is detected
        results.show()
        #db.collection(u'detectionResult').document(u'5znMimhJJau6OYtILKM4').set(results)
        #results.save(labels=True,save_dir= img_path)
        #uploadPhoto(results)
        #db.collection(u'detectionResult').document(str_now).set(objects)
        print("DB upload")

#upload photo on firebase storage        
def uploadPhoto(file):
    blob = bucket.blob('pictures/' + file)
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}  # access token 필요
    blob.metadata = metadata
    blob.upload_from_filename(filename=picture_directory+file)
    print("Photo delete on local space & Uploade Complete")
    print(blob.public_url)

#do capture 
while True: 
    print("listening...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST,PORT))
        s.listen(1)
        conn,addr = s.accept()
        with conn:
            print(f'Connected by {addr}')
            while True:
                data = conn.recv(1024)
                if data:
                    print(data.decode())
                    key = data['id']
                   
                    response_data = "Received POST request"   # Response
                    response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}".format(len(response_data), response_data)
                    conn.sendall(response.encode())
                
                    capture_cnt = 5       # number of times capturing objs
                    while capture_cnt :
                        capture_cnt-= 1 
                       
                        now = datetime.datetime.now()
                        str_now = str(now.strftime('%y%m%d%H%M%S'))
                        filename = str_now + '.jpg'   # name filename
                        
                        ret, img= cap.read()      #read img
                        img_path = ""
                        
                        #save & upload photo
                        savePhoto(img, filename, img_path)
                        uploadPhoto(filename)
                        
                        #find one optimal detection out of 5 times of capturing
                        objects = {}
                        max_obj_num = 0
                        max_objects = {}
                        runModel(img_path, objects)     # run yolov5 coop-go model
                        
                        if(objects.size() > max_obj_num):    #update max_obj_num & max_objects
                            max_obj_num = objects.size()
                            max_objects = objects
                            
                        #send data 
                        conn.sendall(max_objects)
                        #upload collection on firestore
                        db.collection(u'detectionResult').document(key).set(max_objects)
