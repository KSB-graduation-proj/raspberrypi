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
import json, threading
from PIL import Image
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
from uuid import uuid4
from flask import Flask,request

app = Flask(__name__)

#load model
#model =torch.load('/home/ksb/yolov5/coopgo/coopgo_model/weights/best.pt')
model_name = '/home/ksb/yolov5/coopgo/coopgo_model/weights/best.pt'
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_name, force_reload = True)

#firebase initialization
PROJECT_ID = "coopgo-b58bd"
cred = credentials.Certificate("/home/ksb/coopgo-b58bd-firebase-adminsdk-q4m0h-913cfe20db.json")
default_app = firebase_admin.initialize_app(cred,{'storageBucket':f"{PROJECT_ID}.appspot.com"})
db = firestore.client()
bucket = storage.bucket()#기본 버킷 사용
print("firebase setup end")

#capture
threshold_move = 50
diff_compare = 10 

#listening firestore 
pre_document = "hello"
callback_done=threading.Event()
col_query = db.collection(u'qr')

try : 
    cap = cv2.VideoCapture(-1)
except:
    print("No Camera")
    quit()
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
picture_directory = "/home/ksb/webcam/" 

print("camera setup end")

def savePhoto(frame,filename):
    img_path = picture_directory+filename
    cv2.imwrite(img_path, frame)
    return img_path

def runModel(img_path) :
    objects = {} 
    #img = Image.open(img_path)
    print("Run Model")  
    results = model(img_path)
    detections = results.pandas().xyxy[0]
            
    for _, detection in detections.iterrows():
        cls_name = detection['name']
        if cls_name not in objects:
            objects[cls_name] = 0
        objects[cls_name]+= 1
    #print the objects    
    print(objects)    
    if(objects) :
        results.show()
        
    return objects    

def uploadPhoto(filename):
    blob = bucket.blob('pictures/' + filename)
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}  
    blob.metadata = metadata
    blob.upload_from_filename(filename=picture_directory+filename)
    print("사진 delete & 업로드 완료")
    print(blob.public_url)

def capture(key):
    print(key)
    objects = {}
    max_obj_num = 0
    max_objects = {}
    capture_cnt = 5 # take 5 pictures in a row
    
    while capture_cnt > 0: 
        capture_cnt -= 1 
        now = datetime.datetime.now()
        str_now = str(now.strftime('%y%m%d%H%M%S'))
        filename = str_now + '.jpg' 
        ret, img= cap.read() # take a picture
        img_path = savePhoto(img, filename)
        print(img_path)
        print("image saved")
        #count num of objects
        objects = runModel(img_path) # model
        if(len(objects) > max_obj_num):
            max_obj_num = len(objects)
            max_objects = objects
            print(f'max objects : {0}'.format(objects))
            
    db.collection(u'detection').document(key).set(max_objects)
    print("firestore upload")
    uploadPhoto(filename)        
    print("storage upload")   
    
                        
# Create a callback on_snapshot function to capture changes

def on_snapshot(col_snapshot, changes, read_time):
    global pre_document
    print('ready')
    if len(changes) == 1:
        for change in changes:
            if change.type.name == 'ADDED':
                print(f'{change.document.id}')
                cur_document = change.document.id
                if pre_document == cur_document:
                    break
                pre_document = cur_document
                doc_ref = db.collection(u'qr').document(cur_document)
                #capture(doc_ref.id)
                capture(cur_document)
    

# Watch the document
query_watch = col_query.on_snapshot(on_snapshot)
while True:
    callback_done.wait() 
    callback_done.clear()
