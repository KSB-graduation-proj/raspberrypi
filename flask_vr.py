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
import json
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
bucket = storage.bucket()#기본 버킷 사용
print("firebase setup end")

#capture
threshold_move = 50
diff_compare = 10 

try : 
    cap = cv2.VideoCapture(-1)
except:
    print("No Camera")
    quit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
picture_directory = "/home/ksb/webcam/" 
print("capture setup end")

def savePhoto(frame,filename):
    img_path = picture_directory+filename
    cv2.imwrite(img_path, frame)
    return img_path

def runModel(objects, img_path) : 
    print("Run Model")  
    results = model(img_path)
    detections = results.pandas().xyxy[0]
    for _, detection in detections.iterrows():
        cls_name = detection['name']
        if cls_name not in objects:
            objects[cls_name] = 0
        objects[cls_name]+= 1
    print(objects)    
    if(objects != {}) :
        results.show()
        results.save(labels=True,save_dir= img_path+'res')
        uploadPhoto(results)
    return objects    

def uploadPhoto(file):
    blob = bucket.blob('pictures/' + file)
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}  
    blob.metadata = metadata
    blob.upload_from_filename(filename=picture_directory+file)
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
        objects = runModel(objects, img_path) # model
        if(len(objects) > max_obj_num):
            max_obj_num = objects.size()
            max_objects = objects
    db.collection(u'detection').document(key).set(max_objects)
    print("firestore upload")
    uploadPhoto(filename)        
    print("storage upload")        

                       
#flask
@app.route('/api/qr-detection',methods=['POST'])
def handle_post_request():
    if request.method=='POST':
        parsed = json.loads(request.data)
        print(parsed)
        if(parsed != None):
            capture("2076196") #test
        response_data="Received Post request"
        return response_data,200

if __name__=='__main__':
    app.run(host='192.168.137.114',port=8000)  



