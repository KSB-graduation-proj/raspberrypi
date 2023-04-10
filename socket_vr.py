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
bucket = storage.bucket()#기본 버킷 사용
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
    print("Run Model")
    results = model(img_path)
    detections = results.pandas().xyxy[0]

    for _, detection in detections.iterrows():
        cls_name = detection['name']
        if cls_name not in objects:
            objects[cls_name] = 0
        objects[cls_name]+= 1
        
    if(objects != {}) :
        results.show()
        #db.collection(u'detectionResult').document(u'5znMimhJJau6OYtILKM4').set(results)
        #results.save(labels=True,save_dir= img_path)
        #uploadPhoto(results)
        #db.collection(u'detectionResult').document(str_now).set(objects)
        print("DB upload")
        
    #if(results):
        #results.show()
        #db.collection(u'detectionResult').document(u'5znMimhJJau6OYtILKM4').set(results)
        #

def uploadPhoto(file):
    #파베 스토리지 주소: /pictures/ + file
    blob = bucket.blob('pictures/' + file)
    #blob.make_public()
    # new token, metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}  # access token 필요
    blob.metadata = metadata
    blob.upload_from_filename(filename=picture_directory+file)
    print("사진 delete & 업로드 완료")
    print(blob.public_url)

#flask
@app.route('/api/qr-detection',methods=['POST'])
def handle_post_request():
    if request.method=='POST':
        data = request.json
        print(data)
        response_data="Received Post request"
        return response_data,200
if __name__=='__main__':
    app.run(host='192.168.137.114',port=8000)  

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
                    # 응답할 데이터
                    response_data = "Received POST request"
                    # 클라이언트에게 응답
                    response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}".format(len(response_data), response_data)
                    conn.sendall(response.encode())
                    capture_cnt = 5
                    while capture_cnt > 0:
                        now = datetime.datetime.now()
                        str_now = str(now.strftime('%y%m%d%H%M%S'))
                        filename = str_now + '.jpg'
                        ret, img= cap.read()
                        img_path = ""
                        savePhoto(img, filename, img_path)
                        uploadPhoto(filename)        
                        objects = {}
                        max_obj_num = 0
                        max_objects = {}
                        runModel(img_path, objects)
                        if(objects.size() > max_obj_num):
                            max_obj_num = objects.size()
                            max_objects = objects
                        conn.sendall(max_objects)
                        db.collection(u'detectionResult').document(key).set(max_objects)
                        capture_cnt-= 1
                        
 
