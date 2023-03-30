import cv2
import numpy as np
import datetime
import sys, os
import requests
import firebase_admin
import torch
import json
import yolov5.utils.torch_utils

from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
from uuid import uuid4

################load model###########################
#model = torch.hub.load('.','custom',path='/home/ksb/yolov5/coopgo/coopgo_model/weights/best.pt', source='local')
model_name='/home/ksb/yolov5/coopgo/coopgo_model/weights/best.pt'
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_name, force_reload = True)

################firebase initialization##############
PROJECT_ID = "coopgo-b58bd"
cred = credentials.Certificate("/home/ksb/coopgo-b58bd-firebase-adminsdk-q4m0h-913cfe20db.json")
default_app = firebase_admin.initialize_app(cred,{'storageBucket':f"{PROJECT_ID}.appspot.com"})
db = firestore.client()
bucket = storage.bucket()#기본 버킷 사용
print("firebase setup end")

################capture #############################

def savePhoto(frame,str_now,filename):
    img_path = picture_directory+filename
    cv2.imwrite(img_path, frame)
    #print('사진 저장 완료 :' +filename)
    print("Run Model")
    results = model(img_path)
    detections = results.pandas().xyxy[0]
    #results = yolov5.utils.torch_utils.postprocess(results, input_size, 0.4, 0.5)
    objects = {}
    for _, detection in detections.iterrows():
        cls_name = detection['name']
        if cls_name not in objects:
            objects[cls_name] = 0
        objects[cls_name]+= 1
        
    #with open('/home/ksb/results.json', 'w') as f:
    #json.dump(objects, f) 
    #label 추출
    if(objects != {}) :
        db.collection(u'detectionResult').document(str_now).set(objects)
        print("DB upload")
    #if(results):
        #results.show()
        #f = open("/home/ksb/objects.txt","w")
        #print(results)
        #f.write(results)
        #db.collection(u'detectionResult').document(u'5znMimhJJau6OYtILKM4').set(results)
        #results.save(labels=True,save_dir= picture_directory + 'exp')

def uploadPhoto(now,now_search,file):
    #파베 스토리지 주소: /pictures/ + file
    blob = bucket.blob('pictures/' + file)
    #blob.make_public()
    # new token, metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}  # access token 필요
    blob.metadata = metadata
    blob.upload_from_filename(filename=picture_directory+file)
    os.remove(picture_directory+file) #remove the image
    print("사진 delete & 업로드 완료")
    print(blob.public_url)

threshold_move = 50
diff_compare = 10 
cap = cv2.VideoCapture("http://192.168.0.232:8081/?action=stream")
#cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
picture_directory = "/home/ksb/webcam/" 
ret, img_first = cap.read()
ret, img_second = cap.read()
print("okay")
 
while True:
    ret,img_third = cap.read()
    scr = img_third.copy()

    img_first_gray = cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
    img_second_gray = cv2.cvtColor(img_second, cv2.COLOR_BGR2GRAY)
    img_third_gray = cv2.cvtColor(img_third, cv2.COLOR_BGR2GRAY)
	 
    diff_1 = cv2.absdiff(img_first_gray, img_second_gray)
    diff_2 = cv2.absdiff(img_second_gray, img_third_gray)
	 
    ret,diff_1_thres=cv2.threshold(diff_1,threshold_move,255,cv2.THRESH_BINARY);
    ret,diff_2_thres=cv2.threshold(diff_2,threshold_move,255,cv2.THRESH_BINARY);
	 
    diff = cv2.bitwise_and(diff_1_thres,diff_2_thres)
    diff_cnt = cv2.countNonZero(diff)
	 
    if diff_cnt > diff_compare:
          nzero = np.nonzero(diff)
          print("Object Detected")
          now = datetime.datetime.now()
          str_now = str(now.strftime('%y%m%d%H%M%S'))
          str_now_search = str(now.strftime('%Y%m%d%H%M%S'))
          filename = str_now + '.jpg'
          savePhoto(img_third,str_now,filename)
          uploadPhoto(str_now,str_now_search,filename)
          #cv2.rectangle(scr,(min(nzero[1]),min(nzero[0])),(max(nzero[1]),max(nzero[0])),(0,255,0),1)
		  #cv2.putText(scr,"Motion Detected", (10,10),cv2.FONT_HERSHEY_DUPLEX, 0.3,(0,255,0))
         
    cv2.imshow('scr',img_third)
 
