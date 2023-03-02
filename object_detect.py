import cv2
import numpy as np
import datetime
import sys, os
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from uuid import uuid4

################firebase initialization##############
PROJECT_ID = "coopgo-b58bd"
cred = credentials.Certificate("/home/ksb/webcam/coopgo-b58bd-firebase-adminsdk-q4m0h-913cfe20db.json")
default_app = firebase_admin.initialize_app(cred,{'storageBucket':f"{PROJECT_ID}.appspot.com"})

bucket = storage.bucket()#기본 버킷 사용
print("firebase setup end")
################capture #############################
def fileUpload(file):
    blob = bucket.blob('image_store/'+file) #저장한 사진을 파이어베이스 storage의 image_store라는 이름의 디렉토리에 저장
    #new token and metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token} #access token이 필요하다.
    blob.metadata = metadata
 
    #upload file
    blob.upload_from_filename(filename='/home/pi/webcam/'+file, content_type='image/png') #파일이 저장된 주소와 이미지 형식(jpeg도 됨)
    #debugging hello
    print("hello ")
    print(blob.public_url) 
 
def savePhoto(frame,filename):
    cv2.imwrite(picture_directory+filename, frame)
    print('사진 저장 완료 :' +filename)

def uploadPhoto(now,now_search,file):
    #파베 스토리지 주소: /pictures/ + file
    blob = bucket.blob('pictures/' + file)
    #blob.make_public()
    # new token, metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}  # access token 필요
    blob.metadata = metadata

    blob.upload_from_filename(filename=picture_directory+file)
    print("사진 업로드 완료")
    print(blob.public_url)
    #스토리지 이미지를 public으로 접근 권한 부여하여 db에 메타정보를 연동하여 이를 이용
    #db.child("image_Data").child(now).child("image").set(blob.public_url)
    #db.child("image_Data").child(now).child("title").set(now)
    #db.child("image_Data").child(now).child("description").set(now+'에 찍힌 사진입니다.')
    #db.child("image_Data").child(now).child("search").set(now_search)

threshold_move = 50
diff_compare = 10 
cap = cv2.VideoCapture("http://192.168.123.116:8081/?action=stream")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
picture_directory = "/home/ksb/webcam/" 
ret, img_first = cap.read()
ret, img_second = cap.read()
print("okay")
 
while True:
    ret,img_third = cap.read()
    scr = img_third.copy()
    print("okay1")
	 
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
          str_now = str(now.strftime('%Y-%m-%d %H:%M:%S'))
          str_now_search = str(now.strftime('%Y%m%d%H%M%S'))
          filename = str_now + '.jpg'
          savePhoto(img_third,filename)
          uploadPhoto(str_now,str_now_search,filename)
          #cv2.rectangle(scr,(min(nzero[1]),min(nzero[0])),(max(nzero[1]),max(nzero[0])),(0,255,0),1)
		  #cv2.putText(scr,"Motion Detected", (10,10),cv2.FONT_HERSHEY_DUPLEX, 0.3,(0,255,0))
         
    cv2.imshow('scr',img_third)
 
