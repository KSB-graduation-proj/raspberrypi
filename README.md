### 라즈베리파이 4b

firestore에 qr정보가 업로드되면 사진을 찍어 yolov5모델을 돌린 결과를 firestore detection에 저장.


savePhoto(frame, filename):
save frame as filename and return img_path

runModel(img_path) :
returns list of detected objects in the image

uploadPhoto(filename):
uploade file to firebase

capture(key):
capture images (key for identification)

on_snapshot(col_snapshot, changes, read_time):
create a callback on_snapshot function to capture changes on database
and run capture(database_id)

