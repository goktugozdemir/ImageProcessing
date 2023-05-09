import cv2 as cv
import numpy as np
from datetime import date
import psycopg2
import os
import datetime



def insert(today,time,path,box,obj):
    sql = """INSERT INTO Entry(date_today,time_now, file_path, box_id, object_type)
                    VALUES (%s, %s, %s, %s,%s);"""
    try:
        conn = psycopg2.connect(database="Entry",
                                host="localhost",
                                user="postgres",
                                password="1",
                                port="5432")
        
#Farklı veri tabanları için bilgilerin değişmesi gerekli
        
        cursor = conn.cursor()
        cursor.execute(sql, (today, time,path, box, obj))
        conn.commit()
        #Insert yapılıp kaydedildi
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()





cap=cv.VideoCapture("test.mp4")
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 20)
#Video kaynağı verildi, ayarlar değiştirilebilir.

red=(0, 0, 255)
color = (0, 0, 0)
i=0

start_point1 = (300, 170)
end_point1 = (460, 275)
color1 = (0, 0, 0)
thickness = 2
#Çizilecek olan A kutusunun kordinatları
start_point2 = (100, 130)
end_point2 = (250, 250)
color2 = (0, 0, 0)
thickness = 2
#Çizilecek olan B kutusunun kordinatları

weightsPath = ("yolov3.weights")
configPath = ("yolov3.cfg")
namesPath= ("coco.names")
#Gerekli dosylaraın yüklenimi. Aynı dosya konumundalar.

net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
#Varsa intel gpu kullanımı.

classes = open(namesPath).read().strip().split('\n')
layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]



while True:
    #Frame by frame video kontrol ediliyor.
    isTrue,frame=cap.read()
    height, width, channel = frame.shape

    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer)
    #İşlem yapılabilmesi için blob çıkarılıyor.

    class_ids = []
    conf = []
    boxes = []
    #Bulunan objeler için sınıf isimleri ve kutular çiziliyor.%50 güven bakılıyor.

    cv.rectangle(frame, start_point1, end_point1, color1, thickness)
    cv.rectangle(frame, start_point2, end_point2, color2, thickness)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                conf.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, conf, 0.5, 0.4)
    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            #Arabalar kutulara girerse gerekli işlemler yapılıyor.
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, label, (x, y + 30), font, 3, color, 3)
            if(label=="car"):
                if(300 <= x <= 460 and 170 <= y <= 275):
                    if(color1!=(0,255,0)):
                        color1=(0,255,0)
                        cv.imwrite('car' + str(i) + '.jpg', frame)
                        i += 1
                        today = date.today()
                        dt = datetime.datetime.now()
                        time_str = dt.strftime('%H:%M:%S')
                        t = datetime.datetime.strptime(time_str, '%H:%M:%S').time()
                        name="\\"+'car' + str(i) + '.jpg'
                        path=os.getcwd()+(name)
                        insert(today,t,path,'A','car')
                else:
                    color1 = (0, 0, 0)
                if (100 <= x <= 250 and 130 <= y <= 250):
                    cv.putText(frame, "Object in wrong box", (start_point2[0], end_point2[0] - 200), font, 3,red, 3)
            color = (0, 0, 0)

    cv.imshow('Video',frame)

    if(cv.waitKey(1) & 0xFF == ord('q')):
        break



cap.release()
cv.destroyAllWindows()