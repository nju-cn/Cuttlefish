#### **start server**

​	python server/server.py 

***you need to open this file to change the binding address***



#### start client

​	python A3C/client.py

***you need to open this file to change the server address, and you can choose the testing video in this file.***



#### Train

​	python A3C/main.py

#### Note

you should put the yolov3.weight into  server/weights/, and add the corresponding video files in server/. In addition, you should get the correspong feature map files .npy, and put them in traindata/pedestrian.avi/feature_map/ and traindata/exp.avi/feature_map/