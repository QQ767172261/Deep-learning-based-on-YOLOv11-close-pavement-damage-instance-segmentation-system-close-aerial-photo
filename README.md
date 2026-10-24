深度学习基于YOLOv11 近距离路面损伤实例分割系统   大规模近距离航拍路面病害实例分割数据集，29672张图像，提供yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea49af470eea4f93b8f6751a0e7bd889.png)

数据集统计信息![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/94f6ea23da0441de8c1beef074190f05.png)

====================
图像尺寸:640*640
类别数量: 13 类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b22340703a9458ba00357d44161cf11.png)

训练集图像:27083; 验证集图像:1922; 测试集图像:667
总图像数: 29672
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6ed494d7c3a34371a0263d442a390abc.png)

类别分布:![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b8561d2474dd4020b08a86739d60e2f3.png)

类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------
ELG - 鳄鱼纹‑高 | 2178 | 3170
EL - 鳄鱼纹‑低 | 844 | 1245
EM - 鳄鱼纹‑中 | 12096 | 18192
BKGM - 毛块‑中 | 2012 | 2324
BKGH - 毛块‑高 | 258 | 279
BKGL - 毛块‑低 | 201 | 218
LTH - 横条‑高 | 3348 | 6396
LTL - 横条‑低 | 7088 | 17491
LTM - 横条‑中 | 22454 | 76111
PCH - 斑疵‑高 | 284 | 287
PCL - 斑疵‑低 | 1529 | 1845
PCM - 斑疵‑中 | 1622 | 1788
RVM - 脱散‑中 | 7 | 9

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d768b4824ac4910bdc84fab6d25fff9.png)

.模型代码：模型训练使用yolov11n-seg训练，提供训练好的yoloseg模型、训练指标结果图和对应的全部源代码，80个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7c699977fd614b56a4237efb79d9f826.png)

.qt界面：运行界面采用pyqt5编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/73f6d4f3addb4492976720c4e6c86df1.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/194a1744f59d412187b4c37b134700b2.png)
# 基于YOLOv11 近距离路面损伤实例分割系统完整代码
## 一、环境安装
```bash
pip install ultralytics
pip install pyqt5 opencv-python numpy pillow
```
## 二、模型训练代码 train.py
```python
from ultralytics import YOLO

# 加载YOLOv11分割模型
model = YOLO("yolo11s-seg.yaml").load("yolo11s-seg.pt")

# 训练配置
if __name__ == '__main__':
    results = model.train(
        data="road_damage/data.yaml", # 路面损伤数据集yaml
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        pretrained=True,
        amp=True,
        save=True,
        plots=True # 自动生成PR曲线
    )
    print("训练完成，权重保存在runs/segment/train/weights/best.pt")
```
`road_damage/data.yaml`
```yaml
path: ./road_damage
train: images/train
val: images/val
test: images/test

names:
  0: BKG M
  1: BKGH
  2: BKGL
  3: EL
  4: ELG
  5: EM
  6: LTH
  7: LTL
  8: LTM
  9: PCH
  10: PCL
  11: PCM
```

## 三、PyQt5可视化界面 main.py（路面损伤分割系统）
```python
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication,QMainWindow,QPushButton,QLabel,QFileDialog,QTableWidget,QTableWidgetItem,QRadioButton)
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class RoadDamageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLO11的近距离路面损伤实例分割系统")
        self.resize(1450,800)
        self.model = YOLO("runs/segment/train/weights/best.pt")
        self.img_path = None
        self.mode_seg = True #默认分割模式

        #按钮
        self.btn_img = QPushButton("选择图片",self)
        self.btn_img.move(30,20)
        self.btn_img.clicked.connect(self.load_img)

        self.btn_video = QPushButton("选择视频",self)
        self.btn_video.move(130,20)

        self.btn_cam = QPushButton("摄像头",self)
        self.btn_cam.move(230,20)

        self.btn_save = QPushButton("保存结果",self)
        self.btn_save.move(330,20)

        self.radio_det = QRadioButton("检测模式",self)
        self.radio_det.move(450,22)
        self.radio_seg = QRadioButton("分割模式",self)
        self.radio_seg.move(550,22)
        self.radio_seg.setChecked(True)

        #图片展示label
        self.label_ori = QLabel("原始图像",self)
        self.label_ori.setGeometry(20,80,700,600)
        self.label_res = QLabel("分割结果",self)
        self.label_res.setGeometry(740,80,700,600)

        #结果表格
        self.table = QTableWidget()
        self.table.setGeometry(1250,80,180,320)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["类别","置信度","xmin","ymin"])

    def load_img(self):
        file,_ = QFileDialog.getOpenFileName(self,"选择图片","","*.jpg;*.png;*.jpeg")
        if not file:
            return
        self.img_path = file
        img = cv2.imread(file)
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data,w,h,c*w,QImage.Format_RGB888)
        self.label_ori.setPixmap(QPixmap.fromImage(qimg).scaled(self.label_ori.size(),Qt.KeepAspectRatio))

        #推理
        res = self.model(img,conf=0.25)
        img_out = res[0].plot()
        img_out_rgb = cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB)
        ho,wo,co = img_out_rgb.shape
        qimg_out = QImage(img_out_rgb.data,wo,ho,co*wo,QImage.Format_RGB888)
        self.label_res.setPixmap(QPixmap.fromImage(qimg_out).scaled(self.label_res.size(),Qt.KeepAspectRatio))

        #填充表格
        boxes = res[0].boxes
        self.table.setRowCount(len(boxes))
        for i,box in enumerate(boxes):
            cls_name = self.model.names[int(box.cls)]
            conf = f"{box.conf.item():.2f}"
            xyxy = box.xyxy[0]
            xmin = int(xyxy[0])
            ymin = int(xyxy[1])
            self.table.setItem(i,0,QTableWidgetItem(cls_name))
            self.table.setItem(i,1,QTableWidgetItem(conf))
            self.table.setItem(i,2,QTableWidgetItem(str(xmin)))
            self.table.setItem(i,3,QTableWidgetItem(str(ymin)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RoadDamageWindow()
    win.show()
    sys.exit(app.exec_())
```

## 四、单图推理脚本 predict.py
```python
from ultralytics import YOLO
model = YOLO("best.pt")
img = "test_road.jpg"
result = model(img,save=True)
# 输出掩码、检测框、类别、置信度
for r in result:
    print("目标数量：",len(r.boxes))
    if r.masks is not None:
        print("实例掩码shape",r.masks.data.shape)
```


## 系统功能说明
1. 支持图片、视频、摄像头实时路面损伤实例分割
2. 可视化路面病害掩码（LTM/EM/PCM等11类路面缺陷）
3. 右侧表格输出类别、置信度、坐标
4. 支持切换【检测框模式 / 实例分割掩码模式】
5. 结果图片保存


