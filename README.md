
### 大四统计学毕业设计：改进的YOLOv5结合实例分割
本文主要通过改进后的YOLOv5模型结合Mask R-CNN模型实现外周血细胞的计数任务，但是由于两部分算法模型是分开的两个python项目操作起来较繁琐，所以编写一个桌面应用将二者整合起来。
开发环境为Python 3.6，编译器Pycharm CE 2021,程序开发工具为tkinter,模型训练使用的数据集使用的是kaggle的血细胞数据集。   

### 创新点：模型中加入颜色注意力机制  

血细胞在染色后呈现的颜色特征不同，而借助颜色信息可以更好的进行识别和分类，比如白细胞的颜色相对红细胞更深，红细胞由于其双凹盘结构中心颜色较浅接近于背景色。  

（1）颜色注意力实现原理：  
加入全卷积网络后，模型在训练过程中会对其认为重要(使损失函数更小)的颜色(像素)自动赋予更合适的权重，基本结构如下图：      
After adding the full convolutional network, the model will automatically assign more appropriate weights to the colors (pixels) it considers important (to make the loss function smaller) during training. The basic structure is as follows:  

<img width="576" alt="截屏2024-10-23 16 12 03" src="https://github.com/user-attachments/assets/d916d167-0e4e-4f33-865d-10e81b74d47d">

（2）加入颜色注意力的优势：  
经过实验对比，加入颜色注意力相比加入常用的**<font color=#0000FF>SENet通道注意力机制</font>**，得到的mAP@0.5差距并不大，但加入颜色注意力机制的YOLOv5模型训练时间更短（如下300轮实验数据）    
After experimental comparison, adding color attention is not much different from adding the commonly used **<font color=#0000FF>SENet channel attention mechanism</font>** in terms of mAP@0.5, but the YOLOv5 model with the color attention mechanism takes less time to train (as shown in the following 300 rounds of experimental data)
<img width="480" alt="截屏2024-10-23 16 08 54" src="https://github.com/user-attachments/assets/38eac34e-8013-46cd-8c79-e787f19be92a">
<img width="728" alt="截屏2024-10-23 16 09 22" src="https://github.com/user-attachments/assets/3a5c5b8c-2747-4ff7-9817-a510c6c2792d">


  
### 应用程序主要内容：
（1）输入：在程序主窗口输入所要处理的血细胞图片路径  


 <img width="268" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/96fae405-0058-491b-9985-9f27f9c7496e">

（2）YOLOv5目标检测：将已经训练好的权重文件和预测代码嵌入程序中，点击按钮【YOLO】完成目标检测操作，并生成带有Bounding boxes的图片，不同类别血细胞数量信息将会显示在窗口右侧。  

<img width="256" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/6e5f7b35-e3b4-44b0-8aec-d39a12ad2397">  

<img width="255" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/a9cf6b45-b7c9-4948-a368-64aa76463a30">

（3）粘连细胞实例分割：将训练好的Mask R-CNN权重以及相关文件嵌入，点击【Mask】按钮，对之前检测到的每一个RBC2区域进行实例分割，并且可以查看每一个区域的实例分割带掩码图片，比如下图4个按钮对应4个重叠的红细胞，点击【0】即可查看第1个重叠红细胞实例分割后的图片。  

<img width="302" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/25898723-ac30-4bc1-988c-9bf86b6e8fb8">    


<img width="1027" alt="截屏2024-05-25 13 34 43" src="https://github.com/JiachenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/6e863a8d-0b12-4b28-9dcc-beef88010acd">


