
### 改进的YOLOv5结合实例分割
本文主要通过改进后的YOLOv5模型结合Mask R-CNN模型实现外周血细胞的计数任务，但是由于两部分算法模型是分开的两个python项目操作起来较繁琐，所以编写一个桌面应用将二者整合起来。
开发环境为Python 3.6（conda），编译器Pycharm CE 2021,程序开发工具为tkinter。

### 程序主要内容：
（1）输入：在程序主窗口输入所要处理的血细胞图片路径  


 <img width="268" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/96fae405-0058-491b-9985-9f27f9c7496e">

（2）YOLOv5目标检测：将已经训练好的权重文件和预测代码嵌入程序中，点击按钮【YOLO】完成目标检测操作，并生成带有Bounding boxes的图片，不同类别血细胞数量信息将会显示在窗口右侧。  

<img width="256" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/6e5f7b35-e3b4-44b0-8aec-d39a12ad2397">  

<img width="255" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/a9cf6b45-b7c9-4948-a368-64aa76463a30">

（3）粘连细胞实例分割：将训练好的Mask R-CNN权重以及相关文件嵌入，点击【Mask】按钮，对之前检测到的每一个RBC2区域进行实例分割，并且可以查看每一个区域的实例分割带掩码图片，比如下图4个按钮对应4个重叠的红细胞，点击【0】即可查看第1个重叠红细胞实例分割后的图片。  

<img width="302" alt="image" src="https://github.com/JiaChenWong/Blood-Cells-Count-YOLO-MaskRCNN/assets/115158062/25898723-ac30-4bc1-988c-9bf86b6e8fb8">

<font color=#8A2BE2 size=7 face="黑体"> PS:运行请先修改各种路径 </font>

