
from tkinter import Tk, Label,Button, Entry,ttk,Frame,Toplevel,Listbox
from PIL import Image, ImageTk
import os
import re
import subprocess
import ast
import  cv2
from collections import Counter

def showImg(path,x,y):
    image = Image.open(path)
    image = image.resize((400,400), Image.LANCZOS)  # 使用高质量的缩放算法
    photo = ImageTk.PhotoImage(image)

    # 创建一个Label用于显示图片
    label = Label(root, image=photo, borderwidth=4, relief="solid",
                  highlightbackground='green', highlightcolor='red', highlightthickness=4)
    label.image = photo  # 保持对photo的引用
    label.place(x=x,y=y)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_file_paths(directory):
    file_paths = []  # 用于存储路径
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # 将文件的绝对路径添加到列表中
    return file_paths

def YOLO(image_path):
    # 将要预测图片的路径保存到【图片路径.txt】
    with open('/Users/wangjiachen/PycharmProjects/毕设/YoloV5s/图片路径.txt', 'w') as file:
        file.write(image_path)  # 写入文件路径并换行

    # 使用Python命令运行脚本
    subprocess.run(['python', '/Users/wangjiachen/PycharmProjects/毕设/YoloV5s/Predict.py'])

    # 得到YOLO输出的图片
    filelist = os.listdir('/Users/wangjiachen/PycharmProjects/毕设/runs/detect')
    filelist.sort(key=natural_sort_key)
    directory = '/Users/wangjiachen/PycharmProjects/毕设/runs/detect/' + filelist[-1]
    print(directory)
    # 获取文件绝对路径
    file_paths = get_file_paths(directory)
    output_image_path= file_paths[0]
    print(output_image_path)

    # 显示细胞个数
    with open('图片坐标.txt', 'r') as file:
        # 读取文件内容
        file_content = file.read()
        # 解析内容为Python列表
        data_list = ast.literal_eval(file_content)
        CellList = []
        for i in data_list:
           CellList.append( i[2].split()[0] )
        counter = Counter(CellList)
        # 将结果转换为字典
        count_dict = dict(counter)
    text = Label(root, bd=2, fg='grey', bg='white', text='数量统计', font=('Times New Roman', 15))
    text.place(x=845, y=20)  # 绝对位置，放置文本
    #                字典组件
    listrs = Listbox(root, height=20, yscrollcommand=True, bg='white',width=15)
    count_dict = sorted(count_dict.items(), key=lambda x: x[1])  # 将两个字典按照字典的值进行排序
    for i in count_dict:  # 第一个小部件插入数据
        s = '{} : {}'.format(i[0], round(i[1], 4))
        listrs.insert(0, s)
    listrs.place(x=850,y=45)  # 将小部件放置到主窗口中
    #                   字典组件
    return output_image_path

def getRBC2():
    with open('图片坐标.txt', 'r') as file:
        # 读取文件内容
        file_content = file.read()
        # 解析内容为Python列表
        data_list = ast.literal_eval(file_content)
        RBC2list = []
        for i in data_list:
            if i[2].split()[0]=='RBC2':
                RBC2list.append( [i[0],i[1]] )
    if RBC2list==[]:
        print('没有重叠的红细胞')
    print(RBC2list)
    return RBC2list

def resize_and_pad_image(image, target_size):
    """
    调整图像大小并填充到固定的分辨率。

    :param image: 输入的PIL图像对象
    :param target_size: 目标分辨率 (宽度, 高度)
    :return: 调整后的PIL图像对象
    """
    # 原始图像尺寸
    original_size = image.size
    # 创建白色背景的图像
    background = Image.new('RGB', target_size, (255, 255, 255))
    # 计算将原图居中粘贴到背景图上的位置
    offset = ((target_size[0] - original_size[0]) // 2, (target_size[1] - original_size[1]) // 2)
    # 将原图粘贴到背景图上
    background.paste(image, offset)
    return background

def crop_one_image(image_path,point1,point2,index):
    """
    从图像中裁剪出以point1和point2为对角线的矩形区域。

    :param image_path: 输入图像文件的路径
    :param output_path: 输出裁剪后的图像文件的路径
    :param point1: 第一个点的坐标 (x1, y1)
    :param point2: 第二个点的坐标 (x2, y2)
    """

    # pList=getRBC2()
    # if len(pList)!=0:
    # point1=pList[0]
    # point2 = pList[1]
    # 打开图像文件
    with Image.open(image_path) as img:
        # 获取矩形区域的边界
        left = min(point1[0], point2[0])
        upper = min(point1[1], point2[1])
        right = max(point1[0], point2[0])
        lower = max(point1[1], point2[1])

        # 裁剪图像
        cropped_img = img.crop((left, upper, right, lower))
        cropped_img2=resize_and_pad_image(cropped_img,(360,360))
        # 保存裁剪后的图像
        cropped_img2.save('mask_img/'+str(index)+'.png')

def crop_all_img(image_path):
    pointList=getRBC2()
    if len(pointList)!=0:
        for index,pp in enumerate(pointList):
            point1=pp[0]
            point2=pp[1]
            crop_one_image(image_path,point1,point2,index)
            print(f'第{index}个crop已经保存')

    else:
        print('没有重叠红细胞！')
    return len(pointList)

def Mask(img_path,index):
    from config import Config
    import model as modellib
    image = cv2.imread(img_path)
    class BalloonConfig(Config):
        """Configuration for training on the toy  dataset.
        Derives from the base Config class and overrides some values.
        """
        # Give the configuration a recognizable name
        NAME = "balloon"
        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # Background + balloon

        # Number of training steps per epoch
        STEPS_PER_EPOCH = 1

        # Skip detections with < 90% confidence
        DETECTION_MIN_CONFIDENCE = 0.
    config = BalloonConfig()
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir='logs')
    model.load_weights(
        '/Users/wangjiachen/Downloads/Mask_RCNN-master/logs/balloon20240515T1618/mask_rcnn_balloon_0096.h5',
        by_name=True)

    result = model.detect([image])
    print(result[0])

    class_names = ['BG', 'RBC']

    from visualize import display_instances

    save_path = 'mask_img2/'+str(index)+'.png'
    print(save_path)
    display_instances(image, result[0]['rois'], result[0]['masks'], result[0]['class_ids'],
                      class_names,
                      scores=None, title="RBCs",
                      figsize=(5, 5), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, save_path=save_path)
    print(f'【{img_path}已完成MASK掩码】')

def AllMask(image_path):
    num=crop_all_img(image_path)  # num是重叠红细胞的个数
    all_path=get_file_paths('mask_img')
    if all_path!=[]:
        for i in range(num):
            path = 'mask_img/'+str(i)+'.png'
            Mask(path,i)
    else:
        print('没有重叠红细胞路径')
    return num

def  AllMask_show(image_path):
    num=AllMask(image_path)
    if num > 0:
        frame=Frame(root, width=num*60, height=60, bd=3,relief='groove',bg='lightgreen')
        frame.place(x=250, y=490)
        # 创建一个按钮
        ax=10
        for i in range(num):
            button = Button(frame, text=str(i), command=lambda i=i:Show_each_maskIMG(i) , width=2, height=2, bg='grey',fg='navy',
                            font=('Times New Roman', 12), relief='ridge')
            button.place(x=ax, y=10, anchor='nw')
            ax +=52
        text = Label(root, bd=2, fg='red', bg='white', text='共有{}个粘连细胞'.format(num), font=('Times New Roman', 15))
        text.place(x=250, y=550)  # 绝对位置，放置文本

    else:
        text = Label(root, bd=2, fg='red', bg='white', text='无粘连红细胞', font=('Times New Roman', 15))
        text.place(x=250,y=500)     # 绝对位置，放置文本

def Show_each_maskIMG(index):
    image = Image.open('mask_img2/'+str(index)+'.png')
    image = image.resize((350,350), Image.LANCZOS)  # 使用高质量的缩放算法
    photo = ImageTk.PhotoImage(image)
    new_window = Toplevel(root)
    new_window.title("MASK实例分割")
    new_window.geometry('400x400')
    new_window.configure(background='green')
    # 创建一个Label用于显示图片
    label = Label(new_window, image=photo, borderwidth=4, relief="solid",
                  highlightbackground='green', highlightcolor='red', highlightthickness=4)
    label.image = photo  # 保持对photo的引用
    label.place(x=10,y=10)


#-*-*-*-*-*-*-*-*-*-*-*-*-*-*GUI界面设置-*-*-*-*-*-*-*-*-*-*-*-*-*-*
root = Tk()
root.title('外周血细胞计数')
root.geometry('1000x555')
#root.resizable(0,0)
root.configure(background='white')
# 输入图片的路径
text = Label(root, bd=2, fg='navy', bg='white', text='请输入血细胞图片绝对路径', font=('Times New Roman', 15))
text.place(x=10, y=8)  # 绝对位置，放置文本
entryFpath = Entry(root, width=45,bg='lavender')
entryFpath.config(highlightbackground="blue", highlightcolor="red")
entryFpath.place(x=10 , y=30 , anchor='nw')
button = Button(root, text='确认', command=lambda:[showImg(entryFpath.get(),x=10,y=70)], width=6, height=2, bg='grey', fg='blue',
                font=('Times New Roman', 15), relief='ridge')
button.place(x=430, y=20, anchor='nw')
#   /Users/wangjiachen/Downloads/BA_56068.jpg

# YOLO 算法的代码运行YOLO(entryFpath.get())
button = Button(root, text='YOLO', command=lambda:showImg(YOLO(entryFpath.get()),x=430,y=70), width=8, height=2, bg='grey', fg='red',
                font=('Times New Roman', 15), relief='ridge')
button.place(x=20, y=500, anchor='nw')


# MaskRCNN
button = Button(root, text='Mask', command=lambda:  AllMask_show(entryFpath.get())   , width=8, height=2, bg='grey', fg='red',
                font=('Times New Roman', 15), relief='ridge')
button.place(x=140, y=500, anchor='nw')

























root.mainloop()



