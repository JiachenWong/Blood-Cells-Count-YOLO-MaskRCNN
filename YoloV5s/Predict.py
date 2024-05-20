import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import time

def detect(save_img=False):
    Lcoordinate=[]
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        Info = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        print('【'+str(Info)+'】')
                        Lcoordinate.append(Info)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    #print(Lcoordinate)
    return Lcoordinate

from PIL import Image
def get_image_resolution(image_path):
    # 打开图像文件
    with Image.open(image_path) as img:
        # 获取图像分辨率
        width, height = img.size

    return width, height
BloodWeightsppath= '/Users/wangjiachen/Desktop/yolov5-5.0/runs/train/exp/weights/best.pt'

# 文件路径
txt_file_path = '/Users/wangjiachen/PycharmProjects/毕设/YoloV5s/图片路径.txt'
# 读取所有行为列表，每行一个路径
with open(txt_file_path, 'r') as file:
    image_paths = [line.strip() for line in file]

image_path=  image_paths[0]

width, height = get_image_resolution(image_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # --weights: 权重的路径地址
    parser.add_argument('--weights', nargs='+', type=str, default=BloodWeightsppath, help='model.pt path(s)')
    # --source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    parser.add_argument('--source', type=str,
    default=image_path, help='source')  # file/folder, 0 for webcam

    # --img-size:网络输入图片大小
    parser.add_argument('--img-size', type=int, default=max(width, height), help='inference size (pixels)')
    # --conf-thres:置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # --iou - thres: 做nms的iou阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    #--device: 是用GPU还是CPU做推理
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # --view-img:是否展示预测之后的图片/视频，默认False
    parser.add_argument('--view-img', action='store_true', help='display results')
    # --save-txt:是否将预测的框坐标以txt文件形式保存，默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # --classes:设置只保留某一部分类别，形如0或者0 2 3
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # --agnostic - nms: 进行nms是否也去除不同类别之间的框，默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # --augment:推理的时候进行多尺度，翻转等操作(TTA)推理
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    #--update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--update', action='store_true', help='update all models')
    #--project：推理的结果保存在runs / detect目录下
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    #--name：结果保存的文件夹名称
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                A=(detect())
                strip_optimizer(opt.weights)
        else:
            A=(detect())
    print('【detect OK】')

    # 使用 writelines() 方法写入所有元素
    with open('图片坐标.txt', 'w') as file:
        file.writelines(str(A))
    file.close()
    print('【detect OK，坐标信息已经写入！】')


