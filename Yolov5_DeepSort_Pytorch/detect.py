# limit the number of cpus used by high performance libraries
import os
from re import L
from tkinter.font import names
from cv2 import sort
from flask import Flask, render_template, Response, redirect, url_for
import subprocess
import keyboard
from threading import Thread  
from matplotlib.transforms import Bbox
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["KMP_DUPLICATE_LIB_OK"]='1'
id_roi = []
cls_roi = []
import sys
sys.path.insert(0, './yolov5')
from datetime import datetime, timedelta
import mysql.connector
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import prune, select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

app = Flask(__name__)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
count0, count1, count2, count3, count4 = 0, 0, 0, 0, 0
global current_time
current_time = datetime.now()
data = []
roi = []

percent_mask = 0

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            global im0
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        #count
                        c = int(cls)  # integer class
                        count_obj(im0,bboxes,w,h,id, c)
                        

                        label = f'{names[c]} {c} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                #LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                #LOGGER.info('No detections')

            # Stream results
            
            im0 = annotator.result()
            
            if show_vid:
                global count, count0, count1, count2, count3, count4, id_roi, cls_roi
                color=(0,255,0)
                start_point = (w-300, h)
                end_point = (w-300, 0)
                #cv2.line(im0, start_point, end_point, color, thickness=2) #ปรับเส้น
                thickness = 3
                org = (30, 100)
                font = cv2.FONT_HERSHEY_DUPLEX
                fontScale = 2
                cv2.putText(im0, 'Total: ' + str(count), org, font,  # เลขตัวใหญ่
                   fontScale, color, thickness, cv2.LINE_AA)

                cv2.rectangle(im0, (0, 160), (300 , 420), (0,0,0), -1)
                # class 1 -5
                cv2.putText(im0, 'cloth_mask: ' + str(count0), (0, 200), font,  
                   0.8, color, 2, cv2.LINE_AA)
                
                cv2.putText(im0, 'mask_under_chin: ' + str(count1), (0, 250), font,  
                   0.8, color, 2, cv2.LINE_AA)
                
                cv2.putText(im0, 'mask_under_nose: ' + str(count2), (0, 300), font,  
                   0.8, color, 2, cv2.LINE_AA)

                cv2.putText(im0, 'masked: ' + str(count3), (0, 350), font,  
                   0.8, color, 2, cv2.LINE_AA)

                cv2.putText(im0, 'no_mask: ' + str(count4), (0, 400), font,  
                   0.8, color, 2, cv2.LINE_AA)
                
                #cv2.putText(im0, 'Inside ROI: ' + ('id: '+str(id_roi) +'  '+ 'cls: '+str(cls_roi)), (0, 500), 2,  
                #   1, (255,69,0), 2, cv2.LINE_AA)

                #cv2.putText(im0, str(round(percent_mask*100, 2))+' %', (30, 600), 2,  
                #   3, (247, 71, 71), 2, cv2.LINE_AA)
                
                   
                thickness = 2
                color = (255, 0, 0)
                if len(roi) == 4 :
                    cv2.polylines(im0, [np.array(roi, np.int32)], True, (15, 220, 10), 2)
                    #for i in range(len(roi)-1):
                    #    cv2.putText(im0, str(roi[i][0]) + ',' +
					#        str(roi[i][1]), (roi[i][0],roi[i][1]), font,
					#        0.4, (255, 0, 0), 2)
                    #cv2.putText(im0, str(roi[-1][0]) + ',' +
					#        str(roi[-1][1]), (roi[-1][0],roi[-1][1]), font,
					#        0.4, (255, 0, 0), 2)
                        
                            
                
                ret, buffer = cv2.imencode('.jpg', im0)
                global frame
                frame = buffer.tobytes()
                
                
                cv2.imshow(str(p), im0)
                 
                cv2.setMouseCallback(str(p), click_event)
                
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
                    StopIteration
                
                #if keyboard.is_pressed("a"):
                #    print('kub')  
                    
                   
                   

            # Save results (image with detections)
            if save_vid:
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

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
            
            

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

def count_obj(im0,box,w,h,id, c):

    global count,data, id_roi, cls_roi
    global count0, count1, count2, count3, count4
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    if len(roi) >= 4:
        #print('center',center_coordinates)
        #print('width:',w)
        #print('height:',h)
        #print(roi[0][0])
        #print(roi[2][0])
        #print(np.array(roi))
        inside_region = cv2.pointPolygonTest(np.array(roi), center_coordinates, False)
        if inside_region > 0 :

            if id not in data:
                count += 1
                data.append(id)
                print('class:', c)
                if c == 0:
                    count0 += 1
                    
                if c == 1:
                    count1 += 1
                    
                if c == 2:
                    count2 += 1
                    
                if c == 3:
                    count3 += 1
                    
                if c == 4:
                    count4 += 1
                id_roi.append(id) 
                cls_roi.append(c)
                

                    

                # Getting the current date and time
                ts = datetime.now()
                global percent_mask
                def calcuate_percent(c0,c1,c2,c3,c4):
                    total = c0+ c1+ c2+ c3+ c4
                    mask_true = c0 + c3
                    mask_false = c1+ c2 + c4

                    return mask_false/total #((cls.count(0) * 46) + (cls.count(1) * 95) + (cls.count(2) * 95) + (cls.count(3) * 34) + (cls.count(4) * 95))/ (100*len(cls))  คำนวนความเสี่ยงภายใน ROI
                percent_mask = calcuate_percent(count0,count1,count2,count3,count4)
                sent_value(c, ts)

        elif id in data and inside_region == -1:
            idx = id_roi.index(id)
            print('taking cls out', idx)
            del data[data.index(id)]
            del id_roi[idx]
            del cls_roi[idx]
        
        
        #if len(cls_roi) == 0:
        #    percent_mask = 0
            
                           
                
                   
                
def click_event(event, x, y, flags, params):
	global roi  
    
	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:
        
		# displaying the coordinates
		# on the Shell
		
		roi_cor = [x, y]
		roi.append(roi_cor)


    

def sent_value( c, ts):
    #global current_time
    #skip_time = current_time.copy()
    #if current_time > current_time + timedelta(seconds=10): # อัปเดททุก 10 วิ

    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="",
      database="project_database"
    )

    mycursor = mydb.cursor()
    
    sql = "INSERT INTO count_mask (class, time) VALUES (%s, %s)"
    val = (c, str(ts))
    mycursor.execute(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "record inserted.")



@app.route('/track')
def track():
    global frame
    
      
def gen_frames():  # generate frame by frame from camera
    while True:
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    
@app.route('/dashboard')
def dash():
    def get_value():
        mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="project_database"
        )

        mycursor = mydb.cursor()
        sql = "SELECT * from count_mask order by time desc "
        mycursor.execute(sql)
        DBData = mycursor.fetchall()
        time = DBData[0][1]
        count_class = []
        
        for row in DBData:
            count_class.append(row[0])
        c_0 = count_class.count(0) 
        c_1 = count_class.count(1)
        c_2 = count_class.count(2)
        c_3 = count_class.count(3)
        c_4 = count_class.count(4)
        total_c = c_0+c_1+c_2+c_3+c_4
        if total_c == 0:
            risk = 0
        else:
            risk = int(round((c_1 + c_2 +c_4)/ total_c, 2)* 100)
        line_time = []
        for t in range(1,24+1):
            if t < 10:
                sql = f"SELECT * from count_mask where time >= '{str(datetime.now())[0:11]+'0'+str(t)+':00:00.000000'}' and time < '{str(datetime.now())[0:11]+'0'+str(t+1)+':00:00.000000'}'"
            else:
                if t == 24:
                    sql = f"SELECT * from count_mask where time >= '{str(datetime.now())[0:11]+'00:00:00.000000'}' and time < '{str(datetime.now())[0:11]+'01:00:00.000000'}'"  
                else:
                    sql = f"SELECT * from count_mask where time >= '{str(datetime.now())[0:11]+str(t)+':00:00.000000'}' and time < '{str(datetime.now())[0:11]+str(t+1)+':00:00.000000'}'" 
            mycursor.execute(sql)
            DBData = mycursor.fetchall()
            total_line = len(DBData)
            line_time.append(total_line)
            
            
                
        #DBData = count
        return DBData, time, c_0,c_1,c_2,c_3,c_4,total_c,risk,line_time
    
    
    DBData, time, c_0,c_1,c_2,c_3,c_4,total_c,risk,line_time = get_value()
    return render_template('dashboard.html', ScrapedBookData = DBData, last_time = time, count0 = c_0, count1 = c_1, count2 = c_2, count3 = c_3, count4 = c_4, total_class = total_c, p_risk = risk, totalline = line_time)

@app.route('/video_feed')
def video_feed():
    global frame
    
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/')
def tohome():
    def get_value():
        mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="project_database"
        )

        mycursor = mydb.cursor()
        sql = "SELECT * from count_mask order by time desc "
        mycursor.execute(sql)
        DBData = mycursor.fetchall()
        time = DBData[0][1]
        count_class = []
        for row in DBData:
            count_class.append(row[0])
        c_0 = count_class.count(0) 
        c_1 = count_class.count(1)
        c_2 = count_class.count(2)
        c_3 = count_class.count(3)
        c_4 = count_class.count(4)
        total_c = c_0+c_1+c_2+c_3+c_4
        if total_c == 0:
            risk = 0
        else:
            risk = int(round((c_1 + c_2 +c_4)/ total_c, 2)* 100)
        
        #DBData = count
        return DBData, time, c_0,c_1,c_2,c_3,c_4,total_c,risk
    
    global percent_mask
    DBData, time, c_0,c_1,c_2,c_3,c_4,total_c,risk = get_value()
    return render_template('index.html', ScrapedBookData = DBData, last_time = time, count0 = c_0, count1 = c_1, count2 = c_2, count3 = c_3, count4 = c_4, total_class = total_c, p_risk = risk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/Traffic.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    global opt
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    with torch.no_grad():
        subprocess.run(detect(opt)) 


    
        
        
 






