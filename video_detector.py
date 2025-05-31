import time
from ultralytics import YOLO
import cv2
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt


bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax, class_id, class_name, conf):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.class_id = class_id
        self.class_name = class_name
        self.conf = conf

    def get_points(self):
        return ((self.xmin, self.ymin), (self.xmax, self.ymax))


def draw_bounding_box(frame, bb, current_object=-1,):
    color = bbox_colors[bb.class_id % 10]
    cv2.rectangle(frame, (bb.xmin, bb.ymin), (bb.xmax, bb.ymax), color, 2)

    label = f'{bb.class_name}: {int(bb.conf*100)}%'
    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Get font size
    # Make sure not to draw label too close to top of window
    label_ymin = max(bb.ymin, labelSize[1] + 10)
    # Draw white box to put label text in
    cv2.rectangle(
        frame, (bb.xmin, label_ymin-labelSize[1]-10), (bb.xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
    cv2.putText(frame, label, (bb.xmin, label_ymin-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Draw label text
    if current_object != -1:
        cv2.putText(frame, str(current_object), (int((bb.xmin+bb.xmax)/2), int((bb.ymin+bb.ymax)/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_frame(frame, model, labels,  draw_bounding_boxes=True):
    results = model(frame, verbose=False)
    detections = results[0].boxes
    bounding_boxes = []
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
        # Extract individual coordinates and convert to int
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > 0.5:
            bb = BoundingBox(xmin, ymin, xmax, ymax, classidx, classname, conf)
            object_count += 1
            bounding_boxes.append(bb)
            if draw_bounding_boxes:
                draw_bounding_box(frame, bb)

    cv2.putText(frame, f'Number of objects: {object_count}', (
        10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    return bounding_boxes


def process_continuous_form_input(cap, model):
    resize = True
    resW = 1280
    resH = 720
    ret = cap.set(3, resW)
    ret = cap.set(4, resH)
    labels = model.names
    print(labels)
    avg_frame_rate = 0
    frame_rate_buffer = []
    fps_avg_len = 200

    while True:

        t_start = time.perf_counter()
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

        if resize == True:
            frame = cv2.resize(frame, (resW, resH))

        bbs = process_frame(frame, model, labels, draw_bounding_boxes=False)

        if bbs:
            for i in range(len(bbs)):
                draw_bounding_box(frame, bbs[i])

        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)  # Draw framerate

        cv2.imshow('YOLO detection results', frame)  # Display image

        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
            break
        elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
            cv2.waitKey()

        avg_frame_rate = np.mean(frame_rate_buffer)


def process_cam(model_path='my_model.pt'):
    cap = cv2.VideoCapture(0)  # 0 is video cam usb
    model = YOLO(model_path, task='detect')
    process_continuous_form_input(cap, model)
    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path, model_path='my_model.pt'):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path, task='detect')
    process_continuous_form_input(cap, model)
    cap.release()
    cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()


def get_folder(folder):
    filelist = glob.glob(folder + '/images/*')

    return filelist


def read_image(imgs_list, current_image):
    if current_image >= len(imgs_list):
        print('All images have been processed. Exiting program.')
        sys.exit(0)
    img_filename = imgs_list[current_image]
    frame = cv2.imread(img_filename)
    return frame


def plot_frame(frame):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def process_folder(folder, model_path='my_model.pt', visualize=True):
    images = get_folder(folder)
    model = YOLO(model_path, task='detect')
    for cur in range(len(images)):
        frame = read_image(images, cur)
        bounding_boxes = process_frame(frame, model, model.names)
        if visualize:
            plot_frame(frame)
        yield frame, bounding_boxes


if __name__ == '__main__':
    process_video('test_video.mp4')
