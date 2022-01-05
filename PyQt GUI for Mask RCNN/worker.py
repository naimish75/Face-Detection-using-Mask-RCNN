import os
import glob
import time
import cv2
import visualize
import final
import tensorflow as tf
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from enums import Actions, Requests
from enum import Enum
from Mask_RCNN.mrcnn import model as modellib
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


class_names = ["BG", "Rahul Patel", "Nilesh"]

config = final.CustomConfig()


class InferenceConfig(config.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
#config.display()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "logs\models\mask_rcnn_object_0065.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class Worker(QThread):
    communicator = pyqtSignal(Enum)
    frameChanged = pyqtSignal()

    def __init__(self, parent=None):
        print("Model initialised")
        QThread.__init__(self, parent=parent)

        self.loadedWeights = False
        self.stopped = True
        self.paused = False
        self.detectObjects = False
        self.showMasks = False
        self.showBoxes = False
        self.saveVideo = False
        self.fps = 0

    def setVideo(self, filePath):
        self.filePath = filePath

    def setSave(self, filePath):
        self.savePath = filePath

    def handleRequest(self, request):
        if request is Requests.START:
            self.stopped = False
            self.paused = False

        if request is Requests.STOP:
            self.stopped = True
            self.paused = False

        if request is Requests.PAUSE:
            self.paused = True

        if request is Requests.RESUME:
            self.paused = False

        if request is Requests.DETECT_ON:
            self.detectObjects = True

        if request is Requests.DETECT_OFF:
            self.detectObjects = False

        if request is Requests.MASKS_ON:
            self.showMasks = True

        if request is Requests.MASKS_OFF:
            self.showMasks = False

        if request is Requests.BOXES_ON:
            self.showBoxes = True

        if request is Requests.BOXES_OFF:
            self.showBoxes = False

        if request is Requests.SAVE_ON and self.stopped:
            self.saveVideo = True

        if request is Requests.SAVE_OFF and self.stopped:
            self.saveVideo = False


    def random_colors(N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        return colors

    def apply_mask(image, mask, color, alpha=0.5):
        
        for n, c in enumerate(color):
            image[:, :, n] = np.where(
                mask == 1,
                image[:, :, n] * (1 - alpha) + alpha * c,
                image[:, :, n]
            )
        return image

    def display_instances(image, boxes, masks, ids, names, scores):
        
        n_instances = boxes.shape[0]
        colors = Worker.random_colors(n_instances)
 
        if not n_instances:
            print('NO INSTANCES TO DISPLAY')
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
 
        for i, color in enumerate(colors):
            if not np.any(boxes[i]):
                continue
 
            y1, x1, y2, x2 = boxes[i]
            label = names[ids[i]]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            print(caption)
            mask = masks[:, :, i]
 
            image = Worker.apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
            )
 
        return image

    
    def make_video(outvid, images=None, fps=30, size=None, is_color=True, format="mp4v"):
        
        fourcc = VideoWriter_fourcc(*format)
        vid = None
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            img = imread(image)
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        return vid


    def run(self):

        self.communicator.emit(Actions.LOADING_WEIGHTS)

        with tf.device("/gpu:0"):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        model.load_weights(MODEL_PATH, by_name=True)

        self.loadedWeights = True

        self.communicator.emit(Actions.LOADED_WEIGHTS)

        # Never exiting loop
        while True:
            # Do nothing until we should be playing a video
            if self.stopped:
                continue
            # Load the video file
            self.capture = cv2.VideoCapture(self.filePath)

            # Tell the main thread the video has loaded
            self.communicator.emit(Actions.LOADED_VIDEO)

            while not self.stopped:
                file_name = os.path.basename(os.path.splitext(self.filePath)[0])
                VIDEO_SAVE_DIR = os.path.join(self.savePath + f"/save/{file_name}")
                batch_size = 1
                try:
                    if not os.path.exists(VIDEO_SAVE_DIR):
                        os.makedirs(VIDEO_SAVE_DIR)
                except OSError:
                    print ('Error: Creating directory of data')
                frames = []
                frame_count = 0
 
                while True:
                    if self.stopped:
                        continue

                    ret, frame = self.capture.read()
                    # Bail out when the video file ends
                    if not ret:
                        break

                    if frame is None:
                        self.stopped = True
                        break
        
                    # Save each frame of the video to a list
                    frame_count += 1
                    frames.append(frame)
                    print('frame_count :{0}'.format(frame_count))
                    if len(frames) == batch_size:
                        results = model.detect(frames, verbose=0)
                        print('Predicted')
                        for i, item in enumerate(zip(frames, results)):
                            frame = item[0]
                            r = item[1]
                            frame = Worker.display_instances(
                                frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                            )
                            name = '{0}.jpg'.format(frame_count + i - batch_size)
                            name = os.path.join(VIDEO_SAVE_DIR, name)
                            cv2.imwrite(name, frame)
                            print('writing to file:{0}'.format(name))
                            # Clear the frames array to start the next batch
                        frames = []

                    images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
                    # Sort the images by integer index
                    images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

                    outvid = os.path.join(self.savePath, f"{file_name}.mp4")
                    Worker.make_video(outvid, images, fps=30)
 
                #self.capture.release()
                
                if self.detectObjects:
                    pass

                if self.saveVideo:
                    pass

                while self.paused:
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.saveVideo:
                pass

            self.capture.release()
            cv2.destroyAllWindows()
            self.communicator.emit(Actions.FINISHED)