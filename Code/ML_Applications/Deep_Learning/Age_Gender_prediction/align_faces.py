import cv2
import dlib
import imutils
import numpy as np
from imutils.face_utils import FaceAligner as FA
from imutils.face_utils import rect_to_bb

from config import config, parser

# import face_recognition
# face_recognition.face_locations()

class FaceAligner:

  def __init__(self):
    self.desiredFaceWidth = int(parser['DATA']['aligned_out_size']) # output size
    self.face_threshold   = float(parser['DATA']['face_threshold'])
    self.expand_margin    = float(parser['DATA']['expand_margin'])

    self.Path2ShapePred   = config.model + "shape_predictor_68_face_landmarks.dat"
    # self.Path2Detecor     = config.model + "mmod_human_face_detector.dat"

    self.detector         = dlib.get_frontal_face_detector()
    # self.cnn_detector     = dlib.cnn_face_detection_model_v1(self.Path2Detecor)
    self.predictor        = dlib.shape_predictor(self.Path2ShapePred)
    self.fa               = FA(self.predictor, desiredFaceWidth=self.desiredFaceWidth)

  def getAligns(self,
                img,
                use_cnn = False,
                savepath = None,
                return_info = False):
    """
    get face alignment picture
    :param img: original BGR image or a path to it
    :param use_cnn: using CNN to extract aligned faces, if set, dlib 
                    be compiled with cuda support
    :param savepath: savepath, format "xx/xx/xx.png"
    :param return_info: if set, return face positinos [(x, y, w, h)] 
    :return: aligned faces, (opt) rects
    """
    if type(img) == str:
      img = cv2.imread(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    unsample = 1 if img.shape[0] * img.shape[1] < 512 * 512 else 0

    if use_cnn:
      dets = self.cnn_detector(img, unsample)
      scores = [d.confidence for d in dets if d.confidence > self.face_threshold]
      rects = [d.rect for d in dets if d.confidence > self.face_threshold]
    else:
      rects, scores, _ = self.detector.run(img,
                                           unsample,
                                           self.face_threshold)

    # expand rects by some margin
    exp_rects = []
    for rect in rects:
      x, y, w, h = rect_to_bb(rect)

      # make sure bounds are within the image
      x = max(0, x)
      y = max(0, y)
      w = min(img.shape[1] - x, w)
      h = min(img.shape[0] - y, h)

      exp = min(int(w * self.expand_margin), x, img.shape[1] - x - w,
                int(h * self.expand_margin), y, img.shape[0] - y - h)
      exp = max(0, exp)

      x, y = x - exp, y - exp
      w, h = w + 2 * exp, h + 2 * exp

      exp_rects.append(dlib.rectangle(x, y, x + w, y + h))

    aligned = [self.fa.align(img, gray, rect) for rect in exp_rects]

    if savepath:
      if len(aligned) == 1:
        cv2.imwrite(savepath, aligned)
      else:
        for i, al in enumerate(aligned):
          cv2.imwrite("{}_{}.{}".format(savepath[:-4], i, savepath[-3:]), aligned)

    if return_info:
      return aligned, [rect_to_bb(rect) for rect in exp_rects], scores
    return aligned # BGR faces, cv2.imshow("Aligned", faceAligned)


  def example(self):
    image = cv2.imread("images/example_02.jpg")
    # image = imutils.resize(image, width=self.resizeWidth)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale image
    cv2.imshow("Input", image)
    rects = self.detector(gray, 2)

    # loop over the face detections
    for rect in rects:
      # extract the ROI of the *original* face, then align the face
      # using facial landmarks
      (x, y, w, h) = rect_to_bb(rect)
      faceOrig = imutils.resize(image[y:y + h, x:x + w], width=self.desiredFaceWidth)
      faceAligned = self.fa.align(image, gray, rect)

      import uuid
      f = str(uuid.uuid4())
      cv2.imwrite("foo/" + f + ".png", faceAligned)

      # display the output images
      cv2.imshow("Original", faceOrig)
      cv2.imshow("Aligned", faceAligned)
      cv2.waitKey(0)


if __name__ == "__main__":
  ts = FaceAligner()
  # ts.internet_example()
  pass




