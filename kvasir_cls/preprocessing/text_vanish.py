import cv2
import numpy as np
import tesserocr as tr
from PIL import Image


class Vanisher:

    def __init__(self, impath):
        self.impath = impath

    def do_vanish(self, dark=False):
        """
        detect text in image and remove this applying a mask
        :param dark: default param for different operations based on image brightness
        :return:
        """
        org_img = cv2.imread(self.impath, cv2.IMREAD_UNCHANGED)
        # rename original image
        cv_img = org_img

        # if parameter is not True, try text detection with additional thresholding
        if not dark:
            cv_img = cv2.threshold(org_img, 209, 255, cv2.THRESH_BINARY)[1]

        # convert image to single channel
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # convert opencv image to pil for processing with tesserocr
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))

        # initialize tesserocr api
        api = tr.PyTessBaseAPI()

        try:
            # set pil image for ocr class
            api.SetImage(pil_img)
            # get bounding boxes of text
            boxes = api.GetComponentImages(tr.RIL.TEXTLINE, True)
            # make copy of image for processing
            rec_img = cv_img.copy()
            # try text detection with tesserocr api
            text = api.GetUTF8Text()

            # try again with other param if no text found
            if not text.strip() and dark==False:
                self.do_vanish(dark=True)

            # iterate over returned list, draw rectangles
            for (im, box, _, _) in boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(rec_img, (x, y), (x + w, y + h), color=(0, 0, 255))

            # convert bounding boxes to coordinates
            coordinates = self.boxes_to_coordinates(boxes)
            # create mask of detected characters in image
            mask = self.create_mask(gray, coordinates)

            # convert to 8bit for inpainting function
            gray_8 = (mask).astype('uint8')
            # paint over the mask null values
            dst_TELEA = cv2.inpaint(org_img, gray_8, 3, cv2.INPAINT_TELEA)

        finally:
            api.End()
            return dst_TELEA

    @staticmethod
    def boxes_to_coordinates(boxes):
        co_dict = dict()
        bounds = [bound for im, bound, _, _ in boxes]
        # iterate bounds of all boxes
        for bnd in bounds:
            # for every y within the bounds
            for y in range(bnd['y'], bnd['y']+bnd['h'] + 1):
                # array of every x within the bounds
                xs = [x for x in (range(bnd['x'], bnd['x']+bnd['w']+1))]
                # create dict with y coordinate as key, array of x coordinates as value
                if y in co_dict:
                    co_dict[y].extend(xs)
                else:
                    co_dict[y] = xs
        # return coordinates dict
        return co_dict

    @staticmethod
    def create_mask(bnw, coordinates):
        # get dimensions of image
        x_axis = bnw.shape[1]
        # turn numpy array into 2D list
        bnw = list(bnw)
        # iterate array of pixels
        for e, row in enumerate(bnw):
            # if y coordinate within character bounds, keep the pixel info
            if e in coordinates:
                xs = coordinates.pop(e)
                for e_x, x in enumerate(row):
                    if e_x not in xs:
                        bnw[e][e_x] = 0
            # else discard info (turn all y coordinates into 0)
            else:
                bnw[e] = [0 for _ in range(x_axis)]
        # return mask a numpy array
        return np.array(bnw)
