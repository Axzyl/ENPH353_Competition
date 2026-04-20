import cv2 as cv
import numpy as np
import tensorflow as tf

class ZhaoOCR():
    def __init__(self):
        self.area_thresh = 12000

        self.interpreter = tf.lite.Interpreter(model_path="/home/fizzer/ros_ws/src/2025_competition/2025_comp_controller/scripts/ENPH353_comp_cnn.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def update_area_thresh(self,area_thresh):
        self.area_thresh = area_thresh
    
    def order_points(pts):
        """
        Return points ordered as:
        top-left, top-right, bottom-right, bottom-left
        """
        pts = np.array(pts, dtype=np.float32)
        # print(f"{pts}")

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    

    def get_homography(self,img):
        orig = img.copy()
        _,img = cv.threshold(img,127,255,0)
        contours,_ = cv.findContours(img,1,2)

        if contours is None or contours == []:
            return None

        

        def white_pixel_count(cnt):

            perim = cv.arcLength(cnt,True)
            epsilon = 0.02*perim
            approxCorners = cv.approxPolyDP(cnt,epsilon,True)

            mask = np.zeros(img.shape[:2],dtype=np.uint8)
            cv.fillPoly(mask,[approxCorners],255)
            masked = cv.bitwise_and(img,img,mask=mask)
            pixel_sum = np.sum(masked)

            return pixel_sum
        


        best_match = min(
            (x for x in contours if cv.contourArea(x) > self.area_thresh),
            key=white_pixel_count
        )

        if best_match is None:
            return None

        perim = cv.arcLength(best_match,True)
        epsilon = 0.02*perim
        approxCorners = cv.approxPolyDP(best_match,epsilon,True)
        approxCorners = approxCorners.reshape(-1,2)

        # if len(approxCorners) != 4:
        #     return None

        box = np.array(approxCorners, dtype=np.float32)

        ordered_box = ZhaoOCR.order_points(box)

        # Compute output size from side lengths
        (tl, tr, br, bl) = ordered_box

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = int(max(width_top, width_bottom))

        height_right = np.linalg.norm(br - tr)
        height_left = np.linalg.norm(bl - tl)
        max_height = int(max(height_right, height_left))

        # Destination rectangle for "forward view"
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Homography / perspective transform
        H = cv.getPerspectiveTransform(ordered_box, dst)
        warped = cv.warpPerspective(orig, H, (max_width, max_height))
        return warped

    def normalize_letter(self,img, target_h=50, target_w=50):
        h, w = img.shape[:2]

        # scale to fit inside box
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

        # compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # pad to final size
        padded = cv.copyMakeBorder(
            resized, top, bottom, left, right,
            cv.BORDER_CONSTANT,
            value=0  # black background
        )

        return padded
    
    def box_area(box):
        x, y, w, h = box
        return w * h


    def is_fully_contained(inner, outer):
        """
        Return True if `inner` is fully inside `outer`.
        Boxes are [x, y, w, h].
        """
        x1, y1, w1, h1 = inner
        x2, y2, w2, h2 = outer

        return (
            x1 >= x2 and
            y1 >= y2 and
            x1 + w1 <= x2 + w2 and
            y1 + h1 <= y2 + h2
        )

    def get_boundBoxes(self,homography):
        if homography is None:
            return None

        homography_area = homography.shape[:2][0]*homography.shape[:2][1]
        letter_estimate = homography_area/120

        _,homography = cv.threshold(homography,127,255,0)
        contours,_ = cv.findContours(homography,1,2)

        bounding_boxes = []

        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            bounding_boxes.append([x,y,w,h])

        # filter by area and aspect ratio
        bounding_boxes = [box for box in bounding_boxes if np.abs(box[2]*box[3]-letter_estimate)/letter_estimate < 0.5]
        aspect_ratios = [box[2]/box[3] for box in bounding_boxes]
        # print(aspect_ratios)
        aspect_median = np.median(aspect_ratios)
        bounding_boxes = [box for box in bounding_boxes if np.abs(box[2]/box[3]-aspect_median)/aspect_median < 0.4]

        # need to filter out boxes that overlap too much
        # we can check to see which boxes have too similar of an x value and then take the larger one (larger height should do)
        bounding_boxes.sort(key=lambda x:x[0])

        # Sort largest area first so larger boxes get considered first
        boxes = sorted(bounding_boxes, key=ZhaoOCR.box_area, reverse=True)

        filtered_boxes = []

        for box in boxes:
            contained = False

            for kept_box in filtered_boxes:
                if ZhaoOCR.is_fully_contained(box, kept_box):
                    contained = True
                    break

            if not contained:
                filtered_boxes.append(box)

        return sorted(filtered_boxes,key=lambda x:x[0])
    
    def draw_Boxes(self, homography):
        img = homography.copy()
        boxes = self.get_boundBoxes(img)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for box in boxes:
            cv.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)
        
        return img



    def filter_img(img):

        lh = 110
        ls = 50
        lv = 50
        uh = 130
        us = 255
        uv = 235

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([lh, ls, lv], dtype=np.uint8)
        upper_hsv = np.array([uh, us, uv], dtype=np.uint8)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)

        # cv.imshow("Thresholded Image", cv.cvtColor(mask, cv.COLOR_GRAY2BGR))
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        kernel = np.ones((2,2),np.uint8)
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        return opening


    def get_text(self,img,vocal=False):
        homography = self.get_homography(img)

        bounding_boxes = self.get_boundBoxes(homography)
        if bounding_boxes is None:
            print("homography failed") if vocal else None
            return None

        mid_height = homography.shape[:2][0]/2

        key_letters = [box for box in bounding_boxes if box[1] < mid_height]
        value_letters = [box for box in bounding_boxes if box[1] >= mid_height]

        # key_letters = sorted(key_letters,key=lambda x:x[0])
        # value_letters = sorted(value_letters,key=lambda x:x[0])

        key_segments = []
        value_segments = []

        for box in key_letters:
            x, y, w, h = box
            roi = homography[y:y+h, x:x+w]
            key_segments.append(roi)

        for box in value_letters:
            x, y, w, h = box
            roi = homography[y:y+h, x:x+w]
            value_segments.append(roi)

        # resize/pad so all images are same size
        key_segments = [self.normalize_letter(letter) for letter in key_segments]
        value_segments = [self.normalize_letter(letter) for letter in value_segments]

        key_batch = np.stack(key_segments).astype(np.float32)
        key_batch = np.expand_dims(key_batch, axis=-1)   # (N, 50, 50) -> (N, 50, 50, 1)

        value_batch = np.stack(value_segments).astype(np.float32)
        value_batch = np.expand_dims(value_batch, axis=-1)   # (N, 50, 50) -> (N, 50, 50, 1)

        text = []

        for batch in [key_batch, value_batch]:
            self.interpreter.resize_tensor_input(
                self.input_details[0]['index'],
                batch.shape
            )
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.interpreter.set_tensor(self.input_details[0]['index'], batch)
            self.interpreter.invoke()
            outputs = self.interpreter.get_tensor(self.output_details[0]['index'])

            pred_indices = np.argmax(outputs, axis=1)
            pred_chars = [self.classes[i] for i in pred_indices]
            text.append("".join(pred_chars))

        return text