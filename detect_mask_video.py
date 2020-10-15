import cv2
import imutils
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference

# model = load_pytorch_model('models/face_mask_detection.pth');
model = load_pytorch_model('models/model360.pth');
# anchor configuration
#feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

#prediction threshold
predThreshold = 0.6

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):

    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        pred = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result and pred > predThreshold:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, id2class[class_id], (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
        output_info.append([class_id, pred, xmax-xmin, ymax-ymin])

    if show_result:
        Image.fromarray(image).show()
    return output_info


def show_png(x, y, path, frame):
	img = cv2.imread(path, -1)
	img_height, img_width, _ = img.shape
	y1, y2 = y, y + img.shape[0]
	x1, x2 = x, x + img.shape[1]
	alpha_s = img[:, :, 3] / 255.0
	alpha_l = 1.0 - alpha_s

	for c in range(0, 3):
		frame[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] +
								  alpha_l * frame[y1:y2, x1:x2, c])


def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    status = True
    idx = 0
    iter = 0  # threshold for prediction
    flagTitleOff = 0  # counter for title 'Наденьте маску!', the higher the value, the longer the title lasts (now = 15)
    weightMask = 0
    weightNoMask = 0
    minDistX = 100
    minDistY = 130

    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return

    while status:
        status, img_raw = cap.read()
        frame = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        if (status):
            result = inference(frame,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)

            if(len(result) > 0):
                (class_id, pred, distX, distY) = result[0]
                if class_id == 0:
                    weightMask += pred + 0.5
                else:
                    weightNoMask += pred
                iter += 1
            else:
                weightNoMask = weightMask = 0

            if flagTitleOff > 0:
                flagTitleOff -= 1
                show_png(20, 20, "/Users/kuklavodovich/Desktop/FaceMaskDetection-master/mask_please.png", frame[:, :, ::-1])

            if (iter == 10):
                iter = 0
                if (weightMask >= weightNoMask):
                    print(1)
                else:
                    flagTitleOff = 10
                    show_png(20, 20, "/Users/kuklavodovich/Desktop/FaceMaskDetection-master/mask_please.png", frame[:, :, ::-1])
                weightNoMask = weightMask = 0


            cv2.imshow('image', imutils.resize(frame[:, :, ::-1], width=1400))
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            cv2.waitKey(1)
            idx += 1



if __name__ == "__main__":
    video_path = 0
    run_on_video(video_path, '', conf_thresh=0.5)
