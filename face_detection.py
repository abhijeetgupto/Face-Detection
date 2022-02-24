import cv2
import mediapipe as mp


def face_detector(img):
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection()

    img = cv2.flip(img, 1)
    results = faceDetection.process(img)
    if results.detections:
        for id, detection in enumerate(results.detections):
            if detection.score[0] > 0.50:
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, f"{int(detection.score[0] * 100)}%",
                            (int(bboxc.xmin * iw), int(bboxc.ymin * ih - 15)),
                            cv2.FONT_ITALIC, 1.5, (255, 255, 255), 2)
    return img
