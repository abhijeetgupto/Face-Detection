import cv2 as cv
import mediapipe as mp


def face_detector(img):

    video = cv.VideoCapture(0)
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection()

    img = cv.flip(img, 1)
    results = faceDetection.process(img)
    if results.detections:
        for id, detection in enumerate(results.detections):
            if detection.score[0] > 0.50:
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                cv.rectangle(img, bbox, (0, 255, 0), 2)
                cv.putText(img, f"{int(detection.score[0] * 100)}%",
                           (int(bboxc.xmin * iw), int(bboxc.ymin * ih - 15)),
                           cv.FONT_ITALIC, 1.5, (255, 255, 255), 2)

    cv.waitKey(0)
    return img

def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)