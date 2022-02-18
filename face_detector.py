import cv2 as cv
import time
import mediapipe as mp

def face_detector() :
    pTime = 0
    video = cv.VideoCapture(0)
    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection()

    while True:
        success, img = video.read()
        if not success :
            return "Please turn on your camera and refresh the page"
        else:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
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
                                   (int(bboxc.xmin * iw), int(bboxc.ymin * ih-15)),
                                   cv.FONT_ITALIC, 1.5, (255, 255, 255), 2)

            cv.putText(img, f"FPS:{int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
            pTime = cTime
            if cv.waitKey(30) and 0xFF == ord('d'):
                break

            ret, buffer = cv.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield b'--frame\r\n'\
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

    cv.waitKey(0)
