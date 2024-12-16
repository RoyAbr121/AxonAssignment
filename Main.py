import cv2
from multiprocessing import Process, Queue
import imutils

def streamer(video_path, output_queue):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output_queue.put(frame)

    cap.release()
    output_queue.put(None)


def detector(input_queue, output_queue):
    first_frame = None

    while True:
        frame = input_queue.get()
        if frame is None:
            output_queue.put(None)
            break

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray
            output_queue.put(frame)  # Pass the original frame forward
            continue

        frame_delta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_queue.put(frame)

def presenter(input_queue):
    while True:
        frame = input_queue.get()
        if frame is None:
            break

        cv2.imshow("Presenter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    streamer_to_detector_queue = Queue(maxsize=10)
    detector_to_presenter_queue = Queue(maxsize=10)

    video_path = 'People - 6387.mp4'

    streamer_process = Process(target=streamer, args=(video_path, streamer_to_detector_queue))
    detector_process = Process(target=detector, args=(streamer_to_detector_queue, detector_to_presenter_queue))
    presenter_process = Process(target=presenter, args=(detector_to_presenter_queue,))


    streamer_process.start()
    detector_process.start()
    presenter_process.start()

    streamer_process.join()
    detector_process.join()
    presenter_process.join()

if __name__ == "__main__":
    main()