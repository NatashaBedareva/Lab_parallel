import threading
import queue

import time

import cv2
from ultralytics import YOLO

import argparse


class KeyPontDetector:
    def __init__(self):
        self.frames_queue = queue.Queue()
        self.frames_annotation_queue = queue.Queue()
        self.processed_frames = []
        self.frames_dict = {}
        self.model = YOLO('yolov8s-pose.pt', verbose=False)
        self.stop_event = threading.Event()

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            i += 1
            if not ret:
                break
            self.frames_queue.put((i, frame))
        cap.release()
        self.stop_event.set()


    def annotate_frame(self, thread_num):
        while not self.stop_event.is_set() or not self.frames_queue.empty():
            try:

                id, frame = self.frames_queue.get(timeout=1)
                results = self.model(frame, verbose=False)
                annotated_frame = results[0].plot()

                self.frames_annotation_queue.put((id, frame, annotated_frame))
                print(f"Thread {thread_num} frame {id}")
            except queue.Empty:
                continue

    def run_video(self):
        while not self.frames_annotation_queue.empty() :
            try:
                id, original_frame, annotated_frame = self.frames_annotation_queue.get(timeout=1)

                self.frames_dict[id] = (original_frame, annotated_frame)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

        for id in sorted(self.frames_dict.keys()):
            original_frame, annotated_frame = self.frames_dict[id]
            new_frame = cv2.addWeighted(original_frame, 0.5, annotated_frame, 0.5, 0)
            cv2.imshow('Video', new_frame)
            if cv2.waitKey(24) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


    def save_video(self,output_path, fps, frame_size):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        for id in sorted(self.frames_dict.keys()):
            original_frame, annotated_frame = self.frames_dict[id]
            new_frame = cv2.addWeighted(original_frame, 0.5, annotated_frame, 0.5, 0)
            out.write(new_frame)
        out.release()

def main(video_path,is_parallel,output_path):

    detector = KeyPontDetector()

    if is_parallel:
        thread_count = 6
    else:
        thread_count = 1

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    reader_thread = threading.Thread(target=detector.read_video, args=(video_path,))
    reader_thread.start()

    start_time = time.time()
    threads = [threading.Thread(target=detector.annotate_frame, args=(i,)) for i in range(thread_count)]

    for t in threads:
        t.start()

    reader_thread.join()

    for t in threads:
        t.join()

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Time: {processing_time} seconds")

    detector.run_video()
    detector.save_video(output_path, fps, frame_size)

#python main.py videoShort.mp4 1 out2.mp4

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sensor Data Display")
    parser.add_argument("video_name", type=str)
    parser.add_argument("is_paralel_work", type=int)
    parser.add_argument("output_video_name", type=str)

    args = parser.parse_args()

    main(args.video_name, args.is_paralel_work, args.output_video_name)