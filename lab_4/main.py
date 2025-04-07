import threading
import queue
import time
import argparse
import logging
import os

import cv2


if not os.path.exists('log'):
    os.makedirs('log')
logging.basicConfig(filename='log/app.log', level=logging.ERROR)

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorX(Sensor):
    def __init__(self,delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, camera_name: str, HW: tuple):
        self.camera_name = camera_name
        self.resolution = HW
        if camera_name == '0':
            camera_name = 0
        self.cap = cv2.VideoCapture(camera_name)

        if not self.cap.isOpened():
            logging.error(f"Camera {camera_name} didnt open.")

            raise RuntimeError(f"Camera {camera_name} didnt open.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, HW[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HW[1])

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Camera read failed.")
            raise RuntimeError("Camera read failed.")
        return frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class WindowImage:
    def __init__(self, frequency: float):
        self.frequency = frequency
        self.window_name = "video"
        cv2.namedWindow(self.window_name)

    def show(self, img):
        cv2.imshow(self.window_name, img)
        cv2.waitKey(int(1000 / self.frequency))

    def __del__(self):
        cv2.destroyWindow(self.window_name)

def sensor_thread(sensor: Sensor, data_queue: queue.Queue):
    while True:
        try:
            data = sensor.get()
            data_queue.put(data)
        except Exception as e:
            logging.error(f"Error in sensor thread: {e}")
            break


def main(camera_name, resolution, display_frequency):
    sensor0 = SensorX(0.01)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(1)

    sensor_cam = SensorCam(camera_name, resolution)

    window_image = WindowImage(display_frequency)

    data_queue0 = queue.Queue()
    data_queue1 = queue.Queue()
    data_queue2 = queue.Queue()

    threads = [
        threading.Thread(target=sensor_thread, args=(sensor0, data_queue0)),
        threading.Thread(target=sensor_thread, args=(sensor1, data_queue1)),
        threading.Thread(target=sensor_thread, args=(sensor2, data_queue2)),
    ]

    for i in threads:
        i.start()

    try:
        while True:
            data0 = data_queue0.get() if not data_queue0.empty() else None
            data1 = data_queue1.get() if not data_queue1.empty() else None
            data2 = data_queue2.get() if not data_queue2.empty() else None
            frame = sensor_cam.get()

            img_with_data = frame.copy()
            if data0 is not None:
                cv2.putText(img_with_data, f'Sensor 0: {data0}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if data1 is not None:
                cv2.putText(img_with_data, f'Sensor 1: {data1}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if data2 is not None:
                cv2.putText(img_with_data, f'Sensor 2: {data2}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            window_image.show(img_with_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                for thread in threads:
                    thread.join()

                del sensor_cam
                del window_image
                exit(0)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:

        for thread in threads:
            thread.join()

        del sensor_cam
        del window_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Data Display")
    parser.add_argument("camera_name", type=str)
    parser.add_argument("resolution", type=str)
    parser.add_argument("display_frequency", type=float)

    args = parser.parse_args()

    resolution_tuple = tuple(map(int, args.resolution.split('x')))

    main(args.camera_name, resolution_tuple, args.display_frequency)