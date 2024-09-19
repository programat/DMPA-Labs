import cv2
import numpy as np
from enum import Enum


class ColorMode(Enum):
    RED = 1


class WindowMode(Enum):
    ORIGINAL = 1
    FILTERED = 2
    MORPHED = 3
    TRACKED = 4


def task(number):
    def decorator(func):
        func.task_number = number
        return func

    return decorator


class RedObjectTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])

    @task(1)
    def read_and_convert_to_hsv(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    @task(2)
    def apply_color_filter(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.lower_red, self.upper_red)
        return mask

    @task(3)
    def apply_morphological_operations(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing

    @task(4)
    def find_moments(self, processed_frame):
        moments = cv2.moments(processed_frame)
        area = moments['m00']
        return moments, area

    @task(5)
    def track_object(self, original_frame, moments, area):
        if area > 500:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            cv2.rectangle(original_frame, (cx - 50, cy - 50), (cx + 50, cy + 50), (0, 0, 0), 2)
        return original_frame

    def process_frame(self):
        hsv_frame = self.read_and_convert_to_hsv()
        if hsv_frame is None:
            return None

        filtered_frame = self.apply_color_filter(hsv_frame)
        morphed_frame = self.apply_morphological_operations(filtered_frame)
        moments, area = self.find_moments(morphed_frame)
        tracked_frame = self.track_object(cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR), moments, area)

        return tracked_frame


class Menu:
    def __init__(self, tracker):
        self.tracker = tracker
        self.choices = {
            '1': ('Запустить трекинг', self.run_tracking),
            '2': ('Выход', self.exit_program)
        }

    def display_menu(self):
        print("\n" + "=" * 40)
        print(" Меню трекинга красного объекта")
        print("=" * 40)
        for key, value in self.choices.items():
            print(f"{key}. {value[0]}")
        print("=" * 40)

    def run_tracking(self):
        while True:
            frame = self.tracker.process_frame()
            if frame is None:
                break
            cv2.imshow('Tracked Object', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.tracker.cap.release()
        cv2.destroyAllWindows()

    def exit_program(self):
        print("Выход из программы.")
        exit()

    def run(self):
        while True:
            self.display_menu()
            choice = input("Выберите опцию: ")
            action = self.choices.get(choice)
            if action:
                action[1]()
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")


def main():
    tracker = RedObjectTracker()
    menu = Menu(tracker)
    menu.run()


if __name__ == '__main__':
    main()
