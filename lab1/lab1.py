import os
import time
import cv2
import numpy as np
from enum import Enum
from functools import wraps


def task(number):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            print(f"\nВыполнение задания {number}")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class ColorMode(Enum):
    NORMAL = 0
    GRAYSCALE = 1
    INVERTED = 2


class WindowMode(Enum):
    NORMAL = 0
    FULLSCREEN = 1
    SMALL = 2


class ImageVideoProcessor:
    def __init__(self, base_filename, image_extensions):
        self.base_filename = base_filename
        self.image_extensions = image_extensions
        self.image_files = self._ensure_image_formats()
        self.color_mode = ColorMode.NORMAL
        self.window_mode = WindowMode.NORMAL
        self.current_image_index = 0

    def _ensure_image_formats(self):
        existing_files = [f"{self.base_filename}.{ext}" for ext in self.image_extensions if
                          os.path.exists(f"{self.base_filename}.{ext}")]
        missing_extensions = [ext for ext in self.image_extensions if not os.path.exists(f"{self.base_filename}.{ext}")]

        if existing_files:
            source_image = cv2.imread(existing_files[0])
            for ext in missing_extensions:
                new_filename = f"{self.base_filename}.{ext}"
                print(f"Создание файла: {new_filename}")
                cv2.imwrite(new_filename, source_image)
        else:
            print(f"Ошибка: Не найдено ни одного изображения с именем {self.base_filename}")

        return [f"{self.base_filename}.{ext}" for ext in self.image_extensions]

    @task(2)
    def display_images(self):
        print("Элементы управления:")
        print("w/стрелка вверх - изменить цветовой режим")
        print("s/стрелка вниз - изменить режим окна")
        print("a/стрелка влево - предыдущее изображение")
        print("d/стрелка вправо - следующее изображение")
        print("q - выход")

        while True:
            image_path = self.image_files[self.current_image_index]
            img = cv2.imread(image_path)
            if img is None:
                print(f"Не удалось прочитать изображение: {image_path}")
                return

            processed_img = self._process_image(img)

            window_name = f'Image - {os.path.basename(image_path)} - Color: {self.color_mode.name} - Window: {self.window_mode.name}'
            self._set_window_properties(window_name)
            cv2.imshow(window_name, processed_img)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w') or key == 82:  # 'w' или стрелка вверх
                self.color_mode = ColorMode((self.color_mode.value + 1) % len(ColorMode))
            elif key == ord('s') or key == 84:  # 's' или стрелка вниз
                self.window_mode = WindowMode((self.window_mode.value + 1) % len(WindowMode))
            elif key == ord('a') or key == 81:  # 'a' или стрелка влево
                self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            elif key == ord('d') or key == 83:  # 'd' или стрелка вправо
                self.current_image_index = (self.current_image_index + 1) % len(self.image_files)

        cv2.destroyAllWindows()

    @task(3)
    def display_video(self, video_path):
        print("Элементы управления:")
        print("r - изменить размер видео")
        print("c - изменить цветовой режим")
        print("q - выход")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Ошибка при открытии видео")
            return

        size_mode = 0  # 0: оригинальный, 1: уменьшенный, 2: увеличенный
        color_mode = 0  # 0: BGR, 1: Grayscale, 2: HSV

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if size_mode == 1:
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            elif size_mode == 2:
                frame = cv2.resize(frame, (int(frame.shape[1] * 1.5), int(frame.shape[0] * 1.5)))

            if color_mode == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif color_mode == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            cv2.imshow('Video', frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                size_mode = (size_mode + 1) % 3
                print(f"Размер: {['Оригинальный', 'Уменьшенный', 'Увеличенный'][size_mode]}")
            elif key == ord('c'):
                color_mode = (color_mode + 1) % 3
                print(f"Цвет: {['BGR', 'Grayscale', 'HSV'][color_mode]}")

        cap.release()
        cv2.destroyAllWindows()

    @task(4)
    def convert_video(self, input_path, output_path=None, output_format='mp4'):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Ошибка при открытии входного видео: {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if output_format == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif output_format == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            print(f"Неподдерживаемый формат: {output_format}")
            return

        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_filename = os.path.basename(input_path)
            output_filename = f"converted_{os.path.splitext(input_filename)[0]}.{output_format}"
            output_path = os.path.join(input_dir, output_filename)
        elif os.path.isdir(output_path):
            input_filename = os.path.basename(input_path)
            output_filename = f"converted_{os.path.splitext(input_filename)[0]}.{output_format}"
            output_path = os.path.join(output_path, output_filename)
        else:
            output_path = f"{os.path.splitext(output_path)[0]}.{output_format}"

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            cv2.imshow('Converting...', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Видео успешно сконвертировано и сохранено: {output_path}")

    @task(5)
    def display_hsv(self, image_path=''):
        if not image_path:
            image_path = self.image_files[self.current_image_index]

        if not os.path.exists(image_path):
            print(f"Файл не найден: {image_path}")
            return

        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка при чтении изображения: {image_path}")
            return

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', img)

        cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)
        cv2.imshow('HSV Image', hsv_img)

        print("Нажмите любую клавишу для завершения...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @task(6)
    def display_red_cross(self, dynamic_color=False):
        print("Элементы управления:")
        print("q - выход")

        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка при чтении кадра с камеры")
                break

            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2

            if dynamic_color:
                center_color = frame[center_y, center_x]
                b, g, r = center_color
                if r >= g and r >= b:
                    cross_color = (0, 0, 255)  # Красный
                elif g >= r and g >= b:
                    cross_color = (0, 255, 0)  # Зеленый
                else:
                    cross_color = (255, 0, 0)  # Синий
            else:
                cross_color = (0, 0, 255)  # Всегда красный для задания 6

            cross = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(cross, (center_x - 15, center_y - 100), (center_x + 15, center_y + 100), cross_color, -1)
            cv2.rectangle(cross, (center_x - 100, center_y - 15), (center_x + 100, center_y + 15), cross_color, -1)

            blurred_cross = cv2.GaussianBlur(cross, (15, 15), 0)
            result = cv2.addWeighted(frame, 1, blurred_cross, 0.7, 0)

            if dynamic_color:
                cv2.putText(result, f"Center RGB: ({r}, {g}, {b})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Cross on Camera Feed', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    @task(7)
    def webcam_to_file_with_cross(self):
        print("Запись видео с крестом...")
        print("q - для досрочного завершения записи")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка при открытии веб-камеры")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_with_cross.mp4', fourcc, fps, (width, height))

        start_time = time.time()
        duration = 10  # Длительность записи в секундах

        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка при чтении кадра")
                break

            cross = np.zeros((height, width, 3), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            cv2.rectangle(cross, (center_x - 15, center_y - 100), (center_x + 15, center_y + 100), (0, 0, 255), -1)
            cv2.rectangle(cross, (center_x - 100, center_y - 15), (center_x + 100, center_y + 15), (0, 0, 255), -1)

            blurred_cross = cv2.GaussianBlur(cross, (15, 15), 0)
            result = cv2.addWeighted(frame, 1, blurred_cross, 0.7, 0)

            cv2.imshow('Recording with Cross...', result)
            out.write(result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("Запись видео с крестом завершена")
        print("Воспроизведение записанного видео...")
        print("q - для завершения воспроизведения")

        cap = cv2.VideoCapture('output_with_cross.mp4')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Recorded Video with Cross', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    @task(8)
    def color_cross_based_on_center(self):
        self.display_red_cross(dynamic_color=True)

    @task(9)
    def display_phone_camera(self):
        print("Элементы управления:")
        print("q - выход")

        cap = cv2.VideoCapture(1)  # Попробуйте 0, 1, 2 и т.д., если не работает
        if not cap.isOpened():
            print("Ошибка при открытии камеры телефона")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры телефона")
                break

            cv2.imshow('Phone Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Вспомогательные методы
    def _process_image(self, img):
        if self.color_mode == ColorMode.NORMAL:
            return img
        elif self.color_mode == ColorMode.GRAYSCALE:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.color_mode == ColorMode.INVERTED:
            return cv2.bitwise_not(img)

    def _set_window_properties(self, window_name):
        if self.window_mode == WindowMode.FULLSCREEN:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif self.window_mode == WindowMode.SMALL:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 400, 300)
        else:  # NORMAL
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

class Menu:
    def __init__(self, processor):
        self.processor = processor
        self.choices = {
            '1': ('Отобразить изображения', self.processor.display_images),
            '2': ('Отобразить видео', lambda: self.processor.display_video("marc-rebillet.mp4")),
            '3': ('Конвертировать видео',
                  lambda: self.processor.convert_video("marc-rebillet.mp4", "./../", output_format='avi')),
            '4': ('Отобразить HSV', self.processor.display_hsv),
            '5': ('Отобразить красный крест', self.processor.display_red_cross),
            '6': ('Запись с веб-камеры с крестом', self.processor.webcam_to_file_with_cross),
            '7': ('Крест на основе центрального пикселя', self.processor.color_cross_based_on_center),
            '8': ('Камера телефона', self.processor.display_phone_camera),
            '0': ('Выход', exit)
        }

    def display_menu(self):
        print("\n" + "=" * 40)
        print(" Меню обработки изображений")
        print("=" * 40)
        for key, value in self.choices.items():
            print(f"{key}. {value[0]}")
        print("=" * 40)

    def run(self):
        while True:
            self.display_menu()
            choice = input("Выберите опцию: ")
            action = self.choices.get(choice)
            if action:
                action[1]()
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")
            input("\nНажмите Enter, чтобы продолжить...")

def main():
    processor = ImageVideoProcessor("zhuk", ['jpeg', 'png', 'bmp'])
    menu = Menu(processor)
    menu.run()

if __name__ == '__main__':
    main()
