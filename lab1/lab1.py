import os
import cv2
import numpy as np
from enum import Enum


class ColorMode(Enum):
    NORMAL = 0
    GRAYSCALE = 1
    INVERTED = 2


class WindowMode(Enum):
    NORMAL = 0
    FULLSCREEN = 1
    SMALL = 2


class ImageProcessor:
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

    def display_images(self):
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

        cv2.destroyAllWindows()


def main():
    base_filename = "zhuk"
    image_extensions = ['jpeg', 'png', 'bmp']
    processor = ImageProcessor(base_filename, image_extensions)
    processor.display_images()


if __name__ == '__main__':
    main()
