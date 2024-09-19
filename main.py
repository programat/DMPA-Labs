import os
from lab1.lab1 import main as lab1_main
from lab2.lab2 import main as lab2_main


def display_menu():
    print("=" * 40)
    print("DMPA-Labs: Меню лабораторных работ")
    print("=" * 40)
    print("1. Лабораторная работа 1: Основы обработки изображений и видео")
    print("2. Лабораторная работа 2: [Краткое описание]")
    print("0. Выход")
    print("=" * 40)


def main():
    while True:
        display_menu()
        choice = input("Выберите лабораторную работу: ")

        if choice == '1':
            lab1_main()
        elif choice == '2':
            lab2_main()
        elif choice == '0':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Пожалуйста, попробуйте снова.")

        input("\nНажмите Enter, чтобы вернуться в главное меню...")


if __name__ == "__main__":
    main()
