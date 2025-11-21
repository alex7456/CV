import cv2

def process_image(path):
    # Считываем изображение
    img = cv2.imread(path)
   
    
    # Перевод в черно-белый формат
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Вывод чёрно-белого изображения
    cv2.imshow("Черно-белое изображение", gray)
    
    # Применение размытия по Гауссу
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Вывод размытые картинки
    cv2.imshow("Размытие по Гауссу", blurred)

    # Ожидание нажатия клавиши
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image("C:/Users/Alexandr/Desktop/cvision/4/img/2.png")