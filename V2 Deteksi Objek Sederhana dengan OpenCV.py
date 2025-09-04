import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rentang warna HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    # Perlebar rentang biru
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Masking untuk masing-masing warna
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Gabungkan semua mask untuk result
    mask_all = cv2.bitwise_or(mask_red, mask_blue)
    mask_all = cv2.bitwise_or(mask_all, mask_green)
    mask_all = cv2.bitwise_or(mask_all, mask_yellow)
    result = cv2.bitwise_and(frame, frame, mask=mask_all)
    
    # Fungsi untuk menggambar bounding box hanya pada persegi
    def draw_square_bounding_box(mask, color_name, box_color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:  # Cek jumlah sisi
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.9 < aspect_ratio < 1.1:  # Rasio mendekati persegi
                        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                        cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    draw_square_bounding_box(mask_red, "Merah", (0,0,255))
    draw_square_bounding_box(mask_blue, "Biru", (255,0,0))
    draw_square_bounding_box(mask_green, "Hijau", (0,255,0))
    draw_square_bounding_box(mask_yellow, "Kuning", (0,255,255))
    
    # Menampilkan hasil
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask_all)
    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()