import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from sort.sort import Sort
import pytesseract
from openpyxl import Workbook

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

car_model = YOLO('yolov8n.pt')
license_plate_model = YOLO('license_plate.pt')
mot_tracker = Sort()

cap = cv2.VideoCapture(0)

save_folder = 'license_plates'  
os.makedirs(save_folder, exist_ok=True)  

excel_folder_path = 'xl'
os.makedirs(excel_folder_path, exist_ok=True)
excel_file_path = os.path.join(excel_folder_path, 'file.xlsx')
wb = Workbook()
ws = wb.active
ws.append(['Text'])

frame_nmr = -1
start_time = None

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

while True: 
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    box_width = frame.shape[1] // 2
    box_height = frame.shape[0]
    box_x1 = frame.shape[1] // 2
    box_y1 = 0
    box_x2 = frame.shape[1]
    box_y2 = frame.shape[0]
    detection_box = [(box_x1, box_y1), (box_x2, box_y2)]

    frame_with_box = frame.copy()  

    car_detections = car_model(frame)[0]
    car_detections_ = []
    for detection in car_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == 2 and detection_box[0][0] < x1 < detection_box[1][0] and detection_box[0][1] < y1 < detection_box[1][1]:
            car_detections_.append([x1, y1, x2, y2, score])

            cv2.rectangle(frame_with_box, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if start_time is None:
                start_time = datetime.now()

    if len(car_detections_) > 0:
        for car_detection in car_detections_:
            x1, y1, x2, y2, _ = car_detection

            car_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            license_plate_detections = license_plate_model(car_crop)[0]
            license_plate_detections_ = []
            for detection in license_plate_detections.boxes.data.tolist():
                x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = detection
                if int(class_id_lp) == 0 and detection_box[0][0] < (x1_lp + x1) < detection_box[1][0] and \
                   detection_box[0][1] < (y1_lp + y1) < detection_box[1][1]:
                    license_plate_detections_.append([x1_lp, y1_lp, x2_lp, y2_lp, score_lp])

                    license_plate_bbox_adjusted = [x1_lp + x1, y1_lp + y1, x2_lp + x1, y2_lp + y1]

                    cv2.rectangle(frame_with_box, (int(license_plate_bbox_adjusted[0]), int(license_plate_bbox_adjusted[1])),
                                  (int(license_plate_bbox_adjusted[2]), int(license_plate_bbox_adjusted[3])), (255, 0, 0),
                                  2)
                    cv2.putText(frame_with_box, 'License Plate',
                                (int(license_plate_bbox_adjusted[0]), int(license_plate_bbox_adjusted[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    if start_time is not None and (datetime.now() - start_time).total_seconds() >= 3:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(save_folder, f"license_plate_{timestamp}.jpg")
                        plate_crop = frame[int(license_plate_bbox_adjusted[1]):int(license_plate_bbox_adjusted[3]), 
                                           int(license_plate_bbox_adjusted[0]):int(license_plate_bbox_adjusted[2])]
                        cv2.imwrite(filename, plate_crop)
                        print(f"License plate saved as {filename}")

                        extracted_text = pytesseract.image_to_string(plate_crop, config='--psm 8')
                        print("Extracted Text:", extracted_text)

                        ws.append([extracted_text.strip()])

                        start_time = None  

    cv2.imshow('Frame', frame_with_box)

    if frame_nmr == 0:
        resized_width = int(frame.shape[1] / 2)
        resized_height = int(frame.shape[0] / 2)
        cv2.resizeWindow('Frame', resized_width, resized_height)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

wb.save(excel_file_path)  
cap.release()
cv2.destroyAllWindows()
