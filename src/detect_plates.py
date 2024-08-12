import cv2
import os
import easyocr

def detect_plates(image_path):
    reader = easyocr.Reader(['en'])
    harcascade = "model/haarcascade_russian_plate_number.xml"
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        img_roi = img[y: y + h, x: x + w]
        
        # Preprocess image for OCR
        img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_roi_preprocessed = cv2.adaptiveThreshold(img_roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # OCR
        results = reader.readtext(img_roi_preprocessed, detail=0)
        plate_text = ' '.join(results)
        print(f"Detected Number Plate: {plate_text}")

        # Save image if desired
        if not os.path.exists('plates'):
            os.makedirs('plates')
        plate_image_path = f"plates/scanned_img_{plate_text}.jpg"
        cv2.imwrite(plate_image_path, img_roi)

