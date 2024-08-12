import cv2
import os
import easyocr

def detect_plates():
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    harcascade = "model/haarcascade_russian_plate_number.xml"

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    cap.set(3, 640)  # width
    cap.set(4, 480)  # height

    min_area = 500
    count = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                img_roi = img[y: y + h, x: x + w]
                cv2.imshow("ROI", img_roi)

                # Only process OCR when 's' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    # Preprocess image
                    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    img_roi_preprocessed = cv2.adaptiveThreshold(img_roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                    # OCR
                    results = reader.readtext(img_roi_preprocessed, detail=0)
                    print("OCR Results:", results)

                    plate_text = ' '.join(results)
                    print(f"Detected Number Plate: {plate_text}")

                    # Save image if desired
                    if not os.path.exists('plates'):
                        os.makedirs('plates')
                    plate_image_path = "plates/scaned_img_" + str(count) + ".jpg"
                    cv2.imwrite(plate_image_path, img_roi)
                    cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                    cv2.imshow("Results", img)
                    cv2.waitKey(500)
                    count += 1

        # Check for ESC key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

        cv2.imshow("Result", img)

    cap.release()
    cv2.destroyAllWindows()
