# Nameplate Number Detection

This project detects number plates using a Haar Cascade classifier and predicts numbers using an OCR model.

## Project Structure

- `images/`: Folder containing the 5 images.
- `model/`: Folder containing the Haar Cascade XML file.
- `src/`: Folder containing source code.
  - `detect_plates.py`: Script for detecting number plates.
  - `predict.py`: Script for OCR prediction.
  - `preprocess_images.py`: Optional script for preprocessing images.
- `requirements.txt`: List of dependencies.

## Installation

1. Clone the repository.
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
3. Use conda env
  - conda create --name ocr_project python=3.12
  - conda activate ocr_project
  - git initpip install easyocr opencv-python