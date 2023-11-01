import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from typing import Optional


def find_roi(img: np.array) -> Optional[np.array]:
    
    MIN_AREA = 15000
    
    orig = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    kernel = np.ones((9, 21), np.uint8)

    dilate = cv2.morphologyEx(threshold, cv2.MORPH_DILATE, kernel, iterations=3)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=False)
    
    rect_image = orig.copy()
    roi = None
    max_height = 0

    for c in contours:
        if cv2.contourArea(c) > MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)
            if h > max_height and 0.25 < h / w < 1.5 and 0.3 < w / (img.shape[1] * 0.8) < 0.65:
                max_height = h
                roi = (x, y, w, h)
            cv2.rectangle(rect_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
    
    cv2.drawContours(orig, contours, -1, (255, 0, 0), 2)
    
    dilate = cv2.cvtColor(dilate, cv2.COLOR_BGR2RGB)
    
    combined = cv2.hconcat([dilate, orig])
    combined = cv2.hconcat([combined, rect_image])
    
    cv2.imshow(f"{kernel.shape}", combined)
    
    if not roi:
        return None
    
    x, y, w, h = roi
    roi = img[y : y + h, x : x + w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_threshold = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # cv2 DEMO OF ALGORITHM
    # cv2.imshow(f"ROI", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return roi_threshold
        

def get_roi(images: list[np.array]) -> Optional[list[np.array]]:
            
    roi_list = []
    for i in range(len(images)):
        roi = find_roi(images[i])
        if roi is not None:
            roi_list.append((i, roi))

    return roi_list


def extract_text(img: np.array) -> str:
    return pytesseract.image_to_string(img).strip()

def get_images_from_pdf(path: str) -> list[np.array]:
    images = []
    images.extend(
        list(
            map(
                lambda img: cv2.cvtColor(
                    np.asarray(img), code=cv2.COLOR_RGB2BGR
                ),
                convert_from_path(path)
            )
        )
    )
    return images