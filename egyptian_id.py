from fastapi import FastAPI, HTTPException
import pytesseract
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from pytesseract import image_to_string
from ultralytics import YOLO
from PIL import Image
import easyocr
import shutil

# Load the YOLO model and EasyOCR reader once
model = YOLO('best.pt')
reader = easyocr.Reader(['ar'])

# Helper functions
def arabic_to_english_number(arabic_num):
    translation_table = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return arabic_num.translate(translation_table)

def place_of_birth(id):
    id = arabic_to_english_number(id)
    governorates = {
        "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد", "04": "السويس",
        "11": "دمياط", "12": "الدقهلية", "13": "الشرقية", "14": "القليوبية",
        "15": "كفر الشيخ", "16": "الغربية", "17": "المنوفية", "18": "البحيرة",
        "19": "الإسماعيلية", "21": "الجيزة", "22": "بني سويف", "23": "الفيوم",
        "24": "المنيا", "25": "أسيوط", "26": "سوهاج", "27": "قنا", "28": "أسوان",
        "29": "الأقصر", "31": "البحر الأحمر", "32": "الوادي الجديد", "33": "مطروح",
        "34": "شمال سيناء", "35": "جنوب سيناء"
    }
    gov_id = id[7:9]
    return governorates.get(gov_id, "خارج الجمهورية")

def gen(id_number):
    id_number = ''.join(filter(str.isdigit, id_number))
    if len(id_number) < 14:
        return "Unknown"
    return "أنثى" if int(id_number[12]) % 2 == 0 else "ذكر"

def extract_date_of_birth(id):
    id = arabic_to_english_number(id)
    year = '19' + id[1:3] if id[0] == '2' else '20' + id[1:3]
    month = id[3:5].zfill(2)
    day = id[5:7].zfill(2)
    return f"{year}/{month}/{day}"

def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        np_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Decoded image is None")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def cleanup_temp_files(directory="/app/temp_files"):
    shutil.rmtree(directory, ignore_errors=True)

# FastAPI app initialization
app = FastAPI()

# Request model
class IDRequest(BaseModel):
    front_image: str  # Base64-encoded image
    back_image: str   # Base64-encoded image

def process_front_image(frame):
    detected_info = {"name": "", "address": "", "id": []}
    names_with_boxes = []
    addresses_with_boxes = []

    results = model(frame, iou=0.4, conf=0.5)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu()
        cls = results[0].boxes.cls.int().cpu().tolist()

        for box, c in zip(boxes, cls):
            box = box.int().tolist()
            roi = frame[max(0, box[1] - 15):box[3] + 15, max(0, box[0] - 15):box[2] + 15]
            if roi.size == 0:
                continue

            detected_results = reader.readtext(roi, detail=1)
            for result in detected_results:
                text = result[1]  # The detected text
                bbox = result[0]  # The bounding box coordinates for the word

                # Calculate the word's x-coordinate relative to the full image
                word_x = box[0] + int((bbox[0][0] + bbox[2][0]) / 2)  # Use the center of the word's bounding box

                # Store the word and its x-coordinate
                if c == 11:  # Name
                    names_with_boxes.append({
                        "text": text,
                        "x": word_x
                    })
                elif c == 5:  # Additional name class
                    detected_info["name"] += f"{text} "
                elif c in [0, 1]:  # Address
                    addresses_with_boxes.append({
                        "text": text,
                        "x": word_x
                    })

            # Perform OCR using Tesseract for ID detection
            if c == 8:  # ID
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_pil = roi_pil.convert('L')  # Convert to grayscale
                boxes = pytesseract.image_to_boxes(roi_pil, lang='ara_num_test', config='--psm 6')
                detected_digits = []
                for box_line in boxes.splitlines():
                    char, x1, y1, x2, y2, _ = box_line.split(' ')
                    if char.isdigit():  # Only consider numeric characters
                        detected_digits.append(char)
                detected_text = ''.join(detected_digits)
                print("Detected text from Tesseract:", detected_text)  # Debug log
                detected_info["id"].append(detected_text)

    # Sort names by x-coordinate and join
    names_with_boxes.sort(key=lambda item: item["x"], reverse=True)
    sorted_names = [item["text"] for item in names_with_boxes]
    detected_info["name"] = detected_info["name"] + " ".join(sorted_names)

    # Sort addresses by x-coordinate and join
    addresses_with_boxes.sort(key=lambda item: item["x"], reverse=True)
    sorted_addresses = [item["text"] for item in addresses_with_boxes]
    detected_info["address"] = " ".join(sorted_addresses)

    # Return the detected information
    return detected_info

def process_back_image(frame):
    detected_jobs = []
    results = model(frame, iou=0.4, conf=0.5)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu()
        cls = results[0].boxes.cls.int().cpu().tolist()

        for box, c in zip(boxes, cls):
            box = box.int().tolist()
            roi = frame[max(0, box[1] - 15):box[3] + 15, max(0, box[0] - 15):box[2] + 15]
            if roi.size == 0:
                continue

            detected_results = reader.readtext(roi, detail=1)
            for result in detected_results:
                text = result[1]
                word_x = box[0]
                if c in [9, 10]:  # Job
                    detected_jobs.append({"text": text, "x": word_x})

    detected_jobs.sort(key=lambda item: item['x'], reverse=True)
    return " ".join(item['text'] for item in detected_jobs)

@app.post("/process_image")
def process_image(request: IDRequest):
    try:
        front_image = decode_base64_image(request.front_image)
        back_image = decode_base64_image(request.back_image)

        front_info = process_front_image(front_image)
        job_info = process_back_image(back_image)

        id_number = ''.join(front_info["id"])
        english_id = arabic_to_english_number(id_number) if id_number else ""

        response_data = {
            "first_name": front_info["name"].split(" ")[0] if front_info["name"] else "",
            "last_name": " ".join(front_info["name"].split(" ")[1:]) if front_info["name"] else "",
            "address": front_info["address"],
            "id": english_id,
            "birth_date": extract_date_of_birth(english_id) if english_id else "",
            "place_of_birth": place_of_birth(english_id) if english_id else "",
            "gender": gen(english_id) if english_id else "",
            "job": job_info
        }

        return {"isSuccess": True, "message": "ID details processed successfully.", "data": response_data}
    except Exception as e:
        return {"isSuccess": False, "message": str(e), "data": None}
