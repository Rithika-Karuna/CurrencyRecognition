import cv2
import pytesseract

# Load the image
image = cv2.imread("currency_note.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply OCR
text = pytesseract.image_to_string(gray)

print("Detected Text:", text)
