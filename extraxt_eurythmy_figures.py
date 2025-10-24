# extraxt_eurythmy_figures.py
#
# Change to see what GIT does
# 15:49 2025-10-22
#
image_name = 'Eurythmie-Figuren.jpg'
import cv2
import os
import numpy as np

# Load the image
image = cv2.imread(image_name)
output_dir = 'extracted_figures_centered'
os.makedirs(output_dir, exist_ok=True)

# Convert to HSV and create mask
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_bound = (0, 50, 50)
upper_bound = (180, 255, 255)
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Clean up mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

index = 1
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 50:
        figure = image[y:y+h, x:x+w]

        # Resize by 20%
        scale = 1.2
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(figure, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create square canvas and center the figure
        canvas_size = max(new_w, new_h)
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # white background
        x_offset = (canvas_size - new_w) // 2
        y_offset = (canvas_size - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        cv2.imwrite(f'{output_dir}/figure_{index}.jpg', canvas)
        index += 1
