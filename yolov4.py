import cv2
import numpy as np

# Load model YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# List các tên lớp
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Đọc ảnh từ file
img = cv2.imread("example2.jpg")

# Lấy kích thước ảnh
height, width, _ = img.shape

# Tạo input blob từ ảnh
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Đưa input blob vào mô hình
net.setInput(blob)

# Chạy forward pass và lấy output
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_layers)

# Tính toán các thông tin về bounding box và confidence
boxes = []
confidences = []
class_ids = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Áp dụng non-maximum suppression để loại bỏ các bounding box trùng lặp
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Vẽ bounding box lên ảnh
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = classes[class_ids[i]]
    confidence = confidences[i]
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Hiển thị ảnh kết quả
cv2.imshow("YOLOv4", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
