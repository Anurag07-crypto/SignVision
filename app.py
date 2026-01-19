from torchvision.transforms import transforms
import cv2
import torch 
import torchvision.models as models 
from PIL import Image
import torch.nn.functional as F

val_transforms = transforms.Compose(transforms=[
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

pred_model = models.mobilenet_v2(pretrained=True)
num_ftrs = pred_model.classifier[1].in_features
pred_model.classifier[1] = torch.nn.Linear(num_ftrs, 28)

model_path = "Models\\sign_language.pth"
pred_model.load_state_dict(torch.load(model_path, map_location="cpu"))
pred_model.eval()

class_names = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N",
               "Nothing","O","P","Q","R","S","Space","T","U","V","W","X","Y","Z"]

cap = cv2.VideoCapture(0)

# 🔹 NEW: Define ROI (adjust if needed)
x1, y1, x2, y2 = 100, 100, 400, 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 🔹 NEW: Crop ROI
    roi = frame[y1:y2, x1:x2]

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image_tensor = val_transforms(image).unsqueeze(0)

    with torch.no_grad():
        output = pred_model(image_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = class_names[pred.item()]
    conf_score = conf.item() * 100

    # 🔹 NEW: Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(frame,
                f"{label} ({conf_score:.2f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Live AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
