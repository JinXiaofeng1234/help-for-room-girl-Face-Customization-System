import cv2
import torchvision.transforms as transforms
def img_process(img_path):
    global count
    global unrecognized_img_ls
    global face_tensor
    face_cascade=cv2.CascadeClassifier(r"C:\Program Files\OpenCV\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml")
    image = cv2.imread(img_path)
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 如果检测到人脸，则裁剪人脸区域并转换为PyTorch矩阵
    if len(faces) > 0:
        # 获取第一个人脸的坐标和尺寸
        (x, y, w, h) = faces[0]
        face_image=image[y:y+h,x:x+w]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 对人脸图像进行预处理
        face_tensor = transform(face_image).unsqueeze(0)
        # print(face_tensor)


    return face_tensor
# face_tensor=img_process("test.jpg")
# print(face_tensor.shape)