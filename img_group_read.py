import cv2
import torch
import torchvision.transforms as transforms
from os import listdir
from re import search
def custom_sort_key(s):
    math=search(f"female(\d+)",s)
    if math:
        return int(math.group(1))
    else:
        return s
img_folder='img'
file_names=listdir(img_folder)
file_names=sorted(file_names,key=custom_sort_key)
# 加载人脸检测器

def img_proces(img_path):
    global count
    global unrecognized_img_ls
    global face_tensor
    face_cascade=cv2.CascadeClassifier(r"C:\Program Files\OpenCV\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml")
    image = cv2.imread(fr'img\{img_path}')
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

unrecognized_img_ls=[]
count=0
res=[]
for i in file_names:
    img_tensor=img_proces(i)
    if img_tensor is not None:
        res.append(img_tensor)
    else:
        count+=1
        unrecognized_img_ls.append(i)
if len(res)>0:
    res=torch.stack(res,dim=0)
else:
    res=None
print(res)
print(count)
print("以下图片未被识别:"+" ".join(unrecognized_img_ls))
torch.save(res, '../face_tensors.pt')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# face_tensor=.to(device)
