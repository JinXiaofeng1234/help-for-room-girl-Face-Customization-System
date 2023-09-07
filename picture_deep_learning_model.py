import pickle

from img_read import img_process
from torch import save
from torch import load
from torch import from_numpy
from torch.utils.data import DataLoader,TensorDataset
from torch import nn
from torch.optim import SGD
from torch import no_grad
from sys import exit
#读取女角色各项数值的csv文件和女角色头像的tensor
try:
    with open("female_data.pkl","rb") as f:
        female_features_data=pickle.load(f)
        female_picture_tensor = load("face_tensors.pt")
except Exception as e:
    print(e)
#把训练集转换成dataset类型
train_data=TensorDataset(female_picture_tensor)
#将从csv文件读取进来的adarry文件转换成tensor,然后转换成dataset
test_data=TensorDataset(from_numpy(female_features_data))

train_data_loader=DataLoader(train_data,batch_size=64)
test_data_loader=DataLoader(test_data,batch_size=64)




#搭建神经网络
class neural_network(nn.Module):
    def __init__(self):
        super(neural_network, self).__init__()
        #卷积层
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        #全连接层
        self.fc1=nn.Linear(32*7*7,128)
        self.relu3=nn.ReLU()
        self.fc2=nn.Linear(128,59)

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)

        x=x.view(-1,32*7*7)

        x=self.fc1(x)
        x=self.relu3(x)
        x=self.fc2(x)

        return x





train_x=[]
for data_loader in train_data_loader:
    for group_tensor in data_loader:
        for tensor in group_tensor:
            train_x.append(tensor)
# print(train_x[0])
#无名者的悲伤
# print(train_x[0].shape)
# train_y=[]
# for i in range(31):
#     tensor=zeros(64,59)
#     train_y.append(tensor)



train_y=[]
for data_loader in test_data_loader:
    for group_tensor in data_loader:
        for tensor in group_tensor:
            train_y.append(tensor)

def model_train():

    model = neural_network()
    model = model.cuda()
    # 损失函数设置
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.cuda()
    # 优化器

    learning_rate = 1e-3
    optimizer = SGD(model.parameters(), lr=learning_rate)
    print("-0 继续训练模型/ -1 训练新模型:\n")
    ans=input("")

    ModelName=''

    if ans=='0':
        ModelName=input("请输入字典模型名字:")
        model.load_state_dict(load(f"{ModelName}_dict.pth"))
    elif ans=='1':
        ModelName = input("请输入新模型名字:") or "new_untitled_model"
    elif ans!='0' or ans !='1':
        print("请勿输入其它内容")
        return
    round_num=1
    iteration_count=0
    # num_epochs=40#*31才是总的训练次数

    avg_loss=1.0
    save_interval = 10000
    while not(0.01<avg_loss<0.02):
        print(f"-----------------------------第{round_num}轮训练开始-----------------------------")
        round_num+=1
        model.train()

        # total_loss = 0
        tmp_ls=[]
        for x,y in zip(train_x,train_y):
                x=x.cuda()
                y=y.cuda()
                outputs = model(x)
                y=y.unsqueeze(0).expand_as(outputs)
                y=y.float()
                loss=loss_fn(outputs,y)
                tmp_ls.append(loss)
                #设置优化器
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration_count += 1
        avg_loss=(sum(tmp_ls))/len(train_x)
        print("迭代次数:{},平均损失值:{:.2f}".format(iteration_count,avg_loss))
        if iteration_count==save_interval:
            qs=input("训练次数已达10000次，请选择:-0 暂停训练并退出保存模型 -1 继续训练")
            if qs=='0':
                break
            elif qs=='1':
                save_interval+=10000
            else:
                print("请不要输入其它内容")
    print("模型训练完毕")
    save(model, f"{ModelName}.pth")
    save(model.state_dict(),f"{ModelName}_dict.pth")
    print("模型已经保存")
# model_train()



#读取模型并定义模型预测
def model_predict(ModelName):
    # 损失函数设置
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.cuda()

    model_pth=ModelName
    Model=neural_network()
    Model=Model.cuda()
    Model.load_state_dict(load(model_pth))
    Model.eval()
    total_test_loss=0
    iteration_count=0
    with no_grad():
        for x,y in zip(train_x,train_y):
            x=x.cuda()
            y=y.cuda()
            outputs=Model(x)
            y=y.unsqueeze(0).expand_as(outputs)
            y=y.float()
            loss=loss_fn(outputs,y)
            total_test_loss+=loss
            iteration_count+=1
            print("循环次数:{},损失值:{}".format(iteration_count,loss))
    average=total_test_loss/len(train_x)
    print("整体损失值:{}".format(total_test_loss))
    print("平均损失值:{}".format(average))
def model_applicate(face_tensor,model_dict):
    model=neural_network()
    model=model.cuda()
    model.load_state_dict(load(model_dict))
    model.eval()
    inputs=face_tensor
    inputs=inputs.cuda()
    outputs=model(inputs)
    return outputs
def txt_read():
    f=open("face.txt","r",encoding="utf-8")
    lines=f.readlines()
    ls=[]
    for line in lines:
        words=line.strip()
        ls.append(words)
    f.close()
    return ls

print("请选择你要做什么\n")
print("-0 训练模型\n"
      "-1 读取模型并测试\n"
      "-2 应用模型\n"
      "-3 判断人脸相似度")
choice=input("请输入你的选择:")
if choice=='0':

    model_train()
elif choice=='1':
    ModleName=input("请输入模型名字:")
    model_predict(ModleName)
elif choice=='2':
    img_path=input("请输入图片地址:")
    model_dict=input("请输入字典模型地址:")
    face_tensor=img_process(img_path)
    res=model_applicate(face_tensor,model_dict)
    tensor_ls=[]
    tmp_ls=[]
    for subtensor in res:
        tensor_ls.append(subtensor.tolist())
    for i in zip(*tensor_ls):
        tmp_ls.append(round(sum(i)/64))


    ls=txt_read()
    f=open("face_data.txt","w",encoding="utf-8")
    for a,b in zip(tmp_ls,ls):
            f.write(f"{b}:{a}\n")
    f.close()
elif choice=='3':
    loss_fn=nn.MSELoss()
    test_img=input("请输入游戏人物图片名称:")
    test_img_tensor=img_process(fr"test_img\{test_img}")
    label_img=input("请输入真实人物图片的名称:")
    label_img_tensor=img_process(fr"test_img\{label_img}")
    loss_fn=loss_fn.cuda()
    test_img_tensor=test_img_tensor.cuda()
    label_img_tensor=label_img_tensor.cuda()
    loss=loss_fn(test_img_tensor,label_img_tensor)
    print("{:.2f}".format(loss))





else:
    print("请不要输入其它内容")




                # inputs=tensor## 获取输入照片张量
                # # 前向传播
                # outputs=model(inputs)
                # print(len(outputs))




