import pyautogui
import cv2
import numpy as np
import os
import time
import ruamel.yaml
import networks
import torchvision.transforms as transforms
import torch
import matplotlib
import dlib 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from PIL import Image
import utils

#paramters for GUI Display
plt.rcParams.update({'font.size': 15})
matplotlib.use("TKAgg")

#check if cuda is available
DEVICE = torch.device("cuda device available:0" if torch.cuda.is_available() else print("No GPU Detected"))
print(DEVICE)

#define resolution
resolution = (1920, 1080)
centerx=1920/2
centery=1080/2



init=[]
framesave=[]
peporcessing=[]
net=[]
totaltime=[]

idfolder=0

idpath="path"+str(idfolder)
folder="saveVids"

pathToSave=os.path.join(folder,idpath)

if not os.path.exists(pathToSave):
    os.makedirs(pathToSave)
 
numberOfFrames=3
idImg=0

predictor_path      = 'shape_predictor_5_face_landmarks.dat'
cnn_face_detector   = 'mmod_human_face_detector.dat'

detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector)
sp = dlib.shape_predictor(predictor_path)

sadImage=Image.open("sad.jpg")
happyimage=Image.open("happy.jpg")
neutralimage=Image.open("neutral.jpg")
yaml = ruamel.yaml.YAML()   

with open('parameters.yaml') as file:
    parameters = yaml.load(file)  

networkStructure=parameters['network3']
model = networks.Network(networkStructure)
print(parameters['loadnet_path'])
model.load_state_dict(torch.load(parameters['loadnet_path']))
model.cuda()
model.eval()
transformValidation=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
start = time.time()
frames=[]

centerx=1920/2

centery=1080/2

#plt.get_current_fig_manager().window.setGeometry(centerx-330,centery-290,500,500)
a=str(int(centerx-260))
b=str(int(centery-290))
plt.get_current_fig_manager().window.wm_geometry("+"+a+"+"+b)

plt.show()



end = time.time()
#print("initialization",end - start)
#raise NameError("hi")
good=0
for i in range(1000):

    startend=time.time()
    while(good<numberOfFrames):

        end3=time.time()
        # Take screenshot using PyAutoGUI
        img = pyautogui.screenshot(region=(centerx-260,centery-290,500,500))
        
        # Convert the screenshot to a numpy array
        frame = np.array(img)
        end2 = time.time()
        framesave.append(end2-end3)
        #cv2.imwrite(pathToSave+"/imgbeforecrop"+str(idImg)+".jpg", frame)


        end4=time.time()
        
        frame=face_alignment(frame,detector,cnn_face_detector,sp)
        
        # Convert it from BGR(Blue, Green, Red) to
        # RGB(Red, Green, Blue)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("face not found")
            continue
        #cv2.imwrite(pathToSave+"/img"+str(idImg)+".jpg", frame)

        frame=transformValidation(Image.fromarray(frame))
        frames.append(frame)
        end5=time.time()
        peporcessing.append(end5-end4)
        endnet1=time.time()
        print("endnet1",endnet1)
        idImg+=1
        #end = time.time()
        #print(end - start)
        #start=end
        good+=1

    good=0
    

    #start=time.time()
    with torch.no_grad():
        input_var=torch.stack(frames, dim=3)
        input_var=input_var.unsqueeze(0)
        #print(input_var.size())

        input_var = input_var.to(DEVICE)
        pred_score = model(input_var)

        emotion=torch.argmax(pred_score)
        #print(pred_score.size())
        probabilities=F.softmax(pred_score, dim=0).cpu().numpy()
        #print(probabilities)

    

    plt.close()
    fig, ax = plt.subplots(1,2)
    ax[0].bar(["+","neutral","-"], probabilities, align='center')

    ax[0].set_xlabel('Emotions',fontdict = {'fontsize' : 15})
    ax[0].set_ylabel('Predicted score',fontdict = {'fontsize' : 15})
    ax[0].set_ylim(0,1)
    ax[0].set_title('Emotion Predicted',fontdict = {'fontsize' : 15})
    if(emotion==2):
        print("-")
        ax[1].imshow(sadImage)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].axis('off')
        plt.draw()
        plt.pause(0.05)
        #plt.savefig("image"+str(i)+".png")
    elif(emotion==1):
        print("neutral")
        ax[1].imshow(neutralimage)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].axis('off')
        plt.draw()
        plt.pause(0.05)
        #plt.savefig("image"+str(i)+".png")

    else:
        print("+")
        ax[1].imshow(happyimage)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].axis('off')
        plt.draw()
        plt.pause(0.05)
        #plt.savefig("image"+str(i)+".png")

    #end = time.time()
    #print(end - start)
    idfolder+=1
    idpath="path"+str(idfolder)
    pathToSave=os.path.join(folder,idpath)

    if not os.path.exists(pathToSave):
        os.makedirs(pathToSave)
    
    endnet2=time.time()
    print("endnet2",endnet2)
    print("endnet1",endnet1)
    net.append(endnet2-endnet1)
    frames=[]

    end7=time.time()
    totaltime.append(end7-startend)
    print(i)
    if(i>10):
        print(net)
        print(totaltime)
        print(framesave)
        print(peporcessing)
        raise NameError("stop")
    #print("total",end7-end)
    #raise NameError("end")
# Destroy all windows
cv2.destroyAllWindows()

