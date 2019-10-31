import cv2
import numpy as np
import math


def func(x):
    pass

listy=[]
k1=7
def distance():
    global centroidx, centroidy,listy,cnt,approx
    listy=[]
    for ind,a in enumerate(approx):
        for i,j in a:
            d=math.sqrt(((i-centroidx)**(2))+((j-centroidy)**(2)))
            listy.append((d,(i,j)))
    listy.sort(reverse=True)
    dis=int(listy[0][0]/1.5)
    val=[]
    for i in range(len(listy)):
        a,b=listy[i]
        if a>dis and b[1]<260:
            val.append(listy[i])
    return val


cap=cv2.VideoCapture(0)
if cap.isOpened():
    ret,background=cap.read()
    background=cv2.flip(background,1)
    bg=cv2.GaussianBlur(background,(k1,k1),0)
    bg=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    bg=bg.astype('float')
else:
    ret=False

   
counter=0
while counter<55:
    ret,first=cap.read()
    first=cv2.flip(first,1)
    frame=cv2.GaussianBlur(first,(k1,k1),0)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(frame,bg,0.02)
    counter+=1


cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.createTrackbar('k','image',4,255,func)
cv2.createTrackbar('thresh','image',48,255,func)
bg=cv2.convertScaleAbs(bg)
roi=bg[0:280,400:640]
print('done')
while True:
    thumb=False
    detect=False
    total=0
    c=cv2.getTrackbarPos('thresh','image')
    k=cv2.getTrackbarPos('k','image')
    if k<1:
        k=1
    if k%2==0:
        k+=1
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    ret,img=cap.read()
    img=cv2.flip(img,1)
    frame1=cv2.GaussianBlur(img,(k1,k1),0)
    frame1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    roi1=frame1[0:280,400:640]
    mask=cv2.absdiff(roi,roi1)
    ret,mask=cv2.threshold(mask,c,255,cv2.THRESH_BINARY)
    mask=cv2.erode(mask,kernel)
    mask=cv2.dilate(mask,kernel)
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=0:
        detect=True
        cnt=max(contours,key=cv2.contourArea)
        M=cv2.moments(cnt)
        centroidx=int(M['m10']/M['m00'])
        centroidy=int(M['m01']/M['m00'])
        cv2.circle(img[0:280,400:640],(centroidx,centroidy),20,(0,0,255),3)
        epsi=0.02*cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,epsi,True)
        val=distance()
        conv_hull=cv2.convexHull(cnt)
        top=tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
        bottom=tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
        left=tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
        right=tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
        cX=(left[0]+right[0])//2
        cY=(top[1]+bottom[1])//2
        solidity=float(cv2.contourArea(cnt))/cv2.contourArea(conv_hull)
        if val is not None:
            if solidity<0.87 and len(val)<=5:
                for ind,i in enumerate(val):
                    a,b=i
                    d=math.sqrt(((b[0]-left[0])**(2))+((b[1]-left[1])**(2)))
                    if d<50:
                        thumb=True
                    cv2.circle(img[0:280,400:640],b,20,[0,255,0],-1)
                    total+=1
    if total==0 and detect==False:
        rps=cv2.imread('abc.png')
    elif thumb==True and total==5:
        rps=cv2.imread('2sci.png')
    elif thumb==True and total==3:
        rps=cv2.imread('3stone.png')
    elif total==0:
        rps=cv2.imread('1paper.png')
    else:
        rps=cv2.imread('abc.png')
    rps=cv2.resize(rps,(240,280))
    rp1=np.zeros_like(rps)
    rp1[:,:,0]=mask.copy()
    rp1[:,:,1]=mask.copy()
    rp1[:,:,2]=mask.copy()
    cv2.putText(img,str(total),(70,70),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),10,cv2.LINE_AA)
    cv2.rectangle(img,(400,0),(640,280),(0,255,0),3)
    img3=np.vstack((rps,rp1))
    cv2.imshow('img3',img3)
    cv2.imshow('image',img)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
cap.release()
