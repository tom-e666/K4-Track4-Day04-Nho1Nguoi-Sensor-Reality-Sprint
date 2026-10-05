from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"

def synthetic_road(w=960,h=540):
    img=np.full((h,w,3),185,np.uint8)
    cv2.rectangle(img,(0,330),(w,h),(55,55,55),-1)
    cv2.line(img,(w//2,340),(w//2,540),(235,235,235),8)
    cv2.rectangle(img,(120,250),(350,390),(30,90,190),-1)
    cv2.rectangle(img,(165,285),(305,350),(185,220,245),-1)
    cv2.circle(img,(165,395),30,(20,20,20),-1); cv2.circle(img,(305,395),30,(20,20,20),-1)
    cv2.rectangle(img,(650,235),(700,365),(45,45,45),-1)
    cv2.circle(img,(675,205),25,(70,70,70),-1)
    cv2.putText(img,"ADAS CAMERA HEALTH",(35,70),cv2.FONT_HERSHEY_SIMPLEX,1.2,(20,20,20),3)
    return img

def glare(img):
    x=img.copy(); overlay=np.zeros_like(x)
    cv2.circle(overlay,(700,170),175,(255,255,255),-1)
    mask=cv2.GaussianBlur(overlay,(0,0),55)
    return cv2.addWeighted(x,1.0,mask,0.95,0)

def salt_pepper(img,p=0.025,seed=19):
    rng=np.random.default_rng(seed); x=img.copy(); r=rng.random(img.shape[:2])
    x[r<p/2]=0; x[(r>=p/2)&(r<p)]=255
    return x

def variants(img):
    return [
      ("clean",img),
      ("blur_k5",cv2.GaussianBlur(img,(5,5),0)),
      ("blur_k11",cv2.GaussianBlur(img,(11,11),0)),
      ("blur_k21",cv2.GaussianBlur(img,(21,21),0)),
      ("strong_glare",glare(img)),
      ("salt_pepper",salt_pepper(img)),
    ]

def entropy(gray):
    hist=cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p=hist/hist.sum(); p=p[p>0]
    return float(-(p*np.log2(p)).sum())

def metrics(img,base_edges):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    sat=float((gray>=245).mean())
    ent=entropy(gray)
    edge=float((cv2.Canny(gray,80,160)>0).mean())
    return lap,sat,ent,edge/max(base_edges,1e-9)

def yolo_metrics(model,img):
    r=model.predict(img,verbose=False)[0]
    conf=r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
    return int(len(conf)), float(conf.mean()) if len(conf) else 0.0

def main(use_yolo=False):
    OUT.mkdir(exist_ok=True)
    base=synthetic_road()
    bg=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    base_edges=float((cv2.Canny(bg,80,160)>0).mean())
    model=None
    if use_yolo:
        from ultralytics import YOLO
        model=YOLO("yolov8n.pt")
    rows=[]
    for level,(name,img) in enumerate(variants(base)):
        lap,sat,ent,edge=metrics(img,base_edges)
        row={"level":level,"condition":name,"laplacian_variance":lap,
             "saturation_ratio":sat,"entropy":ent,"edge_retention":edge}
        if model:
            n,c=yolo_metrics(model,img); row.update(detections=n,mean_confidence=c)
        rows.append(row); cv2.imwrite(str(OUT/f"{level}_{name}.jpg"),img)
    # normalized demo health: sharpness + edge retention - saturation penalty
    base_lap=rows[0]["laplacian_variance"]
    for r in rows:
        sharp=min(r["laplacian_variance"]/max(base_lap,1e-9),1.0)
        r["health_score"]=float(np.clip(.55*sharp+.35*min(r["edge_retention"],1)-.30*r["saturation_ratio"],0,1))
    with open(OUT/"metrics.csv","w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    x=[r["level"] for r in rows]
    plt.figure(figsize=(8,4.5))
    plt.plot(x,[r["health_score"] for r in rows],marker="o",label="health score")
    plt.plot(x,[min(r["edge_retention"],1) for r in rows],marker="o",label="edge retention")
    plt.xticks(x,[r["condition"] for r in rows],rotation=25,ha="right")
    plt.ylim(0,1.05); plt.ylabel("normalized proxy"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"degradation_metrics.png",dpi=160); plt.close()
    before=np.hstack([rows and base, glare(base)])
    cv2.putText(before,"CLEAN",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.putText(before,"STRONG GLARE",(980,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.imwrite(str(OUT/"before_after.png"),before)
    print("Wrote",OUT/"metrics.csv")
    for r in rows: print(r)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--yolo",action="store_true")
    main(p.parse_args().yolo)
