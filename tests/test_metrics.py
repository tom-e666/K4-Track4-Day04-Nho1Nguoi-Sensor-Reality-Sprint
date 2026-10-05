import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.run_demo import synthetic_road, variants, metrics
import cv2

def test_blur_reduces_laplacian_variance():
    img=synthetic_road()
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edge=float((cv2.Canny(g,80,160)>0).mean())
    vs=variants(img)
    clean=metrics(vs[0][1],edge)[0]
    blur=metrics(vs[3][1],edge)[0]
    assert blur < clean
