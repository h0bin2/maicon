---

cpu.bat / gpu.bat 실행

or 

cd py
python startpack.py true/false cpu/gpu

1. true 한번 실행할 때 여러번, false 실행 당 1번
2. gpu 있으면 gpu, 없으면 cpu
3. 그냥 python 실행할 땐 py/models, py/logs 에 불러오기/저장이 된다.
4. bat 실행은 가장 밖에 있는 models, logs 에서 불러오기/저장됨

5. 모델을 추가로 넣고 실험하고 싶으면 어떻게 실행할 건지에 따라 맞는 models 폴더에 넣으면 됨. 
    대신 기본으로 불러오는(yolo11m.pt, rtdetr-l.pt) 
    것과 같이 ultralytics에 내장된 pt 파일 제외 이름으로 저장해서 넣기

    사실 불러는 와지는데 파일에 있는거 우선으로 load 될 것.

6. engine 모델의 경우 gpu에서만 동작하는 것 같다.
7. bat file은 default 가 false 다.
8. 완전 처음 즉 models 에 모델이 없다면 기본 모델을 불러와서(다운) 사용.

---
