<Anaconda 가상환경 설정>

# 드라이브 성능에 맞는 GPU API 설치하기
# 본 팀은 CUDA 11.2 와 cudnn 8.1 설치하였습니다.

# 제출한 enviornment.yml파일에는 anaconda 가상환경이 담겨있습니다.
# 따라서 아래의 코드를 cmd에 입력하시면 본 팀의 가상환경을 사용하실 수 있습니다.

conda env create -f environment.yml
conda activate mee
python AIM.py

2. .ipynb 확장자는 jupyter notebook, .py 확장자는 로컬 환경(cmd 등)을 통해서 실행 가능합니다.