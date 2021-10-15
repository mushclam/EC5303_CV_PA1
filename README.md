# Computer Vision Programming Assingment 1
본 자료는 광주과학기술원 AI대학원 `EC5303 Computer Vision`의 `Programming Assignment 1 - Spatial Propagation using Optimization`에 대한 코드 및 정답을 포함한다.

본 자료에 대한 보고서는 `report.ipynb`에 서술되어 있으며, 해당 파일에 각 문항에 대한 요구사항이 실행 가능하도록 구현되어 있다.

코드에서 포함하는 내용은 다음과 같다.

1. `main.py` 전체 코드를 실행하기 위한 종합적인 조작을 제공
2. `neighborhood.py` neighborhood matrix의 계산에 관련한 함수
3. `least_square_solution.py` least-square solver와 관련된 함수
4. `graph_cut.py` graph-cut에 관련된 함수
5. `weight_functions.py` 계산에 사용되는 weight functions을 포괄
6. `metric.py` IoU score를 계산하는 함수
7. `utils.py` 기타 코드 작동에 부가적으로 필요한 편의성 함수

## Installation

```bash
# In virtual environment
pip install -r requirement.txt
```

## Usage

```
usage: main.py [-h] [--data_dir DATA_DIR] [--matrix_dir MATRIX_DIR] [--scribble_dir SCRIBBLE_DIR] [--gpu GPU] [--n_label {2,7}] [--method {lstsq,graph-cut}] [--operator {none,lsqr,lsmr}]
               [--threshold THRESHOLD] [--precision PRECISION] [--n_modal N_MODAL] [--weight-function {w1,w2,laplacian}]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -d DATA_DIR
  --matrix_dir MATRIX_DIR, -md MATRIX_DIR
  --scribble_dir SCRIBBLE_DIR, -sd SCRIBBLE_DIR
  --gpu GPU, -g GPU
  --n_label {2,7}, -nl {2,7}
  --method {lstsq,graph-cut}, -m {lstsq,graph-cut}
  --operator {none,lsqr,lsmr}, -o {none,lsqr,lsmr}
  --threshold THRESHOLD, -t THRESHOLD
  --precision PRECISION, -p PRECISION
  --n_modal N_MODAL, -nm N_MODAL
  --weight-function {w1,w2,laplacian}, -wf {w1,w2,laplacian}
```

### binary label Least square solution

```bash
python3 main.py --n_label 2 --method lstsq --operator lsqr --threshold 5 --precision 1e-6
```

### binary label graph cut

```bash
python3 main.py --n_label 2 --method graph-cut --threshold 5
```

### Multi-label Least square solution

```bash
python3 main.py --n_label 7 --method lstsq --operator lsqr --threshold 5 --precision 1e-6
```

### Multi-label graph cut

```bash
python3 main.py --n_label 7 --method graph-cut --threshold 5
```