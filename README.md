# Pytorch-Recommender
**Neural Recommendation System Using PyTorch**

본 repo는 추천시스템을 공부하며, 대표적인 알고리즘들을 Pytorch를 사용하여 직접 구현해본 실습 예제 코드입니다. 해당 코드 및 데이터셋은논문에 나온 코드의 동작 구조 재현 자체에 초점을 맞추었으며,  별도의 전처리 및 튜닝을 거치지 않았기 때문에 좋은 성능을 기대하기는 어렵습니다.

**실제 코드의 흐름만 파악해주셨으면 합니다!** 

## Algorithms

- [Neural-Collaborative-Filtering](https://arxiv.org/pdf/1708.05031.pdf)
- [Wide-Deep-Learning](https://arxiv.org/pdf/1606.07792.pdf)
- [DeepFM](https://arxiv.org/abs/1703.04247)
- [AutoEncoder-Meet-Collaborative-Filtering](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)



## Get Started

각 파트에 맞는 디렉터리에 들어가셔서 일관적으로 train.py를 실행하시면 됩니다. 기본적으로 모든 default param이 설정되어 있고, 아래와 같이 --help option을 사용하여 연결된 input param을 확인하실 수 있습니다.

```shell
$ python train.py -h
usage: train.py [-h] [--model_fn MODEL_FN] [--data_path DATA_PATH]
                [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS]
                [--embed_dim EMBED_DIM] [--mlp_dims MLP_DIMS]
                [--train_ratio TRAIN_RATIO] [--valid_ratio VALID_RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --model_fn MODEL_FN   Model file name to save. Additional information would
                        be annotated to the file name.
  --data_path DATA_PATH
                        Dataset Path,
                        Default=../data/kmrd/kmr_dataset/datafile/kmrd-
                        small/rates.csv
  --batch_size BATCH_SIZE
                        Mini batch size for gradient descent. Default=256
  --n_epochs N_EPOCHS   Number of epochs to train. Default=30
  --embed_dim EMBED_DIM
                        Embedding Vector Size. Default=100
  --mlp_dims MLP_DIMS   MultiLayerPerceptron Layers size. Default=[16, 16, 16]
  --train_ratio TRAIN_RATIO
                        Train data ratio. Default=0.8
  --valid_ratio VALID_RATIO
                        Valid data ratio. Default=0.1
```



## Dataset

다음과 같은 방식으로 각각의 데이터셋을 다운로드할 수 있습니다. <br>
데이터셋의 설치 경로는 자유이지만, 실행전에 각 실행 코드의 DATA_PATH를 확인해주세요.

### KMRD
```shell
$ mkdir src/data && cd data/
$ git clone https://github.com/lovit/kmrd && cd kmrd/
$ python setup install
# ~\Pytorch-Recommender\src\data\kmrd\kmr_dataset\datafile\kmrd-small
```



## References

- [torchfm](https://pypi.org/project/torchfm/)
- [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- [DeepRecommender](https://github.com/NVIDIA/DeepRecommender)
- [FactorizationMachine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

