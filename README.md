# Enhancing Abstractive Summarization of Implicit Datasets with Contrastive Attention

본 연구는 abstractive summarization task에서 explicit/implicit datasets을 구분짓고 contrastive attention module을 통해 implicit datasets의 요약 성능을 더욱 향상시키는 방법을 제안한다.

-------------

## Explicit/Implicit Datasets
- Implicit Datasets(CNNDM, NYT)
  - gold summary와 원문의 문장들 간의 중요도 편차가 작은 데이터셋
  - 문서 내에 gold summary와 유사도가 큰 명확한 주제 문장이 있고 추가적인 부연 설명으로 이루어진 데이터셋

- Explicit Datasets(XSum, Reddit, SAMSum)
  - gold summary와 원문의 문장들 간의 중요도 편차가 큰 데이터셋
  - 문서 내에 summary가 implicitly 내재된 데이터셋

![dataset_dist](https://github.com/Kwonrince/Implicit_summarization__ESWA/assets/72617445/6a4f6128-4035-40c3-b778-90b141f53c3f)
![dataset_box](https://github.com/Kwonrince/Implicit_summarization__ESWA/assets/72617445/93bb5e28-003a-48ef-a9fe-c894314a5957)


## Summary of proposed method
- Explicit Datasets은 원문에서 core part와 incidental part가 잘 구분되고, Implicit Datasets은 원문에서 그 둘의 구분이 명확하지 않다.
- 기존 원문에서 core part를 반영하기 위한 시도가 있었으나 incidental part 정보를 학습에 활용한 경우는 없어 실제 Implicit Datasets에서 성능 향상이 거의 없었다.(모델이 핵심부분을 파악하기 힘듦)
- 따라서, 본 연구에서는 학습 시 core part와 incidental part에 대한 정보를 명확히 반영함으로써 Implicit Datasets의 요약 성능을 개선하는 모델을 제안한다.

<p align="center"><img src="https://github.com/Kwonrince/Implicit_summarization__ESWA/assets/72617445/eed0949a-fbe3-434a-b160-0ebaa7bdfb38" width="80%" height="70%"></p>

## Model architecture
- 제안한 모델은 생성 요약을 위한 transformer 기반의 encoder-decoder model에 core part인 positive와 incidental part인 negative를 학습하기 위한 추가적인 module(Contrastive Attention Module)을 추가하였다.
- 학습 시에 모델은 일반적인 생성 요약 모델의 fine-tuning 방식으로 학습하며, 추가적으로 salience와 non-salience를 학습하도록 multi-task learning 방식으로 end-to-end 학습한다.
- 이렇게 학습된 모델은 일반적인 fine-tuning 방식보다 문서의 key point를 집중해서 학습했을 것이라고 기대한다.
- 추론 시에는 추가 module을 제거하고 요약문을 생성한다.

<p align="center"><img src="https://github.com/Kwonrince/Implicit_summarization__ESWA/assets/72617445/5266f051-aab8-44fd-b4be-415b6d5b75b4" width="60%" height="60%"></p>


------------
## Requirements

```bash
$ pip install -r requirements.txt
```

## Data Preprocessing
Dataset download and preprocessing for salience and non-salience sentence extraction

```bash
$ python preprocess_xsum.py
```

## Train
See `main.py` for detailed arguments.

It works natively with pytorch's distributed data parallel.

Note that the evaluation results of each epoch are inaccurate because they are not calculated based on the actual generated summary.

```bash
$ python main.py --devices [0_1_2_3_4_5_6_7] --batch_size [total_batch] --model_dir [./save/dir/name] --triplet [True/False]
```

## Inference

```bash
$ python inference.py --save_file [./save/dir/name/.pt]
```
