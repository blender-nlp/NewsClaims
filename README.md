# NewsClaims

This repository contains code and data for the paper:
```
NewsClaims: A New Benchmark for Claim Detection from News with Background Knowledge 
```
Arxiv link: [https://arxiv.org/pdf/2112.08544.pdf](https://arxiv.org/pdf/2112.08544.pdf)


### Data

We have released an updated version of the corpus with 889 claims over 143 news articles.

We release dev and test sets with 18 articles containing 103 claims and 125 articles containing 786 claims, respectively.

You can find the data [here](https://drive.google.com/file/d/1jlQ0kQLS0kLbrXIC1fh6oT2HsWppx5QT/view?usp=sharing).

The data contains the following items:

-  `dev.json`: This is the development file.
-  `test.json`: This is the test file.
-  `all_sents.json`: This file contains the list of sentences for each news article (to be used for identifying claim sentences).
-  `text`: This folder contains the raw text for each news article.

### Evaluation Script

The evaluation script for this benchmark is present at `eval/eval_claim_detection.py` which can be used as follows:

```
python eval_claim_detection.py --gold_file <path to dev/test.json> --predictions_file <path to outputs to evaluate> --<sub task to evaluate>
```
where the `<sub task to evaluate>` flag can be the following:
- `--eval_claim`: Evaluate claim sentence detection. (Numbers [here](https://github.com/uiucnlp/NewsClaims/blob/main/README.md#claim-detection))
- `--eval_claimer`: Evaluate claimer detection. (Numbers [here](https://github.com/uiucnlp/NewsClaims/blob/main/README.md#claimer-detection))
- `--eval_claim_object`: Evaluate claim object detection. (Numbers [here](https://github.com/uiucnlp/NewsClaims/blob/main/README.md#stance-detection))
- `--eval_claim_span`: Evaluate claim span detection. (Numbers [here](https://github.com/uiucnlp/NewsClaims/blob/main/README.md#span-detection))
- `--eval_stance`: Evaluate stance detection. (Numbers [here](https://github.com/uiucnlp/NewsClaims/blob/main/README.md#stance-detection))

### Numbers

We release updated numbers for each sub-task using the new version of the dataset:

#### Claim Detection

|System|	P|	R|	F1|
|-------|-------|-------|-------|
ClaimBuster | 13.0 | <strong>86.5</strong> |22.6 |
ClaimBuster + Zero-shot NLI | <strong>21.8</strong> | 53.3 | <strong>30.9</strong> |
Human | 52.7 | 70.0 | 60.1 |

#### Claim Object Detection

|Approach | Model | Type|		F1|
|-------|-------|-------|-------|
Prompting | GPT-3 | Zero-shot | 15.2|
Prompting | T5 | Zero-shot | 11.4|
In-context learning | GPT-3 | Few-Shot | <strong>51.9</strong> |
Prompt-based fine-tuning | T5 | Few-Shot | 51.6 |
Human | - | - | 67.7 |

#### Stance Detection

|Model | Affirm F1 | Refute|		Accuracy|
|-------|-------|-------|-------|
Majority class |  82.5 | 0.0 | 70.3 |
NLI (No topic) | 89.1 | 68.0 | 83.8 |
NLI (With topic) | <strong>91.1</strong> | <strong>78.8</strong> | <strong>87.5</strong> |
Human | 97.0 | 84.2 | 94.9 |

#### Claim Span Detection

|Model | Precision | Recall|		F1|
|-------|-------|-------|-------|
PolNeAR-Content | 67.0 | 42.8 | 52.3 |
Debater Boundary Detection | <strong>75.7</strong> | <strong>77.7</strong> | <strong>76.7</strong> |
Human | 82.7 | 90.9 | 86.6 |

#### Claimer Detection

|Model | F1 | Reported|	Journalist |
|-------|-------|-------|-------|
SRL | 41.7 | 23.5  | <strong>67.2</strong> |
PolNeAR-Source | <strong>42.3</strong> | <strong>25.5</strong> | 65.9 |
Human | 85.8 | 81.3 | 88.9 |

|Model | In-Sentence F1 | Out-of-Sentence F1|	
|-------|-------|-------|
SRL | 35.8 | 2.4 |
PolNeAR-Source | <strong>38.9</strong> | <strong>2.7</strong> |

### Citation

If you used this dataset in your work, please consider citing our paper:
```
@article{reddy2021newsclaims,
  title={NewsClaims: A New Benchmark for Claim Detection from News with Background Knowledge},
  author={Reddy, Revanth Gangi and Chinthakindi, Sai and Wang, Zhenhailong and Fung, Yi R and Conger, Kathryn S and Elsayed, Ahmed S and Palmer, Martha and Ji, Heng},
  journal={arXiv preprint arXiv:2112.08544},
  year={2021}
}
```

