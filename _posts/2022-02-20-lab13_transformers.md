---
layout: single
title:  "[딥러닝기초] 14. 트랜스포머 (transformers) with huggingface"
categories: DL
tag: [python, deep learning, pytorch, transformers]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# Lab 13 – Transformers using Hugging Face

- We can use available pre-trained models and datasets from Hugging Face to perform NLP tasks using transformers.




```python
# Ensure that transformers are installed
!pip install transformers
```

<pre>
Collecting transformers
  Downloading transformers-4.16.2-py3-none-any.whl (3.5 MB)
[K     |████████████████████████████████| 3.5 MB 5.3 MB/s 
[?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2019.12.20)
Collecting pyyaml>=5.1
  Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
[K     |████████████████████████████████| 596 kB 37.6 MB/s 
[?25hCollecting sacremoses
  Downloading sacremoses-0.0.47-py2.py3-none-any.whl (895 kB)
[K     |████████████████████████████████| 895 kB 38.6 MB/s 
[?25hCollecting tokenizers!=0.11.3,>=0.10.1
  Downloading tokenizers-0.11.5-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.8 MB)
[K     |████████████████████████████████| 6.8 MB 5.6 MB/s 
[?25hRequirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers) (3.4.2)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers) (4.62.3)
Collecting huggingface-hub<1.0,>=0.1.0
  Downloading huggingface_hub-0.4.0-py3-none-any.whl (67 kB)
[K     |████████████████████████████████| 67 kB 2.0 MB/s 
[?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (1.21.5)
Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers) (4.11.0)
Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers) (2.23.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers) (21.3)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0,>=0.1.0->transformers) (3.10.0.2)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->transformers) (3.0.7)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers) (3.7.0)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2021.10.8)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (1.24.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (3.0.4)
Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (7.1.2)
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (1.15.0)
Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (1.1.0)
Installing collected packages: pyyaml, tokenizers, sacremoses, huggingface-hub, transformers
  Attempting uninstall: pyyaml
    Found existing installation: PyYAML 3.13
    Uninstalling PyYAML-3.13:
      Successfully uninstalled PyYAML-3.13
Successfully installed huggingface-hub-0.4.0 pyyaml-6.0 sacremoses-0.0.47 tokenizers-0.11.5 transformers-4.16.2
</pre>
### Required for Korean translation model



```python
!pip install sentencepiece
```

<pre>
Collecting sentencepiece
  Downloading sentencepiece-0.1.96-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
[K     |████████████████████████████████| 1.2 MB 4.7 MB/s 
[?25hInstalling collected packages: sentencepiece
Successfully installed sentencepiece-0.1.96
</pre>
**IMPORTANT!** `Restart Runtime` before continuing 




```python
from transformers import pipeline
import torch
import torch.nn.functional as F
```

### Use pipeline

- The `pipeline` function returns an end-to-end object that performs an NLP task on one or several texts

- Supply the task that you want to perform

- See a list of available tasks and how to run them [HERE](https://huggingface.co/transformers/main_classes/pipelines.html)

- Since no model was supplied, the default model is used: `distilbert-base-uncased-finetuned-sst-2-english`



```python
classifier = pipeline("sentiment-analysis")
classifier("We are very happy to be a part of this experience!")
```

<pre>
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
</pre>
<pre>
Downloading:   0%|          | 0.00/629 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/255M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/48.0 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/226k [00:00<?, ?B/s]
</pre>
<pre>
[{'label': 'POSITIVE', 'score': 0.999871015548706}]
</pre>
Classify sentiments for multiple text inputs

- Supply sentences in a list 

- They will be parsed together in a batch



```python
results = classifier(["We are very happy to be a part of this experience!",
                      "I'm not too sure the movie was that good.",
                      "Only the start was good."])

for result in results:
  print(result)
```

<pre>
{'label': 'POSITIVE', 'score': 0.999871015548706}
{'label': 'NEGATIVE', 'score': 0.9991068243980408}
{'label': 'POSITIVE', 'score': 0.9976555109024048}
</pre>
### Zero-shot classification pipeline

- Zero-shot learning broadly means get a model to do something that it wasn't explicitly trained to do

- Requires providing some kind of descriptor for an unseen class

- The "zero-shot-classification" pipeline allows you to provide the labels for the classification

- Here the classifier has given a high probability for "education" and lower probabilitiesy for the two other labels




```python
classifier = pipeline("zero-shot-classification")
classifier("This Deep Learning course has been chanllenging and fun!",
           candidate_labels=["education", "sports", "religion"]
           )
```

<pre>
No model was supplied, defaulted to facebook/bart-large-mnli (https://huggingface.co/facebook/bart-large-mnli)
</pre>
<pre>
Downloading:   0%|          | 0.00/1.13k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/1.52G [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/26.0 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/878k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/1.29M [00:00<?, ?B/s]
</pre>
<pre>
{'labels': ['education', 'sports', 'religion'],
 'scores': [0.6662465333938599, 0.24374939501285553, 0.0900040715932846],
 'sequence': 'This Deep Learning course has been chanllenging and fun!'}
</pre>
### Text generation

- Let's get it to generate text!

- By default it uses `gpt-2`

- Run it several times for the same input to see different text generated



```python
generator = pipeline("text-generation")
generator("Once upon a time, a dancer")
```

<pre>
No model was supplied, defaulted to gpt2 (https://huggingface.co/gpt2)
</pre>
<pre>
Downloading:   0%|          | 0.00/665 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/523M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/0.99M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/1.29M [00:00<?, ?B/s]
</pre>
<pre>
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
</pre>
<pre>
[{'generated_text': "Once upon a time, a dancer is forced to step back and allow her own audience to enjoy her performance. If you're in favor of this approach for your audience, then here are some ideas that you might learn to do while watching your own chore"}]
</pre>
Supply model, length and number of outputs



```python
generator = pipeline("text-generation", model='distilgpt2')
generator("Once upon a time, a dancer", 
          max_length=30,
          num_return_sequences=2)
```

<pre>
Downloading:   0%|          | 0.00/762 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/336M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/0.99M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/1.29M [00:00<?, ?B/s]
</pre>
<pre>
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
</pre>
<pre>
[{'generated_text': "Once upon a time, a dancer and a girl might be in the same place when you meet your master, but there aren't any other kinds of"},
 {'generated_text': 'Once upon a time, a dancer would be standing up behind one another and that should be your business plan. Now, with the intent of offering to'}]
</pre>
### Fill in the blanks

- supply a sentence and blanks out one work by specifying `<mask>`

- `top_k` argument specifies the top `k` most likely words according to the model, so it will return `k` solutions



```python
unmasker = pipeline("fill-mask")
unmasker("My <mask> was not good until I decided to cry out loud.", top_k=5)
```

<pre>
No model was supplied, defaulted to distilroberta-base (https://huggingface.co/distilroberta-base)
</pre>
<pre>
Downloading:   0%|          | 0.00/480 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/316M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/878k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/1.29M [00:00<?, ?B/s]
</pre>
<pre>
[{'score': 0.39760929346084595,
  'sequence': 'My mood was not good until I decided to cry out loud.',
  'token': 6711,
  'token_str': ' mood'},
 {'score': 0.040549106895923615,
  'sequence': 'My voice was not good until I decided to cry out loud.',
  'token': 2236,
  'token_str': ' voice'},
 {'score': 0.02257687598466873,
  'sequence': 'My sleep was not good until I decided to cry out loud.',
  'token': 3581,
  'token_str': ' sleep'},
 {'score': 0.019748961552977562,
  'sequence': 'My reaction was not good until I decided to cry out loud.',
  'token': 4289,
  'token_str': ' reaction'},
 {'score': 0.018136844038963318,
  'sequence': 'My handwriting was not good until I decided to cry out loud.',
  'token': 39615,
  'token_str': ' handwriting'}]
</pre>
### Named entity recognition (NER)

- Locates and classifies named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.




```python
ner = pipeline("ner")
sentence = "This Swiss chocolate is bittersweet, I bought it from Sam the chocolatier in Hyehwa, Seoul."
ner(sentence)
```

<pre>
No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)
</pre>
<pre>
Downloading:   0%|          | 0.00/998 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/1.24G [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/60.0 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/208k [00:00<?, ?B/s]
</pre>
<pre>
[{'end': 10,
  'entity': 'I-MISC',
  'index': 2,
  'score': 0.99901676,
  'start': 5,
  'word': 'Swiss'},
 {'end': 57,
  'entity': 'I-PER',
  'index': 13,
  'score': 0.99431854,
  'start': 54,
  'word': 'Sam'},
 {'end': 78,
  'entity': 'I-LOC',
  'index': 20,
  'score': 0.9952434,
  'start': 77,
  'word': 'H'},
 {'end': 81,
  'entity': 'I-LOC',
  'index': 21,
  'score': 0.9695425,
  'start': 78,
  'word': '##yeh'},
 {'end': 83,
  'entity': 'I-LOC',
  'index': 22,
  'score': 0.97633684,
  'start': 81,
  'word': '##wa'},
 {'end': 90,
  'entity': 'I-LOC',
  'index': 24,
  'score': 0.9990404,
  'start': 85,
  'word': 'Seoul'}]
</pre>
### Question-Answering

- extracts answers to a question from a given context 



```python
qa = pipeline("question-answering")
qa(question="Where do you work?",
    context="My name is Jane and I am a professor at SKKU")
```

<pre>
No model was supplied, defaulted to distilbert-base-cased-distilled-squad (https://huggingface.co/distilbert-base-cased-distilled-squad)
</pre>
<pre>
Downloading:   0%|          | 0.00/473 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/249M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/29.0 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/208k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/426k [00:00<?, ?B/s]
</pre>
<pre>
{'answer': 'SKKU', 'end': 44, 'score': 0.9915756583213806, 'start': 40}
</pre>
### Translation



```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
translator("공원에서 친구를 만났어요")
```

<pre>
Downloading:   0%|          | 0.00/1.12k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/298M [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/44.0 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/822k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/794k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/1.64M [00:00<?, ?B/s]
</pre>
<pre>
[{'translation_text': 'I met a friend in the park.'}]
</pre>
### Let's do some of this manually!

- Use a concrete model and tokenizer

- A list of all the models are available [HERE](https://huggingface.co/models).

- You may want to select a specific task on the left, e.g. 'Text Classification' to see which set of models you should choose from

- Below two models have been given as options. Use either one of them and test.

- You could also change the input text sentences to see how they perform on different sentences.



```python
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

classifier = pipeline("sentiment-analysis", model=model_name)
results = classifier(["We are very happy to be a part of this experience!",
                      "I'm not too sure the movie was that good",
                      "Only the start was good."])

for result in results:
  print(result)
```

<pre>
{'label': 'POSITIVE', 'score': 0.999871015548706}
{'label': 'NEGATIVE', 'score': 0.9991569519042969}
{'label': 'POSITIVE', 'score': 0.9976555109024048}
</pre>
Import generic class for tokenizer and a class for sequence classification

- `from_pretrained()` is a very important function

- This will produce the same results because they are the default values for model and tokenizer

- `input_id` contains the token id for each word in the sentence

- `attention_mask` shows which word are considered for processing, here all if them  are



```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english" # default model used
# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

results = classifier(["We are very happy to be a part of this experience!",
                      "I'm not too sure the movie was that good",
                      "Only the start was good."])

for result in results:
  print(result)

token_ids = tokenizer("We are very happy to be a part of this experience!")
print(token_ids)

# token_ids include BOS (101), EOS (102)
# token_ids is a dict of input_ids and attention_mask
```

<pre>
{'label': 'POSITIVE', 'score': 0.999871015548706}
{'label': 'NEGATIVE', 'score': 0.9991569519042969}
{'label': 'POSITIVE', 'score': 0.9976555109024048}
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2022, 1037, 2112, 1997, 2023, 3325, 999, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
</pre>
### Prepare train set and run tokenizer over it

- Since all sentences do not have the same length, set padding and truncation to True

- Now the attention mask will contain zeros for words that are not considered for tokenisation because they are just padding



```python
X_train = ["We are very happy to be a part of this experience!",
            "She drives a red car",
            "Only the start was good."]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt") # pt for Pytorch
print(batch)
```

<pre>
{'input_ids': tensor([[ 101, 2057, 2024, 2200, 3407, 2000, 2022, 1037, 2112, 1997, 2023, 3325,
          999,  102],
        [ 101, 2016, 9297, 1037, 2417, 2482,  102,    0,    0,    0,    0,    0,
            0,    0],
        [ 101, 2069, 1996, 2707, 2001, 2204, 1012,  102,    0,    0,    0,    0,
            0,    0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])}
</pre>
### Use for prediction

- The actual logits are returned

- Pass them through a softmax to get probability distribution

- Get the maximum probability's numerical label for each sentence

- Convert numerical label to text



```python
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

with torch.no_grad():
  outputs = model(**batch) #, labels = torch.tensor([1,2,0])) # unpack dict
  print(outputs)
  predictions = F.softmax(outputs.logits, dim=1)
  print(f'Predictions {predictions}') # the actual probabilities
  labels = torch.argmax(predictions, dim=1)
  print(labels)
  labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
  print(labels)
```

<pre>
SequenceClassifierOutput(loss=None, logits=tensor([[-4.3036,  4.6521],
        [-0.8797,  0.8328],
        [-2.9104,  3.1430]]), hidden_states=None, attentions=None)
Predictions tensor([[1.2898e-04, 9.9987e-01],
        [1.5284e-01, 8.4716e-01],
        [2.3444e-03, 9.9766e-01]])
tensor([1, 1, 1])
['POSITIVE', 'POSITIVE', 'POSITIVE']
</pre>
### Save in directory



```python
dir = "./"
tokenizer.save_pretrained(dir)
model.save_pretrained(dir)

tokenizer = AutoTokenizer.from_pretrained(dir)
model = AutoModelForSequenceClassification.from_pretrained(dir)
```

### Exercise: Load another model

- view all models in `huggingface.co/models`

- pick a german model

- text classification (= sentiment analysis)

- copy link from top of page after clicking on model name



```python
model_name = "oliverguhr/german-sentiment-bert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = ["Mit keinem guten Ergebnis", "Das war unfair", "Das war gut!", "Das auto"]

batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt") # pt for Pytorch
print(batch)
```

<pre>
Downloading:   0%|          | 0.00/161 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/665 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/249k [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/112 [00:00<?, ?B/s]
</pre>
<pre>
Downloading:   0%|          | 0.00/416M [00:00<?, ?B/s]
</pre>
<pre>
{'input_ids': tensor([[    3,   304,  8524,  5569,  2011,     4,     0],
        [    3,   295,   185,   174,  8716,   124,     4],
        [    3,   295,   185,  1522, 26982,     4,     0],
        [    3,   295,  4874, 26910,     4,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0]])}
</pre>

```python
with torch.no_grad():
  outputs = model(**batch) #, labels = torch.tensor([1,0,0])) # unpack dict
  print(outputs)
  predictions = F.softmax(outputs.logits, dim=1)
  print(predictions) # the actual probabilities
  label_ids = torch.argmax(predictions, dim=1)
  print(label_ids)
  labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
  print(labels)
```

<pre>
SequenceClassifierOutput(loss=None, logits=tensor([[-1.2243,  4.9076, -4.7593],
        [-1.5162,  2.9544, -1.5254],
        [ 3.8754, -0.2011, -4.4488],
        [ 3.2615,  0.5646, -4.6103]]), hidden_states=None, attentions=None)
tensor([[2.1676e-03, 9.9777e-01, 6.3205e-05],
        [1.1185e-02, 9.7773e-01, 1.1083e-02],
        [9.8308e-01, 1.6679e-02, 2.3847e-04],
        [9.3651e-01, 6.3134e-02, 3.5712e-04]])
tensor([1, 1, 0, 0])
['negative', 'negative', 'positive', 'positive']
</pre>
Some PyTorch notebooks are available in Hugging Face [HERE](https://youtu.be/zHvTiHr506c)

