---
layout: single
title:  "[딥러닝기초] 12. RNN으로 감성분석하기 (feat. IMDB)"
categories: DL
tag: [python, deep learning, pytorch, RNN]
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


# Lab 11: Using `CountVectorizer()` & RNN Text Classification

- We wiill walk through an example to understand how sklearn's `CountVectorizer()` works

- And then we will use `CountVectorizer` and RNNs to perform Sentiment Analysis on IMDB reviews



```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
```

### CountVectorizer()

- Converts a collection of text documents to a matrix of token counts in each document



```python
corpus = ['Hello there world!',
'There is a cat',
'I\'m a dreamer...',
'There there dreamer cat',
'And the answer is...42']

count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(corpus)
# Convert matrix to array
count_array = count_matrix.toarray()
# Convert array to data frame, dtm (document term matrix)
dtm = pd.DataFrame(data=count_array, 
                   columns=count_vect.get_feature_names())
print(dtm)
```

<pre>
   42  and  answer  cat  dreamer  hello  is  the  there  world
0   0    0       0    0        0      1   0    0      1      1
1   0    0       0    1        0      0   1    0      1      0
2   0    0       0    0        1      0   0    0      0      0
3   0    0       0    1        1      0   0    0      2      0
4   1    1       1    0        0      0   1    1      0      0
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>

```python
# Q1. How many tokens were created by CountVectorizer()?
  ## ANSWER: 10

# Q2. Apart from splitting by white space, what other pre-processing was done by CountVectorizer?
  ## ANSWER: lowering letter cases / removing special symbols and punctuation marks
```

View the tokens and vocabulary



```python
print(count_vect.get_feature_names())

print(count_vect.vocabulary_)
```

<pre>
['42', 'and', 'answer', 'cat', 'dreamer', 'hello', 'is', 'the', 'there', 'world']
{'hello': 5, 'there': 8, 'world': 9, 'is': 6, 'cat': 3, 'dreamer': 4, 'and': 1, 'the': 7, 'answer': 2, '42': 0}
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>
### Dealing with **stop words** in `CountVectorizer()`

- custom stop words list

- `sklearn`'s built in stop words list in English

- using `max_df` and `min_df`



```python
# Q3. Redo the above task by specifying "a" and "the" as stop words. Print out the document term matrix as before
count_vect = CountVectorizer(stop_words=['a', 'the'])

count_matrix = count_vect.fit_transform(corpus)
count_array = count_matrix.toarray()
dtm = pd.DataFrame(data = count_array,
                   columns = count_vect.get_feature_names())
print(dtm)
```

<pre>
   42  and  answer  cat  dreamer  hello  is  there  world
0   0    0       0    0        0      1   0      1      1
1   0    0       0    1        0      0   1      1      0
2   0    0       0    0        1      0   0      0      0
3   0    0       0    1        1      0   0      2      0
4   1    1       1    0        0      0   1      0      0
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>

```python
# Q4. Redo the above task by adding all known stop words in sklearn's built-in list in 'english'. 
# Print out the document term matrix as before
count_vect = CountVectorizer(stop_words='english')

count_matrix = count_vect.fit_transform(corpus)
count_array = count_matrix.toarray()
dtm = pd.DataFrame(data = count_array,
                   columns = count_vect.get_feature_names())
print(dtm)
```

<pre>
   42  answer  cat  dreamer  hello  world
0   0       0    0        0      1      1
1   0       0    1        0      0      0
2   0       0    0        1      0      0
3   0       0    1        1      0      0
4   1       1    0        0      0      0
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>

```python
# Q5. Which words were removed from the original list of words?
  ## ANSWER: 'there', 'a', 'and', 'the', 'is'
```


```python
# Q6. Remove any words that occur more than twice in the document and print out the document term matrix as before
count_vect = CountVectorizer(max_df=2)

count_matrix = count_vect.fit_transform(corpus)
count_array = count_matrix.toarray()
dtm = pd.DataFrame(data = count_array,
                   columns = count_vect.get_feature_names())
print(dtm)
```

<pre>
   42  and  answer  cat  dreamer  hello  is  the  world
0   0    0       0    0        0      1   0    0      1
1   0    0       0    1        0      0   1    0      0
2   0    0       0    0        1      0   0    0      0
3   0    0       0    1        1      0   0    0      0
4   1    1       1    0        0      0   1    1      0
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>

```python
# Q7. Remove any words that occur in less than 20% of the document and print out the document term matrix as before
count_vect = CountVectorizer(min_df=0.2)

count_matrix = count_vect.fit_transform(corpus)
count_array = count_matrix.toarray()
dtm = pd.DataFrame(data = count_array,
                   columns = count_vect.get_feature_names())
print(dtm)
```

<pre>
   42  and  answer  cat  dreamer  hello  is  the  there  world
0   0    0       0    0        0      1   0    0      1      1
1   0    0       0    1        0      0   1    0      1      0
2   0    0       0    0        1      0   0    0      0      0
3   0    0       0    1        1      0   0    0      2      0
4   1    1       1    0        0      0   1    1      0      0
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>

```python
# Q8. Which words were removed and why?
  ## ANSWER: 'a'
  ## ANSWER: it appears in less than 20% of the document
```

Use bigrams instead of single words

- specify the range of n-grams in the `ngram_range` argument in `CountVectorizer()`



```python
count_vect = CountVectorizer(ngram_range=(2,2))
count_matrix = count_vect.fit_transform(corpus)
count_array = count_matrix.toarray()
dtm = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names())
print(dtm)
```

<pre>
   and the  answer is  dreamer cat  ...  there is  there there  there world
0        0          0            0  ...         0            0            1
1        0          0            0  ...         1            0            0
2        0          0            0  ...         0            0            0
3        0          0            1  ...         0            1            0
4        1          1            0  ...         0            0            0

[5 rows x 11 columns]
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>

```python
# View the tokens in this vector
count_vect.get_feature_names()
```

<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>
<pre>
['and the',
 'answer is',
 'dreamer cat',
 'hello there',
 'is 42',
 'is cat',
 'the answer',
 'there dreamer',
 'there is',
 'there there',
 'there world']
</pre>

```python
# View the vocabulary of this vector
count_vect.vocabulary_
```

<pre>
{'and the': 0,
 'answer is': 1,
 'dreamer cat': 2,
 'hello there': 3,
 'is 42': 4,
 'is cat': 5,
 'the answer': 6,
 'there dreamer': 7,
 'there is': 8,
 'there there': 9,
 'there world': 10}
</pre>

```python
# Q9. Redo the above task by using unigrams and bigrams and removing stop words from the built-in list in 'english' 
# to build the vocabulary. Print out the document term matrix as before.
count_vect = CountVectorizer(ngram_range=(1,2), stop_words='english')

count_matrix = count_vect.fit_transform(corpus)
count_array = count_matrix.toarray()
dtm = pd.DataFrame(data = count_array,
                   columns = count_vect.get_feature_names())
print(dtm)
```

<pre>
   42  answer  answer 42  cat  dreamer  dreamer cat  hello  hello world  world
0   0       0          0    0        0            0      1            1      1
1   0       0          0    1        0            0      0            0      0
2   0       0          0    0        1            0      0            0      0
3   0       0          0    1        1            1      0            0      0
4   1       1          1    0        0            0      0            0      0
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
</pre>

```python
count_vect.vocabulary_
```

<pre>
{'42': 0,
 'answer': 1,
 'answer 42': 2,
 'cat': 3,
 'dreamer': 4,
 'dreamer cat': 5,
 'hello': 6,
 'hello world': 7,
 'world': 8}
</pre>
Use the vocab created on the simple training data in Q9 above on a new sentence



```python
test = ["The cat there is ginger"]
result = count_vect.transform(test)
result.toarray()
```

<pre>
array([[0, 0, 0, 1, 0, 0, 0, 0, 0]])
</pre>

```python
# Run this code cell
count_vect.transform(['Something completely new.']).toarray()

# Q10. Why is the result like so?
  ## ANSWER: Each words in the new sequence are not in the vocabularay. So the CountVectorizer assigned 0
```

<pre>
array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
</pre>
# Lab 11: RNN Text Classification – Sentiment Analysis of IMDB Movie Reviews

- We will use Python `pandas` to read a csv file containing IMDB movie reviews available from a public Google drive folder

- We will train on the WHOLE dataset instead of splitting into train/test

- We will create a custom Dataset for the IMDB sequences

- We will use utilities from `torch.nn.utils.rnn` for padding

- We will use `scikit-learn.feature_extraction.text.CountVectorizer` to convert a collection of text documents to a matrix of token counts

- We will define a Recurrent Neural Network (RNN) that utilises GRUs

- We will use DataLoader to load the data in batches

- We will use `tqdm.notebook` utilities to add a progress bar when training

- We will test on existing reviews and any user provided review

- NOTE: Please connect to the GPU for this part



```python
from pathlib import Path

from google_drive_downloader import GoogleDriveDownloader as gdd

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```

<pre>
device(type='cuda')
</pre>
## Download the training data



This is a dataset of positive and negative IMDB reviews. We can download the data from a public Google Drive folder by providing a id of the sharable link from Google Drive and a destination path.



```python
DATA_PATH = 'data/imdb_reviews.csv'
if not Path(DATA_PATH).is_file():
    gdd.download_file_from_google_drive(
        file_id='1zfM5E6HvKIe7f3rEt1V2gBpw5QOSSKQz',
        dest_path=DATA_PATH,
    )

# Q11. Where is the csv file saved?
  ## ANSWER: 'data/'
```

<pre>
Downloading 1zfM5E6HvKIe7f3rEt1V2gBpw5QOSSKQz into data/imdb_reviews.csv... Done.
</pre>
## Preprocess the text

- create a custom Dataset class for this data which takes in a maximum length for the sequence

  - `__init__(), __len__(), __getitem__()`

  - contains vocab, encode pad and sequences attributes

- read the CSV file using pandas

- tokenizing strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators

- add padding



```python
class IMDB_sequence(Dataset):
    def __init__(self, path, max_seq_len):
        self.max_seq_len = max_seq_len
        df = pd.read_csv(path)
        # Print the first ten lines
        print(df.head(10))
        # Data shape
        print(df.shape)

        # Use CountVectorizer()
        vectorizer = CountVectorizer(stop_words='english', min_df=0.015, ngram_range=(1,2))
        vectorizer.fit(df.review.tolist())
        
        self.vocab = vectorizer.vocabulary_
        # Add padding token
        self.vocab['<PAD>'] = max(self.vocab.values()) + 1

        # Create a tokenizer instance that lets you extract the tokenizing step from the pipeline wrapped in CountVectorizer  
        tokenizer = vectorizer.build_tokenizer()
        self.encode = lambda x: [self.vocab[token] for token in tokenizer(x)
                                 if token in self.vocab]
        self.pad = lambda x: x + (max_seq_len - len(x)) * [self.vocab['<PAD>']]
        
        # Get sequence from each row and prepare it to have length max_seq_len
        sequences = [self.encode(sequence)[:max_seq_len] for sequence in df.review.tolist()]
        sequences, self.labels = zip(*[(sequence, label) for sequence, label
                                    in zip(sequences, df.label.tolist()) if sequence])
        self.sequences = [self.pad(sequence) for sequence in sequences]

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]
    
    def __len__(self):
        return len(self.sequences)
```


```python
# Q12. Why is fit() and not fit_transform() used here?
  ## ANSWER: Because we do not want to transform the data, it is also not a training task

# Q13. What would happen to reviews that are made up of more than max_seq_len tokens?
# Show which line(s) of code handles this.
  ## ANSWER: It is sliced from start to max_seq_len.
  ### sequences = [self.encode(sequence)[:max_seq_len] for sequence in df.review.tolist()]
```

### Create Dataset Instance



```python
# This could take a little while (20-30 secs)
dataset = IMDB_sequence(DATA_PATH, max_seq_len=128)
```

<pre>
                                              review  label
0  Once again Mr. Costner has dragged out a movie...      0
1  This is an example of why the majority of acti...      0
2  First of all I hate those moronic rappers, who...      0
3  Not even the Beatles could write songs everyon...      0
4  Brass pictures (movies is not a fitting word f...      0
5  A funny thing happened to me while watching "M...      0
6  This German horror film has to be one of the w...      0
7  Being a long-time fan of Japanese film, I expe...      0
8  "Tokyo Eyes" tells of a 17 year old Japanese g...      0
9  Wealthy horse ranchers in Buenos Aires have a ...      0
(62155, 2)
</pre>

```python
# Q14. How many rows and columns are there in this dataset? What do the columns represent?
  ## ANSWER: 62155 rows / 2 columns
  ## Each column means: review sequence / sentiment label
```


```python
print(len(dataset.vocab))

# Q15. How many unique words have been extracted from this dataset to build the vocabulary?
  ## ANSWER: 1104 words
```

<pre>
1158
</pre>
### Custom Collate Function & DataLoader

- Convert reviews and labels to tensors



```python
def collate(batch):
    inputs = torch.LongTensor([item[0] for item in batch])
    target = torch.FloatTensor([item[1] for item in batch])
    return inputs, target

batch_size = 2048
train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
```

### Create RNN class



```python
class RNN_sentiment(nn.Module):
    def __init__(
        self,
        vocab_size,
        batch_size,
        embedding_dimension=100,
        hidden_size=128, 
        num_layers=2,
        device='cpu',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size # number of features in the hidden state h
        self.device = device
        self.batch_size = batch_size
        
        self.encoder = nn.Embedding(vocab_size, embedding_dimension)
        self.gru = nn.GRU(
            embedding_dimension,
            hidden_size,
            num_layers=num_layers,
            batch_first=True, # provide input and output tensors as (batch, seq, hidden_size)
        )
        self.decoder = nn.Linear(hidden_size, 1)
        
    def init_hidden(self):
        # return torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
    
    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            
        encoded = self.encoder(inputs)
        output, hidden = self.gru(encoded, self.init_hidden())
        # output[batch_size, seq, hidden_size]
        # Take all from batch and seq but only output from last layer
        output = self.decoder(output[:, :, -1]).squeeze()
        return output

# Q15. Why does the decoder layer have an output size of 1?
  ## That's because we are doing bianry classification(positive/negative)
```


```python
model = RNN_sentiment(
    hidden_size=128,
    vocab_size=len(dataset.vocab),
    device=device,
    batch_size=batch_size,
)
model = model.to(device)
model
```

<pre>
RNN_sentiment(
  (encoder): Embedding(1158, 100)
  (gru): GRU(100, 128, num_layers=2, batch_first=True)
  (decoder): Linear(in_features=128, out_features=1, bias=True)
)
</pre>
### Loss and Optimiser

- `BCEWithLogitsLoss()` combines a Sigmoid layer and the BCELoss (Binary Cross Entropy Loss) in one single class

- **Logits** are unscaled (unnormalised) outputs of the model. It means, in particular, the sum of these outputs may not equal 1, or the values are not probabilities.



```python
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Train the model

- Add a progress bar by using utilities from `tqdm.notebook`

- Add gradient clipping




```python
from tqdm.notebook import tqdm, tqdm_notebook

EPOCHS = 50

model.train()
train_losses = []
for epoch in range(EPOCHS):
    progress_bar = tqdm_notebook(train_loader, leave=False)
    losses = []
    total = 0
    for inputs, target in progress_bar:
        inputs, target = inputs.to(device), target.to(device)

        model.zero_grad()

        output = model(inputs)
    
        loss = criterion(output, target)
        
        loss.backward()
              
        nn.utils.clip_grad_norm_(model.parameters(), 3)

        optimizer.step()
        
        progress_bar.set_description(f'Epoch {epoch+1} Loss: {loss.item():.4f}')
        
        losses.append(loss.item())
        total += 1
    
    epoch_loss = sum(losses) / total
    train_losses.append(epoch_loss)

    tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')
```

<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #1	Train Loss: 0.809
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #2	Train Loss: 0.729
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #3	Train Loss: 0.720
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #4	Train Loss: 0.716
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #5	Train Loss: 0.712
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #6	Train Loss: 0.702
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #7	Train Loss: 0.673
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #8	Train Loss: 0.617
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #9	Train Loss: 0.577
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #10	Train Loss: 0.527
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #11	Train Loss: 0.487
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #12	Train Loss: 0.464
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #13	Train Loss: 0.444
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #14	Train Loss: 0.427
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #15	Train Loss: 0.414
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #16	Train Loss: 0.402
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #17	Train Loss: 0.393
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #18	Train Loss: 0.383
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #19	Train Loss: 0.374
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #20	Train Loss: 0.366
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #21	Train Loss: 0.358
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #22	Train Loss: 0.351
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #23	Train Loss: 0.342
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #24	Train Loss: 0.335
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #25	Train Loss: 0.326
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #26	Train Loss: 0.314
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #27	Train Loss: 0.305
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #28	Train Loss: 0.295
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #29	Train Loss: 0.291
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #30	Train Loss: 0.271
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #31	Train Loss: 0.259
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #32	Train Loss: 0.240
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #33	Train Loss: 0.220
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #34	Train Loss: 0.202
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #35	Train Loss: 0.180
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #36	Train Loss: 0.160
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #37	Train Loss: 0.144
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #38	Train Loss: 0.136
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #39	Train Loss: 0.152
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #40	Train Loss: 0.126
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #41	Train Loss: 0.116
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #42	Train Loss: 0.120
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #43	Train Loss: 0.113
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #44	Train Loss: 0.100
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #45	Train Loss: 0.083
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #46	Train Loss: 0.080
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #47	Train Loss: 0.073
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #48	Train Loss: 0.067
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #49	Train Loss: 0.071
</pre>
<pre>
  0%|          | 0/31 [00:00<?, ?it/s]
</pre>
<pre>
Epoch #50	Train Loss: 0.078
</pre>

```python
# Q16. Explain what this line of code in the training loop is doing and why is it needed:
#nn.utils.clip_grad_norm_(model.parameters(), 3)
  ## ANSWER: It applies gradient clipping.
  ### It is required because in RNNs the gradient might be over large by the length. 
  ### Therefore, limiting the gradient-updating speed is needed.
  ### Instead of putting too-low learning rate, gradient clipping gets L2-norm of gradient.
  ### If it is larger than threshold, (threshold / L2-norm of gradient) is multiplied to the gradient vector.
  ### Gradient always become less than before, and the descent direction does not change.
  ### By doing so, handling the update of gradient is possible.
```

### Test the Model

- `predict_sentiment` function takes in a piece of text and determines if it is positive or negative based on what it has learned from the training data



```python
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        # Encode, pad and convert to tensor
        test_vector = torch.LongTensor([dataset.pad(dataset.encode(text))]).to(device)
        
        output = model(test_vector)
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            print(f'{prediction:0.3f}: Positive sentiment')
        else:
            print(f'{prediction:0.3f}: Negative sentiment')

# Q17. Why do we need to run the sigmoid function (torch.sigmoid) on the predictions?
  ## It is a task that defines whether this review is positive or negative(binary classification)
```

### Test on Reviews from "Cool Cat Saves the Kids" using Trained RNN:



![](https://m.media-amazon.com/images/M/MV5BNzE1OTY3OTk5M15BMl5BanBnXkFtZTgwODE0Mjc1NDE@._V1_UY268_CR11,0,182,268_AL_.jpg)



Run the following code cells to see how well the model can tell apart positive reviews from negative ones



```python
test_text = """
This poor excuse for a movie is terrible. It has been 'so good it's bad' for a
while, and the high ratings are a good form of sarcasm, I have to admit. But
now it has to stop. Technically inept, spoon-feeding mundane messages with the
artistic weight of an eighties' commercial, hypocritical to say the least, it
deserves to fall into oblivion. Mr. Derek, I hope you realize you are like that
weird friend that everybody know is lame, but out of kindness and Christian
duty is treated like he's cool or something. That works if you are a good
decent human being, not if you are a horrible arrogant bully like you are. Yes,
Mr. 'Daddy' Derek will end on the history books of the internet for being a
delusional sour old man who thinks to be a good example for kids, but actually
has a poster of Kim Jong-Un in his closet. Destroy this movie if you all have a
conscience, as I hope IHE and all other youtube channel force-closed by Derek
out of SPITE would destroy him in the courts.This poor excuse for a movie is
terrible. It has been 'so good it's bad' for a while, and the high ratings are
a good form of sarcasm, I have to admit. But now it has to stop. Technically
inept, spoon-feeding mundane messages with the artistic weight of an eighties'
commercial, hypocritical to say the least, it deserves to fall into oblivion.
Mr. Derek, I hope you realize you are like that weird friend that everybody
know is lame, but out of kindness and Christian duty is treated like he's cool
or something. That works if you are a good decent human being, not if you are a
horrible arrogant bully like you are. Yes, Mr. 'Daddy' Derek will end on the
history books of the internet for being a delusional sour old man who thinks to
be a good example for kids, but actually has a poster of Kim Jong-Un in his
closet. Destroy this movie if you all have a conscience, as I hope IHE and all
other youtube channel force-closed by Derek out of SPITE would destroy him in
the courts.
"""
predict_sentiment(test_text) # Ground truth: Negative
```

<pre>
0.007: Negative sentiment
</pre>

```python
test_text = """
Cool Cat Saves The Kids is a symbolic masterpiece directed by Derek Savage that
is not only satirical in the way it makes fun of the media and politics, but in
the way in questions as how we humans live life and how society tells us to
live life.

Before I get into those details, I wanna talk about the special effects in this
film. They are ASTONISHING, and it shocks me that Cool Cat Saves The Kids got
snubbed by the Oscars for Best Special Effects. This film makes 2001 look like
garbage, and the directing in this film makes Stanley Kubrick look like the
worst director ever. You know what other film did that? Birdemic: Shock and
Terror. Both of these films are masterpieces, but if I had to choose my
favorite out of the 2, I would have to go with Cool Cat Saves The Kids. It is
now my 10th favorite film of all time.

Now, lets get into the symbolism: So you might be asking yourself, Why is Cool
Cat Orange? Well, I can easily explain. Orange is a color. Orange is also a
fruit, and its a very good fruit. You know what else is good? Good behavior.
What behavior does Cool Cat have? He has good behavior. This cannot be a
coincidence, since cool cat has good behavior in the film.

Now, why is Butch The Bully fat? Well, fat means your wide. You wanna know who
was wide? Hitler. Nuff said this cannot be a coincidence.

Why does Erik Estrada suspect Butch The Bully to be a bully? Well look at it
this way. What color of a shirt was Butchy wearing when he walks into the area?
I don't know, its looks like dark purple/dark blue. Why rhymes with dark? Mark.
Mark is that guy from the Room. The Room is the best movie of all time. What is
the opposite of best? Worst. This is how Erik knew Butch was a bully.

and finally, how come Vivica A. Fox isn't having a successful career after
making Kill Bill.

I actually can't answer that question.

Well thanks for reading my review.
"""
predict_sentiment(test_text) # Ground truth: Positive
```

<pre>
0.992: Positive sentiment
</pre>

```python
test_text = """
Don't let any bullies out there try and shape your judgment on this gem of a
title.

Some people really don't have anything better to do, except trash a great movie
with annoying 1-star votes and spread lies on the Internet about how "dumb"
Cool Cat is.

I wouldn't be surprised to learn if much of the unwarranted negativity hurled
at this movie is coming from people who haven't even watched this movie for
themselves in the first place. Those people are no worse than the Butch the
Bully, the film's repulsive antagonist.

As it just so happens, one of the main points of "Cool Cat Saves the Kids" is
in addressing the attitudes of mean naysayers who try to demean others who
strive to bring good attitudes and fun vibes into people's lives. The message
to be learned here is that if one is friendly and good to others, the world is
friendly and good to one in return, and that is cool. Conversely, if one is
miserable and leaving 1-star votes on IMDb, one is alone and doesn't have any
friends at all. Ain't that the truth?

The world has uncovered a great, new, young filmmaking talent in "Cool Cat"
creator Derek Savage, and I sure hope that this is only the first of many
amazing films and stories that the world has yet to appreciate.

If you are a cool person who likes to have lots of fun, I guarantee that this
is a movie with charm that will uplift your spirits and reaffirm your positive
attitudes towards life.
"""
predict_sentiment(test_text) # Ground truth: Positive
```

<pre>
0.704: Positive sentiment
</pre>

```python
tricky_test = """
It is not a great movie to be honest. 
I wish it was a bit longer with more plot twists.
The characters were not well developed. 
I will not be seeing the sequel. 
"""

predict_sentiment(tricky_test) # Ground truth: Negative
```

<pre>
0.013: Negative sentiment
</pre>
### Test with your own sentiments!



### Improve the model above by one or more of the following:

- Increasing `num_layers` to 2 (stacking two RNNs instead of 1)

- Initialising h0 to random values rather than 0s (code line provided as a comment in RNN_sentiment()

- Using a GRU layer instead of a vanilla RNN

- Using bigrams (or unigrams and bigrams) instead of just single words when creating the vocabulary

- Using an LSTM layer instead of vanilla RNN/GRU (add your own LSTM class and create an instance)

- Anything else you can think of

- **NOTE:** You MUST run predict_sentiment() after training your model and before testing with the test texts each time

- **NOTE:** Train for about 30-50 epochs (max 80) to save time and GPU limit




```python
# Q18. Which model was able to perform the best on all the test texts, especially on tricky_test, and your own review? 
# Comment on the overall performance of RNN-based models.

  ## ADJUSTMENTS: num_layers=2 / num_epochs=50(I tried more than 60, but I was worried about overfitting as there is 'tricky_test') / GRU layer / Used unigrams & bigrams
  ## test(neg): 0.007 / negative
  ## test(pos_1st): 0.992 / positive
  ## test(pos_2nd): 0.704 / positive
  ## test(tricky(neg)): 0.013 / negative!!
```
