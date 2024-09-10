---
layout: single
title:  "[모각코] 파이썬기초 4일차: 산술연산자"
categories: Python
tag: [python, 모각코]
toc: true
author_profile: false
nav: "docs"
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



```python
price = 3420

paper_1000 = price//1000
coin_100 = (price - (paper_1000*1000))//100
coin_10 = (price - (paper_1000*1000) - (coin_100*100))//10

print("1000원:",paper_1000)
print("100원:",coin_100)
print("10원:",coin_10)
print("필요한 동전의 개수:",coin_100+coin_10)
```

<pre>
1000원: 3
100원: 4
10원: 2
필요한 동전의 개수: 6
</pre>
# 리스트로 한 자리수 별 접근



```python
price = [3,4,2,0]

paper_1000 = price[0]//1
coin_100 = price[1]//1
coin_10 = price[2]//1

print("1000원:",paper_1000)
print("100원:",coin_100)
print("10원:",coin_10)
print("필요한 동전의 개수:",coin_100+coin_10)
```

<pre>
1000원: 3
100원: 4
10원: 2
필요한 동전의 개수: 6
</pre>

```python
```
