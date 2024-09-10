---
layout: single
title:  "[모각코] 파이썬기초 3일차: 자료형과 변수"
categories: Python
tag: [python, 모각코]
toc: true
author_profile: false
nav: "docs"
---



## 정수/실수/불/문자열 변수 할당하고 타입 출력하기



```python
## 정수 / 실수 / 불 / 문자열 변수 할당하기

# 정수형 변수
x_int = 970908
# 실수형 변수
x_float = 97.0908
# 불리언 변수
x_boolean = sum([10,11,12]) < 40
# 문자열 변수
x_string = "97-09-08"

## 변수 타입 출력하기
print(type(x_int))
print(type(x_float))
print(type(x_boolean))
print(type(x_string))
```

<pre>
<class 'int'>
<class 'float'>
<class 'bool'>
<class 'str'>
</pre>
## "제 이름은 ㅇㅇㅇ입니다"에서 "ㅇㅇㅇ"만 슬라이싱해 출력하기



```python
myname = "제 이름은 신성구입니다"
print(myname[-6:-3])
```

<pre>
신성구
</pre>