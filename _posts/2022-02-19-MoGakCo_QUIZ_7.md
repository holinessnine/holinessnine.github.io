---
layout: single
title:  "[모각코] 파이썬기초 7일차: 반복문"
categories: Python
tag: [python, 모각코, 비교연산자, for]
toc: true
author_profile: false
nav: "docs"
---

  

  

  

## 오늘의 문제는 두 가지 입니다.

  

  

**오늘의 문제를 해결하기 위해서 필요한 7일차의 내용은 다음과 같습니다.**

  

> **반복문 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/7-1.-%EB%B0%98%EB%B3%B5%EB%AC%B8)**

> 그 외에도 리스트/산술연산자/비교연산자/input() 등을 활용했습니다.

  

  

  

## 첫 번째 문제입니다.

  

> *1부터 100까지의 수 중에서 짝수이면서 7의 배수가 아닌 수의 개수를 출력해보세요.* 

  

  

  

#### **해결 과정 - 1**

  

- 먼저, 조건을 만족하는 숫자를 저장하기 위한 빈 리스트를 생성합니다. (a.k.a 'answer')

- 다음으로 for문을 활용해 루프를 만들고, range()를 통해 1~100까지의 수를 루프에 넣어줍니다.

- if 조건문을 사용해, 첫 번째 조건인 짝수를 만족하는 수를 임시변수 'even'에 저장합니다.

- if 조건문 내부에 두번째 조건을 위한 if문을 하나 더 만들어줍니다. 'even'에 저장된 수 중에서 '7의 배수가 아닌 수'를 결과 리스트에 저장합니다.

- 조건 외의 수는 pass를 사용해 넘깁니다.

- 마지막으로 len(리스트)를 사용해 'answer'리스트에 저장된 수의 개수를 출력하는 과정으로 이루어집니다.

    

    
  
    

#### **코드 - 1**

  

```python
# 결과 리스트 생성
answer = []

# for문으로 1~100까지의 숫자 반복하기
for num in range(1,101):
    
    # 짝수라면 임시변수 'even'에 저장
    if num % 2 == 0:
        even = num
        # 짝수이면서 7의 배수가 아닌 경우, 'answer'리스트에 저장
        if even % 7 != 0:
            answer.append(even)

    # 조건 외의 수는 넘김
    else:
        pass
 
# 결과 출력   
print(f"1부터 100까지의 수 중에서 짝수이면서 7의 배수가 아닌 수: {len(answer)}개")
```

  

  

  

#### **결과 - 1**

  

- 먼저, 조건을 만족하는 숫자의 개수를 출력했습니다.


  

  

<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1644900819234/7%EC%9D%BC%EC%B0%A81.JPG"></center>

​    

  

- 다음으로, 혹시 결과 리스트에 저장된 수 중에서 조건에 맞지 않는 수가 있는지 확인해 보았습니다.


  

  

<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1644900832905/7%EC%9D%BC%EC%B0%A82.JPG"></center>

​    

  

- **조건, 리스트에 저장된 숫자 모두 이상이 없었습니다!**

    

    
  
    

## 두 번째 문제입니다.

  

> *0이 입력될 때까지 숫자를 계속 입력 받아 입력 받은 숫자들의 합을 출력하는 프로그램을 만들어보세요*

  

  

#### **해결 과정 - 2**

  

- 먼저, 결과 변수인 'result'를 생성하고, 0으로 초기화를 합니다.
- 다음으로 'while True:' 를 사용해 특정 조건을 만족할 때까지 루프를 반복시킵니다.
- input()으로 숫자를 입력값으로 받아옵니다.
- if 조건문으로, 받아온 숫자가 0이 아닌 경우에 'result' 변수에 해당 숫자를 더해줍니다.
- 입력값이 0인 경우에는 break가 되면서 while 루프를 탈출합니다.
- 0 이전의 입력값들의 총합인 'result' 변수에 저장된 값을 출력합니다.

  

  

  

#### **코드 - 2**

  

```python
# 결과 변수 'result'를 생성하고, 0으로 초기화
result = 0

# while문으로 루프 생성
while True:
    num = int(input("숫자를 입력하세요:"))

    # 0이 아닌 경우에는 'result' 변수에 해당 숫자 더하기
    if num != 0:
        result += num

    # 0인 경우에는 루프 탈출
    else:
        break

# 결과 출력    
print(f"\n{result}")
```

  

  

  

#### **결과 - 2**

  

- 순서대로 10,20,30을 입력했고, 네 번째로 0을 입력했습니다.


  

  

<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1644901294779/7%EC%9D%BC%EC%B0%A83.JPG"></center>

  

- 세번째 입력값까지의 총합인 10+20+30=60을 성공적으로 계산했고, 0이 입력된 순간 루프를 탈출하여 총합을 출력해 주고 있었습니다. 
