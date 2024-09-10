---
layout: single
title: "[모각코] 파이썬기초 13일차: 클래스"
categories: Python
tag: [python, 모각코, class]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

  

  

## 오늘의 문제: 계산기 클래스 만들기

  

> *여러분만의 계산기 클래스를 만들어서 입력과 출력까지 보여주세요. 정수가 아니여도 괜찮겠죠?*

  

  

**위 문제를 해결하기 위해서 필요한 13일차의 내용은 다음과 같습니다.**

  

> **클래스 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/13-1.-%ED%81%B4%EB%9E%98%EC%8A%A4%EC%99%80-%EC%83%9D%EC%84%B1%EC%9E%90)**  

  

  

  

#### **해결 과정**

  

- 먼저, 계산기 클래스로 사용될 "**Calculator()**"를 생성합니다.
- 다음으로 초깃값을 설정합니다. 제가 만든 계산기는 인자로 [**a**(첫번째 숫자), **b**(두번째 숫자), **cal_type**(계산방식)] 총 3가지를 받습니다.
- 덧셈, 뺄셈, 곱셈, 나눗셈에 해당하는 함수를 선언해 줍니다. 연산 기호만 알면 되니 어렵지 않습니다.
- 이후 **calculate() 함수**를 정의해, 클래스가 계산과 출력의 기능을 동시에 갖도록 해줍니다.
- calculate() 함수는 계산 방식(사칙연산)에 따라 해당 방식에 맞는 결과를 출력하는 함수입니다. 조건문이 사용되었습니다.

  

  

  


#### **코드**

  

- 먼저 계산기 클래스를 정의합니다.

```python
# 계산기 클래스 생성
class Calculator():
    
    # 초깃값 설정
    def __init__(self, a, b, cal_type):
        self.a = a
        self.b = b
        self.cal_type = cal_type
        
    # 덧셈
    def add(self):
        return self.a + self.b
    
    # 뺄셈
    def sub(self):
        return self.a - self.b
    
    # 곱셈
    def mul(self):
        return self.a * self.b
    
    # 나눗셈
    def div(self):
        return self.a / self.b
    
    # 계산방식에 따라 계산하는 함수 생성
    def calculate(self):
        
        print("\n계산이 완료되었습니다.\n")
        
        # 계산방식 = "더하기"
        if self.cal_type == "더하기":
            print(self.add())
            
        # 계산방식 = "빼기"
        elif self.cal_type == "빼기":
            print(self.sub())
            
        # 계산방식 = "곱하기"
        elif self.cal_type == "곱하기":
            print(self.mul())
            
        # 계산방식 = "나누기"
        else:
            print(self.div())
```

  

  

  

- 다음으로는 정의한 계산기 클래스를 가지고 실제 연산을 수행하는 코드입니다.
- 계산할 두 숫자인 x,y를 입력받고 / 계산 방식인 cal_type을 입력받습니다.
- Calculator() 클래스의 인스턴스인 "**calculator**"를 생성하고, 입력받은 숫자와 계산방식을 넣어줍니다.
- 위에서 정의한 **calculate() 함수를 통해 결과를 출력**합니다.

```python
# 계산할 두 수 입력받기
x = float(input("수를 입력하세요.:"))
y = float(input("수를 입력하세요.:"))

# 계산방식 입력받기
cal_type = input("계산방식을 선택하세요 [더하기, 빼기, 곱하기, 나누기]:")

# 계산기 클래스에 인자 넣기
calculator = Calculator(x, y, cal_type)

# 계산 결과 출력
calculator.calculate()
```

  

  

  

#### **결과**

  

- 121과 11을 입력했고, 11의 제곱수가 121이므로 계산방식은 "나누기"로 지정해 보았습니다.






<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1645622952243/13%EC%9D%BC%EC%B0%A8.JPG"></center>