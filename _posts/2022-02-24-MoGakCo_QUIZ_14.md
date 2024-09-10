---
layout: single
title: "[모각코] 파이썬기초 14일차: 클래스의 상속"
categories: Python
tag: [python, 모각코, class]
toc: true
author_profile: false
sidebar:
    nav: "docs"

---

  

 

## 오늘의 문제: 클래스 생성하기

  

> 저번 시간에 여러분이 만든 나만의 계산기 클래스를 상속해 새로운 클래스를 만들어보거나, 달콤한 파이썬 본문에 있는 "완벽한 계산기"를 이용하여 클래스를 새로 만들어보세요!

  

  

**위 문제를 해결하기 위해서 필요한 14일차의 내용은 다음과 같습니다.**

  

> **클래스 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/14-1.-%ED%81%B4%EB%9E%98%EC%8A%A4%EC%9D%98-%EC%83%81%EC%86%8D)**  

  

  

  

### **해결 과정**

  

- 먼저, 13일차에 생성한 클래스 "Calculator"를 상속받는 "**new_calculator**"를 생성합니다.
- 나머지 함수는 동일하도록 건드리지 않고, 나눗셈 함수 "**div()**"의 기능을 나눗셈의 몫 -> 나머지 반환으로 변경해 보았습니다.
- "Calculator"의 클래스 변수 "a", "new_calculator"의 클래스 변수 "b"를 생성합니다.
- 각각의 경우에서 5 나누기 2의 반환값을 비교합니다.

  

  

  

### **코드**

  

- 먼저, 13주차에 작성한 계산기 클래스입니다.

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

  

  

- 다음으로는 앞에서 정의한 계산기 클래스를 상속받는 클래스를 만들어 봅니다.

```python
# 자식 클래스 "new_calculator"
class new_calculator(Calculator):

    # 나누기 함수 "div()"를 오버라이딩 - 몫 반환으로 변경 
    def div(self):
        return self.a // self.b
    
# 결과값 비교를 위해 클래스변수 a, b 생성    
a = Calculator(5,2,"나누기")
b = new_calculator(5,2,"나누기")

# 클래스 별 결과 출력
print("5 나누기 2의 결과 비교")
print(f"\nCalculator의 5 나누기 2 결과: {a.div()}")
print(f"\nnew_calculator의 5 나누기 2 결과: {b.div()}")
```

  

  

  

### **결과**

  

- 두 클래스 모두 5에서 2를 나누어 보았습니다.
- 의도대로라면 "Calculator"는 5/2의 값인 2.5, "new_calculator"는 5/2의 몫인 2를 출력해 주어야 겠네요.






<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1645825791852/14%EC%9D%BC%EC%B0%A8.JPG"></center>