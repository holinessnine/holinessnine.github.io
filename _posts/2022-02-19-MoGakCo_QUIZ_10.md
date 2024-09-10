---

layout: single
title: "[모각코] 파이썬기초 10일차: 함수"
categories: Python
tag: [python, 모각코, function]
toc: true
author_profile: false
sidebar:
    nav: "docs"

---

  

  

  

## 오늘의 문제: 정수 n까지의 합을 구하는 함수 만들기

  

> *정수 n을 하나 입력받고, 0부터 n까지의 합을 구하는 함수를 만들어 보세요.*

  

  

**위 문제를 해결하기 위해서 필요한 10일차의 내용은 다음과 같습니다.**

  

> **함수 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/10-1.-%ED%95%A8%EC%88%98)**

  

  

  

#### **해결 과정**

  

- 먼저, 함수의 이름과 매개변수를 선언합니다: 매개변수는 정수 n이므로, 편의상 'num'으로 이름을 붙였습니다.

- 다음으로 정수 n까지의 합이 저장될 'result' 변수를 할당하고, 0으로 초기화합니다.

- for 반복문을 선언하고, 루프가 돌 범위를 range()로 제한해 줍니다: 0~n까지의 정수이므로, range(0,n+1)로 지정합니다.

- 반복문이 돌면서 각 단계의 i(0 ~ n까지의 정수)값이 'result' 변수에 더해지도록 합니다.

- 최종적으로 0부터 n까지의 숫자가 더해진 'result'를 반환하면서 함수의 정의를 끝마칩니다.

  

- 정의한 함수를 사용하기 위해, input()을 통해 원하는 정수를 입력으로 받아옵니다.

- 마지막으로, 받아온 정수에 대해 함수가 반환한 결과를 출력합니다.

  

  

  

#### **코드**

  

```python
# 함수의 이름과 매개변수를 선언
def add_until_n(num):
    result = 0
    for i in range(0,num+1): # range()를 통해 정수의 범위를 제한
        result += i
    return result

# n으로 사용될 정수 받아오기
input_num = int(input("정수를 입력하세요.:"))

# 결과값 출력
print(f"\n{add_until_n(input_num)}")
```

  

  

  

#### **결과**

  

- n의 값으로 5를 입력해 보았습니다.


  

<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1645194945528/10%EC%9D%BC%EC%B0%A8.JPG"></center>

  

- **0부터 5까지의 합은: 0+1+2+3+4+5 = 15이므로, 결과가 알맞게 출력되는 것을 확인할 수 있습니다.**
