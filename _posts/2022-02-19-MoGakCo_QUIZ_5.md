---
layout: single
title:  "[모각코] 파이썬기초 5일차: input()으로 입력받기 & 자료형의 변환"
categories: Python
tag: [python, 모각코, data type]
toc: true
author_profile: false
nav: "docs"  
---

  

  

  

## 오늘의 문제: 숫자로 생일을 입력받아 연도와 월, 일을 출력하기

  

> 숫자로 생일을 입력받아 연도와 월, 일을 출력해보세요
>
> 추가조건: *input을 받은 후에 자료형을 변환하는 것이 아니라, input을 받는 과정에서 자료형을 변환하라*

  

  

**위 문제를 해결하기 위해서 필요한 5일차의 내용은 다음과 같습니다.**

> **입력받기 & 자료형 변환 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/5-1.-%EC%9E%85%EB%A0%A5%EB%B0%9B%EA%B8%B0-%EC%9E%90%EB%A3%8C%ED%98%95-%EB%B3%80%ED%99%98)**

  

  

  

#### **해결 과정**

  

* input()을 이용해 사용자가 입력하는 생년월일을 받아옵니다.
* 연도는 생년월일 8자리 중 앞 4자리, 월은 5~6번째 자리, 일은 7~8번째 자리에 위치합니다.
* 추가조건에 의해 받아온 생년월일이 string 형태가 아니라 int형태라 슬라이싱은 힘들고, '//' 연산자를 사용합니다.

- 연도: 생년월일 // 10000 [생년월일 8자리를 10000으로 나눈 몫]
- 월: 생년월일 - (연도x10000) // 100 [생년월일 8자리에서 연도 4자리를 없애고, 나머지를 100으로 나눈 몫]
- 일: 생년월일 - (연도x10000) - (월x100) [생년월일에서 연도와 월의 숫자를 뺀 나머지 부분]

  

  

  

#### 코드

```python
birth_date = int(input("생일을 입력하세요: "))

year = birth_date//10000 # 
month = (birth_date - (year*10000))//100
day = ((birth_date - (year*10000)) - (month*100))
```

  

  

* 출력은 f-string을 사용해서 한줄로 처리했습니다.


  

  

```python
# 결과 출력하기
print(f"\n{year}년\n{month}월\n{day}일")
```

  

  

  

#### 결과

  

* 실제 제 생일 대신에, 오늘 태어난 아이의 생일을 축하한다는 의미로 오늘 날짜를 입력으로 받아보겠습니다.


  

  

<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1644552531557/5%EC%9D%BC%EC%B0%A8.JPG"></center>
