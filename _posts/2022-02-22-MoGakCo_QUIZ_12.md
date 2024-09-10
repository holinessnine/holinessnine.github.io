---

layout: single
title: "[모각코] 파이썬기초 12일차: 튜플과 집합, 딕셔너리"
categories: Python
tag: [python, 모각코, tuple, dictionary]
toc: true
author_profile: false
sidebar:
    nav: "docs"

---

   

  

  

## 오늘의 문제: **튜플과 딕셔너리로 문자열 길이 출력하기**

  

> *3개의 문자열이 담긴 튜플을 선언하고, 딕셔너리를 통해 각각의 문자열의 길이를 저장한 뒤 출력해 보세요.*

  

  

**위 문제를 해결하기 위해서 필요한 12일차의 내용은 다음과 같습니다.**

  

> **튜플 & 집합 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/12-1.-%ED%8A%9C%ED%94%8C-%EC%A7%91%ED%95%A9)**

> **딕셔너리 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/12-2.-%EB%94%95%EC%85%94%EB%84%88%EB%A6%AC)**

  

  

  

#### **해결 과정**

  

- 먼저, 3개의 문자열이 담긴 튜플을 선언합니다.
- 결과가 저장될 빈 딕셔너리도 하나 선언해줍니다.
- 반복문이 돌면서 튜플에 담긴 문자열과 그 길이가 딕셔너리의 키-밸류로 추가되도록 합니다.
- 마지막으로 그 결과를 출력합니다.

  

  

  

#### **코드**

  

```python
# 3개의 문자열이 담긴 튜플 선언하기
str_tuple = ('Manner', 'Maketh', 'Man')

# 결과가 저장될 빈 딕셔너리 생성하기
str_dict = {}

# 튜플의 길이만큼 반복하는 for문 선언 
for i in range(len(str_tuple)):

    # 해당 단계의 문자열을 '키', 그 문자열의 길이를 '값'으로 하여 딕셔너리에 추가
    str_dict[str_tuple[i]] = len(str_tuple[i])

# 결과 출력하기
print(str_dict)
```

  

  

  

#### **결과**

  

- 각각의 문자열과 그 길이가 잘 매칭이 되어 출력되었습니다.






<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1645532571958/12%EC%9D%BC%EC%B0%A8.JPG"></center>

