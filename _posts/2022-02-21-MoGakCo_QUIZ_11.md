---

layout: single
title: "[모각코] 파이썬기초 11일차: 리스트"
categories: Python
tag: [python, 모각코, list]
toc: true
author_profile: false
sidebar:
    nav: "docs"

---

  



   

## 오늘의 문제: **입력받은 수의 평균 구하기**

  

> *7개의 수를 입력받고 평균을 구하는 프로그램을 작성해보세요. (리스트 사용)*

  

  

**위 문제를 해결하기 위해서 필요한 11일차의 내용은 다음과 같습니다.**

  

> **리스트 학습자료: [링크](https://codemate.kr/project/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A9%94%EC%9D%B4%ED%8A%B8-%EA%B8%B0%EC%B4%88%ED%8E%B8/11-1.-%EB%A6%AC%EC%8A%A4%ED%8A%B8)**

  

  

  

#### **해결 과정**

  

- 먼저, 입력받은 정수들을 저장하기 위한 빈 리스트 'mean_list'를 생성합니다.
- for 반복문을 선언하고, 루프가 돌 범위를 range()로 제한해 줍니다: 정수를 7번 받으므로, range(0,7)이면 되겠네요.
- 반복문이 돌면서 각 단계의 num(입력받은 정수)값이 'mean_list' 에 삽입되도록 합니다: append()
- 파이썬에는 리스트의 평균을 바로 구하는 함수가 없기 때문에, sum()과 len()을 사용해서 'mean_list'의 평균을 구합니다.
- 마지막으로 그 결과를 출력합니다.

  

  

​    

#### **코드**

  

```python
# 빈 리스트 생성
mean_list = []

# 7개의 수를 받도록 반복문 선언
for i in range(0,7):
    num = int(input("정수를 입력해주세요.:"))
    
    # 입력받은 수를 리스트에 삽입
    mean_list.append(num) 
    
# 삽입이 완료된 리스트의 평균을 출력
print(f"\n평균:{sum(mean_list, 0) / len(mean_list)}")
```

  

  

  

#### **결과**

  

- 7개의 정수를 입력해 보았습니다.


  

  

<center><img src="https://s3.ap-northeast-2.amazonaws.com/images.codemate.kr/images/seg3981/post/1645429062741/11%EC%9D%BC%EC%B0%A8.JPG"></center>  