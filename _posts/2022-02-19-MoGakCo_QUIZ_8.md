---
layout: single
title:  "[모각코] 파이썬기초 8일차: 도전과제 해결하기-1"
categories: Python
tag: [python, 모각코]
toc: true
author_profile: false
nav: "docs"
---

키와 몸무게를 입력받아 BMI 지수가 25 ~는 '비만입니다.', BMI 지수가 23 ~ 25는 '과체중입니다.', BMI 지수가 18.5 ~ 23은 '정상체중입니다.', BMI 지수가 18.5 미만일 경우에는 '저체중입니다.'를 출력해주는 프로그램을 작성하시오.



BMI 지수는 몸무게(kg) / {키(m) * 키(m)} 로 계산한다.




👉 입력예시



키를 입력하세요. : 185
몸무게를 입력하세요. : 75


👉 출력예시



정상체중입니다.



### 코드


```python
height = int(input("키를 입력하세요. :"))
weight = int(input("몸무게를 입력하세요. :"))
bmi = weight/height*height

if bmi >= 18.5 and bmi <23:
    print('정상체중입니다.')
elif bmi >= 23 and bmi < 25:
    print('과제충입니다.')
else:
    print('비만입니다.')
```



### 결과

<pre>
키를 입력하세요. :180
몸무게를 입력하세요. :70
비만입니다.
</pre>
