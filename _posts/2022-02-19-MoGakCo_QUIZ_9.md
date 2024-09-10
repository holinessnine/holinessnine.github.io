---
layout: single
title:  "[모각코] 파이썬기초 9일차: 도전과제 해결하기-2"
categories: Python
tag: [python, 모각코]
toc: true
author_profile: false
nav: "docs"
---





```python
year = int(input("연도를 입력하세요.:"))

# 4로 나누어 떨어지면서 100으로 나누어 떨어지지 않는 경우는 윤년이다.
if year % 4 == 0 and year % 100 != 0:
    print(f"{year}년은 윤년입니다.")

# 400으로 나누어 떨어지는 경우는 무조건 윤년이다.    
elif year % 400 == 0:
    print(f"{year}년은 윤년입니다.")

# 그 외는 윤년이 아니다.    
else:
    print(f"{year}년은 윤년이 아닙니다.")
    
```
