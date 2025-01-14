---
layout: single
title:  "[모각코] 파이썬기초 1일차: Hello Python!"
categories: Python
tag: [python, 모각코]
toc: true
author_profile: false
nav: "docs"
---

  

## 모각코 2월 과정 시작!!

* 지금껏 학회활동을 하면서 필요했던 부분이나 전공수업에서 활용되는 코딩 지식을 구글링으로, 강의자료로 독학해왔습니다.

* 이제는 어느 정도 코딩에 익숙해졌다는 느낌을 받았지만, 보다 창의적인 코딩을 하는 것에 한계를 느꼈던 것 같습니다.

* 그렇기 때문에 체계적인 커리큘럼을 통해 기초부터 다시 코딩 공부를 하고 싶었고, 복습을 하는 차원에서 내가 가장 많이 사용하는 언어인 <파이썬>의 기초 과정을 이수하게 되었습니다!!

  

* 온라인 모각코 과정은 매달 진행되고, 15일이라는 기간 동안 매일 학습과 실습과제를 꾸준히 한다는 점에서 코딩하는 습관을 들이기에 좋은 기회였는데요.

* 생각보다 매일매일 코딩하는 것이 즐거웠고, 3월부터는 <파이썬 코딩테스트 대비과정>을 수료할 생각입니다. 매일매일 힘내 보겠습니다!!


  

  

  

## 오늘의 문제: 파이썬 개발환경 인증샷 찍어올리기

* 학습자료에서는 VSCode를 권장했으나, 이미 사용하고 있던 다른 툴이 익숙하다면 그것을 사용해도 상관이 없다고 나와있었습니다.
* 그래서 가장 손에 익은 Jupyter Notebook으로, 현재 시각과 간단한 자기소개를 출력해 보았습니다.


  

  

  

### 코드

```python
import datetime

cur = datetime.datetime.now()
print(cur)
```

  

  

  

### 결과

<pre>
2022-02-04 01:02:01.975630
</pre>
