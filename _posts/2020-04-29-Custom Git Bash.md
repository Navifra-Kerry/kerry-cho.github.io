---
title: "Custom Git Bash"
use_math: true
tags: [Bash]
header:
---

안녕하세요. 조대희 입니다.
블로그 방문을 환영 합니다.

이번 포스트는 Windows 개발 환경에서 Git Bash에 Custom Theme 적용 하는 방법에 대해 소개 하려 합니다.

그럼 시작하겠습니다.

결과는 아래와 같습니다. Visual Studio Code와 Git Bash Shell이 적용 된 모습니다.  
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/result.png" alt="result">

Theme적용은 방법은 Open Source 인 Bash-it 사용 하겠습니다.  
이 포스트는 2020.4.29에 작성 되었기 때문에 많은 시간이 경과 했을 경우 [여기](https://github.com/Bash-it/bash-it)를 참고하세요.  
VS Code와 같이 사용 하기 위해 VS Code 실행 합니다.

F1 누른 후 TerminalSelect:Default Shell을 선택 하고 Git Bash를 선택 후 New Terminal를 열어 주세요.  
그러면 아래와 같이 Git Bash가 실행 된 모습을 볼 수 있을 겁니다.  
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/1.png" alt="1">
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/2.png" alt="2">
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/3.png" alt="3">

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/4.png" alt="4">

그 후 터미널에 아래의 명령어를 입력 해주세요 (Git 설치 되어 있지 않다면 [여기](https://git-scm.com/download/win)에서 Git을 다운 받아 설치 해주세요.)

```
git clone --depth=1 https://github.com/Bash-it/bash-it.git ~/.bash_it
```

아래와 같이 나와야 정상이에요.  
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/5.png" alt="5">

다음 명령은 아래와 같이 입력 해주세요.

```
~/.bash_it/install.sh
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/6.png" alt="6">
설치시에 누르면 파일의 경로를 찾을 수 없다는 말이 나오는데 일단은 무시하고 진행 합니다.

설치가 완료 되면 새로운 Terminal을 열어 주세요.

Git Bash의 테마가 변경 된 것을 확인 할 수 있습니다.  
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/7.png" alt="7">

```
윈도우 탐색기를 이용하여 아래의 경로에서 .bashrc 파을은 메모장으로 열어 줍니다.
%UserProfile%

아래 내용의 bobby 부분이 테마의 이름입니다. [여기](https://github.com/Bash-it/bash-it/wiki/Themes)에서 원하는 테마를 확인하고 적용 해보겠습니다.
저는 Brunton 테마를 적용 해 보겠습니다. (일부 테마의 경우 FONT가 없으면 깨질수 있습니다.)
export BASH_IT_THEME='bobby'
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/8.png" alt="8">

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/9.png" alt="9">

Brunton 테마가 적용된 콘솔의 모습니다.  
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/10.png" alt="10">

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/11.png" alt="11">

추가로 자동 완성 기능을 적용 하는 방법에 대해 알려 드리겠습니다.  
자동 완성이란 Bash의 특성상 많은 명령어가 존재 하는데 앞에 몇글자만 치면 존재하는 모든 명령어를 보여 주는 기능 입니다.

먼저 적용 되어 있는 자동 완성 옵션을 확인 하는 명령어는 아래와 같습니다.

```
bash-it help completions
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/12.png" alt="12">  
저는 아무것도 활성화 되어 있지 않네요.  
활성화 하는 방법은 아래 명령어와 같이 입력합니다.

```
bash-it enable completion (적용할대상)
저는 Git을 사용 하게위해 아래와 같이 입력 해 보겠습니다.
bash-it enable completion git
```

자동 완성 기능을 적용 하고 나서 아래와 같이 일부 명령어만 작성 후 탭키를 두번 누르면 실행이 가능한 명령어에 대해 알려 줍니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/bash/13.png" alt="13">

지금 까지 Gti Bash Cutom Theme 적용 방법에 대한 내용에 대해 알아 봤습니다. (참고로 리눅스에도 동일하게 적용 가능합니다.)
