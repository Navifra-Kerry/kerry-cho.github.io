---
title: "Windows terminal"
use_math: true
tags: [Windows]
header:
---


안녕하세요. 조대희 입니다.
블로그 방문을 환영 합니다.

이번에 소개 해드릴 내용은 2019년 5월 발표 된 Windows terminal에 관한 내용을 설명 드리겠습니다.

Windows termianal은 Windows의 환경 내에서 Command Line Interface의 사용자 환경을 개선 하기 위해 발표 되었습니다.  


[![Video Label](https://img.youtube.com/vi/8gw0rXPMMPE/0.jpg)](https://www.youtube.com/watch?v=8gw0rXPMMPE)


저는 최근에 Console로 업무를 진행 하는 경우가 많은데 Windows의 Builtin Console을 사용 하지 않고 
Git Bash를 사용해서 많은 일을 진행 했습니다. 

그 이유는 Git Bash의 경우 Custom이 가능 하고 자동 완성, vim 등 Console 작업을 하는데 도움을 주는 소소한 유틸들 때문이 였습니다. 

Windows Terminal의 경우 위에서 말한 모든 내용들이 가능하고 더욱 강력 한 기능들을 포함 하고 있습니다. (물론 파워쉘에서...)  

그 사용 방법들을 공유 하고자 합니다.
오늘 공유 드릴 내용은 아래와 같습니다.

- [Windows terminal 설치](#Windows-terminal-설치)
- [Windows terminal custom theme 설정](#Windows-terminal-custom-theme-설정)
- [VS Code Default terminal 설정](#VS-Code-Default-terminal-설정)
- [Visual Studio terminal 설정](#Visual-Studio-terminal-설정)

# Windows terminal 설치
Windows terminaldml 경우 Microsoft 앱스토에서 무료로 설치가 가능 합니다.  
설치를 하기 위해서는 Microsoft의 계정이 필요합니다.  

아래의 링크를 접속 하여 설치를 진행 해주세요.
[설치하기](https://www.microsoft.com/ko-kr/p/windows-terminal/9n0dx20hk701?activetab=pivot:overviewtab)

# Windows terminal custom theme 설정
Custom Theme의 경우 Powershell을 통해 설정이 가능 한데요 Powershell의 경우 보완 관련 설정을 해주지 않을 경우 사용 하지 못하는 명령어가 있습니다. 

아래의 명령을 실행 해 현재 사용자의 원격 다운로드 권한을 제한을 해제 합니다.  


```powershell
Set-ExecutionPolicy RemoteSigned
```


다음으로 Git 과 oh-my-posh 라는 Nuget Module을 설치 합니다.  
 _Windows는 패키지 매니저가 너무 많아 혼란 스럽긴 합니다._  



 ```powershell
Install-Module posh-git -Scope CurrentUser
Install-Module oh-my-posh -Scope CurrentUser
 ```

사용자 지정 Theme를 사용 하기 위한 Module을 설치 합니다.  



 ```powershell
 Install-Module -Name PSReadLine -Scope CurrentUser -Force -SkipPublisherCheck
 ```


 모든 모듈의 설치가 완료 되면 PowerShell profile Script를 작성합니다.
 PowerShell profile Script란 PowerShell이 처음 실행 될 때 한번 실행 되는 Script로 Custom Theme를 설정 하기 위한 명령어를 작성해 놓습니다.  



 ```powershell
 아래의 명령어 실행 후 노트 패드에 스크립트를 작성해 넣습니다.
 notepad $PROFILE

Import-Module posh-git
Import-Module oh-my-posh
Set-Theme Paradox
 ```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/script.gif" alt="script">

 Powerline Theme를 사용 하기 위해 Cascadia Font가 필요 합니다.  
 해당 Font의 경우 Windows Github [저장소](https://github.com/microsoft/cascadia-code/releases)에서 다운 받을 수 있습니다.

 최신 릴리즈를 다운 받은 후 아래의 모든 파일을 C:\Windows\Font 경로에 복사 해주세요.
 폴더째 복사 하는것이 아니라 파일만 옴기도록 합니다. 

 복사가 완료 되면 Windows terminal 사용자 설정 파일을 수정해 Theme를 적용 합니다.

 Windows termianl의 Setting.json 파일에서 
 Powershell 항목을 아래와 같이 변경 합니다. 

 ```json
 {
    // Make changes here to the powershell.exe profile.
    "guid": "{61c54bbd-c2c6-5271-96e7-009a87ff44bf}",
    "name": "Windows PowerShell",
    "commandline": "powershell.exe",
    "fontFace": "Cascadia Code PL",
    "hidden": false
},
 ```
 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/setting.gif" alt="setting">

적용이 완료 되면 아래와 같은 모습을 볼 수 있습니다.

 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/result.gif" alt="result">

# VS Code Default terminal 설정
VS Code에 Custom termianl을 적용 하게 위해서는 Git Bash 적용 방법과 비슷합니다. Powshell에서만 사용 가능 합니다.
- WSL 시스템의 Linux console 적용이 가능 합니다. 

VSCode를 실행 하고 나서 F1 -> Default terminal을 Windows Powershell로 설정 합니다.  
그 다음 terminal에서 사용 할 글꼴을 설정 해줍니다.

 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/default.gif" alt="default">


아래의 이미지를 참고하여 사용자 설정 .json파일에 아래의 내용을 추가 해주세요.

```json
"terminal.integrated.fontFamily": "Cascadia Code PL",
```

 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/fonts.gif" alt="fonts">

모든 적용이 완료 되면 아래와 같은 모습을 볼 수 있습니다.

 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/vscoderesult.gif" alt="vscoderesult">

# Visual Studio terminal 설정

Visual Studio 경우 2019 버전 [16.3](https://devblogs.microsoft.com/visualstudio/say-hello-to-the-new-visual-studio-terminal/)업데이트 이상이 설치 되어 있어야 합니다.

```
도구-> 환경 -> 글꼴 및 색 -> Terminal 에서
"Cascadia Code PL" 글꼴로 선택 하세요.
```

 <img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/visualsetting.gif" alt="visualsetting">


 적용이 완료 되면 Visual Studio 에서도 아래와 같이 Custom Console창을 사용 가능 합니다.

  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/windowsterminal/visualresult.gif" alt="visualresult">

  지금 까지 Windows terminal에 Custom theme르 적용 하는 방법에 대해 알아 보았습니다. 
