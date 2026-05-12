# 운동 루틴 앱 (5×5 / nSuns) — 실행 가이드

`Streamlit_Routine.py`는 메인 리프트 1RM 기반으로 5×5, nSuns 루틴의 세트/무게를 자동 계산하고 진행 상황을 기록하는 Streamlit 앱입니다.

새 컴퓨터(아무것도 안 깔린 Windows)에서 처음 실행하는 절차입니다.

---

## 1. Python 설치

1. https://www.python.org/downloads/ 접속
2. **Download Python 3.12.x** (또는 3.11) 클릭
3. 설치파일 실행 시 **첫 화면 하단의 "Add python.exe to PATH" 반드시 체크**
4. `Install Now` 클릭
5. 설치 후 PowerShell 새 창에서 확인:
   ```powershell
   python --version
   ```

---

## 2. 저장소 ZIP 다운로드

1. https://github.com/dydwls4769/AI-TEAM 접속
2. 우측 상단 녹색 **`<> Code`** 버튼 클릭
3. **`Download ZIP`** 클릭
4. 다운받은 `AI-TEAM-main.zip` 압축 풀기
5. 풀린 `AI-TEAM-main` 폴더를 **바탕화면**으로 이동

---

## 3. 라이브러리 설치 & 실행

PowerShell 열고:

```powershell
cd "$env:USERPROFILE\Desktop\AI-TEAM-main\routine"
pip install streamlit pandas plotly
streamlit run Streamlit_Routine.py
```

브라우저가 자동으로 열리며 `http://localhost:8501` 에서 앱이 뜹니다.

종료: PowerShell에서 `Ctrl + C`.

---

## 4. `pip` / `streamlit` 명령이 안 먹힐 때

Windows 앱 실행 별칭 문제 또는 PATH 이슈입니다. `python -m`으로 우회:

```powershell
python -m pip install streamlit pandas plotly
python -m streamlit run Streamlit_Routine.py
```

---

## 5. 기존 운동기록 옮기기 (선택)

GitHub에는 `data/user_data.json`이 **포함되어 있지 않습니다** (개인 운동기록 보호).

**새로 시작해도 된다면** → 이 단계 건너뛰기. 앱 첫 실행 시 1RM 다시 입력하면 됩니다.

**기존 기록을 유지하려면**:

1. 기존 컴퓨터의 `user_data.json`을 USB / OneDrive / 카톡 등으로 옮기기
2. 새 컴퓨터의 `AI-TEAM-main\routine\` 안에 `data` 폴더 만들기:
   ```powershell
   mkdir "$env:USERPROFILE\Desktop\AI-TEAM-main\routine\data"
   ```
3. 옮긴 `user_data.json`을 그 안에 붙여넣기

최종 경로:
```
AI-TEAM-main\routine\data\user_data.json
```

---

## 6. 두 번째 실행부터

라이브러리 설치는 **한 번만** 하면 됩니다. 이후로는 PowerShell 새 창에서:

```powershell
cd "$env:USERPROFILE\Desktop\AI-TEAM-main\routine"
streamlit run Streamlit_Routine.py
```

---

## 7. 더블클릭으로 실행 (선택)

`routine` 폴더 안에 메모장으로 `run.bat` 파일 만들기:

```bat
@echo off
cd /d "%~dp0"
streamlit run Streamlit_Routine.py
pause
```

> 메모장 저장 시 **파일 형식**을 "모든 파일(*.*)"로 바꾸고 파일명 `run.bat` 입력 (그래야 `.txt`가 안 붙음)

이후 `run.bat` 더블클릭만으로 실행됩니다.

---

## 자주 막히는 부분

| 증상 | 해결 |
|---|---|
| `python` 명령 못 찾음 | 1단계 "Add to PATH" 체크 안 함 → 파이썬 재설치 |
| `cd ...routine` 에서 "경로 없음" | ZIP 압축 안 풀었거나, 중첩 폴더 → `dir`로 확인하며 한 단계 더 들어가기 |
| `pip` "액세스 거부" | `python -m pip install ...` 로 우회 |
| `streamlit` 명령 못 찾음 | `python -m streamlit run Streamlit_Routine.py` |
| 브라우저 자동으로 안 열림 | PowerShell의 `Local URL: http://localhost:8501` 주소 직접 복사해서 브라우저에 붙여넣기 |
| 한글 경로 오류 | `C:\Projects\` 같은 영문 경로로 폴더 옮기기 |
| 앱은 떴는데 기록이 비어있음 | 정상. 5단계 진행하거나 1RM 새로 입력 |

---

## 코드 업데이트 시 (ZIP 방식)

`git pull`이 없으므로 다시 ZIP 받아야 합니다:

1. 새 ZIP 다운로드
2. **`routine\data\user_data.json` 먼저 백업**
3. 기존 `AI-TEAM-main` 폴더 삭제
4. 새 ZIP 압축 풀어 교체
5. 백업한 `user_data.json` 다시 `routine\data\` 안에 넣기

---

## 사용 라이브러리

- `streamlit` — 웹 UI
- `pandas` — 데이터 처리
- `plotly` — 그래프

(`json`, `math`, `os`, `datetime`은 파이썬 기본 내장)
