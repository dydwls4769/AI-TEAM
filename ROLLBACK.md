# 롤백 가이드

만약 FastAPI 마이그레이션이 실패해서 **Streamlit 원본 상태로 돌려야 할 때** 사용하세요.

---

## 시나리오 A. "원본 폴더가 망가졌다"고 의심될 때

원본 폴더(`AI_Exercise_Pose_Feedback-main/`) 자체는 본 작업 중 단 한 번도 수정되지 않았습니다.
의심된다면 백업본과 비교해서 차이가 있는지 먼저 확인:

```bash
# Windows PowerShell
$src = "C:\Users\dydwl\OneDrive\바탕 화면\AI_Exercise_Pose_Feedback-main"
$bak = "C:\Users\dydwl\OneDrive\바탕 화면\AI_Exercise_Pose_Feedback-BACKUP_2026-04-27"
robocopy $src $bak /L /E /NJH /NJS /XD venv __pycache__
# 출력에 "Newer" / "Mismatch" 가 없으면 원본 = 백업
```

차이가 있다면:
```powershell
# 백업본으로 원본 덮어쓰기 (venv는 보존)
robocopy $bak $src /E /XD venv
```

---

## 시나리오 B. "AI_Exercise_FullStack/ 폴더 자체를 없애고 싶다"

```powershell
Remove-Item -Recurse -Force "C:\Users\dydwl\OneDrive\바탕 화면\AI_Exercise_FullStack"
```

원본 폴더는 영향받지 않습니다.

---

## 시나리오 C. "GitHub의 dydwls4769/AI-TEAM 을 마이그레이션 이전으로 되돌리고 싶다"

GitHub 백업 브랜치 `before-fastapi-migration` 을 이용:

```bash
git clone https://github.com/dydwls4769/AI-TEAM.git
cd AI-TEAM
git fetch origin before-fastapi-migration
git checkout main
git reset --hard origin/before-fastapi-migration
git push --force-with-lease origin main   # ⚠️ 팀원 합의 후에만!
```

> ⚠️ `--force` push 는 협업자에게 영향을 줍니다. 본인만 쓰는 저장소가 아니라면 `revert` 커밋 방식을 권장:
> ```bash
> git revert <마이그레이션-커밋-해시>..HEAD
> git push origin main
> ```

---

## 시나리오 D. "Streamlit 모드만 다시 띄우고 싶다"

마이그레이션과 무관하게 원본 Streamlit 앱은 항상 실행 가능:
```bash
cd "C:\Users\dydwl\OneDrive\바탕 화면\AI_Exercise_Pose_Feedback-main"
.\venv\Scripts\activate
streamlit run Streamlit_Upload.py
```

---

## 백업 자료 정리

| 자료 | 위치 | 손상 시 복구 가능? |
|---|---|---|
| 로컬 백업 폴더 | `바탕 화면/AI_Exercise_Pose_Feedback-BACKUP_2026-04-27/` | GitHub 원본에서 재다운로드 가능 |
| GitHub 백업 브랜치 | `dydwls4769/AI-TEAM` 저장소의 `before-fastapi-migration` 브랜치 | 본인 계정 외엔 못 지움 |
| 원본 저장소 | `PSLeon24/AI_Exercise_Pose_Feedback` (외부) | 가장 안전. 항상 fresh clone 가능 |

**최종 안전판:**
원본 코드 자체가 없어진다 해도 `git clone https://github.com/PSLeon24/AI_Exercise_Pose_Feedback.git` 한 줄로 항상 새로 받을 수 있음.

---

## 이 문서 자체를 잃어버렸을 때

본 마이그레이션 정보는 다음 위치에 분산 저장됨:
1. `AI_Exercise_FullStack/MIGRATION_LOG.md` — 코드 매핑 상세
2. `AI_Exercise_FullStack/ROLLBACK.md` — 이 파일
3. `AI_Exercise_FullStack/backend/README.md` — 실행 방법
4. GitHub `dydwls4769/AI-TEAM` 의 커밋 히스토리
