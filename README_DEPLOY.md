# Tetris AI Web 제출 안내

이 폴더는 원본 Python/Pygame Tetris 프로젝트에 Render 배포용 정적 웹 버전을 추가한 제출본입니다.

## Render 설정

Render에서 `New > Static Site`를 선택한 뒤 GitHub 저장소를 연결합니다.

- Build Command: `echo "Static Tetris build complete"`
- Publish Directory: `public`
- Blueprint 사용 시: `render.yaml`이 자동으로 위 설정을 잡습니다.

Render 공식 문서 기준으로 Static Site는 Git 저장소를 연결하면 고유한 `onrender.com` 공개 URL을 제공합니다.

## GitHub 업로드

```powershell
git init
git add .
git commit -m "Submit Tetris web deployment"
git branch -M main
git remote add origin https://github.com/깃허브아이디/저장소이름.git
git push -u origin main
```

## 제출 파일

- 완성 코드 zip: `과제7_학번_송형륜.zip`
- 링크 txt: `공용웹주소_깃허브주소.txt`

`공용웹주소_깃허브주소.txt`에는 Render 배포가 끝난 뒤 받은 주소와 GitHub 저장소 주소를 넣으면 됩니다.

