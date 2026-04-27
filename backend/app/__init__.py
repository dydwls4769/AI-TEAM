import os
import sys


def _setup_ascii_mediapipe_path():
    """Windows에서 site-packages 경로에 한글이 포함되면 mediapipe .binarypb 로딩 실패.
    ASCII 경로로 디렉토리 정션을 만들고 sys.path 앞에 끼움. mediapipe import 전에 실행되어야 함.
    """
    if sys.platform != "win32":
        return
    site_pkg = None
    for candidate in sys.path:
        if candidate.lower().endswith("site-packages") and os.path.isdir(candidate):
            try:
                candidate.encode("ascii")
            except UnicodeEncodeError:
                site_pkg = candidate
                break
    if site_pkg is None:
        return
    link = r"C:\mp_ascii_path"
    if not os.path.exists(link):
        import subprocess
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, site_pkg],
            check=False,
            capture_output=True,
        )
    if os.path.exists(link) and link not in sys.path:
        sys.path.insert(0, link)


_setup_ascii_mediapipe_path()
