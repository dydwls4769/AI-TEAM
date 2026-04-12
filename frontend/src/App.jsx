import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import BlogPage from './pages/BlogPage';
import BlogDetail from './pages/BlogDetail';
import MapPage from './pages/MapPage';
import BioScanPage from './pages/BioScanPage'; // 이름 변경 반영


function App() {
  return (
    <Router>
      <div style={styles.container}>
        {/* 콘텐츠 영역 */}
        <div style={styles.content}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/biology" element={<BioScanPage />} />
            <Route path="/blog" element={<BlogPage />} />
            <Route path="/blog/:id" element={<BlogDetail />} /> {/* 상세 페이지 추가 */}
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>

        {/* 하단 네비게이션 바 */}
        <nav style={styles.navBar}>
          <Link to="/blog" style={styles.navItem}>📝<br/>블로그</Link>
          <Link to="/map" style={styles.navItem}>🗺️<br/>지도</Link>
          <Link to="/biology" style={styles.navItem}>📷<br/>이미지</Link>
          <Link to="/" style={styles.navItem}>🏠<br/>기타</Link>
          <Link to="/settings" style={styles.navItem}>⚙️<br/>설정</Link>
        </nav>
      </div>
    </Router>
  );
}

// 간단한 페이지 컴포넌트들 (나중에 파일로 분리하세요!)
const Home = () => <div style={styles.page}>기타/메인 화면</div>;
// const Blog = () => <div style={styles.page}>블로그 화면</div>;
const Settings = () => <div style={styles.page}>설정 화면</div>;

// 스타일 설정 (모바일 앱 느낌)
const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    // overflow: 'hidden', // 혹시 모르니 이 줄은 그대로 두시거나 살짝 주석처리해보세요.
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    paddingBottom: '80px', // 네비바 높이보다 살짝 더 줘야 마지막 카드가 안 잘려요.
  },
  navBar: {
    position: 'fixed',
    bottom: 0,
    left: 0,           // 👈 화면 왼쪽 끝에 딱 붙임
    right: 0,          // 👈 화면 오른쪽 끝에 딱 붙임
    height: '70px',
    backgroundColor: '#ffffff',
    display: 'flex',
    justifyContent: 'space-around',
    alignItems: 'center',
    borderTop: '1px solid #ddd',
    boxShadow: '0 -2px 5px rgba(0,0,0,0.1)',
    zIndex: 1000,
  },
  navItem: {
    flex: 1,           // 👈 네비 버튼들이 가로 길이를 똑같이 나눠 갖게 함
    textDecoration: 'none',
    color: '#333',
    fontSize: '12px',
    textAlign: 'center',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  page: {
    padding: '20px',
    textAlign: 'center',
  }
};

export default App;