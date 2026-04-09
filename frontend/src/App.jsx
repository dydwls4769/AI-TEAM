import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
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
            <Route path="/blog" element={<Blog />} />
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
const Blog = () => <div style={styles.page}>블로그 화면</div>;
const Settings = () => <div style={styles.page}>설정 화면</div>;

// 스타일 설정 (모바일 앱 느낌)
const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    overflow: 'hidden',
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    paddingBottom: '70px', // 네비게이션 높이만큼 여백
  },
  navBar: {
    position: 'fixed',
    bottom: 0,
    width: '100%',
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
    textDecoration: 'none',
    color: '#333',
    fontSize: '12px',
    textAlign: 'center',
  },
  page: {
    padding: '20px',
    textAlign: 'center',
  }
};

export default App;