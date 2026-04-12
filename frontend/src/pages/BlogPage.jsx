import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function BlogPage() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPosts = async () => {
      try {
        const currentHost = window.location.hostname;
        // 최신순 정렬은 백엔드에서 처리해주는 것이 가장 좋지만, 
        // 여기서도 한번 더 확인하도록 호출합니다.
        const response = await axios.get(`http://${currentHost}:8001/api/observations`); 
        setPosts(response.data);
      } catch (error) {
        console.error("데이터 로딩 실패:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchPosts();
  }, []);

  if (loading) return <div style={styles.loading}>탐사 기록을 불러오는 중...</div>;

  return (
    <div style={styles.container}>
      {/* 상단 헤더 */}
      <div style={styles.header}>
        <span style={styles.headerTitle}>생물 탐사 피드</span>
      </div>

      {/* 3열 그리드 피드 */}
      <div style={styles.gridContainer}>
        {posts.length > 0 ? (
          posts.map((post) => (
            <div 
              key={post.id} 
              style={styles.gridItem} 
              onClick={() => navigate(`/blog/${post.id}`)}
            >
              <img 
                src={`http://${window.location.hostname}:8001${post.image_url}`} 
                alt={post.species_name} 
                style={styles.image}
              />
            </div>
          ))
        ) : (
          <div style={styles.noData}>아직 등록된 기록이 없습니다.</div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    backgroundColor: '#fff',
    minHeight: '100vh',
    paddingBottom: '70px', // 하단 네비게이션 바 공간 확보
  },
  header: {
    height: '50px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderBottom: '1px solid #efefef',
    position: 'sticky',
    top: 0,
    backgroundColor: '#fff',
    zIndex: 10,
  },
  headerTitle: {
    fontWeight: 'bold',
    fontSize: '16px',
    color: '#333',
  },
  gridContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)', // 한 줄에 3개씩
    gap: '2px', // 인스타그램 스타일의 미세한 여백
  },
  gridItem: {
    aspectRatio: '1 / 1', // 정사각형 유지
    overflow: 'hidden',
    backgroundColor: '#f0f0f0',
    cursor: 'pointer',
  },
  image: {
    width: '100%',
    height: '100%',
    objectFit: 'cover', // 찌그러짐 방지
    display: 'block',
  },
  loading: {
    textAlign: 'center',
    marginTop: '50px',
    color: '#999',
    fontSize: '14px',
  },
  noData: {
    gridColumn: 'span 3',
    textAlign: 'center',
    marginTop: '100px',
    color: '#999',
  }
};

export default BlogPage;