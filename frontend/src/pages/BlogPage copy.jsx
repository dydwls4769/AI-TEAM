import React, { useEffect, useState } from 'react';
import axios from 'axios';

function BlogPage() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPosts = async () => {
      try {
        const currentHost = window.location.hostname;
        // 백엔드 routers/blog.py에서 목록 조회 API가 있다고 가정
        const response = await axios.get(`http://${currentHost}:8001/blog/`);
        setPosts(response.data);
      } catch (error) {
        console.error("데이터 로딩 실패:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchPosts();
  }, []);

  if (loading) return <div style={styles.loading}>도감을 불러오는 중... 🌿</div>;

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h2 style={styles.title}>생물도감 피드</h2>
      </header>

      <div style={styles.feed}>
        {posts.length === 0 ? (
          <p style={styles.empty}>아직 등록된 도감이 없어요. 첫 발견자가 되어보세요!</p>
        ) : (
          posts.map((post) => (
            <div key={post.id} style={styles.card}>
              {/* 유저 정보 영역 */}
              <div style={styles.userInfo}>
                <div style={styles.avatar}>🌱</div>
                <span style={styles.userName}>{post.nickname || "익명의 발견자"}</span>
              </div>

              {/* 이미지 영역 */}
              <img 
                src={`http://${window.location.hostname}:8001${post.image_url}`} 
                alt={post.species_name} 
                style={styles.image} 
              />

              {/* 콘텐츠 영역 */}
              <div style={styles.content}>
                <div style={styles.speciesBadge}>#{post.species_name}</div>
                <p style={styles.location}>📍 {post.location_name || "관찰 위치"}</p>
                <p style={styles.date}>{new Date(post.created_at).toLocaleDateString()}</p>
                
                {/* AI 보정 이력이 있다면 표시 (추후 확장용) */}
                {post.has_ai_log && (
                  <span style={styles.aiTag}>✨ AI 고화질 보정됨</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    backgroundColor: '#fafafa',
    minHeight: '100vh',
  },
  header: {
    padding: '15px',
    backgroundColor: '#fff',
    borderBottom: '1px solid #efefef',
    position: 'sticky',
    top: 0,
    zIndex: 10,
    textAlign: 'center',
  },
  title: { fontSize: '18px', margin: 0, color: '#2e7d32' },
  feed: { padding: '10px' },
  card: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    marginBottom: '20px',
    border: '1px solid #efefef',
    overflow: 'hidden',
  },
  userInfo: {
    display: 'flex',
    alignItems: 'center',
    padding: '12px',
  },
  avatar: { width: '30px', height: '30px', marginRight: '10px', fontSize: '20px' },
  userName: { fontWeight: '600', fontSize: '14px' },
  image: { width: '100%', aspectRatio: '1/1', objectFit: 'cover' },
  content: { padding: '15px' },
  speciesBadge: {
    display: 'inline-block',
    padding: '4px 10px',
    backgroundColor: '#e8f5e9',
    color: '#2e7d32',
    borderRadius: '20px',
    fontSize: '13px',
    fontWeight: 'bold',
    marginBottom: '8px',
  },
  location: { margin: '5px 0', fontSize: '13px', color: '#666' },
  date: { fontSize: '11px', color: '#999', marginTop: '5px' },
  aiTag: { fontSize: '11px', color: '#4caf50', fontWeight: 'bold' },
  loading: { textAlign: 'center', marginTop: '50px', color: '#666' },
  empty: { textAlign: 'center', marginTop: '50px', color: '#999' }
};

export default BlogPage;