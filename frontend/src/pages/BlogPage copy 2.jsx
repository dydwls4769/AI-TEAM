import React, { useState, useEffect } from 'react';
import axios from 'axios';

function BlogPage() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPost, setSelectedPost] = useState(null); // 클릭한 포스트 저장

  useEffect(() => {
    const fetchPosts = async () => {
      try {
        const currentHost = window.location.hostname;
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

  if (loading) return <div style={styles.loading}>로딩 중...</div>;

  return (
    <div style={styles.container}>
      <div style={styles.header}><span style={styles.headerTitle}>탐사 기록</span></div>

      {/* 그리드 피드 */}
      <div style={styles.gridContainer}>
        {posts.map((post) => (
          <div key={post.id} style={styles.gridItem} onClick={() => setSelectedPost(post)}>
            <img src={`http://${window.location.hostname}:8001${post.image_url}`} style={styles.image} />
          </div>
        ))}
      </div>

      {/* 상세 보기 모달 (포스트를 클릭했을 때만 나타남) */}
      {selectedPost && (
        <div style={styles.modalOverlay} onClick={() => setSelectedPost(null)}>
          <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            {/* 닫기 버튼 */}
            <button style={styles.closeBtn} onClick={() => setSelectedPost(null)}>✕</button>
            
            {/* 이미지 섹션 */}
            <div style={styles.modalImageWrapper}>
              <img 
                src={`http://${window.location.hostname}:8001${selectedPost.image_url}`} 
                style={styles.modalImage} 
              />
            </div>

            {/* 정보 및 커뮤니티 섹션 (여기에 나중에 하트, 댓글, 필터 버튼이 들어감) */}
            <div style={styles.modalInfo}>
              <h3>{selectedPost.species_name}</h3>
              <p>발견자: {selectedPost.nickname}</p>
              <hr style={{ border: '0.5px solid #efefef' }} />
              
              {/* 여기에 버튼들이 들어갈 자리! */}
              <div style={styles.actionButtons}>
                <span>❤️ 0</span> <span>💬 0</span> <span>🎨 필터</span>
              </div>
              
              <div style={styles.commentSpace}>
                <p style={{ color: '#999', fontSize: '12px' }}>아직 댓글이 없습니다.</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    backgroundColor: '#fff',
    minHeight: '100vh',
    paddingBottom: '70px', // 네비바 공간
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
  },
  gridContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)', // 👈 핵심: 한 줄에 3개
    gap: '2px', // 사진 사이의 얇은 실선 여백
  },
  gridItem: {
    aspectRatio: '1 / 1', // 👈 핵심: 정사각형 유지
    overflow: 'hidden',
    backgroundColor: '#f0f0f0',
  },
  image: {
    width: '100%',
    height: '100%',
    objectFit: 'cover', // 👈 핵심: 비율 깨지지 않게 꽉 채움
    display: 'block',
    cursor: 'pointer',
  },
  loading: {
    textAlign: 'center',
    marginTop: '50px',
    color: '#999',
  },
  modalOverlay: {
    position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
    backgroundColor: 'rgba(0,0,0,0.8)', zIndex: 2000,
    display: 'flex', justifyContent: 'center', alignItems: 'center'
  },
  modalContent: {
    backgroundColor: '#fff', width: '90%', maxWidth: '500px',
    borderRadius: '15px', overflow: 'hidden', position: 'relative'
  },
  closeBtn: {
    position: 'absolute', top: '10px', right: '15px', background: 'none',
    border: 'none', fontSize: '20px', cursor: 'pointer', zIndex: 10
  },
  modalImageWrapper: { width: '100%', aspectRatio: '1/1', backgroundColor: '#000' },
  modalImage: { width: '100%', height: '100%', objectFit: 'contain' },
  modalInfo: { padding: '15px' },
  actionButtons: { display: 'flex', gap: '15px', marginBottom: '10px', fontSize: '18px' },
  commentSpace: { marginTop: '10px', minHeight: '50px' }
};

export default BlogPage;