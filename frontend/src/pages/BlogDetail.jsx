import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

function BlogDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [post, setPost] = useState(null);
  const [filter, setFilter] = useState('none'); 

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        const currentHost = window.location.hostname;
        // 💡 백엔드 라우터 구조에 맞춰 /api/observations/${id} 로 호출
        const res = await axios.get(`http://${currentHost}:8001/api/observations`);
        // 가져온 리스트 중 ID가 일치하는 것만 찾기 (상세 API가 따로 없을 경우를 대비)
        const target = res.data.find(item => item.id === parseInt(id));
        setPost(target);
      } catch (err) {
        console.error("상세 데이터 로드 실패:", err);
      }
    };
    fetchDetail();
  }, [id]);

  if (!post) return <div style={{padding: '20px', textAlign: 'center'}}>데이터를 찾는 중...</div>;

  return (
    <div style={styles.container}>
      {/* 상단바 */}
      <div style={styles.topBar}>
        <button onClick={() => navigate(-1)} style={styles.backBtn}>← 뒤로</button>
        <span style={styles.title}>{post.species_name}</span>
      </div>

      {/* 이미지 영역 */}
      <div style={styles.imageWrapper}>
        <img 
          src={`http://${window.location.hostname}:8001${post.image_url}`} 
          style={{ ...styles.image, filter: filter }} 
          alt="탐사사진"
        />
      </div>

      {/* 정보 영역 */}
      <div style={styles.infoSection}>
        <div style={styles.actions}>
          <span style={{fontSize: '20px'}}>❤️ 0</span>
          <span style={{fontSize: '20px', marginLeft: '10px'}}>💬 0</span>
          <button 
            onClick={() => setFilter(filter === 'none' ? 'grayscale(100%)' : 'none')}
            style={styles.filterBtn}
          >
            🎨 필터 토글
          </button>
        </div>
        
        <div style={styles.textContent}>
          <p><strong>발견자:</strong> {post.author}</p>
          <p><strong>발견 날짜:</strong> {post.created_at}</p>
          <hr style={styles.hr} />
          <p style={styles.desc}>이곳에 생물의 상세 설명이나 댓글이 들어갈 예정입니다.</p>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: { backgroundColor: '#fff', minHeight: '100vh' },
  topBar: { height: '50px', display: 'flex', alignItems: 'center', padding: '0 15px', borderBottom: '1px solid #eee' },
  backBtn: { fontSize: '16px', background: 'none', border: 'none', cursor: 'pointer' },
  title: { marginLeft: '15px', fontWeight: 'bold' },
  imageWrapper: { width: '100%', aspectRatio: '1/1', backgroundColor: '#f9f9f9', display: 'flex', alignItems: 'center' },
  image: { width: '100%', height: '100%', objectFit: 'contain', transition: 'filter 0.3s' },
  infoSection: { padding: '15px' },
  actions: { display: 'flex', alignItems: 'center', marginBottom: '15px' },
  filterBtn: { marginLeft: 'auto', padding: '5px 10px', borderRadius: '5px', border: '1px solid #ddd', backgroundColor: '#fff' },
  textContent: { lineHeight: '1.6' },
  hr: { border: '0', borderTop: '1px solid #eee', margin: '15px 0' },
  desc: { color: '#444' }
};

export default BlogDetail;