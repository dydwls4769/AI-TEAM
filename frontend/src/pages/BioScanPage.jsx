import React, { useState } from 'react';
import axios from 'axios';

function BioScanPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null); 
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setData(null); 
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return alert("사진을 선택하세요!");

    setLoading(true);
    // 📍 1. 현재 위치(GPS) 정보 가져오기
    let lat = null;
    let lng = null;

    try {
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject);
      });
      lat = position.coords.latitude;
      lng = position.coords.longitude;
      console.log("현재 위치:", lat, lng);
    } catch (err) {
      console.log("위치 정보를 가져올 수 없습니다. 기본 위치로 저장합니다.");
    }

    // 2. FormData에 사진과 함께 좌표 추가
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("lat", lat || 0); // 좌표가 없으면 0이라도 보냄
    formData.append("lng", lng || 0);

    try {
      // 💡 주소 수정: backend/routers/biology.py에서 설정한 경로로 보냅니다.
      // prefix가 /biology 이고, @router.post("/predict") 이므로 최종 주소는 /biology/predict 입니다.
      const currentHost = window.location.hostname;
      const backendUrl = `http://${currentHost}:8001/biology/predict`;

      console.log("요청 주소:", backendUrl);
      const response = await axios.post(backendUrl, formData);
      setData(response.data); 
    } catch (error) {
      console.error("에러 발생:", error);
      alert("분석에 실패했습니다. 백엔드 서버 상태를 확인하세요.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      padding: '20px', 
      textAlign: 'center', 
      fontFamily: 'sans-serif', 
      backgroundColor: '#f5f7f5', 
      minHeight: '100vh',
      paddingBottom: '100px' // 하단 네비게이션에 가려지지 않게 여백 추가
    }}>
      <h1 style={{ color: '#2e7d32', marginBottom: '10px' }}>🌳 우리동네 생물도감 🌳</h1>
      <p style={{ color: '#666', marginBottom: '30px' }}>우리동네 생물을 찍으면 AI가 알려줘요!</p>
      
      <div style={{ 
        backgroundColor: 'white', 
        padding: '20px', 
        borderRadius: '20px', 
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        maxWidth: '500px',
        margin: '0 auto'
      }}>
        {/* 파일 선택창을 예쁘게 만들기 위해 스타일 살짝 수정 */}
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange} 
          style={{ 
            marginBottom: '20px', 
            display: 'block', 
            width: '100%',
            padding: '10px',
            border: '2px dashed #4caf50',
            borderRadius: '10px',
            cursor: 'pointer'
          }} 
        />

        {preview && (
          <div>
            <img 
              src={preview} 
              alt="미리보기" 
              style={{ width: '100%', borderRadius: '15px', marginBottom: '20px', maxHeight: '300px', objectFit: 'cover' }} 
            />
            <button 
              onClick={handleSubmit} 
              disabled={loading} 
              style={{ 
                width: '100%',
                padding: '15px', 
                backgroundColor: loading ? '#ccc' : '#4caf50', 
                color: 'white', 
                border: 'none', 
                borderRadius: '12px', 
                fontSize: '18px', 
                fontWeight: 'bold',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: '0.3s'
              }}
            >
              {loading ? "🔍 AI 분석 중..." : "이 생물은 무엇인가요?"}
            </button>
          </div>
        )}
      </div>

      {data && (
        <div style={{ 
          maxWidth: '500px', 
          margin: '30px auto', 
          padding: '25px', 
          borderRadius: '20px', 
          backgroundColor: '#ffffff', 
          textAlign: 'left', 
          boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
          borderLeft: '8px solid #4caf50'
        }}>
          <h2 style={{ color: '#1b5e20', marginTop: 0, borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
            🔍 결과: {data.prediction}
          </h2>
          
          <div style={{ marginTop: '15px' }}>
            <p style={{ color: '#2e7d32', fontWeight: 'bold', marginBottom: '5px' }}>📍 서식지</p>
            <div style={{ backgroundColor: '#f9f9f9', padding: '12px', borderRadius: '10px', color: '#444' }}>
              {data.habitat}
            </div>
          </div>

          <div style={{ marginTop: '15px' }}>
            <p style={{ color: '#2e7d32', fontWeight: 'bold', marginBottom: '5px' }}>📝 상세 설명</p>
            <div style={{ backgroundColor: '#f9f9f9', padding: '12px', borderRadius: '10px', color: '#444', lineHeight: '1.6' }}>
              {data.description}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default BioScanPage;