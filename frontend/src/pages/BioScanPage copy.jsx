import React, { useState } from 'react';
import axios from 'axios';

function BioScanPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // 1. 이미지 선택 핸들러
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile)); // 미리보기 생성
  };

  // 2. 서버로 이미지 전송 (AI 분석 요청)
  const handleUpload = async () => {
    if (!file) return alert("이미지를 선택해주세요!");

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // 백엔드 biology 라우터의 /predict 경로로 전송
      const response = await axios.post("http://localhost:8001/biology/predict", formData);
      setResult(response.data);
    } catch (error) {
      console.error("분석 실패:", error);
      alert("서버와 통신 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center', paddingBottom: '80px' }}>
      <h2>🌱 AI 생물 스캔</h2>
      <p>궁금한 동식물 사진을 찍어 올리세요!</p>

      {/* 이미지 업로드 영역 */}
      <input type="file" accept="image/*" onChange={handleFileChange} style={{ marginBottom: '20px' }} />
      
      {preview && (
        <div style={{ margin: '20px 0' }}>
          <img src={preview} alt="미리보기" style={{ maxWidth: '100%', borderRadius: '10px' }} />
        </div>
      )}

      <button 
        onClick={handleUpload} 
        disabled={loading}
        style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '5px' }}
      >
        {loading ? "분석 중..." : "AI 분석 시작"}
      </button>

      {/* 결과 출력 영역 */}
      {result && (
        <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f9f9f9', borderRadius: '10px', border: '1px solid #ddd' }}>
          <h3>결과: {result.prediction}</h3>
          <p><b>서식지:</b> {result.habitat}</p>
          <p><b>설명:</b> {result.description}</p>
        </div>
      )}
    </div>
  );
}

export default BioScanPage;