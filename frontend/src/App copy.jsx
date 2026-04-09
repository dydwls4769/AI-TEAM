import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

function App() {
  const [chargers, setChargers] = useState([]);
  const mapRef = useRef(null);

  useEffect(() => {
    // 1. 필요한 외부 자원 한꺼번에 로드
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css';
    document.head.appendChild(css);

    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js';
    script.async = true;
    script.onload = () => {
      console.log("클러스터 라이브러리 로드 완료!");
      // 데이터가 이미 있다면 바로 그리기 시도
      if (chargers.length > 0) renderClusters();
    };
    document.body.appendChild(script);

    fetch("http://localhost:8001/api/chargers")
      .then(res => res.json())
      .then(data => {
        console.log("데이터 3219개 수신 완료!");
        setChargers(data);
      });
  }, []);

  const renderClusters = () => {
    const map = mapRef.current;
    if (!map || !window.L.markerClusterGroup || chargers.length === 0) return;

    // 기존에 그려진 클러스터가 있다면 싹 지우기
    map.eachLayer((layer) => {
      if (layer._group && layer instanceof L.Marker) map.removeLayer(layer);
      if (layer instanceof L.MarkerClusterGroup) map.removeLayer(layer);
    });

    const mg = new window.L.MarkerClusterGroup({
      chunkedLoading: true,
      maxClusterRadius: 50
    });

    chargers.forEach(chg => {
      if (chg.lat && chg.lng) {
        const marker = window.L.marker([Number(chg.lat), Number(chg.lng)])
          .bindTooltip(`<b>${chg.statNm}</b>`);
        mg.addLayer(marker);
      }
    });

    map.addLayer(mg);
    console.log("클러스터가 지도에 표시되었습니다!");
  };

  // 데이터나 라이브러리가 준비되면 실행
  useEffect(() => {
    const timer = setTimeout(() => renderClusters(), 500);
    return () => clearTimeout(timer);
  }, [chargers]);

  return (
    <div style={{ height: "100vh", width: "100vw" }}>
      <MapContainer 
        center={[35.1595, 126.8526]} 
        zoom={12} 
        style={{ height: "100%", width: "100%" }}
        ref={mapRef}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      </MapContainer>
    </div>
  );
}

export default App;
