import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

function MapPage() {
  const [chargers, setChargers] = useState([]);
  const mapRef = useRef(null);
  const [isLibLoaded, setIsLibLoaded] = useState(false); // 라이브러리 로딩 상태 추가

  useEffect(() => {
    // 1. CSS 로드
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css';
    document.head.appendChild(css);

    // 2. JS 로드
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js';
    script.async = true;
    script.onload = () => setIsLibLoaded(true); // 로딩 완료 표시
    document.body.appendChild(script);

    // 3. 데이터 가져오기 (localhost 대신 현재 호스트 주소 사용)
    const currentHost = window.location.hostname; // PC에선 localhost, 폰에선 IP주소를 자동으로 잡음
    fetch(`http://${currentHost}:8001/api/chargers`)
      .then(res => res.json())
      .then(data => setChargers(data))
      .catch(err => console.error("데이터 로드 실패:", err));
  }, []);

  const renderClusters = () => {
    const map = mapRef.current;
    // window.L 이나 markerClusterGroup이 없으면 대기
    if (!map || !window.L || !window.L.markerClusterGroup || chargers.length === 0) return;

    // 기존 레이어 정리
    map.eachLayer((layer) => {
      if (layer instanceof L.MarkerClusterGroup) map.removeLayer(layer);
    });

    const mg = new window.L.MarkerClusterGroup({
      chunkedLoading: true,
      maxClusterRadius: 50
    });

    chargers.forEach(chg => {
      const lat = Number(chg.lat);
      const lng = Number(chg.lng);
      if (!isNaN(lat) && !isNaN(lng)) {
        const marker = L.marker([lat, lng])
          .bindTooltip(`<b>${chg.statNm || '충전소'}</b>`);
        mg.addLayer(marker);
      }
    });

    map.addLayer(mg);
  };

  // 데이터가 오거나 라이브러리가 로드되면 다시 그리기
  useEffect(() => {
    renderClusters();
  }, [chargers, isLibLoaded]);

  return (
    <div style={{ height: "calc(100vh - 60px)", width: "100%" }}> 
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

export default MapPage;