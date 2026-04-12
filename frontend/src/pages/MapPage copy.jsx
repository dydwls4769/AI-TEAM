import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

// 마커 아이콘 설정
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

function MapPage() {
  const [observations, setObservations] = useState([]);
  const mapRef = useRef(null);
  const [isLibLoaded, setIsLibLoaded] = useState(false);

  useEffect(() => {
    // 1. 클러스터 라이브러리 CSS/JS 로드
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css';
    document.head.appendChild(css);

    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js';
    script.async = true;
    script.onload = () => setIsLibLoaded(true);
    document.body.appendChild(script);

    // 2. 데이터 가져오기
    const currentHost = window.location.hostname;
    fetch(`http://${currentHost}:8001/api/observations`)
      .then(res => res.json())
      .then(data => {
        console.log("가져온 데이터:", data); // 데이터가 잘 오는지 확인용
        setObservations(data);
      })
      .catch(err => console.error("데이터 로드 실패:", err));
  }, []);

  const renderClusters = () => {
    const map = mapRef.current;
    if (!map || !window.L || !window.L.markerClusterGroup || observations.length === 0) return;

    // 기존 레이어 청소
    map.eachLayer((layer) => {
      if (layer instanceof L.MarkerClusterGroup) map.removeLayer(layer);
    });

    const mg = new window.L.MarkerClusterGroup();

    observations.forEach(obs => {
      // 위치 정보가 유효한지 꼼꼼하게 체크
      const lat = parseFloat(obs.lat);
      const lng = parseFloat(obs.lng);
      
      if (!isNaN(lat) && !isNaN(lng) && lat !== 0) {
        const popupContent = `
          <div style="text-align: center; min-width: 150px;">
            <img src="http://${window.location.hostname}:8001${obs.image_url}" 
                 style="width: 100%; border-radius: 8px; margin-bottom: 8px;" />
            <br/>
            <strong style="font-size: 16px; color: #2e7d32;">${obs.species_name}</strong><br/>
            <span style="font-size: 12px; color: #666;">발견자: ${obs.author}</span>
          </div>
        `;
        const marker = L.marker([lat, lng]).bindPopup(popupContent);
        mg.addLayer(marker);
      }
    });

    map.addLayer(mg);
  };

  useEffect(() => {
    renderClusters();
  }, [observations, isLibLoaded]);

  return (
    <div style={{ height: "calc(100vh - 70px)", width: "100%" }}> 
      <MapContainer 
        center={[35.1595, 126.8526]} // 광주광역시 시청 중심
        zoom={13} 
        style={{ height: "100%", width: "100%" }}
        ref={mapRef}
      >
        {/* <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" /> */}
        <TileLayer 
          // url="https://api.vworld.kr/req/wmts/1.0.0/자신의API키/Base/{z}/{y}/{x}.png" 
          // 키가 없다면 일단 아래 오픈 소스용 주소를 써보세요.
          url="https://xdworld.vworld.kr/2d/Base/service/{z}/{x}/{y}.png"
        />
      </MapContainer>
    </div>
  );
}

export default MapPage;