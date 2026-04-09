import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

function MapPage() { // 1. 이름을 MapPage로 변경
  const [chargers, setChargers] = useState([]);
  const mapRef = useRef(null);

  useEffect(() => {
    // 라이브러리 로드 및 데이터 페치 로직 (기존과 동일)
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css';
    document.head.appendChild(css);

    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js';
    script.async = true;
    script.onload = () => {
      if (chargers.length > 0) renderClusters();
    };
    document.body.appendChild(script);

    // 백엔드 주소 확인 (port 8001)
    fetch("http://localhost:8001/api/chargers")
      .then(res => res.json())
      .then(data => {
        setChargers(data);
      });
  }, []);

  const renderClusters = () => {
    const map = mapRef.current;
    if (!map || !window.L.markerClusterGroup || chargers.length === 0) return;

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
  };

  useEffect(() => {
    const timer = setTimeout(() => renderClusters(), 500);
    return () => clearTimeout(timer);
  }, [chargers]);

  return (
    // 상단 네비게이션 바가 들어갈 자리를 고려해서 height를 조정할 수도 있습니다.
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

export default MapPage; // 2. 내보내기 이름 변경