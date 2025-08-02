/**
 * Consciousness Flow Visualizer - 3D Visualization
 * 意識の流れを美しく可視化するThree.jsベースのビジュアライゼーション
 */

class ConsciousnessVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.particles = null;
        this.flowLines = [];
        this.attentionFoci = [];
        this.isRunning = true;
        this.colorScheme = 'phenomenological';
        this.ws = null;
        this.stateHistory = [];
        this.maxHistoryLength = 1000;
        
        this.init();
        this.connectWebSocket();
        this.animate();
    }
    
    init() {
        // キャンバスの設定
        const canvas = document.getElementById('consciousness-canvas');
        const container = document.getElementById('canvas-container');
        
        // Three.jsの初期設定
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.FogExp2(0x0a0a0a, 0.0008);
        
        // カメラの設定
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 30, 50);
        this.camera.lookAt(0, 0, 0);
        
        // レンダラーの設定
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        // ライティング
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 1);
        pointLight.position.set(50, 50, 50);
        this.scene.add(pointLight);
        
        // パーティクルシステムの初期化
        this.createParticleSystem();
        
        // フローラインの初期化
        this.createFlowLines();
        
        // ウィンドウリサイズ対応
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    createParticleSystem() {
        const particleCount = 5000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);
        
        // 初期位置とプロパティを設定
        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;
            
            // 球体状に配置
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            const radius = 20 + Math.random() * 10;
            
            positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = radius * Math.cos(phi);
            
            // 初期色（青緑のグラデーション）
            colors[i3] = 0.0;
            colors[i3 + 1] = 0.5 + Math.random() * 0.5;
            colors[i3 + 2] = 0.8 + Math.random() * 0.2;
            
            // サイズ
            sizes[i] = Math.random() * 2;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // シェーダーマテリアル
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                integration: { value: 0.5 }
            },
            vertexShader: `
                attribute float size;
                attribute vec3 color;
                varying vec3 vColor;
                uniform float time;
                uniform float integration;
                
                void main() {
                    vColor = color;
                    vec3 pos = position;
                    
                    // 統合度に基づいて粒子を中心に引き寄せる
                    pos *= 1.0 - integration * 0.3;
                    
                    // 時間に基づく波動
                    pos.x += sin(time * 0.001 + position.y * 0.1) * 2.0;
                    pos.y += cos(time * 0.001 + position.z * 0.1) * 2.0;
                    pos.z += sin(time * 0.001 + position.x * 0.1) * 2.0;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z) * (0.5 + integration);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                
                void main() {
                    vec2 uv = gl_PointCoord.xy * 2.0 - 1.0;
                    float d = length(uv);
                    if (d > 1.0) discard;
                    
                    float alpha = 1.0 - smoothstep(0.0, 1.0, d);
                    gl_FragColor = vec4(vColor, alpha * 0.8);
                }
            `,
            blending: THREE.AdditiveBlending,
            depthTest: false,
            transparent: true,
            vertexColors: true
        });
        
        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }
    
    createFlowLines() {
        // フローラインを表現する曲線
        for (let i = 0; i < 20; i++) {
            const curve = new THREE.CatmullRomCurve3([
                new THREE.Vector3(
                    Math.random() * 40 - 20,
                    Math.random() * 40 - 20,
                    Math.random() * 40 - 20
                ),
                new THREE.Vector3(
                    Math.random() * 40 - 20,
                    Math.random() * 40 - 20,
                    Math.random() * 40 - 20
                ),
                new THREE.Vector3(
                    Math.random() * 40 - 20,
                    Math.random() * 40 - 20,
                    Math.random() * 40 - 20
                )
            ]);
            
            const points = curve.getPoints(50);
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            
            const material = new THREE.LineBasicMaterial({
                color: new THREE.Color(0.0, 0.8, 1.0),
                opacity: 0.3,
                transparent: true,
                blending: THREE.AdditiveBlending
            });
            
            const line = new THREE.Line(geometry, material);
            this.flowLines.push({ line, curve, offset: Math.random() * Math.PI * 2 });
            this.scene.add(line);
        }
    }
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket接続確立');
            this.updateConnectionStatus(true);
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket接続終了');
            this.updateConnectionStatus(false);
            // 再接続を試みる
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocketエラー:', error);
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'update':
                this.updateVisualization(message.data);
                break;
            case 'history':
                this.loadHistory(message.data);
                break;
            case 'analysis':
                this.updateAnalysis(message.data);
                break;
            case 'transitions':
                this.showTransitions(message.data);
                break;
        }
    }
    
    updateVisualization(state) {
        // 状態履歴に追加
        this.stateHistory.push(state);
        if (this.stateHistory.length > this.maxHistoryLength) {
            this.stateHistory.shift();
        }
        
        // パーティクルの更新
        if (this.particles) {
            this.particles.material.uniforms.integration.value = state.integration;
            
            // 注意の焦点に基づいて色を更新
            const colors = this.particles.geometry.attributes.color;
            const positions = this.particles.geometry.attributes.position;
            const attentionKeys = Object.keys(state.attention);
            
            for (let i = 0; i < colors.count; i++) {
                const i3 = i * 3;
                
                // 注意の焦点に近い粒子を明るくする
                let brightness = 0.3;
                attentionKeys.forEach((key, index) => {
                    const attentionStrength = state.attention[key];
                    const angle = (index / attentionKeys.length) * Math.PI * 2;
                    const focusX = Math.cos(angle) * 20;
                    const focusZ = Math.sin(angle) * 20;
                    
                    const dx = positions.array[i3] - focusX;
                    const dz = positions.array[i3 + 2] - focusZ;
                    const distance = Math.sqrt(dx * dx + dz * dz);
                    
                    brightness += attentionStrength * Math.exp(-distance * 0.05);
                });
                
                // 現象的性質に基づいて色相を調整
                const phenomenalKeys = Object.keys(state.phenomenal_properties);
                let hue = 0.5;
                phenomenalKeys.forEach(key => {
                    const value = state.phenomenal_properties[key];
                    if (key === 'clarity') hue += value * 0.1;
                    if (key === 'vividness') brightness *= (1 + value * 0.5);
                });
                
                // HSLからRGBに変換
                const color = this.hslToRgb(hue, 0.8, brightness);
                colors.array[i3] = color.r;
                colors.array[i3 + 1] = color.g;
                colors.array[i3 + 2] = color.b;
            }
            
            colors.needsUpdate = true;
        }
        
        // メトリクスの更新
        this.updateMetrics(state);
    }
    
    updateMetrics(state) {
        // 統合度
        const integrationBar = document.getElementById('integration-bar');
        const integrationValue = document.getElementById('integration-value');
        integrationBar.style.width = `${state.integration * 100}%`;
        integrationValue.textContent = state.integration.toFixed(2);
        
        // 認知負荷
        const cognitiveLoadBar = document.getElementById('cognitive-load-bar');
        const cognitiveLoadValue = document.getElementById('cognitive-load-value');
        cognitiveLoadBar.style.width = `${state.cognitive_load * 100}%`;
        cognitiveLoadValue.textContent = state.cognitive_load.toFixed(2);
        
        // メタ認知
        const metaAwarenessBar = document.getElementById('meta-awareness-bar');
        const metaAwarenessValue = document.getElementById('meta-awareness-value');
        metaAwarenessBar.style.width = `${state.meta_awareness * 100}%`;
        metaAwarenessValue.textContent = state.meta_awareness.toFixed(2);
        
        // 現象的性質
        const propertiesList = document.getElementById('properties-list');
        propertiesList.innerHTML = '';
        Object.entries(state.phenomenal_properties).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'property-item';
            item.innerHTML = `${key}: <span class="property-value">${value.toFixed(3)}</span>`;
            propertiesList.appendChild(item);
        });
    }
    
    updateAnalysis(analysis) {
        // 注意の安定性
        if (analysis.attention_stability !== undefined) {
            const stabilityBar = document.getElementById('attention-stability-bar');
            const stabilityValue = document.getElementById('attention-stability-value');
            stabilityBar.style.width = `${analysis.attention_stability * 100}%`;
            stabilityValue.textContent = analysis.attention_stability.toFixed(2);
        }
    }
    
    showTransitions(transitions) {
        const transitionList = document.getElementById('transition-list');
        
        // 最新の遷移を表示
        transitions.slice(-5).reverse().forEach(transition => {
            const item = document.createElement('div');
            item.className = 'transition-item';
            const time = new Date(transition.timestamp * 1000).toLocaleTimeString();
            item.innerHTML = `${time} - ${transition.type} (強度: ${transition.magnitude.toFixed(2)})`;
            transitionList.insertBefore(item, transitionList.firstChild);
        });
        
        // 古い項目を削除
        while (transitionList.children.length > 10) {
            transitionList.removeChild(transitionList.lastChild);
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (!this.isRunning) return;
        
        const time = performance.now();
        
        // パーティクルの更新
        if (this.particles) {
            this.particles.material.uniforms.time.value = time;
            this.particles.rotation.y += 0.0002;
        }
        
        // フローラインの更新
        this.flowLines.forEach((flowLine, index) => {
            const offset = time * 0.0001 + flowLine.offset;
            const points = [];
            
            for (let i = 0; i < 3; i++) {
                const t = (i / 2) + offset;
                points.push(new THREE.Vector3(
                    Math.sin(t) * 30,
                    Math.cos(t * 0.7) * 20,
                    Math.sin(t * 1.3) * 30
                ));
            }
            
            flowLine.curve = new THREE.CatmullRomCurve3(points);
            const newPoints = flowLine.curve.getPoints(50);
            flowLine.line.geometry.setFromPoints(newPoints);
        });
        
        // カメラの緩やかな動き
        this.camera.position.x = Math.sin(time * 0.0001) * 60;
        this.camera.position.z = Math.cos(time * 0.0001) * 60;
        this.camera.lookAt(0, 0, 0);
        
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('#connection-status span:last-child');
        
        if (connected) {
            statusDot.style.background = '#00ff00';
            statusText.textContent = 'リアルタイム接続中';
        } else {
            statusDot.style.background = '#ff0000';
            statusText.textContent = '接続待機中...';
        }
    }
    
    hslToRgb(h, s, l) {
        let r, g, b;
        
        if (s === 0) {
            r = g = b = l;
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1/6) return p + (q - p) * 6 * t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                return p;
            };
            
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        
        return { r, g, b };
    }
}

// グローバル関数
let visualizer;

function toggleVisualization() {
    if (visualizer) {
        visualizer.isRunning = !visualizer.isRunning;
    }
}

function changeColorScheme() {
    if (visualizer) {
        // カラースキームを切り替える
        const schemes = ['phenomenological', 'emotional', 'cognitive', 'abstract'];
        const currentIndex = schemes.indexOf(visualizer.colorScheme);
        visualizer.colorScheme = schemes[(currentIndex + 1) % schemes.length];
        console.log('カラースキーム変更:', visualizer.colorScheme);
    }
}

function resetView() {
    if (visualizer) {
        visualizer.camera.position.set(0, 30, 50);
        visualizer.camera.lookAt(0, 0, 0);
    }
}

// ページ読み込み時に初期化
document.addEventListener('DOMContentLoaded', () => {
    visualizer = new ConsciousnessVisualizer();
    
    // 定期的に分析データを要求
    setInterval(() => {
        if (visualizer.ws && visualizer.ws.readyState === WebSocket.OPEN) {
            visualizer.ws.send(JSON.stringify({ type: 'get_analysis' }));
        }
    }, 1000);
    
    // 定期的に遷移データを要求
    setInterval(() => {
        if (visualizer.ws && visualizer.ws.readyState === WebSocket.OPEN) {
            visualizer.ws.send(JSON.stringify({ type: 'get_transitions' }));
        }
    }, 3000);
});