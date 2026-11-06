import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/** Renderer, camera, lights, ground, landing pad and glide-slope cone. */
export function createWorld(container, { glideSlopeDeg, coneHeight = 1800 }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#2b2f36");
  scene.fog = new THREE.Fog("#2b2f36", 3500, 9000);

  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.5, 20000);
  camera.position.set(2300, 650, 2500);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(60, 650, -30);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;

  scene.add(new THREE.HemisphereLight("#bcd4ff", "#1a1410", 0.9));
  const sun = new THREE.DirectionalLight("#fff3e0", 1.6);
  sun.position.set(-800, 1200, 600);
  scene.add(sun);

  // ground
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(16000, 16000),
    new THREE.MeshStandardMaterial({ color: "#3a3f47", roughness: 1 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  const grid = new THREE.GridHelper(4000, 80, "#555c66", "#474d56");
  grid.position.y = 0.05;
  scene.add(grid);

  scene.add(makePad());
  const cone = makeGlideSlopeCone(glideSlopeDeg, coneHeight);
  scene.add(cone);

  // soft blob shadow under the selected rocket
  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(1, 32),
    new THREE.MeshBasicMaterial({ color: "#000", transparent: true, opacity: 0.5, depthWrite: false })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.2;
  scene.add(shadow);

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  return { renderer, scene, camera, controls, cone, shadow };
}

function makePad() {
  const size = 512;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  g.fillStyle = "#2b323c";
  g.beginPath(); g.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2); g.fill();
  g.strokeStyle = "#e9edf2"; g.lineWidth = 14;
  g.beginPath(); g.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2); g.stroke();
  g.fillStyle = "#e9edf2";
  g.font = `bold ${size * 0.45}px Inter, sans-serif`;
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText("X", size / 2, size / 2 + 6);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const pad = new THREE.Mesh(
    new THREE.CircleGeometry(30, 64),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9 })
  );
  pad.rotation.x = -Math.PI / 2;
  pad.position.y = 0.1;
  return pad;
}

/** Glide-slope constraint ||E r|| tan(γ) <= r_up: an upward cone with apex at the pad. */
function makeGlideSlopeCone(gsDeg, height) {
  const radius = height / Math.tan((gsDeg * Math.PI) / 180);
  const geo = new THREE.ConeGeometry(radius, height, 72, 1, true);
  geo.rotateX(Math.PI); // apex down
  geo.translate(0, height / 2, 0);
  const group = new THREE.Group();
  group.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: "#4cc9f0", transparent: true, opacity: 0.035, side: THREE.DoubleSide, depthWrite: false,
  })));
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.ConeGeometry(radius, height, 24, 4, true).rotateX(Math.PI).translate(0, height / 2, 0), 1),
    new THREE.LineBasicMaterial({ color: "#4cc9f0", transparent: true, opacity: 0.07 })
  );
  group.add(edges);
  return group;
}
