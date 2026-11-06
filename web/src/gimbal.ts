import * as THREE from "three";
import { canvas2d } from "./scene.js";

const DEG = Math.PI / 180;
const DOWN = new THREE.Vector3(0, -1, 0);

type CylMesh = THREE.Mesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>;

interface Actuator {
  anchor: THREE.Vector3;
  local: THREE.Vector3;
  rod: THREE.Mesh;
  cyl: THREE.Mesh;
}

function textSprite(text: string, color = "#a3acb8"): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 128; c.height = 64;
  const g = canvas2d(c);
  g.fillStyle = color;
  g.font = "600 34px JetBrains Mono, monospace";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 64, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  s.scale.set(0.9, 0.45, 1);
  return s;
}

/**
 * Close-up of a single gimbaled engine in the body frame (body axis = up). Frame axes match the
 * main view: +x = pitch direction, +z = pad y = yaw direction. The bell pivots about its gimbal
 * point; two actuators connect it to the thrust structure; the faint ring shows the ±limit.
 */
export class GimbalView {
  private readonly canvas: HTMLCanvasElement;
  private readonly gMax: number;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  private readonly engine = new THREE.Group();
  private readonly plume = new THREE.Group();
  private readonly plumeOuter: CylMesh;
  private readonly plumeCore: CylMesh;
  private readonly exitDot: THREE.Mesh;
  private readonly acts: Actuator[];
  private readonly exitY = -3.0;
  private readonly mountL = 1.6; // actuator attach point below the pivot on the bell
  private readonly _dir = new THREE.Vector3();
  private readonly _q = new THREE.Quaternion();

  constructor(canvas: HTMLCanvasElement, gimbalMaxDeg = 8) {
    this.canvas = canvas;
    this.gMax = gimbalMaxDeg;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = this.scene;
    this.camera.position.set(6.4, 1.5, 8.0);
    this.camera.lookAt(0, -2.0, 0);

    scene.add(new THREE.HemisphereLight("#cfe0ff", "#1a1410", 1.1));
    const key = new THREE.DirectionalLight("#ffffff", 1.6);
    key.position.set(4, 5, 6);
    scene.add(key);

    const metal = new THREE.MeshStandardMaterial({ color: "#8d96a3", roughness: 0.35, metalness: 0.85, side: THREE.DoubleSide });
    const dark = new THREE.MeshStandardMaterial({ color: "#2a3038", roughness: 0.6, metalness: 0.5 });
    const copper = new THREE.MeshStandardMaterial({ color: "#a86a3c", roughness: 0.4, metalness: 0.8 });

    // fixed thrust structure
    const plate = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.16, 40), dark);
    plate.position.y = 0.55;
    scene.add(plate);

    // engine: pivot at the origin, everything hangs below it
    scene.add(this.engine);
    const pivot = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 12), dark);
    this.engine.add(pivot);
    const chamber = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.7, 28), copper);
    chamber.position.y = -0.45;
    this.engine.add(chamber);
    // bell: throat at y = -0.85, exit (radius 0.95) at y = -3.0
    const prof: THREE.Vector2[] = [];
    for (let i = 0; i <= 16; i++) {
      const s = i / 16;
      prof.push(new THREE.Vector2(0.2 + 0.75 * Math.pow(s, 0.7), -0.85 - 2.15 * s));
    }
    prof.unshift(new THREE.Vector2(0.34, -0.8));
    const bell = new THREE.Mesh(new THREE.LatheGeometry(prof, 40), metal);
    this.engine.add(bell);

    // plume (additive cones), hangs from the exit plane
    const pm = (c: string, o: number) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: o, blending: THREE.AdditiveBlending, depthWrite: false });
    const plumeGeo = new THREE.CylinderGeometry(0.9, 0.35, 1, 28, 1, true).translate(0, -0.5, 0);
    this.plumeOuter = new THREE.Mesh(plumeGeo, pm("#ff8a3d", 0.5));
    this.plumeCore = new THREE.Mesh(plumeGeo, pm("#fff1c4", 0.8));
    this.plume.add(this.plumeOuter, this.plumeCore);
    this.plume.position.y = this.exitY;
    this.engine.add(this.plume);

    // gimbal-limit ring around the neutral exit centre, and neutral axis
    const rLim = Math.abs(this.exitY) * Math.tan(this.gMax * DEG);
    const ringPts: THREE.Vector3[] = [];
    for (let i = 0; i <= 64; i++) {
      const a = (i / 64) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(Math.cos(a) * rLim, this.exitY, Math.sin(a) * rLim));
    }
    const lim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color: "#4cc9f0", transparent: true, opacity: 0.7 }));
    scene.add(lim);
    const axis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.4, 0), new THREE.Vector3(0, -4.2, 0)]),
      new THREE.LineDashedMaterial({ color: "#4cc9f0", dashSize: 0.12, gapSize: 0.1, transparent: true, opacity: 0.45 }));
    axis.computeLineDistances();
    scene.add(axis);
    this.exitDot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 12, 8), new THREE.MeshBasicMaterial({ color: "#4cc9f0" }));
    scene.add(this.exitDot);

    // pitch / yaw direction labels on the plate
    const lx = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.66, 0), new THREE.Vector3(2.0, 0.66, 0)]),
      new THREE.LineBasicMaterial({ color: "#a3acb8" }));
    const lz = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.66, 0), new THREE.Vector3(0, 0.66, 2.0)]),
      new THREE.LineBasicMaterial({ color: "#a3acb8" }));
    scene.add(lx, lz);
    const tp = textSprite("+P"); tp.position.set(2.45, 0.7, 0); scene.add(tp);
    const ty = textSprite("+Y"); ty.position.set(0, 0.7, 2.45); scene.add(ty);

    // actuators: fixed anchor on the plate -> attach point on the bell (pitch at +x, yaw at +z)
    this.acts = [new THREE.Vector3(1.0, 0.47, 0), new THREE.Vector3(0, 0.47, 1.0)].map((anchor, i) => {
      const local = i === 0 ? new THREE.Vector3(0.62, -this.mountL, 0) : new THREE.Vector3(0, -this.mountL, 0.62);
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1, 10), dark);
      const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 1, 12), new THREE.MeshStandardMaterial({ color: "#4cc9f0", roughness: 0.5, metalness: 0.3 }));
      scene.add(rod, cyl);
      return { anchor, local, rod, cyl };
    });

    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize(): void {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** gp, gy: gimbal deflection in pitch / yaw (deg); x: exaggerated throttle level 0..1, on: engine lit */
  update(gp: number, gy: number, x: number, on: boolean): void {
    // nozzle exit points opposite the thrust: thrust tilts toward +g, the bell exit toward -g
    const sp = Math.sin(gp * DEG), sy = Math.sin(gy * DEG);
    this._dir.set(-sp, -Math.sqrt(Math.max(0, 1 - sp * sp - sy * sy)), -sy).normalize();
    this._q.setFromUnitVectors(DOWN, this._dir);
    this.engine.quaternion.copy(this._q);
    this.exitDot.position.copy(this._dir).multiplyScalar(Math.abs(this.exitY) / Math.max(1e-6, -this._dir.y)).setY(this.exitY);

    for (const a of this.acts) {
      const p = a.local.clone().applyQuaternion(this._q);
      const mid = a.anchor.clone().lerp(p, 0.5);
      const d = p.clone().sub(a.anchor);
      const len = d.length();
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
      a.cyl.position.copy(a.anchor).addScaledVector(d, 0.3);
      a.cyl.scale.set(1, len * 0.6, 1);
      a.cyl.quaternion.copy(q);
      a.rod.position.copy(mid).addScaledVector(d, 0.15);
      a.rod.scale.set(1, len * 0.7, 1);
      a.rod.quaternion.copy(q);
    }

    this.plume.visible = on;
    if (on) {
      const f = 1 + 0.08 * (Math.random() - 0.5);
      const len = (0.8 + 4.5 * Math.pow(x, 1.2)) * f;
      this.plumeOuter.scale.set(0.9 + 0.6 * x, len, 0.9 + 0.6 * x);
      this.plumeCore.scale.set(0.45 + 0.2 * x, len * 0.5, 0.45 + 0.2 * x);
      this.plumeOuter.material.opacity = 0.2 + 0.55 * x;
      this.plumeCore.material.opacity = 0.35 + 0.6 * x;
    }
    this.renderer.render(this.scene, this.camera);
  }
}

/** Top-down gimbal plot: limit circle, current (pitch, yaw) deflection and a short trail. */
export class GimbalTarget {
  private readonly canvas: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D;
  private readonly gMax: number;

  constructor(canvas: HTMLCanvasElement, gimbalMaxDeg = 8) {
    this.canvas = canvas;
    this.g = canvas2d(canvas);
    this.gMax = gimbalMaxDeg;
  }

  /** trail: recent [pitch, yaw] deflections (deg), oldest first; sat: deflection at the limit. */
  draw(gp: number, gy: number, trail: [number, number][], color: string, sat: boolean): void {
    const c = this.canvas, g = this.g;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 12;
    const X = (p: number) => cx + (p / this.gMax) * R, Y = (y: number) => cy - (y / this.gMax) * R;

    g.strokeStyle = "rgba(255,255,255,0.08)";
    g.lineWidth = 1;
    g.beginPath(); g.arc(cx, cy, R / 2, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(cx - R, cy); g.lineTo(cx + R, cy); g.moveTo(cx, cy - R); g.lineTo(cx, cy + R); g.stroke();
    g.strokeStyle = "rgba(76,201,240,0.7)";
    g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();

    g.font = "600 9px JetBrains Mono, monospace";
    g.fillStyle = "#a3acb8";
    g.textAlign = "left";
    g.fillText("+P", cx + R + 1, cy - 3);
    g.textAlign = "center";
    g.fillText("+Y", cx, cy - R - 3);
    g.fillText(`±${this.gMax}°`, cx, cy + R + 10);

    if (trail.length > 1) {
      for (let i = 1; i < trail.length; i++) {
        g.strokeStyle = color;
        g.globalAlpha = (i / trail.length) * 0.7;
        g.beginPath();
        g.moveTo(X(trail[i - 1][0]), Y(trail[i - 1][1]));
        g.lineTo(X(trail[i][0]), Y(trail[i][1]));
        g.stroke();
      }
      g.globalAlpha = 1;
    }
    const px = X(gp), py = Y(gy);
    g.fillStyle = sat ? "#ffb703" : color;
    g.beginPath(); g.arc(px, py, 4, 0, Math.PI * 2); g.fill();
    g.strokeStyle = "#06080c";
    g.lineWidth = 1.5;
    g.stroke();
  }
}
