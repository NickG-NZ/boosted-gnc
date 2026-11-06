import * as THREE from "three";
import { canvas2d } from "./scene.js";
import type { Geometry } from "./types.js";

const UP = new THREE.Vector3(0, 1, 0);
const DOWN = new THREE.Vector3(0, -1, 0);

type PlumeMesh = THREE.Mesh<THREE.ConeGeometry, THREE.MeshBasicMaterial>;

let _glow: THREE.CanvasTexture | null = null;
function glowTexture(): THREE.CanvasTexture {
  if (_glow) return _glow;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = canvas2d(c);
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.25, "rgba(255,200,140,0.6)");
  grd.addColorStop(1, "rgba(255,120,40,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  _glow = new THREE.CanvasTexture(c);
  return _glow;
}

/**
 * Procedural booster built from the rocket config's `geometry` block.
 * Local origin is at the leg feet / nozzle exit plane so position == guidance r.
 * The body axis is aligned with the thrust vector (thrust acts along the body).
 */
export class Rocket3D {
  readonly group = new THREE.Group();
  readonly body = new THREE.Group();
  private readonly nozzle = new THREE.Group();
  private readonly plume = new THREE.Group();
  private readonly plumeOuter: PlumeMesh;
  private readonly plumeCore: PlumeMesh;
  private readonly glow: THREE.Sprite;
  private readonly engineLight: THREE.PointLight;
  private readonly _cOuterLo = new THREE.Color("#b8330f");
  private readonly _cOuterHi = new THREE.Color("#ffb347");
  private readonly _cCoreLo = new THREE.Color("#ff9a4d");
  private readonly _cCoreHi = new THREE.Color("#ffffff");
  private readonly _dir = UP.clone();

  constructor({ length = 42, diameter = 3.7, n_legs = 4, n_grid_fins = 4 }: Geometry = {}, color = "#4cc9f0") {
    this.group.add(this.body);
    const R = diameter / 2;
    const base = 2.0; // height of nozzle exit above leg feet plane

    const white = new THREE.MeshStandardMaterial({ color: "#e9edf1", roughness: 0.45, metalness: 0.2 });
    const dark = new THREE.MeshStandardMaterial({ color: "#20252c", roughness: 0.7, metalness: 0.4 });
    const accent = new THREE.MeshStandardMaterial({ color, roughness: 0.5, emissive: color, emissiveIntensity: 0.25 });
    const soot = new THREE.MeshStandardMaterial({ color: "#3a332d", roughness: 0.9 });

    const add = (geo: THREE.BufferGeometry, mat: THREE.Material, y = 0) => {
      const m = new THREE.Mesh(geo, mat);
      m.position.y = y;
      this.body.add(m);
      return m;
    };

    // engine section + nozzle (the nozzle pivots about its top so it can show the gimbal)
    this.nozzle.position.y = base + 1.6;
    const bell = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 1.0, 1.6, 24, 1, true), dark);
    bell.position.y = -0.8;
    this.nozzle.add(bell);
    this.body.add(this.nozzle);
    add(new THREE.CylinderGeometry(R, R, 3.0, 32), soot, base + 1.6 + 1.5);
    // main tank
    const tankLen = length - 3.0 - 5.0;
    add(new THREE.CylinderGeometry(R, R, tankLen, 32), white, base + 4.6 + tankLen / 2);
    // colored stripe
    add(new THREE.CylinderGeometry(R + 0.02, R + 0.02, 1.2, 32), accent, base + 4.6 + tankLen * 0.7);
    // interstage
    const topY = base + 4.6 + tankLen;
    add(new THREE.CylinderGeometry(R, R, 5.0, 32), dark, topY + 2.5);

    // grid fins
    for (let i = 0; i < n_grid_fins; i++) {
      const a = (i / n_grid_fins) * Math.PI * 2 + Math.PI / 4;
      const fin = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.4, 0.15), dark);
      fin.position.set(Math.cos(a) * (R + 0.8), topY + 1.6, Math.sin(a) * (R + 0.8));
      fin.rotation.y = -a + Math.PI / 2;
      this.body.add(fin);
    }

    // landing legs: struts from hinge on the tank down/out to the feet plane
    const hinge = new THREE.Vector3(0, base + 9.0, 0);
    for (let i = 0; i < n_legs; i++) {
      const a = (i / n_legs) * Math.PI * 2;
      const dir = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
      const top = hinge.clone().addScaledVector(dir, R);
      const foot = new THREE.Vector3(0, 0, 0).addScaledVector(dir, R + 6.0);
      const len = top.distanceTo(foot);
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.35, len, 0.5), dark);
      leg.position.copy(top).add(foot).multiplyScalar(0.5);
      leg.quaternion.setFromUnitVectors(UP, top.clone().sub(foot).normalize());
      this.body.add(leg);
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 0.2, 16), dark);
      pad.position.copy(foot).setY(0.1);
      this.body.add(pad);
    }

    // exhaust plume: two additive cones (outer glow + hot core), pointing down from the nozzle
    const plumeMat = (c: string, o: number) => new THREE.MeshBasicMaterial({
      color: c, transparent: true, opacity: o, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const coneGeo = new THREE.ConeGeometry(1, 1, 24, 1, true).rotateX(Math.PI).translate(0, -0.5, 0);
    this.plumeOuter = new THREE.Mesh(coneGeo, plumeMat("#ff8a3d", 0.55));
    this.plumeCore = new THREE.Mesh(coneGeo, plumeMat("#fff1c4", 0.85));
    this.glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture(), color: "#ffb070", transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.plume.add(this.plumeOuter, this.plumeCore, this.glow);
    this.plume.position.y = -1.65; // nozzle exit, in the nozzle's pivot frame
    this.nozzle.add(this.plume);
    this.engineLight = new THREE.PointLight("#ff9a4d", 0, 120, 1.5);
    this.engineLight.position.y = base - 4;
    this.body.add(this.engineLight);
  }

  /**
   * @param r three.js position, @param axis three.js body axis (unit), @param throttle 0..1
   * @param level exaggerated throttle level 0..1 (0 = engine floor, 1 = full) driving flame size,
   *        colour and brightness, @param gimbal three.js gimbal vector (perp. to axis, |g| = sin δ)
   */
  update(r: THREE.Vector3, axis: THREE.Vector3 | null, throttle: number, level = throttle, gimbal: THREE.Vector3 | null = null): void {
    this.group.position.copy(r);
    if (axis && axis.lengthSq() > 1e-12) this._dir.copy(axis).normalize();
    this.body.quaternion.setFromUnitVectors(UP, this._dir);

    // nozzle tilt: thrust along t = sqrt(1-|g|^2) b + g, the bell exit points along -t
    if (gimbal && gimbal.lengthSq() > 1e-10) {
      const t = this._dir.clone().multiplyScalar(Math.sqrt(Math.max(0, 1 - gimbal.lengthSq()))).add(gimbal);
      const local = t.negate().applyQuaternion(this.body.quaternion.clone().invert()).normalize();
      this.nozzle.quaternion.setFromUnitVectors(DOWN, local);
    } else this.nozzle.quaternion.identity();

    const on = throttle > 0.01;
    this.plume.visible = on;
    const x = Math.min(1, Math.max(0, level));
    this.engineLight.intensity = on ? 8 + 160 * x * x : 0;
    if (on) {
      const flicker = 1 + 0.07 * (Math.random() - 0.5);
      const len = (6 + 90 * Math.pow(x, 1.3)) * flicker;
      const w = 0.9 + 2.6 * x;
      this.plumeOuter.scale.set(w, len, w);
      this.plumeCore.scale.set(0.45 + 0.55 * x, len * (0.35 + 0.2 * x), 0.45 + 0.55 * x);
      this.plumeOuter.material.color.copy(this._cOuterLo).lerp(this._cOuterHi, x);
      this.plumeCore.material.color.copy(this._cCoreLo).lerp(this._cCoreHi, x);
      this.plumeOuter.material.opacity = 0.18 + 0.6 * x;
      this.plumeCore.material.opacity = 0.3 + 0.65 * x;
      this.glow.scale.setScalar((5 + 22 * x) * flicker);
      this.glow.material.opacity = 0.15 + 0.75 * x;
    }
  }

  setScale(s: number): void {
    this.group.scale.setScalar(s);
  }
}
