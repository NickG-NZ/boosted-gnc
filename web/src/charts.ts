import { canvas2d } from "./scene.js";

export interface Series {
  t: number[];
  y: number[];
  color: string;
  width?: number;
  dashed?: boolean;
  alpha?: number;
}

/** Horizontal reference line. */
export interface Band {
  y: number;
  label: string;
}

/** Vertical event marker (e.g. ignition). */
export interface Marker {
  t: number;
  label: string;
}

export interface TimeChartOptions {
  title: string;
  unit: string;
  yMin?: number | null;
  yMax?: number | null;
  bands?: Band[];
  cursor?: number | null;
}

/** Tiny dependency-free time-series chart on a 2D canvas. */
export class TimeChart {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly title: string;
  private readonly unit: string;
  private readonly yMinFixed: number | null;
  private readonly yMaxFixed: number | null;
  private readonly bands: Band[];
  private series: Series[] = [];
  private cursor: number | null;
  private tMax = 0;
  private yMin = 0;
  private yMax = 1;
  markers: Marker[] = [];

  constructor(canvas: HTMLCanvasElement, { title, unit, yMin = null, yMax = null, bands = [], cursor = 0 }: TimeChartOptions) {
    this.canvas = canvas;
    this.ctx = canvas2d(canvas);
    this.title = title;
    this.unit = unit;
    this.yMinFixed = yMin;
    this.yMaxFixed = yMax;
    this.bands = bands;
    this.cursor = cursor;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  setSeries(series: Series[], tMax: number): void {
    this.series = series;
    this.tMax = tMax;
    const ys = series.flatMap((s) => s.y);
    this.yMin = this.yMinFixed ?? Math.min(...ys);
    this.yMax = this.yMaxFixed ?? Math.max(...ys);
    this.draw();
  }

  setCursor(t: number | null): void {
    this.cursor = t;
    this.draw();
  }

  draw(): void {
    const c = this.canvas, g = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const pad = { l: 38, r: 10, t: 22, b: 18 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const X = (t: number) => pad.l + (t / (this.tMax || 1)) * pw;
    const Y = (y: number) => pad.t + (1 - (y - this.yMin) / (this.yMax - this.yMin || 1)) * ph;

    g.font = "600 10px JetBrains Mono, monospace";
    g.fillStyle = "#a3acb8";
    g.fillText(`${this.title.toUpperCase()} [${this.unit}]`, pad.l, 13);

    // axes ticks
    g.font = "10px JetBrains Mono, monospace";
    g.strokeStyle = "rgba(255,255,255,0.06)";
    g.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = this.yMin + (i / 3) * (this.yMax - this.yMin);
      g.beginPath(); g.moveTo(pad.l, Y(y)); g.lineTo(W - pad.r, Y(y)); g.stroke();
      g.fillText(fmt(y), 4, Y(y) + 3);
    }
    for (let t = 0; t <= this.tMax; t += 5) g.fillText(`${t}`, X(t) - 3, H - 4);

    for (const b of this.bands) {
      g.strokeStyle = "rgba(255,255,255,0.22)";
      g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(pad.l, Y(b.y)); g.lineTo(W - pad.r, Y(b.y)); g.stroke();
      g.setLineDash([]);
      g.fillStyle = "rgba(255,255,255,0.4)";
      g.fillText(b.label, W - pad.r - g.measureText(b.label).width, Y(b.y) - 3);
    }

    for (const mk of this.markers) {
      g.strokeStyle = "rgba(255,154,77,0.7)";
      g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(X(mk.t), pad.t); g.lineTo(X(mk.t), pad.t + ph); g.stroke();
      g.setLineDash([]);
      g.fillStyle = "rgba(255,154,77,0.9)";
      g.fillText(mk.label, X(mk.t) + 3, pad.t + 9);
    }

    for (const s of this.series) {
      g.strokeStyle = s.color;
      g.globalAlpha = s.alpha ?? 1;
      g.lineWidth = s.width ?? 1.5;
      g.setLineDash(s.dashed ? [4, 3] : []);
      g.beginPath();
      s.t.forEach((t, i) => (i ? g.lineTo(X(t), Y(s.y[i])) : g.moveTo(X(t), Y(s.y[i]))));
      g.stroke();
    }
    g.globalAlpha = 1;
    g.setLineDash([]);

    if (this.cursor == null) return;
    g.strokeStyle = "rgba(255,255,255,0.6)";
    g.beginPath(); g.moveTo(X(this.cursor), pad.t); g.lineTo(X(this.cursor), pad.t + ph); g.stroke();
  }
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
