//! Conic problem builder on top of Clarabel.
//!
//! Rust has no modelling layer like CVXPY, so problems are written with affine expressions over
//! scalar variables and a handful of cone constraints:
//!
//! ```text
//! minimize    sum_i w_i * e_i(x)^2  +  c' x
//! subject to  e(x) == 0              (zero cone)
//!             e(x) >= 0              (nonnegative orthant)
//!             e_0(x) >= ||(e_1(x), .., e_k(x))||   (second-order cone)
//! ```
//!
//! Each cone row `e(x) = a'x + c` maps to Clarabel's `A x + s = b, s in K` as `A_row = -a`,
//! `b_row = c`. Convenience helpers add the usual epigraph variables (`abs`, `norm`, `pos`,
//! `huber`, quadratic-over-one bounds).

use std::collections::HashMap;

use clarabel::{
    algebra::CscMatrix,
    solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT},
};

/// Index of a scalar decision variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(pub usize);

/// Affine expression `sum coef * x_var + constant`.
#[derive(Debug, Clone, Default)]
pub struct Expr {
    pub terms: Vec<(usize, f64)>,
    pub c: f64,
}

impl Expr {
    pub fn constant(c: f64) -> Self {
        Self { terms: Vec::new(), c }
    }
    pub fn var(v: Var) -> Self {
        Self { terms: vec![(v.0, 1.0)], c: 0.0 }
    }
    /// `self + a * v`
    pub fn plus_var(mut self, a: f64, v: Var) -> Self {
        if a != 0.0 {
            self.terms.push((v.0, a));
        }
        self
    }
    /// `self + a * e`
    pub fn plus(mut self, a: f64, e: &Expr) -> Self {
        if a != 0.0 {
            self.terms.extend(e.terms.iter().map(|&(i, k)| (i, a * k)));
            self.c += a * e.c;
        }
        self
    }
    pub fn plus_const(mut self, c: f64) -> Self {
        self.c += c;
        self
    }
    pub fn scale(mut self, a: f64) -> Self {
        self.terms.iter_mut().for_each(|t| t.1 *= a);
        self.c *= a;
        self
    }
    pub fn value(&self, x: &[f64]) -> f64 {
        self.c + self.terms.iter().map(|&(i, k)| k * x[i]).sum::<f64>()
    }
}

impl From<Var> for Expr {
    fn from(v: Var) -> Self {
        Expr::var(v)
    }
}

/// `sum coef * var`
pub fn lin(terms: &[(f64, Var)]) -> Expr {
    Expr { terms: terms.iter().filter(|t| t.0 != 0.0).map(|&(a, v)| (v.0, a)).collect(), c: 0.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Solved,
    /// Solved to reduced accuracy (CVXPY's `optimal_inaccurate`).
    AlmostSolved,
    Failed,
}

impl Status {
    pub fn ok(self) -> bool {
        self != Status::Failed
    }
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub status: Status,
    pub x: Vec<f64>,
    /// Objective value including the constant offset.
    pub obj: f64,
    pub solve_time: f64,
    pub iterations: u32,
}

impl Solution {
    pub fn get(&self, v: Var) -> f64 {
        self.x[v.0]
    }
    pub fn eval(&self, e: &Expr) -> f64 {
        e.value(&self.x)
    }
}

#[derive(Debug, Default)]
pub struct Problem {
    n: usize,
    q: Vec<f64>,
    q_const: f64,
    /// upper-triangular P triplets (i <= j); objective is 1/2 x'Px + q'x
    p: HashMap<(usize, usize), f64>,
    zero: Vec<Expr>,
    nonneg: Vec<Expr>,
    socs: Vec<Vec<Expr>>,
}

impl Problem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn var(&mut self) -> Var {
        self.n += 1;
        self.q.push(0.0);
        Var(self.n - 1)
    }

    pub fn vars(&mut self, k: usize) -> Vec<Var> {
        (0..k).map(|_| self.var()).collect()
    }

    pub fn n_vars(&self) -> usize {
        self.n
    }

    // ------------------------------------------------------------------ constraints
    pub fn eq(&mut self, e: Expr) {
        self.zero.push(e);
    }
    /// `e >= 0`
    pub fn ge0(&mut self, e: Expr) {
        self.nonneg.push(e);
    }
    /// `a >= b`
    pub fn ge(&mut self, a: Expr, b: &Expr) {
        self.nonneg.push(a.plus(-1.0, b));
    }
    /// `t >= ||(e_1, .., e_k)||`
    pub fn soc(&mut self, t: Expr, rest: Vec<Expr>) {
        let mut v = Vec::with_capacity(rest.len() + 1);
        v.push(t);
        v.extend(rest);
        self.socs.push(v);
    }
    /// `s >= x^2` (s affine, x affine) as `||(2x, s - 1)|| <= s + 1`.
    pub fn square_le(&mut self, x: &Expr, s: &Expr) {
        self.soc(s.clone().plus_const(1.0), vec![x.clone().scale(2.0), s.clone().plus_const(-1.0)]);
    }

    // ------------------------------------------------------------------ objective
    pub fn add_linear(&mut self, e: &Expr) {
        for &(i, k) in &e.terms {
            self.q[i] += k;
        }
        self.q_const += e.c;
    }
    /// Objective += `w * e^2` (w >= 0).
    pub fn add_square(&mut self, w: f64, e: &Expr) {
        if w == 0.0 {
            return;
        }
        // w (a'x + c)^2 = x' (w a a') x + 2 w c a'x + w c^2 ;  P = 2 w a a'
        let t = merge(&e.terms);
        for (ii, &(i, ai)) in t.iter().enumerate() {
            for &(j, aj) in &t[ii..] {
                let (r, c) = if i <= j { (i, j) } else { (j, i) };
                *self.p.entry((r, c)).or_insert(0.0) += 2.0 * w * ai * aj;
            }
            self.q[i] += 2.0 * w * e.c * ai;
        }
        self.q_const += w * e.c * e.c;
    }

    // ------------------------------------------------------------------ epigraph helpers
    /// New `t >= |e|`.
    pub fn abs(&mut self, e: &Expr) -> Var {
        let t = self.var();
        self.ge0(Expr::var(t).plus(-1.0, e));
        self.ge0(Expr::var(t).plus(1.0, e));
        t
    }
    /// New `t >= ||es||`.
    pub fn norm(&mut self, es: Vec<Expr>) -> Var {
        let t = self.var();
        self.soc(Expr::var(t), es);
        t
    }
    /// New `t >= max(e, 0)`.
    pub fn pos(&mut self, e: &Expr) -> Var {
        let t = self.var();
        self.ge0(Expr::var(t));
        self.ge0(Expr::var(t).plus(-1.0, e));
        t
    }
    /// Objective += `w * huber(e, m)` with huber(x, M) = x^2 if |x| <= M else 2M|x| - M^2 (CVXPY).
    pub fn add_huber(&mut self, w: f64, e: &Expr, m: f64) {
        // huber(e) = min_{s + n = e} s^2 + 2M|n|
        let s = self.var();
        let n = self.var();
        self.eq(Expr::var(s).plus_var(1.0, n).plus(-1.0, e));
        let t = self.abs(&Expr::var(n));
        self.add_square(w, &Expr::var(s));
        self.add_linear(&Expr::var(t).scale(2.0 * m * w));
    }

    // ------------------------------------------------------------------ solve
    pub fn solve(&self) -> Solution {
        let n = self.n;
        let mut rows: Vec<&Expr> = Vec::new();
        let mut cones = Vec::new();
        if !self.zero.is_empty() {
            rows.extend(self.zero.iter());
            cones.push(SupportedConeT::ZeroConeT(self.zero.len()));
        }
        if !self.nonneg.is_empty() {
            rows.extend(self.nonneg.iter());
            cones.push(SupportedConeT::NonnegativeConeT(self.nonneg.len()));
        }
        for s in &self.socs {
            rows.extend(s.iter());
            cones.push(SupportedConeT::SecondOrderConeT(s.len()));
        }
        let (mut ai, mut aj, mut av) = (Vec::new(), Vec::new(), Vec::new());
        let mut b = Vec::with_capacity(rows.len());
        for (r, e) in rows.iter().enumerate() {
            for &(j, k) in &e.terms {
                ai.push(r);
                aj.push(j);
                av.push(-k);
            }
            b.push(e.c);
        }
        let a = CscMatrix::new_from_triplets(rows.len(), n, ai, aj, av);
        let (mut pi, mut pj, mut pv) = (Vec::new(), Vec::new(), Vec::new());
        for (&(i, j), &v) in &self.p {
            pi.push(i);
            pj.push(j);
            pv.push(v);
        }
        let p = CscMatrix::new_from_triplets(n, n, pi, pj, pv);
        let settings = DefaultSettingsBuilder::default().verbose(false).build().expect("valid settings");
        let failed = |n| Solution {
            status: Status::Failed,
            x: vec![f64::NAN; n],
            obj: f64::NAN,
            solve_time: 0.0,
            iterations: 0,
        };
        let mut solver = match DefaultSolver::new(&p, &self.q, &a, &b, &cones, settings) {
            Ok(s) => s,
            Err(_) => return failed(n),
        };
        solver.solve();
        let sol = &solver.solution;
        let status = match sol.status {
            SolverStatus::Solved => Status::Solved,
            SolverStatus::AlmostSolved => Status::AlmostSolved,
            _ => Status::Failed,
        };
        Solution {
            status,
            x: sol.x.clone(),
            obj: sol.obj_val + self.q_const,
            solve_time: sol.solve_time,
            iterations: sol.iterations,
        }
    }
}

/// Sum duplicate variable indices.
fn merge(terms: &[(usize, f64)]) -> Vec<(usize, f64)> {
    let mut m: Vec<(usize, f64)> = Vec::with_capacity(terms.len());
    let mut sorted = terms.to_vec();
    sorted.sort_by_key(|t| t.0);
    for (i, k) in sorted {
        match m.last_mut() {
            Some(last) if last.0 == i => last.1 += k,
            _ => m.push((i, k)),
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_with_soc() {
        // minimize -x - y  s.t. ||(x, y)|| <= 1  -> x = y = 1/sqrt(2)
        let mut p = Problem::new();
        let (x, y) = (p.var(), p.var());
        p.soc(Expr::constant(1.0), vec![x.into(), y.into()]);
        p.add_linear(&lin(&[(-1.0, x), (-1.0, y)]));
        let s = p.solve();
        assert!(s.status.ok());
        assert!((s.get(x) - 0.5f64.sqrt()).abs() < 1e-6);
        assert!((s.obj + 2f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn qp_and_huber() {
        // minimize (x - 3)^2 + huber(x, 1) -> derivative 2(x-3) + 2 = 0 -> x = 2 (|x| > 1)
        let mut p = Problem::new();
        let x = p.var();
        p.add_square(1.0, &Expr::var(x).plus_const(-3.0));
        p.add_huber(1.0, &Expr::var(x), 1.0);
        let s = p.solve();
        assert!((s.get(x) - 2.0).abs() < 1e-6, "{}", s.get(x));
        assert!((s.obj - (1.0 + (2.0 * 2.0 - 1.0))).abs() < 1e-6);
    }

    #[test]
    fn square_le_and_infeasible() {
        // minimize s s.t. s >= (x - 1)^2, x >= 3  -> s = 4
        let mut p = Problem::new();
        let (x, s) = (p.var(), p.var());
        p.square_le(&Expr::var(x).plus_const(-1.0), &Expr::var(s));
        p.ge0(Expr::var(x).plus_const(-3.0));
        p.add_linear(&Expr::var(s));
        let sol = p.solve();
        assert!((sol.get(s) - 4.0).abs() < 1e-6);
        p.ge0(Expr::var(x).scale(-1.0)); // x <= 0 contradicts x >= 3
        assert_eq!(p.solve().status, Status::Failed);
    }
}
