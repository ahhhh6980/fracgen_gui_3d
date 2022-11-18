/*
    Copyright (C) 2022  Shelby Lynn B.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
use std::f64::consts::{E, PI};

use colortypes::*;
use num::complex::{Complex64, ComplexFloat};
use num::traits::MulAdd;
use num::Complex;
use rand::{distributions::Uniform, prelude::Distribution};

use crate::LinearTransformMode;

use super::{RenderData, Renderer};

type Cf64 = Complex64;

pub enum Operator {
    Add(fn(Cf64orI, Cf64orI) -> Cf64orI),
    Sub(fn(Cf64orI, Cf64orI) -> Cf64orI),
    Mul(fn(Cf64orI, Cf64orI) -> Cf64orI),
    Div(fn(Cf64orI, Cf64orI) -> Cf64orI),
    Pow(fn(Cf64orI, Cf64orI) -> Cf64orI),
}
#[derive(Clone, Copy)]
pub struct Cf64orI(Option<Cf64>, Option<i64>);

#[derive(Clone, Copy)]
pub enum ParenEnum {
    Left,
    Right,
}
#[derive(Clone, Copy)]
pub enum TokenType {
    Number(Cf64orI),
    Function(fn(Cf64orI, Option<Cf64orI>, Option<Cf64orI>) -> Cf64orI),
    Operator(&'static str),
    Parenthesis(ParenEnum),
}
#[derive(Clone, Copy)]
pub struct Token(TokenType, usize, bool);

fn shunting_yard(items: Vec<Token>) -> Vec<Token> {
    let mut stack = items;
    let mut output_queue: Vec<Token> = Vec::new();
    let mut operator_stack: Vec<Token> = Vec::new();

    while !stack.is_empty() {
        if let Some(token) = stack.pop() {
            match token.0 {
                TokenType::Number(_) | TokenType::Function(_) => output_queue.push(token),
                TokenType::Operator(_) => {
                    if let Some(mut top) = &operator_stack.pop() {
                        while (token.1 > top.1 || (token.1 == top.1 && top.2))
                            && !matches!(top.0, TokenType::Parenthesis(ParenEnum::Left))
                        {
                            output_queue.push(top);
                            top = if let Some(new_top) = operator_stack.pop() {
                                new_top
                            } else {
                                break;
                            }
                        }
                        operator_stack.push(token);
                    }
                }
                TokenType::Parenthesis(side) => match side {
                    ParenEnum::Left => operator_stack.push(token),
                    _ => {
                        if let Some(mut top) = &operator_stack.pop() {
                            while !matches!(top.0, TokenType::Parenthesis(ParenEnum::Left)) {
                                output_queue.push(top);
                                top = if let Some(new_top) = operator_stack.pop() {
                                    new_top
                                } else {
                                    break;
                                }
                            }
                            if matches!(top.0, TokenType::Parenthesis(ParenEnum::Left)) {
                                if let Some(new_top) = operator_stack.pop() {
                                    if matches!(new_top.0, TokenType::Function(_)) {
                                        output_queue.push(new_top);
                                    }
                                } else {
                                    break;
                                }
                            }
                            operator_stack.push(token);
                        }
                    }
                },
            }
        }
    }
    while !operator_stack.is_empty() {
        if let Some(top) = operator_stack.pop() {
            output_queue.push(top);
        }
    }
    output_queue
}

pub fn dot(a: Cf64, b: Cf64) -> f64 {
    a.re * b.re + a.im * b.im
}
pub fn sdSegment(p: Cf64, a: Cf64, b: Cf64, r: f64) -> f64 {
    let h = ((dot(p - a, b - a) / dot(b - a, b - a)).max(0f64)).min(1f64);
    (p - a - (b - a) * h).norm() - r
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2M {
    pub x: f64,
    pub y: f64,
}

impl Vec2M {
    pub fn new(x: f64, y: f64) -> Self {
        Vec2M { x, y }
    }
}

impl std::ops::Add for Vec2M {
    type Output = Vec2M;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Vec2M {
    type Output = Vec2M;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul for Vec2M {
    type Output = f64;
    fn mul(self, rhs: Self) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y
    }
}
// impl Mat2 {
//     fn new(a:f64,b:f64,c:f64,d:f64)
// }

struct LT {
    mat: Mat2,
    offset: Vec2M,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Mat2 {
    pub r1: Vec2M,
    pub r2: Vec2M,
}

impl Mat2 {
    pub fn new(r1: Vec2M, r2: Vec2M) -> Self {
        Mat2 { r1, r2 }
    }
    pub fn det(&self) -> f64 {
        self.r1.x * self.r2.y - self.r1.y * self.r2.x
    }
    pub fn adjoint(&self) -> Mat2 {
        Mat2::from([self.r2.y, -self.r1.y, -self.r2.x, self.r1.x])
    }
    pub fn inv(&self) -> Option<Mat2> {
        let det = self.det();
        if det != 0.0 {
            Some(self.adjoint() * (1.0 / det))
        } else {
            None
        }
    }
}

impl std::ops::Add for Mat2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Mat2::new(self.r1 + rhs.r1, self.r2 + rhs.r2)
    }
}

impl std::ops::Mul for Mat2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Mat2::from([
            self.r1 * Vec2M::new(rhs.r1.x, rhs.r2.x),
            self.r1 * Vec2M::new(rhs.r1.y, rhs.r2.y),
            self.r2 * Vec2M::new(rhs.r1.x, rhs.r2.x),
            self.r2 * Vec2M::new(rhs.r1.y, rhs.r2.y),
        ])
    }
}

impl std::ops::Mul<Cf64> for Mat2 {
    type Output = Cf64;
    fn mul(self, rhs: Cf64) -> Self::Output {
        Cf64::new(
            self.r1 * Vec2M::new(rhs.re, rhs.im),
            self.r2 * Vec2M::new(rhs.re, rhs.im),
        )
    }
}

impl std::ops::Mul<f64> for Mat2 {
    type Output = Mat2;
    fn mul(self, rhs: f64) -> Self::Output {
        Mat2::new(self.r1 * rhs, self.r2 * rhs)
    }
}

impl std::ops::Mul<f64> for Vec2M {
    type Output = Vec2M;
    fn mul(self, rhs: f64) -> Self::Output {
        Vec2M::new(self.x * rhs, self.y * rhs)
    }
}

impl From<Cf64> for Mat2 {
    fn from(z: Cf64) -> Self {
        Mat2::from([z.re, -z.im, z.im, z.re])
    }
}

// impl From<Mat2> for Cf64 {
//     fn from(m: Mat2) -> Self {
//         Cf64::new(m.0 ^ Vec2M::nw(), m.1.x - m.0.y)
//     }
// }

impl From<(Cf64, Cf64)> for Mat2 {
    fn from(a: (Cf64, Cf64)) -> Self {
        Mat2::from(a.0) + Mat2::from(a.1 * Cf64::new(0.0, 1.0))
    }
}

impl From<[f64; 4]> for Mat2 {
    fn from(arr: [f64; 4]) -> Self {
        Mat2::new(Vec2M::new(arr[0], arr[1]), Vec2M::new(arr[2], arr[3]))
    }
}
impl From<[[f64; 2]; 2]> for Mat2 {
    fn from(arr: [[f64; 2]; 2]) -> Self {
        Mat2::new(
            Vec2M::new(arr[0][0], arr[0][1]),
            Vec2M::new(arr[1][0], arr[1][1]),
        )
    }
}

impl From<[Vec2M; 2]> for Mat2 {
    fn from(arr: [Vec2M; 2]) -> Self {
        Mat2::new(arr[0], arr[1])
    }
}

impl From<(Vec2M, Vec2M)> for Mat2 {
    fn from(arr: (Vec2M, Vec2M)) -> Self {
        Mat2::new(arr.0, arr.1)
    }
}

// impl Default for Mat2 {
//     fn default() -> Self {
//         Mat2(Vec2M::default(), Vec2M::default())
//     }
// }

pub fn mobius(z: Cf64, [a, b, c, d]: [Cf64; 4]) -> Cf64 {
    (a * z + b) / (c * z + d)
}

pub fn inv_mobius(z: Cf64, [a, b, c, d]: [Cf64; 4]) -> Cf64 {
    (d * z - b) / ((-c) * z + a)
}

pub fn fractal(r: &Renderer, p: Cf64) -> Color<Xyz, D65> {
    let c = r.args.test_mat * (r.args.cmap_funct)(r, p);
    // let c = (1.0 / c.powc(Cf64::new(1.0, 0.0)));
    // let jd = (c - self.args.julia).norm();
    // if jd < 0.01 / self.args.zoom {
    //     let v = jd / (0.01 / self.args.zoom);
    //     return Xyz::from_color(Rgb::new([v, v, v, 1.0]));
    // }
    let mut sums = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    // let mut z =

    let (mut z, mut c) = if r.args.julia.1 {
        (c, r.args.julia.0)
    } else {
        (r.args.z_init, c)
    };

    // let mut zs = [z, z, z, z];
    // let mut zr: [Option<(f64, f64)>; 4] = [None, None, None, None];
    let ratio = r.args.size.1 as f64 / r.args.size.0 as f64;
    // let inf = Cf64::new(0.5 * r.args.h, (ratio / 2.0) * r.args.h);
    // zs[0].re -= inf.re / r.args.zoom;
    // zs[1].re += inf.re / r.args.zoom;
    // zs[2].im -= inf.im / r.args.zoom;
    // zs[3].im += inf.im / r.args.zoom;

    let mut der = Cf64::new(1.0, 0.0);
    let mut der2 = Cf64::new(0.0, 0.0);
    let mut der_sum = Cf64::new(0.0, 0.0);

    let mut lastz = z;
    let mut lastder = der;
    let mut lastder2 = der2;

    let mut i = 0.0f64;
    let mut s = 0.0f64;

    let mut old = z;
    let mut period = 1;

    let mut distance_min = f64::MAX;
    let mut distance_max = f64::MIN;
    let mut distance_avg = 0.0;
    let mut last_distance_min = f64::MAX;
    let mut last_distance_max = f64::MIN;
    let mut last_distance_avg = 0.0;
    let mut sum = 0.0;

    let mut old_diff = 0.0;
    let mut avg_diff = 0.0;
    let mut old_angle = 0.0;
    let mut angle_change = 0.0;
    let mut diff_change = 0.0;

    // let mut i2 = 0.0;
    // while i2 < 32.0 {
    //     u = (self.functs.iter_funct)(u, c, (self.functs.cmap_funct)(self, self.args.julia));
    //     i2 += 1.0;
    // }
    let v = 1.0 / ((r.args.size.0 as f64).hypot(r.args.size.1 as f64) * r.args.zoom).powi(2);
    let start_angle = z.arg() + PI;
    let mut angle = 0.0;
    let total_summed = 0.0;
    let mut rotations = 0.0;
    let mut oold = z;

    let min_sub_pixel_spacing = (1.0 / (r.args.size.0 as f64 * r.args.zoom))
        .min(ratio / (r.args.size.0 as f64 * r.args.zoom));
    let min_sub_pixel_spacing_2 = min_sub_pixel_spacing.powi(2);

    while i < r.args.limit
        && (r.args.conditional)(
            r, &z, &lastz, &der, &der_sum, &lastder, &der2, &lastder2, &c, &p,
        )
    {
        sums[2] += z.arg();
        sums[3] += lastz.arg();
        if i > 0.0 {
            sum += 0.5 + 0.5 * (3.0 * z.im.atan2(z.re)).sin();

            lastz = z;
            lastder = der;
            lastder2 = der2;
            sums[0] += (lastz - z).norm();
            sums[1] += (lastder - der).norm();
        }
        // for j in 0..4 {
        //     if zr[j].is_none() {
        //         let result = (r.args.iter_funct)(r, &zs[j], &der, &der2, &c, &p);
        //         zs[j] = result.0;
        //         if zs[j].norm() > r.args.bail {
        //             zr[j] = Some((i, s));
        //         }
        //     }
        // }
        let mut nz = z;
        // for (offset, rotation) in r.args.z_offsets.iter().zip(r.args.z_rotations.iter()) {
        //     if offset.0 .1 {
        //         if offset.1[0] {
        //             nz.re = (nz.re - offset.0 .0.re).abs() + offset.0 .0.re;
        //         } else {
        //             nz.re += offset.0 .0.re;
        //         }
        //         if offset.1[1] {
        //             nz.im = (nz.im - offset.0 .0.im).abs() + offset.0 .0.im;
        //         } else {
        //             nz.im += offset.0 .0.im;
        //         }
        //     }
        //     if rotation.1 {
        //         nz *= Cf64::new(rotation.0.cos(), rotation.0.sin());
        //     }
        // }
        for offset_t in r.args.z_offsets.iter() {
            if offset_t.active {
                match offset_t.mode {
                    // Offset
                    LinearTransformMode::Offset => {
                        nz.re += offset_t.offset.re;
                        nz.im += offset_t.offset.im;
                        if offset_t.abs[0] {
                            nz.re = nz.re.abs();
                            if offset_t.apply_inv {
                                nz.re -= offset_t.offset.re;
                            }
                        }
                        if offset_t.abs[1] {
                            nz.im = nz.im.abs();
                            if offset_t.apply_inv {
                                nz.im -= offset_t.offset.im;
                            }
                        }
                    }
                    // Rotate
                    LinearTransformMode::Rotate => {
                        nz *= Cf64::new(
                            offset_t.scalar_or_angle.cos(),
                            offset_t.scalar_or_angle.sin(),
                        );
                        let mut abs = false;
                        if offset_t.abs[0] {
                            nz.re = nz.re.abs();
                            abs = true;
                        }
                        if offset_t.abs[1] {
                            nz.im = nz.im.abs();
                            abs = true;
                        }
                        if abs && offset_t.apply_inv {
                            nz /= Cf64::new(
                                offset_t.scalar_or_angle.cos(),
                                offset_t.scalar_or_angle.sin(),
                            );
                        }
                    }
                    // Scale
                    LinearTransformMode::Scale => {
                        nz *= offset_t.scalar_or_angle;
                        let mut abs = false;
                        if offset_t.abs[0] {
                            nz.re = nz.re.abs();
                            abs = true;
                        }
                        if offset_t.abs[1] {
                            nz.im = nz.im.abs();
                            abs = true;
                        }
                        if abs && offset_t.apply_inv {
                            nz /= offset_t.scalar_or_angle;
                        }
                    }
                    // Linear Transform
                    LinearTransformMode::Transform => {
                        nz.re += offset_t.offset.re;
                        nz.im += offset_t.offset.im;
                        nz = offset_t.matrix * nz;
                        if offset_t.abs[0] {
                            nz.re = nz.re.abs();
                            if offset_t.apply_inv {
                                nz.re -= offset_t.offset.re;
                            }
                        }
                        if offset_t.abs[1] {
                            nz.im = nz.im.abs();
                            if offset_t.apply_inv {
                                nz.im -= offset_t.offset.im;
                            }
                        }
                        if offset_t.apply_inv {
                            let inv = offset_t.matrix.inv();
                            if let Some(m) = inv {
                                nz = m * nz;
                            }
                        }
                    }
                    // LinearTransformMode::Power => {}
                    LinearTransformMode::Mobius => {
                        nz = mobius(nz, offset_t.mobius);
                        if offset_t.abs[0] {
                            nz.re = nz.re.abs();
                        }
                        if offset_t.abs[1] {
                            nz.im = nz.im.abs();
                        }
                        if offset_t.apply_inv {
                            nz = inv_mobius(nz, offset_t.mobius);
                        }
                    }
                    LinearTransformMode::Power => {
                        nz = nz.powf(offset_t.scalar_or_angle);

                        if offset_t.abs[0] {
                            nz.re = nz.re.abs();
                        }
                        if offset_t.abs[1] {
                            nz.im = nz.im.abs();
                        }
                        if offset_t.apply_inv {
                            nz = nz.powf(1.0 / offset_t.scalar_or_angle);
                        }
                    }
                    _ => {}
                }
            }
        }
        z = nz;
        let result = (r.args.iter_funct)(r, &z, &der, &der2, &c, &p);

        // z = z * z + (c.powc(z / c));
        // der = 2.0 * z + ((z.powc(z / c) * (1.0 + z.ln())) / c);
        if let Some(nc) = result.3 {
            c = nc;
        }
        z = result.0;
        sums[4] += (z - lastz).norm();
        der = if let Some(newv) = result.1 { newv } else { z };
        der2 = if let Some(newv) = result.2 { newv } else { z };
        der_sum += der;

        // start_angle = z.arg() + PI;
        // let new_angle = z.arg();
        // angle += new_angle;
        // if angle.abs() > PI * 2.0 {
        //     angle %= PI * 2.0;
        //     // rotations += 1.0;
        //     let d = (z - oold).norm() * min_sub_pixel_spacing;

        //     // let d = dif.re.hypot(dif.im);
        //     if d < v {
        //         i = r.args.limit;
        //         s = r.args.limit;
        //         break;
        //     } else {
        //         oold = z;
        //     }
        //     period = 0;
        // }
        // total_summed = 0.0;
        // rotations = 0.0;

        // der = (der * 2.0 * z) + 1.0;
        // z = (self.functs.iter_funct)(z, u, (self.functs.cmap_funct)(self, self.args.julia));
        // z = Cf64::new(z.re.abs(), z.im.abs());
        // z = z.powf(-2.0) + c;
        last_distance_min = distance_min;
        last_distance_max = distance_max;
        last_distance_avg = distance_avg;
        let distance = (r.args.orbit_trap)(r, &z, &lastz, &der, &lastder, &der2, &lastder2, &c, &p);
        distance_avg += distance;
        if distance < distance_min {
            distance_min = distance;
        }

        if distance > distance_max {
            distance_max = distance;
        }
        s += (-(z.norm())).exp();

        let d = (z - old).norm();
        let diff = (d - old_diff).abs();
        if diff < avg_diff / i && diff < min_sub_pixel_spacing_2 {
            i = r.args.limit;
            s = r.args.limit;
            break;
        }

        if d < min_sub_pixel_spacing {
            i = r.args.limit;
            s = r.args.limit;
            break;
        }

        period += 1;
        if period > 32 {
            period = 0;
            old = z;
        }
        avg_diff += d;
        old_diff = d;
        i += 1.0;
    }
    // for j in 0..4 {
    //     if zr[j].is_none() {
    //         zr[j] = Some((i, s));
    //     }
    // }

    // let scale_col = 1.0 / CIELab::new([scale, scale, scale, 1.0]);
    if ((!r.args.fill_set || i < r.args.limit) && !r.args.only_set)
        || (r.args.only_set && (i >= r.args.limit))
    {
        Xyz::from_color((r.args.color_funct)(
            r,
            RenderData {
                i,
                s,
                z,
                c,
                o: p,
                der,
                der2,
                der_sum,
                sum,
                lastz,
                lastder,
                lastder2,
                distance_min,
                distance_max,
                distance_avg: distance_avg / (i as f64),
                last_distance_min,
                last_distance_max,
                last_distance_avg,
                sums,
                zr: [None; 4],
            },
        ))
    } else {
        Xyz::from_color(r.args.set_color)
    }
}
pub fn zone_plate(_rend: &Renderer, p: Cf64) -> Color<Xyz, D65> {
    // let r = p.norm();
    // let v = ((r.powi(2) * 64.0) / PI).cos() - 0.25;
    // let mut m = 0.0;
    // if v > 0.0 {
    //     m = 1.0;
    // }
    let m = 0.5 * (1.0 + (p.norm_sqr() / 100.0).sin());
    // let v2 = (r.powi(2) * 12.0 + (2.0 * PI * (1.0 / 3.0))).cos() * 0.5 + 0.5;
    // let v3 = (r.powi(2) * 12.0 + (2.0 * PI * (2.0 / 3.0))).cos() * 0.5 + 0.5;

    Xyz::from_color(Rgb::new([m, m, m, 1.0]))
}
#[allow(unused)]
pub fn gauss(v: f64) -> f64 {
    (-(v * v) / 2.0).exp()
}
pub fn tent(x: f64) -> f64 {
    1.0 - x.abs()
}

pub fn tent_rand(x: f64) -> f64 {
    1.0 - x.sqrt()
}

pub fn tent_r(x: f64, r: f64) -> f64 {
    tent(x / r) / r
}

pub fn test_filter(x: f64) -> f64 {
    ((PI * x).cos() + 1.0) / 2.0
}

pub fn test_filter_r(x: f64, r: f64) -> f64 {
    (((PI * x) / r).cos() + 1.0) / r
}

/*

((12 - 9b - 6c) * |x|^3)
            + ((-18 + 12b + 6c) * |x|^2)
            + (6-2b)


(-1/3 - 2)(|x|^3)
            + (2.0 + 10.0) (|x|^2)
            + (-4 - 38/3) (|x|)
            + (8.0/3 + 24.0/3)

*/

pub fn mitchell(x: f64) -> f64 {
    let absx = x.abs();
    if absx < 1.0 {
        (7.0 * absx.powi(3)) - (12.0 * absx.powi(2)) + (16.0 / 3.0)
    } else if (1.0..2.0).contains(&absx) {
        ((-1.0 / 3.0 - 2.0) * x.abs().powi(3))
            + (12.0 * x * x)
            + (-20.0 * x.abs())
            + (8.0 / 3.0 + 8.0)
    } else {
        0.0
    }
}

pub fn mitchell_ab(x: f64, b: f64, c: f64) -> f64 {
    (if x.abs() < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * x.abs().powi(3))
            + ((-18.0 + 12.0 * b + 6.0 * c) * x * x)
            + (6.0 - 2.0 * b)
    } else if 1.0 <= x.abs() && x.abs() < 2.0 {
        ((-b - 6.0 * c) * x.abs().powi(3))
            + ((6.0 * b + 30.0 * c) * x * x)
            + ((-12.0 * b - 48.0 * c) * x.abs())
            + (8.0 * b + 24.0 * c)
    } else {
        0.0
    }) / 6.0
}

pub fn tent_pdf(x: f64) -> f64 {
    if x == 0.0 {
        x
    } else if x < 0.0 {
        1.0 - (-x).sqrt()
    } else {
        x.sqrt()
    }
}

pub fn tent_pdf_r(x: f64, r: f64) -> f64 {
    tent_pdf(x / r) / r
}

fn radical_inverse_2(a: u64) -> f64 {
    let mut a = a;
    const inv_base: f64 = 0.5;
    let mut reversed_digits: u64 = 0;
    let mut inv_base_n = 1.0;
    while a != 0 {
        let next = a / 2;
        let digit = a - next * 2;
        reversed_digits = reversed_digits * 2 + digit;
        inv_base_n *= inv_base;
        a = next
    }
    (reversed_digits as f64 * inv_base_n).min(1.0 - f64::EPSILON)
}

fn radical_inverse_3(a: u64) -> f64 {
    let mut a = a;
    const inv_base: f64 = 1.0 / 3.0;
    let mut reversed_digits: u64 = 0;
    let mut inv_base_n = 1.0;
    while a != 0 {
        let next = a / 3;
        let digit = a - next * 3;
        reversed_digits = reversed_digits * 3 + digit;
        inv_base_n *= inv_base;
        a = next
    }
    (reversed_digits as f64 * inv_base_n).min(1.0 - f64::EPSILON)
}

fn sinc(x: f64) -> f64 {
    let v = x * PI;
    v.sin() / v
}

// fn graphing(x: f64) -> Color<Xyz, D65> {

//     Color::new()
// }

// pub fn perturbation
#[allow(unused)]
pub fn perturbation_test(
    r: &Renderer,
    p: Cf64,
    pixel: Cf64,
    ref_zf: Vec<Complex<f32>>,
    delta_0f: &[Complex<f32>],
    j: usize,
) -> Color<Xyz, D65> {
    let c = Complex::<f32>::new(p.re as f32, p.im as f32);
    let mut z = Complex::<f32>::default();

    let mut a_n = Complex::<f32>::new(1.0, 0.0);
    let mut b_n = Complex::<f32>::new(0.0, 0.0);
    let mut c_n = Complex::<f32>::new(0.0, 0.0);

    // // Iterate reference orbit

    let mut i = 0;
    while i < r.args.limit as usize && z.re * z.re + z.im * z.im < r.args.bail as f32 {
        // z = z * z + c;
        a_n = 2.0 * ref_zf[i % j] * a_n + 1.0;
        b_n = 2.0 * ref_zf[i % j] * b_n + a_n * a_n;
        c_n = 2.0 * ref_zf[i % j] * c_n + 2.0 * a_n * b_n;
        z = a_n * delta_0f[0] + b_n * delta_0f[1] + c_n * delta_0f[2] + c * delta_0f[3];

        i += 1;
    }
    let nz = (ref_zf[j - 1]);
    let theta = z.arg() as f64;
    let red = theta.cos() as f64;
    let b = theta.sin() as f64;
    let g = (red + b) / 2.0;

    Xyz::from_color(Rgb::new([
        red.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        1.0,
    ]))
}

#[allow(clippy::needless_lifetimes, unused)]
pub fn sampling(r: &Renderer, i: usize, s: usize) -> Color<Xyz, D65> {
    let (x, y) = (i % r.width(), i / r.width());
    let ratio = r.height() as f64 / r.width() as f64;
    let izoom = 1.0 / r.args.zoom;
    let i_size = (1.0 / r.width() as f64, 1.0 / r.height() as f64);
    let mut out = Xyz::new([0.0, 0.0, 0.0, 0.0]);
    // let radius = Uniform::new(0.0, sr);
    // let angle = Uniform::new(0.0, 2.0 * PI);
    // let mut rng = rand::thread_rng();
    let idk = u64::from_ne_bytes(r.time.elapsed().as_secs_f64().to_ne_bytes()) / 2;
    let rng = fastrand::Rng::new();
    let sqr2 = 2f64.sqrt();

    let opx = (x as f64 + 0.5) * i_size.0 - 0.5;
    let opy = (y as f64 + 0.5) * i_size.1 - 0.5;
    let pixel = (Cf64::new(opx * izoom, opy * ratio * izoom)) + r.args.origin;
    // let mut ref_z = Cf64::default();
    // let mut ref_c = (Cf64::new(opx * izoom, opy * ratio * izoom)) + r.args.origin;
    // let mut ref_zf = Vec::new();
    // let mut j = 0;
    // while j < r.args.limit as usize && ref_z.re * ref_z.re + ref_z.im * ref_z.im < r.args.bail {
    //     ref_z = ref_z * ref_z + ref_c;
    //     ref_zf.push(Complex::<f32>::new(ref_z.re as f32, ref_z.im as f32));
    //     j += 1;
    // }
    // ref_z = ref_z * ref_z + ref_c;

    for samplec in 0..s {
        // let theta = (rng.f64() * 2.0 * PI);
        // let mut radius = rng.f64().sqrt() * sr;
        // let mut radius = radius.sample(&mut rng).sqrt();
        // let theta = angle.sample(&mut rng);

        // if !radius.is_normal() {
        //     radius = 0.0;
        // }
        // let (mut rx, mut ry) = (
        //     (radical_inverse_2((r.rendered_samples + samplec) as u64 + 1) - 0.5)
        //         + (rng.f64() - 0.5),
        //     (radical_inverse_3((r.rendered_samples + samplec) as u64 + 1) - 0.5)
        //         + (rng.f64() - 0.5),
        // );
        let (rx, ry) = if samplec + r.rendered_samples == 0 {
            (0.0, 0.0)
        } else {
            (
                (radical_inverse_2(idk + (r.rendered_samples + samplec) as u64) - 0.5) * 2.0,
                (radical_inverse_3(idk + (r.rendered_samples + samplec) as u64) - 0.5) * 2.0,
            )
        };

        // let v = (rx * rx + ry * ry).sqrt() / sqr2;
        // rx -= (rx * (tent(rx)));
        // ry -= (ry * (tent(ry)));
        // let v = PI * sr * sr;
        // let v = 4.0 * tent(rx * rx + ry * rx);
        // let v = 1.5 * 1.5;
        // let v = m * 2.0;
        // let v = ((1.0 - rx.abs()) * (1.0 - ry.abs())) * 4.0;
        // let v = mitchell_ab(rx, 1.0 / 3.0, 1.0 / 3.0) * mitchell_ab(ry, 1.0 / 3.0, 1.0 / 3.0) * 4.0;
        // v =

        let px = (x as f64 + 0.5 + rx) * i_size.0 - 0.5;
        let py = (y as f64 + 0.5 + ry) * i_size.1 - 0.5;

        let p = (Cf64::new(px * izoom, py * ratio * izoom)) + r.args.origin;

        // let delta_0 = ref_c - p;
        // let delta_0_2 = delta_0 * delta_0;
        // let delta_0_3 = delta_0_2 * delta_0;
        // let delta_0_4 = delta_0_3 * delta_0;
        // let delta_0f = [
        //     Complex::<f32>::new(delta_0.re as f32, delta_0.im as f32),
        //     Complex::<f32>::new(delta_0_2.re as f32, delta_0_2.im as f32),
        //     Complex::<f32>::new(delta_0_3.re as f32, delta_0_3.im as f32),
        //     Complex::<f32>::new(delta_0_4.re as f32, delta_0_4.im as f32),
        // ];

        // let scale = Xyz::new::<D65>([v; 4]);
        out += (r.method)(r, p);
    }
    out
}

#[allow(unused)]
#[inline]
pub fn fractal_mandelbrot(
    rend: &Renderer,
    z: &Cf64,
    der: &Cf64,
    der2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> (Cf64, Option<Cf64>, Option<Cf64>, Option<Cf64>) {
    (
        z.mul_add(z, c),
        Some((2.0 * z * der) + 1.0),
        Some(2.0 * (der2 * z + (der * der))),
        None,
    )
}

#[allow(unused)]
#[inline]
pub fn fractal_spade(
    rend: &Renderer,
    z: &Cf64,
    der: &Cf64,
    der2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> (Cf64, Option<Cf64>, Option<Cf64>, Option<Cf64>) {
    (
        z.powc(*z) + (z / (c + (2.0 * f64::EPSILON))),
        Some((2.0 * z * der) + 1.0),
        Some(2.0 * (der2 * z + (der * der))),
        None,
    )
}

// impl TryInto<

#[allow(unused)]
#[inline]
pub fn fractal_burning_ship(
    rend: &Renderer,
    z: &Cf64,
    der: &Cf64,
    der2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> (Cf64, Option<Cf64>, Option<Cf64>, Option<Cf64>) {
    (
        {
            let mut nz = *z;
            nz.re = nz.re.abs();
            nz.im = nz.im.abs();
            nz.mul_add(nz, *c)
        },
        Some((der * 2.0 * z) + 1.0),
        Some(2.0 * (der2.mul_add(z, &(der * der)))),
        None,
    )
}
#[allow(dead_code, unused_variables)]
pub fn OLD_coloring(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let s = (data.s / rend.args.limit).powf(rend.args.cexp);
    let val = (PI * s).cos();
    let val = 1.0 - (val * val);

    CIELab::from_color(CIELch::new::<D65>([
        74.0 - (74.0 * val),
        28.0 + (74.0 * val),
        (s * 360.0).powf(1.5) % 360.0,
        1.0,
    ]))
}

#[allow(dead_code, unused_variables)]
pub fn OLD_normal_map(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    //  Normal mapped variant
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.args.angle.to_radians()).cos(),
        (rend.args.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;
    let t = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    // Normal
    let lo = 0.5 * data.lastz.norm_sqr().ln();
    let u = data.lastz
        * data.lastder
        * ((1.0 + lo) * (data.lastder * data.lastder).conj()
            - lo * (data.lastz * data.lastder2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.args.angle.to_radians()).cos(),
        (rend.args.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;
    let t2 = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let v = (t * f) + (t2 * (1.0 - f));

    let val = (PI * (1.0 - t)).cos();
    let val = 1.0 - (val * val);

    // Color::new(t.to_arr().0, ColorType::RGBA)
    CIELab::from_color(CIELch::new::<D65>([
        74.0 - (74.0 * val),
        28.0 + (74.0 * val),
        ((data.s / rend.args.limit).powf(rend.args.cexp) * 360.0).powf(1.5) % 360.0,
        1.0,
    ]))
}
// #[allow(dead_code, unused_variables)]
// fn hybrid_lober(z: Cf64, dz: Cf64, c: Cf64, j: Cf64) -> (Cf64, Cf64) {
//     let mut nz = z * z * z + c;
//     nz = Cf64::new(nz.re.abs(), nz.im.abs());
//     nz = nz * nz + c;
//     (Cf64::new(0.0, 0.0), Cf64::new(nz.re.abs(), nz.im.abs()) + c)
// }
pub fn fractal_hybrid_lober(
    rend: &Renderer,
    z: &Cf64,
    der: &Cf64,
    der2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> (Cf64, Option<Cf64>, Option<Cf64>, Option<Cf64>) {
    (
        {
            let mut nz = z * z * z + c;
            // nz = Cf64::new(nz.re.abs(), nz.im.abs());
            nz.re = nz.re.abs();
            nz.im = nz.im.abs();
            nz = nz * nz + c;
            nz.re = nz.re.abs();
            nz.im = nz.im.abs();
            nz + c
        },
        Some((der * 2.0 * z) + 1.0),
        Some(2.0 * (der2.mul_add(z, &(der * der)))),
        None,
    )
}

pub fn fractal_Swirly(
    rend: &Renderer,
    z: &Cf64,
    der: &Cf64,
    der2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> (Cf64, Option<Cf64>, Option<Cf64>, Option<Cf64>) {
    (
        {
            // (z * z + c) / ((z - c).sqrt())
            // ((z - c).sqrt()) * (z * z + c)
            //  - c
            let mut nz = *z;
            // let mut nc = c;
            // nz.im = nz.im.abs();
            nz.re = z.re.tan();
            // nc.re = nc.re.cos();
            // nc.im = nc.im.cos();
            nz * nz + c
        },
        Some((der * 2.0 * z) + 1.0),
        Some(2.0 * (der2.mul_add(z, &(der * der)))),
        None,
    )
}

fn fb(z: Cf64) -> Cf64 {
    // let mut t = z * 1.9 - 0.9;
    let mut t = z;
    t.re = t.re * 1.9 - 0.9;
    t.im = t.im * 1.9 - 0.9;
    (t + (1.0 / t)) / Cf64::new(1.0, 2.0)
}

// fn fc(z: Cf64) -> Cf64 {
//     let mut t = z;
//     t.re = t.re * 1.04;
//     t.im = t.im * 1.04;
// }

pub fn fractal_testing(
    rend: &Renderer,
    z: &Cf64,
    der: &Cf64,
    der2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> (Cf64, Option<Cf64>, Option<Cf64>, Option<Cf64>) {
    (
        {
            let mut nz = *z;
            // for (offset, rotation) in rend.args.z_offsets.iter().zip(rend.args.z_rotations.iter()) {
            //     // let (offset, rotation) =
            //     #[allow(clippy::collapsible_if)]
            //     if !rotation.1[0] {
            //         if rotation.1[1] {
            //             if offset.1[0] {
            //                 nz.im = (nz.im - offset.0 .0.im).abs() + offset.0 .0.im;
            //             }
            //         }
            //         nz *= rotation.0 .0;
            //         nz.re = nz.re.abs();
            //         nz.im = nz.im.abs();
            //         if offset.1[0] {
            //             nz.re = (nz.re - offset.0 .0.re).abs() + offset.0 .0.re;
            //         }
            //         if rotation.1[1] {
            //             if offset.1[0] {
            //                 nz.im = (nz.im - offset.0 .0.im).abs() + offset.0 .0.im;
            //             }
            //         }
            //     } else {
            //         if offset.1[0] {
            //             nz.re = (nz.re - offset.0 .0.re).abs() + offset.0 .0.re;
            //         }
            //         if rotation.1[1] {
            //             if offset.1[0] {
            //                 nz.im = (nz.im - offset.0 .0.im).abs() + offset.0 .0.im;
            //             }
            //         }
            //         nz *= rotation.0 .0;
            //         nz.re = nz.re.abs();
            //         nz.im = nz.im.abs();
            //         if rotation.1[1] {
            //             if offset.1[0] {
            //                 nz.im = (nz.im - offset.0 .0.im).abs() + offset.0 .0.im;
            //             }
            //         }
            //     }
            // }

            // nz.re = (nz.re - rend.args.z_offset1.re).abs() + rend.args.z_offset1.re;
            // nz.im = (nz.im - rend.args.z_offset1.im).abs() + rend.args.z_offset1.im;
            // nz.im = nz.im.abs();
            // nz.re = nz.re.tan();

            // // z * z * z.norm().powi(2) + c

            // nz.im = c.im.atan2(z.re).sin() * (z - c).norm();
            // nz.re = z.im.atan2(c.re).cos() * (z - c).norm();
            // nz * nz * nz + (c * z)
            // (z.powc(*z) + c.powc(*c)) / (z - c).norm() + c

            // z * z + c.powc(z / c)

            // (z / (z - c).sin()).powc(z / c) + c

            // z.powc(*z) + (z / c) * ((12.0 / c).cos() * (12.0 / c).sin())

            // c.powc(*c) + (z * z * z) / z.powc(*c).tan()

            // z.powc(z / 1.5) - c.cos() + (z * z * z)
            // c.powc(*c) + (z * z * z) / (z * c).tan()
            // let a = (nz * Cf64::new(nz.im, nz.re)).norm();
            // let mut s = Cf64::default();
            // for i in 0..16 {
            //     if i % 2 == 0 {
            //         s += (z + c).powi(i) / c.powi(i - 1)
            //     } else {
            //         s -= (z + c).powi(i) / c.powi(i - 1)
            //     }
            // }
            // s + c

            // z * z * z - z * z + c

            // let a = (z - c).norm();
            // (z.powf(a + 1.0) / a) * z + c
            // nz.re = (z - p).exp().norm() * z.re + z.re;
            // nz.im = (z - p).exp().norm() * z.im + z.im;
            // nz.im = nz.im.sin();

            //PROMISING
            // z * (2.0 * (c - z).norm() / (c - z)) + c
            //PROMISING^^^

            // z * nz * z.norm() + c

            // (z.cos()) - (z * z) + c
            // ((z / c.norm()) * (c / z.norm())).powi(2) - c
            // nz * nz + c
            ((z * z * z - c) / z) + c
            // z * z + c
        },
        Some((der * 2.0 * z) + 1.0),
        Some(2.0 * (der2.mul_add(z, &(der * der)))),
        // Some(
        //     Cf64::new(
        //         (z - c).norm().cos() * c.norm(),
        //         (z - c).norm().sin() * c.norm(),
        //     ) * (1.0 / c),
        // ),
        None,
    )
}
#[allow(unused)]
#[inline]
pub fn fractal_init(_z: Cf64, _c: Cf64) -> Cf64 {
    Cf64::new(0.0, 0.0)
}
#[allow(unused)]
#[inline]
pub fn fractal_cmap(r: &Renderer, c: Cf64) -> Cf64 {
    c
}
#[inline(always)]
pub fn not_in_gamut(c: Color<Rgb, D65>) -> bool {
    c.0 < 0.0 || c.0 > 1.0 || c.1 < 0.0 || c.1 > 1.0 || c.2 < 0.0 || c.2 > 1.0
}
pub fn lch_mapping(
    rend: &Renderer,
    data: &RenderData,
    m: f64,
    a: Option<f64>,
    b: Option<f64>,
    l: Option<f64>,
    s: Option<f64>,
    invert: bool,
) -> Color<CIELab, D65> {
    let val = (PI * (1.0 - m)).cos();
    let mut val = 1.0 - (val * val);
    if invert {
        val = 1.0 - val;
    }
    let l = if let Some(l2) = l { l2 } else { 100.0 };
    if let Some(val2) = a {
        lch_gamut_mapping(rend, CIELch::new::<D65>([l * val, 128.0 * val, val2, 1.0]))
    } else if let Some(val) = b {
        lch_gamut_mapping(rend, CIELch::new::<D65>([l, 128.0, val, 1.0]))
    } else {
        let s2 = (((if let Some(s2) = s {
            s2
        } else {
            data.s / rend.args.limit
        } + rend.args.hue_offset_pre_exp)
            .powf(rend.args.cexp)
            * rend.args.hue_scale)
            * 360.0)
            + rend.args.hue_offset_post_exp;
        /*
        let h = ((((data.s / rend.args.limit) + rend.args.hue_offset_pre_exp).powf(rend.args.cexp)
                * rend.args.hue_scale)
                * 360.0)
                + rend.args.hue_offset_post_exp;
        */
        lch_gamut_mapping(
            rend,
            CIELch::new::<D65>([l * val, 128.0 * val, s2 % 360.0, 1.0]),
        )
    }
}
/*

let mut a = if data.i >= rend.args.limit {
        1.0
    } else {
        1.0 - (1.0 / (data.s + 1.0))
    };
    // v = 1.0 - (data.s / rend.args.limit).powf(1.5);
    let min = data.distance_min;
    let max = data.distance_max;

    let m = (data.distance_avg - min) / (max - min);
    let m = if !m.is_normal() || m < 0.0 {
        0.0
    } else if m > 1.0 {
        1.0
    } else {
        m
    };

    let m = m.powf(rend.args.h2);

    let val = (PI * (1.0 - m)).cos();
    let val = 1.0 - (val * val);

    let mut color = lch_mapping(rend, &data, m, None, None, None, None);
    color.3 = a;
    color
*/
pub fn newton_coloring(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let color = if data.i < 0.0 || data.i == rend.args.limit {
        Xyz::from_color(Rgb::new([0.0, 0.0, 0.0, 1.0]))
    } else {
        let v2 = 1.0 - (1.0 / (data.s + 1.0)).sqrt();
        let v = 1.0 / (data.z.norm() + 1.0);
        // let v = 1.0 - (((s * PI) / (rend.args.limit / 2.0)).cos() + 1.0);
        // let offset = (rend.args.h2);
        let theta = ((data.z.re.atan2(data.z.im) * rend.args.h2 + rend.args.h) / PI) + 0.5;
        let mut val = 1.0 - (PI * (1.0 - v)).cos().powi(2);
        if !val.is_normal() {
            val = 0.0;
        }
        val = val.clamp(0.0, 1.0).powf(rend.args.cexp);
        let l = 100.0;
        // let mut s = (theta * 360.0).powf(1.5) % 360.0;
        // let mut s = ((theta + rend.args.h) * 360.0).powf(rend.args.cexp) % 360.0;
        let mut s = ((theta * 360.0).powf(rend.args.hue_exponent) * rend.args.hue_scale)
            + rend.args.hue_offset_pre_exp;
        if !s.is_normal() || s > 360.0 {
            s = 360.0;
        }
        Xyz::from_color(lch_gamut_mapping(
            rend,
            CIELch::new::<D65>([
                (l * (1.0 - val)),
                100.0 * v2,
                (s + rend.args.newton_var1) % 360.0,
                1.0,
            ]),
        ))
    };
    CIELab::from_color(if rend.is_viewport {
        let f = 0.9;

        if data.c.norm() < 1.025 && data.c.norm() > 0.975 {
            (f * color) + ((1.0 - f) * Xyz::from_color(Rgb::new([1.0, 1.0, 1.0, 1.0])))
        } else if data.c.norm() < 0.26 && data.c.norm() > 0.24 {
            (f * color) + ((1.0 - f) * Xyz::from_color(Rgb::new([0.0, 0.0, 0.0, 1.0])))
        } else {
            color
        }
    } else {
        color
    })
}
pub fn cool_coloring3(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    if data.i >= rend.args.limit {
        let v = (data.s / rend.args.limit).powf(rend.args.cexp);
        let val = (PI * (1.0 - v)).cos();
        let val = 1.0 - (val * val);
        // let mut color = CIELab::from_color(Xyz::from_color(Rgb::new([v, v, v, 1.0])));
        lch_gamut_mapping(
            rend,
            CIELch::new::<D65>([100.0 * val, 128.0 * val, (data.z.arg() / PI) + 0.5, 1.0]),
        )
    } else {
        let v = (data.s / rend.args.limit).powf(rend.args.cexp2);
        let val = (PI * (1.0 - v)).cos();
        let val = 1.0 - (val * val);
        // let mut color = CIELab::from_color(Xyz::from_color(Rgb::new([v, v, v, 1.0])));
        lch_gamut_mapping(
            rend,
            CIELch::new::<D65>([100.0 * val, 128.0 * val, (data.z.arg() / PI) + 0.5, 1.0]),
        )
    }
    // if data.i < rend.args.limit {
    //     let a = 1.0 - ((1.0 / (data.s + 1.0)).sqrt() + 1.0).log(1.5);
    //     color.3 = a;
    // }
}
pub fn cool_coloring2(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    if data.i >= rend.args.limit {
        let v = (data.s / rend.args.limit).powf(rend.args.h2);
        // CIELab::from_color(Xyz::from_color(Rgb::new([v, v, v, 1.0])))
        lch_mapping(rend, &data, v, None, None, None, None, false)
    } else {
        let v = (data.s / rend.args.limit).powf(rend.args.h);
        // CIELab::from_color(Xyz::from_color(Rgb::new([v, v, v, 1.0])))
        // let mut color = lch_mapping(rend, &data, v, None, None, None, None);
        let val = (PI * (1.0 - v)).cos();
        let val = 1.0 - (val * val);
        let l = 100.0;
        let s = (((data.s / rend.args.limit) * 360.0).powf(rend.args.hue_exponent)
            * rend.args.hue_scale)
            + rend.args.hue_offset_pre_exp;
        let mut color = lch_gamut_mapping(rend, CIELch::new::<D65>([l * val, 128.0 * val, s, 1.0]));
        color.3 = 1.0 - ((1.0 / (data.s + 1.0)).sqrt() + 1.0).log(1.5);
        color
    }
}
pub fn cool_coloring(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // let v = (data.s / rend.args.limit).powf(rend.args.cexp);
    // let mut color = CIELab::from_color(Xyz::from_color(Rgb::new([v, v, v, 1.0])));
    // if data.i < rend.args.limit {
    //     let a = 1.0 - ((1.0 / (data.s + 1.0)).sqrt() + 1.0).log(1.5);

    //     color.3 = a;
    // }
    // color
    let v = (data.s / rend.args.limit).powf(rend.args.h);
    // let mut color = CIELab::from_color(Xyz::from_color(Rgb::new([v, v, v, 1.0])));
    let mut color = lch_mapping(rend, &data, v, None, None, None, None, false);
    if data.i < rend.args.limit {
        let a = 1.0 - ((1.0 / (data.s + 1.0)).sqrt() + 1.0).log(1.5);
        color.3 = a;
    }
    color
}
// pub fn bump_mapping(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
//     let test = (4.0 * data.s)
//         - (data.zr[0].unwrap().1
//             + data.zr[1].unwrap().1
//             + data.zr[2].unwrap().1
//             + data.zr[3].unwrap().1);
//     let v = (test / rend.args.limit).powf(rend.args.cexp);
//     let s = (data.s / rend.args.limit).powf(rend.args.cexp2);
//     // let mut complex = Cf64::new(data.s, data.s);
//     // complex.re = data.zr[1].unwrap().1 - complex.re;
//     // complex.im = data.zr[3].unwrap().1 - complex.im;
//     // let v = ((data.s - complex.norm()).abs() / rend.args.limit)
//     //     .powf(rend.args.cexp2)
//     //     .clamp(0.0, 1.0);
//     // let h = complex.arg();
//     // let v = 1.0 - (data.s / rend.args.limit).powf(rend.args.cexp);
//     // let v = (complex.norm() / rend.args.limit).powf(rend.args.h2);
//     // let v = 0.5;
//     // lch_gamut_mapping(
//     //     rend,
//     //     CIELch::from_color(lch_mapping(rend, &data, h, None, None, None, None, true)),
//     // )
//     // let h = (data.zr[1].unwrap().1 - data.zr[0].unwrap().1)
//     //     .atan2(data.zr[3].unwrap().1 - data.zr[2].unwrap().1);
//         let h2 =
//     CIELab::new([
//         (100.0 * s) - (100.0 * v),
//         ((h).cos() - 0.5) * 200.0,
//         ((h).sin() - 0.5) * 200.0,
//         1.0,
//     ])
//     // CIELab::from_color(Rgb::new([v, v, v, 1.0]))
// }

#[allow(dead_code, unused_variables)]
pub fn fractal_coloring(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // v = 1.0 - (data.s / rend.args.limit).powf(1.5);
    let min = data.distance_min;
    let max = data.distance_max;

    let m = (data.distance_avg - min) / (max - min);
    let m = if !m.is_normal() || m < 0.0 {
        0.0
    } else if m > 1.0 {
        1.0
    } else {
        m
    };

    let m = m.powf(rend.args.h2);

    let val = (PI * (1.0 - m)).cos();
    let mut val = 1.0 - (val * val);
    if !val.is_normal() {
        val = 0.0;
    }
    let mut color = lch_mapping(rend, &data, val, None, None, None, None, false);
    if data.i < rend.args.limit {
        let mut a = 1.0 - ((1.0 / (data.s + 1.0)).sqrt() + 1.0).log(1.5);

        // let v = 1.0 - a;
        color.3 = a;
    }
    color
    // let v = if data.s < rend.args.limit { 1.0 } else { 0.0 };
}
// &z, &lastz, &der, &lastder, &der2, &lastder2, &c, &p
#[allow(dead_code, unused_variables)]
pub fn fractal_default_bail(
    r: &Renderer,
    z: &Cf64,
    lastz: &Cf64,
    der: &Cf64,
    der_sum: &Cf64,
    lastder: &Cf64,
    der2: &Cf64,
    lastder2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> bool {
    z.norm() < r.args.bail
}
#[allow(dead_code, unused_variables)]
pub fn fractal_der_bail(
    r: &Renderer,
    z: &Cf64,
    lastz: &Cf64,
    der: &Cf64,
    der_sum: &Cf64,
    lastder: &Cf64,
    der2: &Cf64,
    lastder2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> bool {
    (der_sum * der_sum).norm_sqr() < r.args.derbail && z.norm_sqr() * z.norm_sqr() < r.args.bail
}
#[allow(unused)]
pub fn fractal_orbit_trap(
    rend: &Renderer,
    z: &Cf64,
    lastz: &Cf64,
    der: &Cf64,
    lastder: &Cf64,
    der2: &Cf64,
    lastder2: &Cf64,
    c: &Cf64,
    p: &Cf64,
) -> f64 {
    // (z - Cf64::new(0.0, 0.0)).norm_sqr()

    let x = z.re;
    let y = z.im;
    let m = x * x + y * y;
    let mut nz = *z;
    let sre = nz.re.signum();
    let sim = nz.im.signum();
    nz.re = nz.re.abs();
    nz.im = nz.im.abs();
    nz = 1.0 / (nz + 1.0);
    let value = nz.norm();
    // let s = 2.0 / 3.0;

    // ((x / s).powi(2) + (y / s).powi(2)).cos() - ((y / s).atan2(x / s).cos().powi(2) + 0.5)
    // (x.powi(2) + y.powi(2)).cos()

    // Original grid!
    // ((2.0 * x).sin() * (x + (3f64.sqrt() * y)).sin() * (x - (3f64.sqrt() * y)).sin()).abs()

    // 1.0 - (x * x + y * y).tan()

    // if x.sin() / rend.args.zoom <= y.cos() {
    //     y.cos() - x.sin()
    // } else {
    //     x.sin() - y.cos()
    // }

    // Circley :3
    // 1.0 - ((E * x * x * y * y * y) / ((x * x + y * y) - 1.0).powi(3))

    // absolutely amazing
    1.0 - ((x * PI).tan())
        .atan2(((4.0 * y) / PI) + (PI / 2.0))
        .powi(2)

    // X's >.>
    // 1.0 - ((x.cos() * y.cos() * y.cos().sqrt()) / (y.sin().powi(2) - 1.0))

    //inf :O
    // 1.0 - ((2.0 * m * m.cos()) / (3.0 * m))
}
fn newt(rend: &Renderer, z: Cf64, c: Cf64) -> Cf64 {
    // (z * z * z - 1.0).sin()

    (z * z * z) + ((c - 1.0) * z) + c

    // z - ((z * z * z - 1.0) / (3.0 * z * z))

    // z.powc(z) - 1.0

    // let a = z * z;
    // let b = a * a * z;

    // let c = (a * b).powf(-2.0);

    // a + b - c + 1.0

    // z.sin().powi(3) + 1.0

    // let a = z.powi(3);
    // let b = a * a;
    // let c = b * b;
    // a + b + c - 3.0

    // z.powf(0.3) + z.powf(3.2)

    // (z * z.sin()) + (z * z)

    // (Cf64::new(0.0, 1.0) * z).exp() + z * z
    // let c = z;
}

fn newt_d(rend: &Renderer, z: Cf64, c: Cf64) -> Cf64 {
    // 3.0 * z * z * (z * z * z - 1.0).cos()

    c + (3.0 * z * z) - 1.0

    // 3.0 * z * z

    // (2.0 / 3.0) - (2.0 / (3.0 * z * z * z))

    // (20.0 / z.powi(21)) + (5.0 * z.powi(4)) + (2.0 * z)
    // 3.0 * z.sin().powi(2) * z.cos()
    // z * (z.powc(z)) * (z.ln() + 1.0)

    // let a = z * z;
    // let b = a * a * z;
    // let c = b * b * z;
    // (12.0 * c) + (6.0 * b) + a

    // (3.3 * z.powf(2.2)) + (0.3 / z.powf(0.7))

    // z.cos() - (z * (z.sin() - 2.0))

    // Cf64::new(0.0, 1.0) * (Cf64::new(0.0, 1.0) * z).exp() + (2.0 * z)
}
fn mandelbrot_set_simple(rend: &Renderer, z: Cf64, c: Cf64) -> (Cf64, Cf64) {
    // let c = z;
    // let mut z_dz = (z, Cf64::new(0.0, 0.0));
    // let mut dz = Cf64::new(0.0, 0.0);
    let mut dz = if rend.args.julia.1 {
        rend.args.julia.0
    } else {
        Cf64::new(1.0, 0.0)
    };
    let mut z = z;
    let mut i = 0.0;
    while i < rend.args.limit && z.norm_sqr() < rend.args.bail {
        dz = (dz * 2.0 * z) + 1.0;
        z = z * z + c;
        i += 1.0;
    }
    (z, dz)
}
#[allow(dead_code, unused_variables)]
pub fn trap_avg(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let min = data.distance_min;
    let max = data.distance_max;

    let m = (data.distance_avg - min) / (max - min);
    let m = if !m.is_normal() || m < 0.0 {
        0.0
    } else if m > 1.0 {
        1.0
    } else {
        m
    };

    let m = m.powf(rend.args.h2);

    let val = (PI * (1.0 - m)).cos();
    let val = 1.0 - (val * val);

    lch_mapping(rend, &data, m, None, None, None, Some(m), false)
}

#[allow(dead_code, unused_variables)]
pub fn trap_scale(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let min = data.distance_min;
    let max = data.distance_max;

    let m = 1.0 - (1.0 / ((max - min).abs() + 1.0));
    let m = if !m.is_normal() || m < 0.0 {
        0.0
    } else if m > 1.0 {
        1.0
    } else {
        m
    };

    let m = m.powf(rend.args.h2);

    let val = (PI * (1.0 - m)).cos();
    let val = 1.0 - (val * val);

    lch_mapping(rend, &data, m, None, None, None, None, false)
}

#[allow(dead_code, unused_variables)]
pub fn trap_sqrt(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let min = data.distance_min;
    let max = data.distance_max;

    let m = (1.0 - (1.0 / (max - min).abs())).sqrt();
    let m = if !m.is_normal() || m < 0.0 {
        0.0
    } else if m > 1.0 {
        1.0
    } else {
        m
    };

    let m = m.powf(rend.args.h2);

    lch_mapping(rend, &data, m, None, None, None, None, false)
}

pub fn lch_gamut_mapping(rend: &Renderer, color: Color<CIELch, D65>) -> Color<CIELab, D65> {
    let mut low = 0.0;
    let mut high = color.1;
    let mut nc = color.1;
    let mut iters = 0;
    while high - low > 0.001 {
        if !not_in_gamut(Rgb::from_color(CIELch::new([color.0, nc, color.2, 1.0]))) {
            low = nc;
        } else {
            high = nc;
        }
        nc = (low + high) * 0.5;
        iters += 1;
    }
    CIELab::from_color(CIELch::new([color.0, nc, color.2, 1.0]))
}
pub fn newton(rend: &Renderer, p: Cf64) -> Color<Xyz, D65> {
    if rend.is_viewport
        && rend.args.julia.1
        && (p - rend.args.julia.0).norm() * (32.0) * rend.args.zoom
            < ((10.0) / ((rend.args.size.0 * rend.args.size.1) as f64)) + 0.5
    {
        return Xyz::from_color(Rgb::new([0.0, 0.0, 0.0, 1.0]));
    }
    // let a = rend.args.julia;
    let a = Cf64::new(1.0, 0.0);
    let mut z = p;
    let mut lastz = Cf64::new(0.0, 0.0);
    let mut lasts;
    let mut i = 0.0;
    let mut s = 0.0;
    while i < rend.args.limit && (lastz - z).norm_sqr() > rend.args.bail2 {
        lasts = s;
        s += (lastz - z).norm();
        if !s.is_normal() {
            s = lasts;
            break;
        }
        lastz = z;
        // Compute m-set resulting Z and derivative of Z
        // returns (z, dz)
        // let result = mandelbrot_set_simple(rend, z, p);
        let result = (
            newt(rend, z, rend.args.julia.0),
            newt_d(rend, z, rend.args.julia.0),
        );
        // Compute newton iteration with f / f'
        // so, z / dz
        z -= (result.0 / result.1) * a;
        if !z.is_normal() {
            z = lastz;
            break;
        }
        i += 1.0;
    }
    Xyz::from_color((rend.args.color_funct)(
        rend,
        RenderData {
            i,
            s,
            z,
            c: p,
            o: p,
            sum: f64::NAN,
            der: Cf64::new(0.0, 0.0),
            der2: Cf64::new(0.0, 0.0),
            der_sum: z,
            lastz,
            lastder: Cf64::new(0.0, 0.0),
            lastder2: Cf64::new(0.0, 0.0),
            distance_avg: f64::NAN,
            distance_min: f64::NAN,
            distance_max: f64::NAN,
            last_distance_avg: f64::NAN,
            last_distance_min: f64::NAN,
            last_distance_max: f64::NAN,
            sums: Vec::new(),
            zr: [None; 4],
        },
    ))

    // Xyz::new([v, v, v, 1.0])
    //     * ((color * v) + ((1.0 - v) * Xyz::from_color(Rgb::new([1.0, 1.0, 1.0, 1.0]))))
}

#[allow(dead_code, unused_variables, unused_parens)]
pub fn floral(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // let v = 1.0 / (data.spread_ch + 1.0);
    let mut v1 = ((((data.sums[2]) / (data.i)) / PI) + 0.5);
    let mut v2 = ((((data.sums[3]) / (data.i)) / PI) + 0.5);
    // let mut v1 = 1.0 - (1.0 / ((data.sums[0] - data.sums[1]).abs() + 1.0));
    let mut f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());
    if !f.is_normal() {
        f = 0.0;
    }
    if !v1.is_normal() {
        v1 = 0.0;
    }
    if !v2.is_normal() {
        v2 = 0.0;
    }
    let v = ((v1 * f) + ((1.0 - f) * v2));
    // let v = v1;
    // let v = (data.s / rend.args.limit).powf(rend.args.cexp);
    // lch_mapping(rend, &data, v, None, None, None, None)
    let val = (PI * (1.0 - v)).cos();
    let val = 1.0 - (val * val);
    let l = 70.0;
    let s = ((data.s / rend.args.limit).powf(rend.args.cexp) * 360.0);
    lch_gamut_mapping(
        rend,
        CIELch::new::<D65>([l * val, 128.0 * val, (s.exp() + 180.0) % 360.0, 1.0]),
    )
    // let v = 1.0 - (data.s / rend.args.limit).powf(rend.args.cexp);
    // let v = if data.s < rend.args.limit { 1.0 } else { 0.0 };
    // CIELab::from_color(Rgb::new([v, v, v, 1.0]))
}

pub fn coloring2(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let theta = (data.z.arg() + PI) % (2.0 * PI);
    let (r, b) = (theta.cos(), theta.sin());
    CIELab::from_color(Rgb::new([r, (r + b) / 2.0, b, 1.0]))
}

#[allow(dead_code, unused_variables)]
pub fn normal_map(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    //  Normal mapped variant
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.args.angle.to_radians()).cos(),
        (rend.args.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;
    let t = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    // Normal
    let lo = 0.5 * data.lastz.norm_sqr().ln();
    let u = data.lastz
        * data.lastder
        * ((1.0 + lo) * (data.lastder * data.lastder).conj()
            - lo * (data.lastz * data.lastder2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.args.angle.to_radians()).cos(),
        (rend.args.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;
    let t2 = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let v = (t * f) + (t2 * (1.0 - f));

    lch_mapping(rend, &data, t, None, None, Some(100.0), None, false)
}

#[allow(dead_code, unused_variables)]
pub fn stripe_normal(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // Triangle inequality average
    let last_orbit = 0.5 + 0.5 * (3.0 * data.lastz.arg()).sin();
    // let last_orbit = idk(data.lastz);
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());
    let v = 1.0
        - ((f * (data.sum / data.i).powf(1.0 / PI))
            + (1.0 - f) * ((data.sum - last_orbit) / (data.i - 1.0)).powf(1.0 / PI));

    let val = (PI * v).cos();
    let val = 1.0 - (val * val);

    let c = 28.0 + (74.0 * val);
    let l = 74.0 - (74.0 * val);
    let h = ((((data.s / rend.args.limit) + rend.args.hue_offset_pre_exp).powf(rend.args.cexp)
        * rend.args.hue_scale)
        * 360.0)
        + rend.args.hue_offset_post_exp;

    let color = CIELch::new::<D65>([l, c, h % 360.0, 1.0]);
    let mut low = 0.0;
    let mut high = color.1;
    let mut nc = color.1;
    let mut iters = 0;
    while high - low > 0.01 {
        if !not_in_gamut(Rgb::from_color(CIELch::new([color.0, nc, color.2, 1.0]))) {
            low = nc;
        } else {
            high = nc;
        }
        nc = (low + high) * 0.5;
    }
    CIELab::from_color(CIELch::new([color.0, nc, color.2, 1.0]))
}

pub fn fade_original(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // v = range1.mapExp( out.i, range2, exponent);
    // rgba = hsv_to_rgb( (v * cMult) + cOffset , 1, 1 - Range<double>{limit}.mapExp(valueChange, Range<double>{}, 0.5 ) );
    // valueChange = lightnessScale * pow((double)(out.i), lightnessExponent);
    let value = rend.args.cexp * ((data.s / rend.args.limit).powf(rend.args.h) * 360.0).powf(1.5);
    let t = data.s.powf(rend.args.h2);
    let h = (value * 1.0) % 360.0;
    /*
    let h = ((((data.s / rend.args.limit) + rend.args.hue_offset_pre_exp).powf(rend.args.cexp)
            * rend.args.hue_scale)
            * 360.0)
            + rend.args.hue_offset_post_exp;
    */
    let val = 1.0 - (PI * (1.0 - (t / rend.args.limit).powf(0.5))).cos().powi(2);

    lch_mapping(rend, &data, val, None, None, None, Some(h), false)
}

pub fn fade_mc_bw(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // v = range1.mapExp( out.i, range2, exponent);
    let mut value = ((data.s / rend.args.limit).powf(rend.args.h) * 360.0)
        .powf(rend.args.cexp)
        .powf(1.5);

    let mut t = data.s.powf(rend.args.h2);
    if t.is_nan() {
        t = 0.0;
    }
    if value.is_nan() {
        value = 0.0;
    }
    let h = (value * 1.0) % 360.0;
    // let s = 1.0;
    // let v = 1.0 - (t / rend.args.limit).powf(0.5);

    let mut val = (PI * (1.0 - (t / rend.args.limit).powf(0.5))).cos();
    if val.is_nan() {
        val = 0.0;
    }
    let val = 1.0 - (val * val);
    lch_mapping(rend, &data, val, None, None, None, None, false)
}

pub fn refract(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // let angle1 = data.z.arg();
    // let angle2 = data.der.arg();
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.args.angle.to_radians()).cos(),
        (rend.args.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h2;

    let mut value = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);
    if value.is_nan() {
        value = 0.0;
    }
    /*
    h: 2.0
    h2: 1.0
    cexp: 0.0
    cexp2: 128
    */
    let value = (value * 2.0 * PI) - PI;
    let mut vangle = ((data.s / rend.args.limit).powf(rend.args.cexp) * (PI / 8.0));
    if vangle.is_nan() {
        vangle = 0.0;
    }
    let b = ((rend.args.h / vangle.cos()).powi(2) - rend.args.h * rend.args.h).sqrt();
    let coord = Cf64::new(
        b * rend.args.cexp2.abs() * value.cos(),
        b * rend.args.cexp2.abs() * value.sin(),
    );
    let ratio = rend.raw.width() as f64 / rend.raw.height() as f64;
    let original_x = ((((data.o.re - rend.args.origin.re) * rend.args.zoom) + 0.5)
        * rend.raw.width() as f64)
        - 0.5;
    let original_y = (((((data.o.im - rend.args.origin.im) * ratio) * rend.args.zoom) + 0.5)
        * rend.raw.height() as f64)
        - 0.5;

    // let b = ((rend.args.h / angle1.cos()).powi(2) - rend.args.h * rend.args.h).sqrt();
    // let coord = data.c - Cf64::new(b * angle2.cos(), b * angle2.sin());

    let img = if rend.args.gif_map {
        let l = rend.args.image_map_textures.len();

        &rend.args.image_map_textures[(rend.ftime.round()) as usize % l]
    } else {
        &rend.args.image_map_textures[0]
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    let coord = Cf64::new(
        (original_x + coord.re) / rend.raw.width() as f64,
        (original_y + coord.im) / rend.raw.height() as f64,
    );
    let (mut wi, mut hi) = ((w as f64 * coord.re), (h as f64 * coord.im));
    if wi < 0.0 {
        wi = ((w as f64) - wi) % w as f64
    }
    if wi > w as f64 {
        wi = ((w as f64) - wi.abs()) % w as f64
    }
    if hi < 0.0 {
        wi = ((h as f64) - hi) % h as f64
    }
    if hi > h as f64 {
        hi = ((h as f64) - hi.abs()) % h as f64
    }
    let l = img.data.len();
    let index = wi.floor() + (hi.ceil() * w as f64);
    let mut color = CIELab::from_color(img.data[index.ceil() as usize % l]);
    // color.0 = ((color.0 / (data.s + 1.0)) + color.0) / 2.0;
    lch_gamut_mapping(rend, CIELch::from_color(color))
}

#[allow(dead_code, unused_variables)]
pub fn image_mapping(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());
    let g_size = rend.args.image_map_textures.len();
    let mut theta = (data.z.im.atan2(data.z.re) + PI) / (PI * 2.0);
    if !theta.is_normal() {
        theta = 0.0
    }
    let mut color = {
        let img = if rend.args.gif_map {
            let l = rend.args.image_map_textures.len();

            &rend.args.image_map_textures[(rend.ftime.round()) as usize % l]
        } else {
            &rend.args.image_map_textures[0]
        };

        let (w, h) = (img.width() as usize, img.height() as usize);
        // let period = whangle * (w as f64);
        let mut width = theta * w as f64;
        // ).round() as u32 % w as u32
        let height = h as f64 - ((data.z.norm() / rend.args.bail).log(rend.args.bail) * h as f64);

        let mut height = height % h as f64;
        let f = if data.i as u32 % 2 == 1 {
            height = h as f64 - height;
            1.0 - f
        } else {
            f
        };
        if !width.is_normal() {
            width = 0.0;
        }
        if !height.is_normal() {
            width = 0.0;
        }
        let mut fx = width - width.floor();
        let mut fy = height - height.floor();
        if !fx.is_normal() {
            fx = 0.0;
        }
        if !fy.is_normal() {
            fy = 0.0;
        }
        let (wi, hi) = (width % w as f64, height % h as f64);
        let l = img.data.len();
        let top = {
            let index = wi.floor() + (hi.ceil() * w as f64);
            let a = Srgb::from_color(img.data[index.ceil() as usize % l]);
            let index = wi.ceil() + (hi.ceil() * w as f64);
            let b = Srgb::from_color(img.data[index.ceil() as usize % l]);

            (fx * b) + ((1.0 - fx) * a)
        };
        let bottom = {
            let index = wi.floor() + (hi.floor() * w as f64);
            let a = Srgb::from_color(img.data[index.round() as usize % l]);
            let index = wi.ceil() + (hi.floor() * w as f64);
            let b = Srgb::from_color(img.data[index.round() as usize % l]);
            (fx * b) + ((1.0 - fx) * a)
        };
        (fy * top) + ((1.0 - fy) * bottom)
    };

    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.args.angle.to_radians()).cos(),
        (rend.args.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;

    let value = (1.0 - (data.s / rend.args.limit).powf(rend.args.cexp))
        * (u.re * v.re + u.im * v.im + norm_height)
        / (1.0 + norm_height);
    let v = value.powf(1.0 / rend.args.cexp);
    let mut color = CIELch::from_color(Rgb::from_color(color));
    if v.is_normal() {
        color.0 *= v;
    }
    // color.to_RGBA()
    lch_gamut_mapping(rend, color)
}

pub fn paper_cut(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let mut f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    if !f.is_normal() {
        f = 0.0;
    }

    let mut val =
        (data.der2.arg() / PI + 0.5) * (data.der.arg() / PI + 0.5) * (data.z.arg() / PI + 0.5);

    if !val.is_normal() {
        val = 0.0;
    }
    let val2 = (data.lastder2.arg() / PI + 0.5)
        * (data.lastder.arg() / PI + 0.5)
        * (data.lastz.arg() / PI + 0.5);

    if !val.is_normal() {
        val = 0.0;
    }
    let value = ((val * f) + (val2 * (1.0 - f))).clamp(0.0, 1.0);
    let mut s = (data.s / rend.args.limit)
        .powf(rend.args.cexp)
        .clamp(0.0, 1.0);
    if !s.is_normal() {
        s = 0.0;
    }

    let color = CIELch::from_color(CIELab::new::<D65>([
        100.0 * (value * s),
        10.0 * (value - 0.5),
        10.0 * (value - 0.5),
        1.0,
    ]));
    lch_gamut_mapping(rend, color)
}

pub fn paper_cut_bw(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let val = ((data.der2.arg() + (PI / 2.0)) / PI)
        * ((data.der.arg() + (PI / 2.0)) / PI)
        * ((data.z.arg() + (PI / 2.0)) / PI);

    let val2 = ((data.lastder2.arg() + (PI / 2.0)) / PI)
        * ((data.lastder.arg() + (PI / 2.0)) / PI)
        * ((data.lastz.arg() + (PI / 2.0)) / PI);

    let value = (val * f) + (val2 * (1.0 - f));

    let color = CIELch::from_color(CIELab::new::<D65>([
        2.0 * value * (data.s / rend.args.limit).powf(rend.args.cexp),
        0.0,
        0.0,
        1.0,
    ]));
    lch_gamut_mapping(rend, color)
}

pub fn idk2(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let mut s = (data.sums[4] / rend.args.limit).powf(rend.args.cexp);
    // let
    let s = s.clamp(0.0, 1.0);
    lch_mapping(rend, &data, s, None, None, None, None, false)
}

pub fn stripe(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // Triangle inequality average
    let last_orbit = 0.5 + 0.5 * (3.0 * data.lastz.arg()).sin();
    // let last_orbit = idk(data.lastz);
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());
    let v = 1.0
        - ((f * (data.sum / data.i).powf(1.0 / PI))
            + (1.0 - f) * ((data.sum - last_orbit) / (data.i - 1.0)).powf(1.0 / PI));

    lch_mapping(rend, &data, v, None, None, None, None, false)

    // let val = (PI * v).cos();
    // let val = 1.0 - (val * val);

    // let c = 28.0 + (74.0 * val);
    // let l = 74.0 - (74.0 * val);
    // let h = (((data.s / rend.args.limit).powf(rend.args.cexp)) * 360.0).powf(1.5);

    // let color = CIELch::new::<D65>([l, c, h % 360.0, 1.0]);
    // let mut low = 0.0;
    // let mut high = color.1;
    // let mut nc = color.1;
    // let mut iters = 0;
    // while high - low > 0.00001 && iters < 128 {
    //     if !not_in_gamut(Rgb::from_color(CIELch::new([color.0, nc, color.2, 1.0]))) {
    //         low = nc;
    //     } else {
    //         high = nc;
    //     }
    //     nc = (low + high) * 0.5;
    //     iters += 1;
    // }
    // CIELab::from_color(CIELch::new([color.0, nc, color.2, 1.0]))
}

#[allow(dead_code, unused_variables)]
pub fn normal_map_dual(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.args.angle.to_radians()).cos(),
        (rend.args.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;

    let value = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    let v = Cf64::new(
        (rend.args.angle_offset.to_radians()).cos(),
        (rend.args.angle_offset.to_radians()).sin(),
    );
    let norm_height = rend.args.h;

    let value2 = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    let bound_a = value.powf(1.0);
    let bound_b = 1.0 - value2.powf(1.0);

    // let val = (PI * ).cos();
    lch_mapping(
        rend,
        &data,
        (data.s / rend.args.limit).powf(rend.args.cexp),
        None,
        None,
        None,
        None,
        false,
    )
}
pub fn testingc(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    /*

    // v = range1.mapExp( out.i, range2, exponent);
    // rgba = hsv_to_rgb( (v * cMult) + cOffset , 1, 1 - Range<double>{limit}.mapExp(valueChange, Range<double>{}, 0.5 ) );
    // valueChange = lightnessScale * pow((double)(out.i), lightnessExponent);
    let value = rend.args.cexp * ((data.s / rend.args.limit).powf(rend.args.h) * 360.0).powf(1.5);
    let t = data.s.powf(rend.args.h2);
    let h = (value * 1.0) % 360.0;
    /*
    let h = ((((data.s / rend.args.limit) + rend.args.hue_offset_pre_exp).powf(rend.args.cexp)
            * rend.args.hue_scale)
            * 360.0)
            + rend.args.hue_offset_post_exp;
    */
    let val = 1.0 - (PI * (1.0 - (t / rend.args.limit).powf(0.5))).cos().powi(2);
    */
    /*
    zs[0].re -= inf.re / r.args.zoom;
    zs[1].re += inf.re / r.args.zoom;
    zs[2].im -= inf.im / r.args.zoom;
    zs[3].im += inf.im / r.args.zoom;
    */
    // let a = data.s - data.zr[1].unwrap_or((0.0, 0.0)).1;
    // let b = data.s - data.zr[3].unwrap_or((0.0, 0.0)).1;
    // let v = 1.0 - 1.0 / ((data.der - data.z).norm() / (data.der - data.c).norm());
    // let v = ((data.z).arg() + PI) / (2.0 * PI);
    let v1 = ((data.distance_avg - data.distance_min) / (data.distance_max - data.distance_min))
        * (((data.z).arg() + PI) / (2.0 * PI));

    let v2 = ((data.last_distance_avg - data.last_distance_min)
        / (data.last_distance_max - data.last_distance_min))
        * (((data.lastz).arg() + PI) / (2.0 * PI));

    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let v = 1.0 / (((f * v1) + ((1.0 - f) * v2)) + 1.0);

    // let v = 1.0 / (a * a + b * b).sqrt().tan();
    // let v = 1.0;
    // CIELab::from_color(Rgb::new([v, v, v, 1.0]))
    lch_mapping(
        rend,
        &data,
        v,
        None,
        None,
        Some((1.0 - (data.s / rend.args.limit).powf(rend.args.h)) * ((1.0 - (v)) * 100.0)),
        None,
        false,
    )
}
#[allow(dead_code, unused_variables)]
pub fn VHS_SMOOTH(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let theta = (data.z.arg() + PI) % (2.0 * PI);
    let r = theta.cos();
    let b = theta.sin();
    let g = (r + b) / 2.0;

    let theta = (data.lastz.arg() + PI) % (2.0 * PI);
    let r2 = theta.cos();
    let b2 = theta.sin();
    let g2 = (r2 + b2) / 2.0;
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let ro = (r * f) + (r2 * (1.0 - f));
    let go = (g * f) + (g2 * (1.0 - f));
    let bo = (b * f) + (b2 * (1.0 - f));

    let ro = if !ro.is_normal() { 0.0 } else { ro };
    let go = if !ro.is_normal() { 0.0 } else { go };
    let bo = if !ro.is_normal() { 0.0 } else { bo };
    // let v = (data.s / rend.args.limit).powf(rend.args.cexp);
    CIELab::from_color(Rgb::new([
        ro.clamp(0.0, 1.0),
        go.clamp(0.0, 1.0),
        bo.clamp(0.0, 1.0),
        1.0,
    ]))
}

#[allow(dead_code, unused_variables)]
pub fn RAINBOW_SMOOTH(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let theta = (data.z.arg() + PI) % (2.0 * PI);
    let r = theta.cos();
    let b = theta.sin();
    let g = (r + b) / 2.0;

    let theta = (data.lastz.arg() + PI) % (2.0 * PI);
    let r2 = theta.cos();
    let b2 = theta.sin();
    let g2 = (r2 + b2) / 2.0;
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let ro = (r * f) + (r2 * (1.0 - f));
    let go = (g * f) + (g2 * (1.0 - f));
    let bo = (b * f) + (b2 * (1.0 - f));

    let ro = if !ro.is_normal() { 0.0 } else { ro };
    let go = if !ro.is_normal() { 0.0 } else { go };
    let bo = if !ro.is_normal() { 0.0 } else { bo };
    // let v = (data.s / rend.args.limit).powf(rend.args.cexp);
    CIELab::from_color(Rgb::new([
        ro.clamp(0.0, 1.0),
        go.clamp(0.0, 1.0),
        bo.clamp(0.0, 1.0),
        1.0,
    ]))
}

#[allow(dead_code, unused_variables)]
pub fn VHS_DISCRETE(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let theta = (data.z.arg() + PI) % (2.0 * PI);
    let r = theta.cos();
    let b = theta.sin();
    let g = (r + b) / 2.0;

    // let v = (data.s / rend.args.limit).powf(rend.args.cexp);
    CIELab::from_color(Rgb::new([
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        1.0,
    ]))
}

#[allow(dead_code, unused_variables)]
pub fn RAINBOW_DISCRETE(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let theta = (data.z.arg() + PI) % (2.0 * PI);
    let r = theta.cos();
    let b = theta.sin();
    let g = (r + b) / 2.0;

    // let v = (data.s / rend.args.limit).powf(rend.args.cexp);
    CIELab::from_color(Rgb::new([
        r.abs().clamp(0.0, 1.0),
        g.abs().clamp(0.0, 1.0),
        b.abs().clamp(0.0, 1.0),
        1.0,
    ]))
}
