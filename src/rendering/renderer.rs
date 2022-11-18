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
use std::{
    env::Args,
    f64::consts::PI,
    fmt::Display,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Mutex,
    time::Instant,
};

use chrono::{DateTime, NaiveDateTime, Utc};
use colortypes::*;
use epaint::Vec2;
use image::Rgba;
use linya::{Bar, Progress};
use nalgebra::{Matrix, Matrix3x4};
use num::complex::Complex64;
use ocl::{
    builders::BuildOpt,
    core::CommandQueue,
    enums::{ImageChannelDataType, ImageChannelOrder, MemObjectType},
    Device, Kernel, OclPrm, Program,
};
use rand::Rng;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    prelude::{IndexedParallelIterator, IntoParallelRefMutIterator},
    slice::ParallelSlice,
    ThreadPool, ThreadPoolBuilder,
};
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
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
/*
theta = a.y / a.x
K = normalize(-u ^ (cos(a),sin(a),0))
L = normalize(u ^ K)
*/
/*
impl Into<Vec3> for (f32, f32, f32) {
    fn into(self) -> Vec3 {
        Vec3 {
            x: self.0,
            y: self.1,
            z: self.2,
        }
    }
}*/
impl From<(f32, f32, f32)> for Vec3 {
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self { x, y, z }
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from([x, y, z]: [f32; 3]) -> Self {
        Self { x, y, z }
    }
}

impl Vec3 {
    pub fn cross(&self, b: Self) -> Self {
        Self {
            x: self.y * b.z - self.z * b.y,
            y: self.z * b.x - self.x * b.z,
            z: self.x * b.y - self.y * b.x,
        }
    }
    pub fn dot(&self, b: Self) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn normalize(self) -> Self {
        self / self.length()
    }
    pub fn rotate_to_around(self, p: Vec3, o: Vec3, theta: f32) -> Self {
        let vmo = self - o;
        let cross_vmo_a = vmo.cross((o + p - self).cross(vmo));
        (vmo.length() * theta.sin() * cross_vmo_a) / cross_vmo_a.length() + (vmo * theta.cos())
    }
    pub fn rotate_to(self, p: Vec3, theta: f32) -> Self {
        let sl = self.length();
        let self_hat = self / sl;
        let p_hat = p / p.length();
        sl * (theta.sin() * self_hat.cross(p_hat.cross(self_hat)) + self_hat * theta.cos())
    }
    pub fn orient(self, alpha: f32, beta: f32) -> Self {
        let theta = self.y.atan2(self.x);
        let sn = self.normalize();
        let k = (self * -1.0).cross((theta.cos(), theta.sin(), 0f32).into());
        let l = self.cross(k).normalize();
        let k = k.normalize();
        let ra = (alpha).sin() * (l.cross(k.cross(l))) + (l * (alpha).cos());
        self.length() * ((beta).sin() * (sn.cross(ra.cross(sn))) + (sn * (beta).cos()))
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Div for Vec3 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl std::ops::Add<f32> for Vec3 {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x + rhs,
            y: self.y + rhs,
            z: self.z + rhs,
        }
    }
}

impl std::ops::Sub<f32> for Vec3 {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x - rhs,
            y: self.y - rhs,
            z: self.z - rhs,
        }
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}
type Cf64 = Complex64;
pub struct RenderData {
    pub i: f64,
    pub s: f64,
    pub z: Cf64,
    pub c: Cf64,
    pub o: Cf64,
    pub sum: f64,
    pub der: Cf64,
    pub der2: Cf64,
    pub der_sum: Cf64,
    pub lastz: Cf64,
    pub lastder: Cf64,
    pub lastder2: Cf64,
    pub distance_min: f64,
    pub distance_max: f64,
    pub distance_avg: f64,
    pub last_distance_min: f64,
    pub last_distance_max: f64,
    pub last_distance_avg: f64,
    /// (lastz - z).norm(), (lastder - der).norm(), z.arg(), lastz.arg(), ??,  z.norm();
    pub sums: Vec<f64>,
    pub zr: [Option<(f64, f64)>; 4],
}
// pub struct RenderDataOption {
//     pub i: Option<f64>,
//     pub s: Option<f64>,
//     pub z: Option<Cf64>,
//     pub c: Option<Cf64>,
//     pub sum: Option<f64>,
//     pub der: Option<Cf64>,
//     pub der2: Option<Cf64>,
//     pub lastz: Option<Cf64>,
//     pub lastder: Option<Cf64>,
//     pub lastder2: Option<Cf64>,
//     pub distance_min: Option<f64>,
//     pub distance_max: Option<f64>,
//     pub distance_avg: Option<f64>,
// }
// impl RenderDataOption {
//     fn to_data(&self) -> RenderData {
//         RenderData {
//             z: self.z.unwrap(),
//             c: self.c.unwrap(),
//             lastz: self.lastz.unwrap(),
//             i: self.i.unwrap_or(f64::NAN),
//             s: self.s.unwrap_or(f64::NAN),
//             sum: self.sum.unwrap_or(f64::NAN),
//             der: self.der.unwrap_or(Cf64::new(0.0, 0.0)),
//             der2: self.der2.unwrap_or(Cf64::new(0.0, 0.0)),
//             distance_min: self.distance_min.unwrap_or(f64::NAN),
//             distance_max: self.distance_max.unwrap_or(f64::NAN),
//             distance_avg: self.distance_avg.unwrap_or(f64::NAN),
//             lastder: self.lastder.unwrap_or(Cf64::new(0.0, 0.0)),
//             lastder2: self.lastder2.unwrap_or(Cf64::new(0.0, 0.0)),
//             sums: Vec::new(),
//             zr: [None; 4],
//         }
//     }
// }

// impl_conversion!(CIELab, CIELch, |color| )

#[derive(Clone, Copy)]
pub enum ParenEnum {
    Left,
    Right,
}

#[derive(Clone, Copy)]
pub enum TokenType {
    Number(&'static str),
    Function(&'static str),
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

pub enum OperatorType {
    Add,
    Sub,
    Div,
    Mul,
    Get,
}

#[allow(unused)]
/*
vec![
    StartFractalLoop,
    Variable,
    Assign(Get),
    Expression(Box::new(vec![
        Token(Function, Some(Rotate), None),
        Token(Variable(Named), Some(Z), None),
        Token(Variable(Declared), None, Some(&my_var2)),
    ])),
    EndFractalLoop,
];*/
/*
pub enum FractalWriteInstruction {
    Variable(Box<FractalWriteInstruction>),
    Assign(OperatorType),
    EndLine,

    Expression(Box<Vec<Token>>),
    Fold(FoldType),

    WriteComment,
    OpenComment,
    CloseComment,

    CallFunction,

    If,
    ElseIf,
    Else,

    TernaryOpen,
    TernaryEnd,

    StartFractalLoop,
    EndFractalLoop,

    StartLoop,
    EndLoop,
}*/

pub enum OpType {
    LessThan,
    GraterThan,
    Equals,
    NotEquals,
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

pub enum FunctionType {
    Box,
    Ball,
    Offset,
    Scale,
    Rotate,

    Sqrt,
    Length,
    Fabs,
    Normalize,
    Recip,
    Mix,
}

#[rustfmt::skip]
pub enum SymbolType<'a> {
    Symbol(&'a str),
    Float4(&'a [f32; 4]),
    Float3(&'a [f32; 3]),
    Float2(&'a [f32; 2]),
    Float(&'a f32),
    Z, C, R, Dr,
    Pi, E, Sqrt2,
}

pub enum FoldToken<'a> {
    Token(SymbolType<'a>),
    Parenthesis(ParenEnum),
    Operator(OpType, usize),
    Function(FunctionType),
}

pub enum FoldEnum<'a> {
    FractalLoop(Box<FoldEnum<'a>>),
    Loop(Box<FoldEnum<'a>>),

    Expression(Vec<FoldToken<'a>>),

    If(Vec<FoldToken<'a>>, Box<FoldEnum<'a>>),
    Else(Vec<FoldToken<'a>>, Box<FoldEnum<'a>>),

    Return(Box<FoldEnum<'a>>),
    EndLine(Box<FoldEnum<'a>>),
    Break,
}

pub enum AngleFoldType {
    Xy,
    Xz,
    Yx,
    Yz,
    Zx,
    Zy,
    None,
}

pub enum Var {
    User(usize),
    KernelDefine(usize),
    Z,
    C,
}

pub enum GuiFold<'a> {
    Variable(Var, bool),

    FractalLoop,
    Loop,
    EndLoop,

    Return,
    Break,

    Assign,
    AddAssign,
    MulAssign,
    Define(usize),

    Scale,
    Offset,
    BoxFold,
    SphereFold,
    AngleFold(AngleFoldType, AngleFoldType, AngleFoldType),
    Expression(Vec<FoldToken<'a>>),

    End,
}

pub fn testing() {
    use AngleFoldType::*;
    use GuiFold::*;
    let my_float = 0.25;
    #[rustfmt::skip]
    let idk = vec![

        Loop, Variable(Var::User(0), false), Variable(Var::User(1), false),

        Scale,                 Variable(Var::Z, false), Variable(Var::User(2), false),
        AngleFold(Xy, Yz, Xz), Variable(Var::Z, false), Variable(Var::User(3), true),
        Offset,                Variable(Var::Z, false), Variable(Var::User(4), false),

        EndLoop,

        FractalLoop, Variable(Var::User(5), false), Variable(Var::User(6), false),

        AngleFold(Xy, Yz, Xz), Variable(Var::Z, false), Variable(Var::User(10), true),
        BoxFold,               Variable(Var::Z, false), Variable(Var::User(7), false), 
        SphereFold,            Variable(Var::Z, false), Variable(Var::User(8), false),  Variable(Var::User(9), false), 
        Scale,                 Variable(Var::Z, false), Variable(Var::User(11), false),
        Offset,                Variable(Var::Z, false), Variable(Var::User(12), false),

        EndLoop,

        Return,
        Expression(
            vec![
                FoldToken::Operator(OpType::Mul, 1),
                FoldToken::Operator(OpType::Div, 1),
                FoldToken::Token(SymbolType::R),
                FoldToken::Token(SymbolType::Dr),
                FoldToken::Token(SymbolType::Float(&0.25)),
            ]
        ),
    ];
}

/*
Var list
    z,c,r,pi,sqrt2

    my_var [float^-]:  [0.f^]
    scale_1 :  |Float3 [2.f]  [2.f]  [2.f]
    angles_1 : |Float3 [2.f]  [2.f]  [2.f]
    offset_1 : |Float3 [2.f]  [2.f]  [2.f]

FractalLoop^ [bail^+] [limit^+]
    assign^ [z^]
        scale^ [z^] [scale_1^+]
        rotate^ [z^] [angles_1^-]   2.f    2.f    2.f
        offset^ [z^] [offset_1^+]
Return^
    [r / dr]

*/

/*
pub fn ParseFoldLang() -> Vec<usize> {
    let mut state: ()
    Loop {
        match
        break;
    }
}
 */
#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct FractalArgs {
    pub run_dur: Instant,
    pub chunk_stop: usize,
    pub chunk_size: (f32, f32),
    pub input_grid: [[f32; 8]; 8],
    pub lookat: Vec3,
    pub cam_up: Vec3,
    pub relative_dir: Vec3,
    pub relative_pos: Vec3,
    pub rotation_mat: nalgebra::Matrix3<f32>,
    pub camera_pos: Vec3,
    pub contrast_scalar: f64,
    // Functions
    pub fill_set: bool,
    pub only_set: bool,
    pub zoom: f64,
    pub origin: Vec3,
    pub size: (usize, usize),
    /// renderer, z, der, der2, c, p -> z, der, der2, c
    // Coloring
    pub set_color: Color<Srgb, D65>,
    pub image_map_textures: Vec<Image<Xyz, D65>>,
    pub gif_map: bool,
    pub newton_var1: f64,
    pub cexp2: f64,
    pub angle: f64,
    pub cexp: f64,
    pub h2: f64,
    pub h: f64,

    pub hue_offset_pre_exp: f64,
    pub hue_offset_post_exp: f64,
    pub hue_scale: f64,
    pub hue_exponent: f64,

    pub display_abs_transforms: bool,

    // Misc
    pub bound_dist: f64,
    pub derbail: f64,
    pub z_init: Cf64,

    pub z_offsets: Vec<LinearTransform>,
    // pub z_rotations: Vec<(f64, bool)>,
    pub z_steps: usize,
    // julia, enabled, draw point
    pub julia: (Cf64, bool, bool),
    pub bail2: f64,
    pub limit: f64,
    pub bail: f64,
    pub angle_offset: f64,

    pub color_funct_choice: usize,
    pub color_funct_page: usize,
    pub iter_funct_choice: usize,
    pub iter_funct_page: usize,
    pub conditional_choice: usize,
    pub conditional_page: usize,

    pub f4s: Vec<[f32; 4]>,
    pub f3s: Vec<[f32; 3]>,
    pub f2s: Vec<[f32; 2]>,
    pub f1s: Vec<f32>,
    pub num_f4s: usize,
    pub num_f3s: usize,
    pub num_f2s: usize,
    pub num_f1s: usize,
    //pub sequence: Vec<FractalWriteInstruction>,
    // IDK
}
#[derive(Clone, Copy)]
pub enum LinearTransformMode {
    Offset,
    Rotate,
    Scale,
    Transform,
    Mobius,
    Power,
}

#[derive(Clone, Copy)]
pub struct LinearTransform {
    pub active: bool,
    pub mode: LinearTransformMode,
    pub scalar_or_angle: f64,
    pub offset: Cf64,
    pub abs: [bool; 2],
    pub apply_inv: bool,
    pub matrix: Mat2,
    pub offset_first: bool,
    pub angles: [f32; 2],
    pub mobius: [Cf64; 4], // pub
}

impl Default for LinearTransform {
    fn default() -> Self {
        LinearTransform {
            active: false,
            mode: LinearTransformMode::Offset,
            scalar_or_angle: 0.0,
            offset: Cf64::default(),
            abs: [false; 2],
            apply_inv: false,
            matrix: Mat2::default(),
            offset_first: true,
            angles: [0.0, PI as f32 / 2.0],
            mobius: [
                Cf64::new(1.0, 0.0),
                Cf64::default(),
                Cf64::default(),
                Cf64::new(1.0, 0.0),
            ],
        }
    }
}

impl Display for FractalArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nan_cplx = Complex64::new(f64::NAN, f64::NAN);
        write!(
            f,
            "
        Set Color: {}\n
        Newton Var 1: {}\n
        Color Exponents: {} {}\n
        Normal Mapping Angle: {}\n
        H1: {} H2: {}\n
        Newton Threshold: {}\n
        Derbail distance: {}\n
        Z init: ({}+i{})\n
        Julia Coord: ({}+i{})\n
        Bailout 1 & 2 {} {}\n
        Limit: {}\n
        hue_offset: {}\n
        hue_scale: {}\n
        hue_exp: {}\n
        angle: {}\n
        angle_offset: {}\n


        cfunct: {}\n
        ifunct: {}\n
        bfunct: {}\n
        ",
            self.set_color,
            self.newton_var1,
            self.cexp,
            self.cexp2,
            self.angle,
            self.h,
            self.h2,
            self.bound_dist,
            self.derbail,
            self.z_init.re,
            self.z_init.im,
            self.julia.0.re,
            self.julia.0.im,
            self.bail,
            self.bail2,
            self.limit,
            self.hue_offset_pre_exp,
            self.hue_scale,
            self.hue_exponent,
            self.angle,
            self.angle_offset,
            self.color_funct_choice,
            self.iter_funct_choice,
            self.conditional_choice,
        )
    }
}
#[allow(unused)]
impl FractalArgs {
    pub fn new(size: (usize, usize)) -> Self
    where
        Self: Sized,
    {
        // let a = image::load(File::open("media/test.jpg").into(), format);
        let args = FractalArgs {
            f4s: Vec::new(),
            f3s: Vec::new(),
            f2s: Vec::new(),
            f1s: Vec::new(),
            num_f4s: 0,
            num_f3s: 0,
            num_f2s: 0,
            num_f1s: 0,
            run_dur: Instant::now(),
            chunk_stop: 8,
            chunk_size: (128.0, 128.0),
            input_grid: [
                [0.22,0.001,119.45,4.98,0.1,0.03,8.,10.,],
                [1.7,1.,6.,13.77,1.31,-2.52,-0.7,0.,],
                [2.5,4.5,5.,12.6,-8.,1.,1.,1.,],
                [2401.15,12800.,-5.,1.,1.,3.72,1.,1.,],
                [1.24, 113.98, 1., 1., 1., 1., 1., 1.],
                [1.01,1.,1.,0.987,1.,1.,0.,2.,],
                [1.98,0.9,2.,0.25,3.,2.25,49.81,1.62],
                [6.59,-3.31,-11.06, 2., 2., 2., 600., 1201.66],
            ],
            lookat: Vec3::from((0., 0., 1.)),
            cam_up: Vec3::from((0., 1., 0.)),
            relative_dir: (0.0, 0.0, 0.0).into(),
            relative_pos: (0.0, 0.0, 0.0).into(),
            rotation_mat: nalgebra::Matrix3::<f32>::new(1., 0., 0., 0., 1., 0., 0., 0., 1.),
            camera_pos: (0.0026905313, -0.47572464, -1.572104).into(),
            contrast_scalar: 1.0,
            only_set: false,
            fill_set: true,
            zoom: 1.,
            // 67707790326206
            // origin: Cf64::new(10.941556709273604, -3.5602927111215754),
            origin: Vec3::from((0.0026905313, -0.47572464, -1.572104)),
            //             Origin:(-0.532152436763584, -0.6882741485944704)
            // Zoom:
            size,
            display_abs_transforms: true,
            z_steps: 1,
            z_offsets: vec![LinearTransform::default(); 1],
            image_map_textures: {
                let mut textures = Vec::new();
                textures.push({
                    let a = image::open(PathBuf::from("media/gradient_bw.png"))
                        .unwrap()
                        .to_rgba32f();
                    let size = (a.width() as usize, a.height() as usize);
                    let mut out: Image<Xyz, D65> = Image::new(size);
                    for (a, b) in out.pixels_mut().zip(a.pixels()) {
                        let arr = b.0;
                        // println!("{},{},{},{}", arr[0], arr[1], arr[2], arr[3]);
                        *a = Xyz::from_color(Rgb::new([
                            arr[0] as f64,
                            arr[1] as f64,
                            arr[2] as f64,
                            arr[3] as f64,
                        ]))
                    }
                    out
                });
                textures
            },
            gif_map: false,
            color_funct_choice: 1,
            iter_funct_choice: 1,
            conditional_choice: 1,
            hue_offset_pre_exp: 0.0,
            hue_offset_post_exp: 0.0,
            hue_scale: 1.0,
            hue_exponent: 1.0,

            iter_funct_page: 0,
            color_funct_page: 0,
            conditional_page: 0,

            bound_dist: 10.0,
            angle: 90f64,
            angle_offset: 0.0f64,
            newton_var1: 1.0,
            h: 1.0,
            h2: 1.0,
            z_init: Cf64::new(0.0, 0.0),
            julia: (
                // Cf64::new(0.39306978245605173, -0.20211411568428914),
                Cf64::new(-2.1, -0.41),
                false,
                false,
            ),
            limit: 17.0,
            bail: 16.0,
            bail2: 1.0 / 16384.0,
            derbail: 10f64.powf(10.0),
            cexp: 0.25674585422838464,
            cexp2: 10.0,
            set_color: Srgb::new([0.0, 0.0, 0.0, 1.0]),
        };
        println!("SizeOf FractalArgs: {} bytes", std::mem::size_of_val(&args));
        args
    }
    pub fn new_random(size: (usize, usize)) -> Self {
        let mut rng = rand::thread_rng();
        let color_funct_choice = rng.gen_range(0..16);
        let iter_funct_choice = rng.gen_range(0..2);
        let conditional_choice = rng.gen_range(0..2);
        // let is_orbit_trap = rng.gen_bool(0.)
        FractalArgs {
            f4s: Vec::new(),
            f3s: Vec::new(),
            f2s: Vec::new(),
            f1s: Vec::new(),
            num_f4s: 0,
            num_f3s: 0,
            num_f2s: 0,
            num_f1s: 0,
            run_dur: Instant::now(),
            lookat: Vec3::from((0., 0., 1.)),
            cam_up: Vec3::from((0., 1., 0.)),

            chunk_stop: 8,
            chunk_size: (128.0, 128.0),
            input_grid: [[0f32; 8]; 8],

            relative_dir: (0.0, 0.0, 0.0).into(),
            relative_pos: (0.0, 0.0, 0.0).into(),
            rotation_mat: nalgebra::Matrix3::<f32>::new(1., 0., 0., 0., 1., 0., 0., 0., 1.),
            camera_pos: (0.0026905313, -0.47572464, -1.572104).into(),
            contrast_scalar: 1.0,
            only_set: false,
            fill_set: true,

            zoom: 1.,
            // origin: Cf64::new(10.941556709273604, -3.5602927111215754),
            origin: Vec3::from((0.0026905313, -0.47572464, -1.572104)),

            size,
            display_abs_transforms: true,
            z_steps: 1,
            z_offsets: vec![LinearTransform::default(); 1],
            image_map_textures: {
                let mut textures = Vec::new();
                textures.push({
                    let a = image::open(PathBuf::from("media/test.jpg"))
                        .unwrap()
                        .to_rgba32f();
                    let size = (a.width() as usize, a.height() as usize);
                    let mut out: Image<Xyz, D65> = Image::new(size);
                    for (a, b) in out.pixels_mut().zip(a.pixels()) {
                        let arr = b.0;
                        *a = Xyz::from_color(Rgb::new([
                            arr[0] as f64,
                            arr[1] as f64,
                            arr[2] as f64,
                            arr[3] as f64,
                        ]))
                    }
                    out
                });
                textures
            },
            gif_map: false,
            color_funct_choice,
            iter_funct_choice,
            conditional_choice,
            iter_funct_page: 0,
            color_funct_page: 0,
            conditional_page: 0,
            hue_offset_pre_exp: rng.gen_range(0.0..360.0),
            hue_offset_post_exp: rng.gen_range(0.0..360.0),
            hue_scale: rng.gen_range(1.0..2.0),
            hue_exponent: 1.0,

            bound_dist: 10.0,
            angle: rng.gen_range(0.0..360.0),
            angle_offset: rng.gen_range(-360.0..360.0),
            newton_var1: 1.0,
            h: rng.gen_range(0.01..2.0),
            h2: rng.gen_range(0.01..2.0),
            z_init: Cf64::new(0.0, 0.0),
            julia: (
                Cf64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
                rng.gen_bool(0.5),
                false,
            ),
            limit: 512.0,
            bail: 16384.0,
            bail2: 16384.0,
            derbail: 16384.0,
            cexp: (1.0 - rng.gen_range(0.001f64..1.0).sqrt()) * 1.0,
            cexp2: (1.0 - rng.gen_range(0.001f64..1.0).sqrt()) * 1.0,
            set_color: Srgb::new([
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ]),
        }
        // Self::default()
    }
    pub fn get_info(&self) -> String {
        let nan_cplx = Complex64::new(f64::NAN, f64::NAN);
        format!(
            "Bailout 1 & 2 {} {}\nLimit: {}\n",
            
            self.bail,
            self.bail2,
            self.limit,
            
        )
    }
    pub fn write_info(&self, path: PathBuf, extra: Option<&str>) {
        let mut np = path;
        np.set_extension("txt");
        File::create(&np);
        fs::write(
            np,
            format!(
                "{}\n{}",
                self.get_info(),
                if let Some(new) = extra { new } else { "" }
            ),
        )
        .expect("Unable to write file");
    }
}

pub fn contrast(color: Color<Rgb, D65>, contrast: f64, avg: &Color<Rgb, D65>) -> Color<Rgb, D65> {
    ((color - *avg) * contrast + *avg).clamp_to_gamut()
}
#[repr(C, packed)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct mat4_t {
    pub r1: ocl::core::Float4,
    pub r2: ocl::core::Float4,
    pub r3: ocl::core::Float4,
    pub r4: ocl::core::Float4,
}
#[repr(C, packed)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct input_grid_t {
    pub r0: [ocl::core::Float; 8],
    pub r1: [ocl::core::Float; 8],
    pub r2: [ocl::core::Float; 8],
    pub r3: [ocl::core::Float; 8],
    pub r4: [ocl::core::Float; 8],
    pub r5: [ocl::core::Float; 8],
    pub r6: [ocl::core::Float; 8],
    pub r7: [ocl::core::Float; 8],
}

#[repr(C, packed)]
#[derive(PartialEq, Copy, Clone, Debug, Default)]
pub struct mfloat2_t {
    x: f32,
    y: f32,
}
unsafe impl ocl::OclPrm for mfloat2_t{}
#[repr(C, packed)]
#[derive(PartialEq, Copy, Clone, Debug, Default)]
pub struct Float3 {
    x: f32,
    y: f32,
    z: f32,
}
unsafe impl ocl::OclPrm for Float3{}
#[repr(C, packed)]
#[derive(PartialEq, Copy, Clone, Debug, Default)]
pub struct Float4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
unsafe impl ocl::OclPrm for Float4{}
#[repr(C, packed)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct ArgsGpu {
    pub input_grid: input_grid_t,
    pub lookat: ocl::core::Float3,
    pub cam_up: ocl::core::Float3,
    pub size: ocl::core::Double2,
    pub zoom: ocl::core::Double,
    pub limit: ocl::core::Int,
    pub origin: ocl::core::Float3,
    pub julia: ocl::core::Double2,
    pub bail: ocl::core::Double,
    pub samples: ocl::core::Int,
    pub target_samples: ocl::core::Int,
    pub time: ocl::core::Ulong,
    pub offset: ocl::core::Int,
    pub sb_size: ocl::core::Float2,
    pub split_pos: ocl::core::Int2,
    pub split_size: ocl::core::Float2,
    pub systime: ocl::core::Uint,
}

impl ArgsGpu {
    pub fn grab(
        &mut self,
        args: &FractalArgs,
        samples: (usize, usize),
        args_gpu_samples_pass: i32,
    ) {
        
        self.systime = (args.run_dur.elapsed().as_micros() as u32).into();
        self.input_grid = input_grid_t {
            r0: args.input_grid[0].map(ocl::core::Float::new),
            r1: args.input_grid[1].map(ocl::core::Float::new),
            r2: args.input_grid[2].map(ocl::core::Float::new),
            r3: args.input_grid[3].map(ocl::core::Float::new),
            r4: args.input_grid[4].map(ocl::core::Float::new),
            r5: args.input_grid[5].map(ocl::core::Float::new),
            r6: args.input_grid[6].map(ocl::core::Float::new),
            r7: args.input_grid[7].map(ocl::core::Float::new),
        };
        self.lookat = [args.lookat.x, args.lookat.y, args.lookat.z].into();
        self.cam_up = [args.cam_up.x, args.cam_up.y, args.cam_up.z].into();
        self.samples = (samples.0 as i32).into();
        self.target_samples = args_gpu_samples_pass.into();
        self.time = (samples.1 as u64).into();
        self.zoom = args.zoom.into();
        self.origin = [args.origin.x, args.origin.y, args.origin.z].into();
        self.julia = [args.julia.0.re, args.julia.0.im].into();
        self.bail = args.bail.into();
        self.limit = (args.limit as i32).into();
    }
}

unsafe impl OclPrm for ArgsGpu {}

impl Default for ArgsGpu {
    fn default() -> Self {
        ArgsGpu {
            systime: 0.into(),
            lookat: [0., 0., 1.].into(),
            cam_up: [0., 1., 0.].into(),
            input_grid: input_grid_t {
                r0: [ocl::core::Float::default(); 8],

                r1: [ocl::core::Float::default(); 8],
                r2: [ocl::core::Float::default(); 8],
                r3: [ocl::core::Float::default(); 8],
                r4: [ocl::core::Float::default(); 8],
                r5: [ocl::core::Float::default(); 8],
                r6: [ocl::core::Float::default(); 8],
                r7: [ocl::core::Float::default(); 8],
            },

            size: ocl::core::Double2::new(3840.0, 2160.0),
            zoom: 1f64.into(),
            limit: 1024.into(),
            origin: ocl::core::Float3::new(0.0026905313, -0.47572464, -1.572104),
            julia: ocl::core::Double2::new(-0.75, 0.0),
            bail: 4.0.into(),
            samples: 18.into(),
            target_samples: 18.into(),
            time: 0.into(),
            offset: 0.into(),
            sb_size: [0.0f32, 0.0f32].into(),
            split_pos: [0, 0].into(),
            split_size: [1.0f32, 1.0f32].into(),
        }
    }
}
#[allow(clippy::redundant_static_lifetimes, unused)]
// static MY_KERNEL: &'static str = ;
#[allow(clippy::type_complexity)]
// #[derive(Copy)]
pub struct Renderer {
    pub lasttime: f64,
    pub animation: Vec<ArgsGpu>,
    pub hdr: bool,
    pub alpha: bool,
    // Rendering procedure
    pub raw: Image<Xyz, D65>,
    pub buffer: image::ImageBuffer<Rgba<f32>, Vec<f32>>,

    pub args: FractalArgs,
    pub pool: ThreadPool,

    // Renderer state
    pub time: Instant,
    pub ftime: f64,
    pub rendering: bool,
    pub rendered_samples: usize,
    pub threads: usize,
    pub is_viewport: bool,
    pub init_rendering: bool,
    pub recording: bool,
    pub recorded_frames: usize,
    pub txt: String,
    pub gif_num: usize,
    pub etime: u64,
    pub error: String,
    // pub old_kernel: String,
    pub use_gpu: bool,

    pub computed_abs: String,
    pub kernel: (
        String,
        ocl::Context,
        Device,
        ocl::Queue,
        Kernel,
        Kernel,
        Program,
    ),
    pub buffers: [Vec<f32>; 5],
    pub gpu_buffs: [ocl::Image<f32>; 3],
    pub depth_buffs: [ocl::Buffer<f32>; 2],
    pub f4s_buff: ocl::Buffer<Float4>,
    pub f3s_buff: ocl::Buffer<Float3>,
    pub f2s_buff: ocl::Buffer<f32>,
    pub f1s_buff: ocl::Buffer<f32>,
    pub skybox: (
        image::ImageBuffer<image::Rgba<f32>, Vec<f32>>,
        ocl::Image<f32>,
    ),
    pub args_gpu: ArgsGpu,
    pub args_gpu_samples: usize,
    pub args_gpu_samples_pass: usize,
    pub split: usize,
    pub output: String,
    pub output_name: (String, bool),
}
unsafe impl std::marker::Sync for Renderer {}

pub union TstUnion {
    ocl: ocl::core::Float,
    ral: f32
}

impl Renderer {
    // pub fn render_with_args(&mut self, args: FractalArgs) {
    //     let old_args = self.args.clone();
    //     self.args = args;
    //     self.export_frame();
    // }
    #[allow(unused)]
    pub fn new(
        size: (usize, usize),
        args: FractalArgs,
        threads: usize,
        init_pool: bool,
        is_viewport: bool,
    ) -> Self {
        // let platform = ;

        //
        let skybox_img = image::open("media/skybox.exr")
            .unwrap()
            .to_rgba32f();
        let sbw = skybox_img.width();
        let sbh = skybox_img.height();

        let kernel_code = fs::read_to_string("kernel.ocl").unwrap();
        if init_pool {
            let err = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global();
            if err.is_err() {
                println!("{} : NO IDEA", rayon::current_num_threads());
            } else {
                println!("SET")
            }
        }

        let context = ocl::Context::builder()
            .platform(ocl::Platform::new(*(ocl::Platform::list()[0].as_core())))
            .devices(Device::specifier().first())
            .build()
            .unwrap();
        let device = context.devices()[0];
        let queue = ocl::Queue::new(&context, device, None).unwrap();
        let program = ocl::Program::builder()
            .cmplr_opt("-cl-std=CL2.0")
            .src(&kernel_code)
            .devices(device)
            .build(&context)
            .unwrap();

        let buffers = [
            vec![0f32; size.0 * size.1 * 4],
            vec![0f32; size.0 * size.1 * 4],
            vec![0f32; size.0 * size.1 * 4],
            vec![0f32; size.0 * size.1],
            vec![0f32; size.0 * size.1],
        ];

        let gpu_buffs = [
            ocl::Image::<f32>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::Float)
                .image_type(MemObjectType::Image2d)
                .dims(size)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&buffers[0])
                .queue(queue.clone())
                .build()
                .unwrap(),
            ocl::Image::<f32>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::Float)
                .image_type(MemObjectType::Image2d)
                .dims(size)
                .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                .copy_host_slice(&buffers[1])
                .queue(queue.clone())
                .build()
                .unwrap(),
            ocl::Image::<f32>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::Float)
                .image_type(MemObjectType::Image2d)
                .dims(size)
                .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                .copy_host_slice(&buffers[2])
                .queue(queue.clone())
                .build()
                .unwrap(),
        ];

        let depth_buffs = [
            ocl::Buffer::<f32>::builder()
                .len(size.0 * size.1)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&buffers[3])
                .queue(queue.clone())
                .build()
                .unwrap(),
            ocl::Buffer::<f32>::builder()
                .len(size.0 * size.1)
                .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                .copy_host_slice(&buffers[4])
                .queue(queue.clone())
                .build()
                .unwrap(),
        ];
        #[rustfmt::skip] 
        let mut f4s = vec![[0f32;4];1].iter().map(|x| Float4{x:x[0],y:x[1],z:x[2],w:x[3]}).collect::<Vec<_>>();
        let mut f3s = vec![[0f32;3];1].iter().map(|x| Float3{x:x[0],y:x[1],z:x[2]}).collect::<Vec<_>>();
        let mut f2s = vec![[0f32;2];1].iter().flatten().cloned().collect::<Vec<f32>>();
        let mut f1s = vec![0f32;1];
        let mut f4s_buffer = ocl::Buffer::<Float4>::builder()
            .len(1).flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
            .copy_host_slice(&f4s)
            .queue(queue.clone())
            .build()
            .unwrap();
        let mut f3s_buffer = ocl::Buffer::<Float3>::builder()
            .len(1).flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
            .copy_host_slice(&f3s)
            .queue(queue.clone())
            .build()
            .unwrap();
        let mut f2s_buffer = ocl::Buffer::<f32>::builder()
            .len(2).flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
            .copy_host_slice(&f2s)
            .queue(queue.clone())
            .build()
            .unwrap();
        let mut f1s_buffer = ocl::Buffer::<f32>::builder()
            .len(1)
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
            .copy_host_slice(&f1s)
            .queue(queue.clone())
            .build()
            .unwrap();
        let mut skybox = ocl::Image::<f32>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::Float)
            .image_type(MemObjectType::Image2d)
            .dims((sbw, sbh))
            .flags(
                ocl::flags::MEM_READ_ONLY
                    | ocl::flags::MEM_HOST_WRITE_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            )
            .copy_host_slice(&skybox_img)
            .queue(queue.clone())
            .build()
            .unwrap();

        let kernel = Kernel::builder()
            .program(&program)
            .name("mandelbrot")
            .queue(queue.clone())
            .global_work_size(size)
            .arg(&gpu_buffs[0])
            .arg(&gpu_buffs[1])
            .arg(&depth_buffs[0])
            .arg(&depth_buffs[1])
            .arg(&skybox)
            .arg_named("args", ArgsGpu::default())
            .arg_named("f2s",&f2s_buffer)
            .arg_named("f1s",&f1s_buffer)
            .build()
            .unwrap();

        let kernel2 = Kernel::builder()
            .program(&program)
            .name("mandelbrot")
            .queue(queue.clone())
            .global_work_size(size)
            .arg(&gpu_buffs[0])
            .arg(&gpu_buffs[1])
            .arg(&depth_buffs[0])
            .arg(&depth_buffs[1])
            .arg(&skybox)
            .arg_named("args",&ArgsGpu::default())
            .arg_named("f2s",&f2s_buffer)
            .arg_named("f1s",&f1s_buffer)
            .build()
            .unwrap();

        let mut rend = Renderer {
            f4s_buff: ocl::Buffer::<Float4>::builder()
                .len(1)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&f4s)
                .queue(queue.clone())
                .build()
                .unwrap(),
            f3s_buff : ocl::Buffer::<Float3>::builder()
                .len(1)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&f3s)
                .queue(queue.clone())
                .build()
                .unwrap(),
            f2s_buff : ocl::Buffer::<f32>::builder()
                .len(2)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&f2s)
                .queue(queue.clone())
                .build()
                .unwrap(),
            f1s_buff : ocl::Buffer::<f32>::builder()
                .len(1)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&f1s)
                .queue(queue.clone())
                .build()
                .unwrap(),
            buffer: image::ImageBuffer::<Rgba<f32>, Vec<f32>>::new(size.0 as u32, size.1 as u32),
            lasttime: 0.,
            animation: Vec::new(),
            skybox: (skybox_img, skybox),
            split: 1,
            computed_abs: "".into(),
            error: "Valid Program Set <3".into(),
            args_gpu_samples: 0,
            args_gpu_samples_pass: 18,
            use_gpu: true,
            kernel: (
                kernel_code.into(),
                context,
                device,
                queue,
                kernel,
                kernel2,
                program,
            ),
            buffers,
            gpu_buffs,
            depth_buffs,
            args_gpu: ArgsGpu {
                systime: 0.into(),
                lookat: [0., 0., 1.].into(),
                cam_up: [0., 1., 0.].into(),
                input_grid: input_grid_t {
                    r0: [ocl::core::Float::new(0f32); 8],

                    r1: [ocl::core::Float::new(0f32); 8],
                    r2: [ocl::core::Float::new(0f32); 8],
                    r3: [ocl::core::Float::new(0f32); 8],
                    r4: [ocl::core::Float::new(0f32); 8],
                    r5: [ocl::core::Float::new(0f32); 8],
                    r6: [ocl::core::Float::new(0f32); 8],
                    r7: [ocl::core::Float::new(0f32); 8],
                },

                size: [size.0 as f64, size.1 as f64].into(),
                zoom: 1f64.into(),
                limit: 300.into(),
                origin: [0.0026905313, -0.47572464, -1.572104].into(),
                julia: [-2.1, -0.41].into(),
                bail: 16.0.into(),
                samples: 18.into(),
                target_samples: 18.into(),
                time: 0.into(),
                offset: 0.into(),
                sb_size: [sbw as f32, sbh as f32].into(),
                split_pos: [0, 0].into(),
                split_size: [1.0f32, 1.0f32].into(),
            },
            hdr: false,
            alpha: false,
            output_name: ("idk".into(), false),
            output: "out/".into(),
            etime: 0,
            gif_num: 0,
            txt: "gif".into(),
            init_rendering: false,
            recording: false,
            recorded_frames: 0,
            pool: ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap(),

            raw: Image::new(size),
            args: FractalArgs::new(size),
            // zoom: 0.4002031965029708,
            // zoom: 0.0025,
            time: Instant::now(),
            ftime: 0.0,
            rendering: false,
            rendered_samples: 0,
            threads,
            is_viewport,
        };

        rend.kernel.1 = ocl::Context::builder()
            .platform(ocl::Platform::new(*(ocl::Platform::list()[0].as_core())))
            .devices(Device::specifier().first())
            .build()
            .unwrap();
        rend.kernel.2 = rend.kernel.1.devices()[0];
        rend.kernel.3 = ocl::Queue::new(&rend.kernel.1, rend.kernel.2, None).unwrap();
        rend.kernel.6 = ocl::Program::builder()
            .cmplr_opt("-cl-std=CL2.0")
            .src(rend.kernel.0.clone())
            .devices(rend.kernel.2)
            .build(&rend.kernel.1)
            .unwrap();

        rend.gpu_buffs = [
            ocl::Image::<f32>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::Float)
                .image_type(MemObjectType::Image2d)
                .dims(size)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&rend.buffers[0])
                .queue(rend.kernel.3.clone())
                .build()
                .unwrap(),
            ocl::Image::<f32>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::Float)
                .image_type(MemObjectType::Image2d)
                .dims(size)
                .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                .copy_host_slice(&rend.buffers[1])
                .queue(rend.kernel.3.clone())
                .build()
                .unwrap(),
            ocl::Image::<f32>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::Float)
                .image_type(MemObjectType::Image2d)
                .dims(size)
                .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                .copy_host_slice(&rend.buffers[2])
                .queue(rend.kernel.3.clone())
                .build()
                .unwrap(),
        ];
        rend.depth_buffs = [
            ocl::Buffer::<f32>::builder()
                .len(size.0 * size.1)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .copy_host_slice(&rend.buffers[3])
                .queue(rend.kernel.3.clone())
                .build()
                .unwrap(),
            ocl::Buffer::<f32>::builder()
                .len(size.0 * size.1)
                .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                .copy_host_slice(&rend.buffers[4])
                .queue(rend.kernel.3.clone())
                .build()
                .unwrap(),
        ];
        rend.skybox.1 = ocl::Image::<f32>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::Float)
            .image_type(MemObjectType::Image2d)
            .dims((sbw, sbh))
            .flags(
                ocl::flags::MEM_READ_ONLY
                    | ocl::flags::MEM_HOST_WRITE_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            )
            .copy_host_slice(&rend.skybox.0)
            .queue(rend.kernel.3.clone())
            .build()
            .unwrap();
        let idfk = ocl::core::Int::from(0);
        rend.kernel.4 = Kernel::builder()
            .program(&rend.kernel.6)
            .name("mandelbrot")
            .queue(rend.kernel.3.clone())
            .global_work_size((size.0, size.1))
            .arg(&rend.gpu_buffs[0])
            .arg(&rend.gpu_buffs[1])
            .arg(&rend.depth_buffs[0])
            .arg(&rend.depth_buffs[1])
            .arg(&rend.skybox.1)
            .arg_named("args", rend.args_gpu)
            .arg_named("f2s",&rend.f2s_buff)
            .arg_named("f1s", &rend.f1s_buff)
            .build()
            .unwrap();
        rend.kernel.5 = Kernel::builder()
            .program(&rend.kernel.6)
            .name("post_process")
            .queue(rend.kernel.3.clone())
            .global_work_size((size.0, size.1))
            .arg(&rend.gpu_buffs[1])
            .arg(&rend.gpu_buffs[2])
            .build()
            .unwrap();
        rend
    }
    pub fn width(&self) -> usize {
        self.raw.width()
    }
    pub fn height(&self) -> usize {
        self.raw.height()
    }
    pub fn clear(&mut self) {
        self.raw = Image::new((self.raw.width(), self.raw.height()));
        self.buffers[0].fill(0.0);
        self.buffers[1].fill(0.0);
        // self
        self.rendered_samples = 0;
    }
    pub fn copy_args(&mut self, other: &Self) {
        self.args = other.args.clone();
        self.hdr = other.hdr;
        self.alpha = other.alpha;
        // self.origin = other.origin;
        // self.zoom = other.zoom;
    }
    #[allow(unused)]
    pub fn render(&mut self, progress: bool, samples: usize, save_frame: bool) {
        if self.rendering {
            return;
        }
        let now = Instant::now();
        let a = self.time.elapsed().as_secs_f64();
        self.ftime = (a / 120.0) % 240.0;
        let fr = 24.0;
        let b = (a / (fr * 2.0));
        // self.ftime = 13.0;
        self.etime = u64::from_ne_bytes(self.time.elapsed().as_secs_f64().to_ne_bytes()) / 2;
        // self.itime = (self.time.elapsed().as_secs_f32()) as usize;

        if self.use_gpu {
            //let f4s = self.args.f4s.clone().into_par_iter().map(ocl::core::Float4::from).collect::<Vec<_>>();
            //let f3s = self.args.f3s.clone().into_par_iter().map(ocl::core::Float3::from).collect::<Vec<_>>();
            //let f2s = self.args.f2s.clone().into_par_iter().map(ocl::core::Float2::from).collect::<Vec<_>>();
            let f1s = self.args.f1s.clone();
            //self.f4s_buff.write(&f4s).enq();
            //self.f3s_buff.write(&f3s).enq();
            //self.f2s_buff.write(&f2s).enq();
            self.f1s_buff.write(&f1s).enq();
            //self.kernel.4.set_arg("f4s", &self.f3s_buff);
            //self.kernel.4.set_arg("f3s", &self.f3s_buff);
            //self.kernel.4.set_arg("f2s", &self.f3s_buff);
            self.kernel.4.set_arg("f1s", &self.f3s_buff);

            //println!("{}", (1.0 / (a - self.lasttime)).floor());
            self.rendering = true;
            if self.rendered_samples == 0 {
                self.buffers[0].par_iter_mut().for_each(|x| *x = 0.);
                self.buffers[3].par_iter_mut().for_each(|x| *x = 0.);
                self.gpu_buffs[0].write(&self.buffers[0]).enq().unwrap();
            }
            let mut new_args = self.args_gpu;
            // new_args.time = (self.time.elapsed().as_millis() as f32).into();
            new_args.grab(&self.args, (samples, self.rendered_samples), samples as i32);
            new_args.time = (self.rendered_samples as u64).into();
            if save_frame && (self.init_rendering || self.recording) {
                let l = self.animation.len();
                let mut dat = l * self.animation.len();
                let mut string = "B";
                match dat {
                    1_000..=999_999 => {
                        string = "KB";
                        dat /= 1_000
                    }
                    1_000_000..=999_999_999 => {
                        string = "MB";
                        dat /= 1_000_000
                    }
                    1_000_000_000..=999_999_999_999 => {
                        string = "GB (Stop!)";
                        dat /= 1_000_000_000
                    }
                    _ => (),
                }
                let mut im_dat = 3 * self.raw.size.0 * self.raw.size.1 * l;
                let mut im_string = "B";
                match im_dat {
                    1_000..=999_999 => {
                        im_string = "KB";
                        im_dat /= 1_000
                    }
                    1_000_000..=999_999_999 => {
                        im_string = "MB";
                        im_dat /= 1_000_000
                    }
                    1_000_000_000..=999_999_999_999 => {
                        string = "GB (Stop!)";
                        im_dat /= 1_000_000_000
                    }
                    _ => (),
                }

                println!(
                    "anim FPS: {:3.6}, Frames: {}, Anim Data: {}{}, Animation Max Size: {}{}, 10:1",
                    1.0 / (a - self.lasttime),
                    l,
                    dat,
                    string,
                    im_dat,
                    im_string
                );
                self.lasttime = a;
                if self.init_rendering {
                    self.animation = Vec::new();
                    self.recorded_frames = 0;
                    // let mut path =
                    //     PathBuf::from(format!("{}_{}/this_exists", self.txt

                    // self.gif_num = i;
                    // path = PathBuf::from(format!("{}_{}/", self.txt, self.gif_num));
                    // let mut path = String::from("gif/");
                    if let Some(new_path) = rfd::FileDialog::new().pick_folder() {
                        self.txt = new_path.display().to_string();
                    }
                    fs::create_dir_all(&self.txt);
                    println!("{}", self.txt.as_str());
                    // fs::create_dir_all(self.gif_path);
                    // /
                }
                //println!("Frame {}", self.recorded_frames);
                self.animation.push(new_args);
                //self.export_image(None);
                self.init_rendering = false;
                self.recording = true;
                self.recorded_frames += 1;
            }
            self.kernel.4.set_arg("args", new_args);
            // let idk = ocl::core::Int::from((self.ftime * 120.0) as i32);
            // self.kernel.4.set_arg("time", idk);
            unsafe {
                self.kernel.4.enq().unwrap();
                self.kernel.5.enq().unwrap();
            }

            // Render Buffers
            self.gpu_buffs[1].read(&mut self.buffers[1]).enq().unwrap();
            self.gpu_buffs[0].write(&self.buffers[1]).enq().unwrap();

            // Post Proc
            self.gpu_buffs[2].read(&mut self.buffer).enq().unwrap();

            // Depth Buffers
            self.depth_buffs[1]
                .read(&mut self.buffers[4])
                .enq()
                .unwrap();
            self.depth_buffs[0].write(&self.buffers[4]).enq().unwrap();
            /*
            println!(
                "FPS: {:3.2} ({:3.2}s), (w: {}, h: {}) @ {}S",
                1.0 / now.elapsed().as_secs_f64(),
                now.elapsed().as_secs_f64(),
                self.raw.width(),
                self.raw.height(),
                self.rendered_samples,
            );*/
            // if self.args.gif_map || self.rendered_samples == 0 {
            // let idk = self.buffers[0];

            // if self.rendered_samples == 0 {
            //     // self.raw.data = self
            //     //     .raw
            //     //     .data
            //     //     .clone()
            //     //     .into_par_iter()
            //     //     .enumerate()
            //     //     .map(|(i, x)| {
            //     //         // let (x, y) = (i % (self.raw.width()), i / (self.raw.width()));
            //     //         Xyz::new::<D65>([
            //     //             self.buffers[1][(i * 4)] as f64,
            //     //             self.buffers[1][(i * 4) + 1] as f64,
            //     //             self.buffers[1][(i * 4) + 2] as f64,
            //     //             self.buffers[1][(i * 4) + 3] as f64,
            //     //         ])
            //     //     })
            //     //     .collect::<Vec<Color<Xyz, D65>>>();
            // } else {
            //     self.raw.data = self
            //         .raw
            //         .data
            //         .clone()
            //         .into_par_iter()
            //         .enumerate()
            //         .map(|(i, x)| {
            //             // let (x, y) = (i % (self.raw.width()), i / (self.raw.width()));
            //             self.raw.data[i]
            //                 + Xyz::new::<D65>([
            //                     self.buffers[1][(i * 4)] as f64,
            //                     self.buffers[1][(i * 4) + 1] as f64,
            //                     self.buffers[1][(i * 4) + 2] as f64,
            //                     self.buffers[1][(i * 4) + 3] as f64,
            //                 ])
            //         })
            //         .collect::<Vec<Color<Xyz, D65>>>();
            // }

            // for (i, e) in out.iter().enumerate() {
            //     let (x, y) = (i % (self.raw.width()), i / (self.raw.width()));
            //     self.raw.data[x + (y * w as usize)] = *e;
            // }
            // } else {
            //     // for (i, e) in out.iter().enumerate() {
            //     //     let (x, y) = (i % self.raw.width(), i / self.raw.width());
            //     //     self.raw.data[x + (y * w as usize)] += *e;
            //     // }
            // }
            // self.args_gpu_samples += samples;
            self.rendered_samples += samples;

            self.rendering = false;
        } /*else {
              let w = self.raw.width();
              let out: Vec<Color<Xyz, D65>> = if progress {
                  let progress = Mutex::new(Progress::new());
                  let bar: Bar = progress
                      .lock()
                      .unwrap()
                      .bar((self.raw.width() * self.raw.height()) as usize, "");
                  self.pool.install(|| {
                      (0..(self.raw.width() * self.raw.height()))
                          .into_par_iter()
                          .map(|i| {
                              progress.lock().unwrap().inc_and_draw(&bar, 1);
                              (self.sampling)(self, i, samples)
                          })
                          .collect()
                  })
              } else {
                  self.pool.install(|| {
                      (0..(self.raw.width() * self.raw.height()))
                          .into_par_iter()
                          .map(|i| (self.sampling)(self, i, samples))
                          .collect()
                  })
              };

              if self.args.gif_map || self.rendered_samples == 0 {
                  for (i, e) in out.iter().enumerate() {
                      let (x, y) = (i % (self.raw.width()), i / (self.raw.width()));
                      self.raw.data[x + (y * w as usize)] = *e;
                  }
              } else {
                  for (i, e) in out.iter().enumerate() {
                      let (x, y) = (i % self.raw.width(), i / self.raw.width());
                      self.raw.data[x + (y * w as usize)] += *e;
                  }
              } /*
                println!(
                    "FPS: {:3.2} ({:3.2}s), (w: {}, h: {}) @ {}S",
                    1.0 / now.elapsed().as_secs_f64(),
                    now.elapsed().as_secs_f64(),
                    self.raw.width(),
                    self.raw.height(),
                    self.rendered_samples,
                ); */

              if self.args.gif_map {
                  self.rendered_samples = samples;
              } else {
                  self.rendered_samples += samples;
              }
              self.rendering = false;
              if save_frame
                  && ((self.init_rendering || self.recording) && self.ftime as u64 % 60 == 0)
              {
                  if self.init_rendering {
                      self.recorded_frames = 0;
                      let mut path =
                          PathBuf::from(format!("{}_{}/this_exists", self.txt, self.gif_num));
                      // let mut i = 0;
                      println!("ok so {}", path.to_str().unwrap());

                      while path.exists() {
                          path = PathBuf::from(format!("{}_{}/this_exists", self.txt, self.gif_num));
                          println!("now {}", path.to_str().unwrap());
                          self.gif_num += 1;
                      }
                      // self.gif_num = i;
                      path = PathBuf::from(format!("{}_{}/", self.txt, self.gif_num));
                      println!("{}", path.to_str().unwrap());
                      fs::create_dir(path);
                      File::create(PathBuf::from(format!(
                          "{}_{}/this_exists",
                          self.txt, self.gif_num
                      )));
                  }
                  self.export_image(None);
                  self.init_rendering = false;
                  self.recording = true;
                  self.recorded_frames += 1;
              }
          }*/
    }
    pub fn render_gpu(&mut self, progress: bool, samples: usize, save_frame: bool) {
        if self.rendering {
            return;
        }
        let now = Instant::now();
        self.ftime = self.time.elapsed().as_secs_f64() * 48.0;
        // self.ftime = 13.0;
        self.etime = u64::from_ne_bytes(self.time.elapsed().as_secs_f64().to_ne_bytes()) / 2;
        // self.itime = (self.time.elapsed().as_secs_f32()) as usize;
        self.rendering = true;

        let mut new_args = self.args_gpu;
        self.buffer = image::ImageBuffer::<Rgba<f32>, Vec<f32>>::new(
            self.raw.size.0 as u32,
            self.raw.size.1 as u32,
        );
        // new_args.time = (self.time.elapsed().as_millis() as f32).into();
        new_args.grab(&self.args, (samples, self.rendered_samples), samples as i32);
        new_args.time = 0.into();
        new_args.samples = ocl::prm::Int::new(self.split as i32);
        new_args.target_samples = ocl::prm::Int::new(samples as i32);
        self.rendered_samples = 0;
        let cw_for_32 = ((self.raw.size.0 as f32) / self.args.chunk_size.0).ceil() as usize;
        let ch_for_32 = ((self.raw.size.1 as f32) / self.args.chunk_size.1).ceil() as usize;
        new_args.split_size = [cw_for_32 as f32, ch_for_32 as f32].into();
        self.kernel
            .4
            .set_default_global_work_size((self.args.chunk_size.0, self.args.chunk_size.1).into());
        for i in 0..(samples / self.split) {
            println!();
            new_args.time = 0.into();
            new_args.offset = ((i * self.split) as i32).into();
            for cy in 0..cw_for_32 {
                for cx in 0..ch_for_32 {
                    print!("|c: {}/{}", cx + (cw_for_32 * cy), cw_for_32 * ch_for_32);
                    new_args.split_pos = [cx as i32, cy as i32].into();
                    self.kernel.4.set_arg("args", new_args).unwrap();
                    unsafe {
                        self.kernel.4.enq().unwrap();
                    }
                }
                if (cy) % self.args.chunk_stop == 0 {
                    println!();
                    self.gpu_buffs[1].read(&mut self.buffers[1]).enq().unwrap();
                    self.gpu_buffs[0].write(&self.buffers[1]).enq().unwrap();
                    println!(
                        "\t[{} / {} ]FPS: {:3.2} ({:3.2}s), (w: {}, h: {}) @ {}S",
                        i,
                        samples / self.split,
                        1.0 / now.elapsed().as_secs_f64(),
                        now.elapsed().as_secs_f64(),
                        self.raw.width(),
                        self.raw.height(),
                        (i * self.split),
                    );
                }
            }
            println!();
            self.gpu_buffs[1].read(&mut self.buffers[1]).enq().unwrap();
            self.gpu_buffs[0].write(&self.buffers[1]).enq().unwrap();

            // Depth Buffers
            self.depth_buffs[1]
                .read(&mut self.buffers[4])
                .enq()
                .unwrap();
            self.depth_buffs[0].write(&self.buffers[4]).enq().unwrap();
            println!(
                "\t[{} / {} ]FPS: {:3.2} ({:3.2}s), (w: {}, h: {}) @ {}S",
                i,
                samples / self.split,
                1.0 / now.elapsed().as_secs_f64(),
                now.elapsed().as_secs_f64(),
                self.raw.width(),
                self.raw.height(),
                (i * self.split),
            );
        }
        unsafe {
            self.kernel.5.enq().unwrap();
        }
        self.gpu_buffs[2].read(&mut self.buffer).enq().unwrap();

        self.rendered_samples = samples;

        self.rendering = false;
    }
    pub fn render_animation(&mut self, samples: usize, animation: &mut Vec<ArgsGpu>) {
        if self.rendering {
            return;
        }
        let now = Instant::now();
        //self.ftime = (self.time.elapsed().as_secs_f64() * 48.0);
        // self.ftime = 13.0;
        self.etime = u64::from_ne_bytes(self.time.elapsed().as_secs_f64().to_ne_bytes()) / 2;
        // self.itime = (self.time.elapsed().as_secs_f32()) as usize;
        self.rendering = true;
        let l = animation.len();
        for (frame, args) in animation.iter().enumerate() {
            self.buffers[0].par_iter_mut().for_each(|x| *x = 0.);
            self.gpu_buffs[0].write(&self.buffers[0]).enq().unwrap();
            let mut new_args = args.clone();
            new_args.size = [self.raw.size.0 as f64, self.raw.size.1 as f64].into();
            // new_args.time = (self.time.elapsed().as_millis() as f32).into();
            //new_args.grab(&self.args, (samples, self.rendered_samples), samples as i32);
            new_args.time = 0.into();
            new_args.samples = ocl::prm::Int::new(self.split as i32);
            new_args.target_samples = ocl::prm::Int::new(samples as i32);
            self.rendered_samples = 0;
            let chunks = 4;
            new_args.split_size = [chunks as f32, chunks as f32].into();
            self.kernel.4.set_default_global_work_size(
                (self.raw.size.0 / chunks, self.raw.size.1 / chunks).into(),
            );
            for i in 0..(samples / self.split) {
                println!();
                new_args.time = 0.into();
                new_args.offset = ((i * self.split) as i32).into();
                for cy in 0..chunks {
                    for cx in 0..chunks {
                        print!("|c: {}/{}", cx + (chunks * cy), chunks * chunks);
                        new_args.split_pos = [cx as i32, cy as i32].into();
                        self.kernel.4.set_arg("args", new_args).unwrap();
                        unsafe {
                            self.kernel.4.enq().unwrap();
                        }
                    }
                }
                println!();
                self.gpu_buffs[1].read(&mut self.buffers[1]).enq().unwrap();
                self.gpu_buffs[0].write(&self.buffers[1]).enq().unwrap();
                println!(
                    "\t[{:6^} / {:6^}][{:6^} / {:6^}]FPS: {:3.2} ({:3.2}s), (w: {:4^}, h: {:4^}) @ {:4^}S",
                    i,
                    samples / self.split,
                    frame,
                    l,
                    1.0 / now.elapsed().as_secs_f64(),
                    now.elapsed().as_secs_f64(),
                    self.raw.width(),
                    self.raw.height(),
                    (i * self.split),
                );
            }
            self.rendered_samples = samples;
            self.export_image(None);
        }
        self.rendering = false;
    }

    pub fn update_gpu(&mut self) {
        self.clear();
        self.rendered_samples = 0;
        // let old: (
        //     String,
        //     ocl::Context,
        //     Device,
        //     ocl::Queue,
        //     ocl::Kernel,
        //     ocl::Program,
        // ) = unsafe { std::mem::transmute_copy(&self.kernel) };

        // self.kernel.1 = ocl::Context::builder()
        //     .devices(ocl::Device::specifier().first())
        //     .build()
        //     .unwrap();
        // self.kernel.2 = self.kernel.1.devices()[0];
        // self.kernel.3 = ocl::Queue::new(&self.kernel.1, self.kernel.2, None).unwrap();
        // let idk = BuildOpt::CmplrDefine { ident: (), val: () }
        let new_p = ocl::Program::builder()
            .cmplr_opt("-cl-std=CL2.0")
            .src(&self.kernel.0)
            // .bo()
            .devices(self.kernel.2)
            .build(&self.kernel.1);
        self.kernel.6 = if let Ok(program) = new_p {
            println!("VALID PROGRAM SET");
            let size = self.raw.size;
            self.args_gpu.size = [size.0 as f64, size.1 as f64].into();
            self.buffers = [
                vec![0f32; size.0 * size.1 * 4],
                vec![0f32; size.0 * size.1 * 4],
                vec![0f32; size.0 * size.1 * 4],
                vec![0f32; size.0 * size.1],
                vec![0f32; size.0 * size.1],
            ];

            self.gpu_buffs = [
                ocl::Image::<f32>::builder()
                    .channel_order(ocl::enums::ImageChannelOrder::Rgba)
                    .channel_data_type(ocl::enums::ImageChannelDataType::Float)
                    .image_type(ocl::enums::MemObjectType::Image2d)
                    .dims(size)
                    .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                    .copy_host_slice(&self.buffers[0])
                    .queue(self.kernel.3.clone())
                    .build()
                    .unwrap(),
                ocl::Image::<f32>::builder()
                    .channel_order(ocl::enums::ImageChannelOrder::Rgba)
                    .channel_data_type(ocl::enums::ImageChannelDataType::Float)
                    .image_type(ocl::enums::MemObjectType::Image2d)
                    .dims(size)
                    .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                    .copy_host_slice(&self.buffers[1])
                    .queue(self.kernel.3.clone())
                    .build()
                    .unwrap(),
                ocl::Image::<f32>::builder()
                    .channel_order(ocl::enums::ImageChannelOrder::Rgba)
                    .channel_data_type(ocl::enums::ImageChannelDataType::Float)
                    .image_type(ocl::enums::MemObjectType::Image2d)
                    .dims(size)
                    .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                    .copy_host_slice(&self.buffers[2])
                    .queue(self.kernel.3.clone())
                    .build()
                    .unwrap(),
            ];

            self.depth_buffs = [
                ocl::Buffer::<f32>::builder()
                    .len(size.0 * size.1)
                    .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                    .copy_host_slice(&self.buffers[3])
                    .queue(self.kernel.3.clone())
                    .build()
                    .unwrap(),
                ocl::Buffer::<f32>::builder()
                    .len(size.0 * size.1)
                    .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
                    .copy_host_slice(&self.buffers[4])
                    .queue(self.kernel.3.clone())
                    .build()
                    .unwrap(),
            ];

            // ocl::Kernel::
            let new_k = ocl::Kernel::builder()
                .program(&program)
                .name("mandelbrot")
                .queue(self.kernel.3.clone())
                .global_work_size((size.0, size.1))
                .arg(&self.gpu_buffs[0])
                .arg(&self.gpu_buffs[1])
                .arg(&self.depth_buffs[0])
                .arg(&self.depth_buffs[1])
                .arg(&self.skybox.1)
                .arg_named("args", self.args_gpu)
                .arg_named("f2s",&self.f2s_buff)
            .arg_named("f1s",&self.f1s_buff)
                .build();

            let new_k_p = ocl::Kernel::builder()
                .program(&program)
                .name("post_process")
                .queue(self.kernel.3.clone())
                .global_work_size((size.0, size.1))
                .arg(&self.gpu_buffs[1])
                .arg(&self.gpu_buffs[2])
                .build();

            if let Ok(new_kernel) = new_k {
                if let Ok(new_kernel_p) = new_k_p {
                    self.kernel.4 = new_kernel;
                    self.kernel.5 = new_kernel_p;
                    program
                } else {
                    self.error = new_k_p.err().unwrap().to_string();
                    println!("INVALID KERNEL, {}", self.error);
                    self.kernel.6.clone()
                }
            } else {
                self.error = new_k.err().unwrap().to_string();
                println!("INVALID KERNEL, {}", self.error);
                self.kernel.6.clone()
            }
            // let a = self.kernel.4.;
        } else {
            self.error = new_p.err().unwrap().to_string();
            println!("INVALID PROGRAM, {}", self.error);
            self.kernel.6.clone()
        };
    }
    fn tone_map(c: Color<Srgb, D65>) -> Color<Srgb, D65> {
        let mut new = Srgb::new([1.0, 1.0, 1.0, 0.0]) - (-c).exp();
        new.3 = c.3;
        new
    }
    // pub fn blur()
    pub fn process_image(&self) -> Image<Rgb, D65> {
        Image::from_vec(
            (self.width(), self.height()),
            // if self.use_gpu {
            self.buffers[1]
                .par_chunks_exact(4)
                .map(|x| {
                    Rgb::from_color(Xyz::new::<D65>([
                        x[0] as f64 / self.rendered_samples as f64,
                        x[1] as f64 / self.rendered_samples as f64,
                        x[2] as f64 / self.rendered_samples as f64,
                        x[3] as f64 / self.rendered_samples as f64,
                    ]))
                })
                .collect::<Vec<Color<Rgb, D65>>>(), // } else {
                                                    // self.buffers[1]
                                                    //     .clone()
                                                    //     .into_par_iter()
                                                    //     .map(|c| Rgb::from_color(c / self.rendered_samples as f64))
                                                    //     .collect::<Vec<Color<Rgb, D65>>>(), // (
                                                    //     self.raw.clone() / self.rendered_samples as f64
                                                    // )
                                                    // .convert::<Srgb>()
                                                    // .map_pixels_to::<Rgb, _>(|c| Rgb::from_color(*c))
                                                    // .collect::<Vec<Color<Rgb, D65>>>(), // },
        )
    }
    pub fn export_image(&self, name: Option<&str>) {
        // let txt = "img";
        // let mut path = PathBuf::from("gif/img.png");
        let mut path = PathBuf::from(format!("{}/img_0.png", self.txt));
        let mut i = 0;
        while path.exists() {
            path.set_file_name(format!("img_{}.png", i));
            i += 1;
        }
        println!("{}", self.txt);
        println!("{}", path.to_str().unwrap());
        // try { fs::create_dir_all(PathBuf::from(format!("{}_{}/", self.txt, self.gif_num))) };
        let file = File::create(path).unwrap();
        let writ = &mut BufWriter::new(file);
        let mut encoder =
            png::Encoder::new(writ, self.raw.width() as u32, self.raw.height() as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let now = Instant::now();
        let mut test = writer.stream_writer_with_size(12).unwrap();
        let mut wrote = 0;

        let raw2 = self.process_image();

        for px in raw2.pixels() {
            let c = px.to_arr8().0;
            let arr = vec![c[0], c[1], c[2]];
            wrote += test.write(&arr).unwrap();
            // wrote += test.write(&arr).unwrap();
        }
        println!(
            "Wrote {} bytes in {:4.4} seconds",
            wrote,
            now.elapsed().as_secs_f32()
        );
    }
    #[allow(clippy::while_immutable_condition)]
    pub fn export_image_with_info(
        &mut self,
        alpha: bool,
        hdr: bool,
        render: Option<usize>,
        name: Option<&str>,
        path_override: Option<&str>,
    ) {
        let img_name = if self.output_name.1 {
            self.output_name.0.clone()
        } else {
            let nt = NaiveDateTime::from_timestamp(Utc::now().timestamp(), 0);
            let dt: DateTime<Utc> = DateTime::from_utc(nt, Utc);
            dt.format("FracgenGUI (%Y_%m_%d) (%H_%M_%S)").to_string()
        };
        let pathtxt = format!(
            "{}{}",
            if let Some(p) = path_override {
                p
            } else {
                &self.output
            },
            img_name
        );
        if !PathBuf::from(&pathtxt).exists() {
            fs::create_dir_all(&pathtxt).unwrap();
        }

        let file = File::create(format!("{0}/{1}.png", &pathtxt, img_name)).unwrap();
        let mut kernel_f = File::create(format!("{0}/kernel.ocl", &pathtxt)).unwrap();
        fs::write(format!("{0}/kernel.ocl", &pathtxt), &self.kernel.0).unwrap();
        let mut grid = "".into();
        for row in self.args.input_grid.iter() {
            let mut g_row = "".into();
            for item in row {
                g_row = format!("{}{}|", g_row, item);
            }
            grid = format!("{}{}", grid, g_row);
        }
        self.args.write_info(
            PathBuf::from(format!("{0}/{1}/", &pathtxt, img_name)),
            Some(&format!(
                "Samples: {}\nOrigin:({}, {}, {})\nCam_Pos:({}, {}, {})\nCam_U:({}, {}, {}),Cam_F:({}, {}, {})\nZoom:{}\nftime:{:3.2}\n{}",
                &render.unwrap_or(0),
                self.args.origin.x,
                self.args.origin.y,
                self.args.origin.z,

                self.args.camera_pos.x,
                self.args.camera_pos.y,
                self.args.camera_pos.z,

                self.args.cam_up.x,
                self.args.cam_up.y,
                self.args.cam_up.z,

                self.args.lookat.x,
                self.args.lookat.y,
                self.args.lookat.z,
                
                self.args.zoom,
                self.ftime,
                grid
            )),
        );

        if let Some(samples) = render {
            let samples = if self.args.gif_map { 32 } else { samples };

            self.render_gpu(true, samples, false);
            while self.rendering {}
        } else if self.args.gif_map {
            self.render(true, 32, false);
            while self.rendering {}
        }

        let writ = &mut BufWriter::new(file);
        println!(
            "raw: {}x{}, stored: ({},{})",
            self.raw.width(),
            self.raw.height(),
            self.args.size.0,
            self.args.size.1
        );
        let mut encoder =
            png::Encoder::new(writ, self.raw.width() as u32, self.raw.height() as u32);
        encoder.set_color(if self.alpha {
            png::ColorType::Rgba
        } else {
            png::ColorType::Rgb
        });
        encoder.set_depth(if self.hdr {
            png::BitDepth::Sixteen
        } else {
            png::BitDepth::Eight
        });
        let mut writer = encoder.write_header().unwrap();
        let now = Instant::now();
        let mut test = writer
            .stream_writer_with_size(if self.alpha { 3 } else { 4 } * if self.hdr { 16 } else { 8 })
            .unwrap();
        let mut wrote = 0;

        //let raw2 = self.process_image();
        if self.hdr {
            for px in self.buffer.pixels() {
                let c = [
                    (px.0[0] * 65536.0) as u16,
                    (px.0[1] * 65536.0) as u16,
                    (px.0[2] * 65536.0) as u16,
                    (px.0[3] * 65536.0) as u16,
                ];

                let mut arr = vec![0; 0];
                arr.extend([(c[0] >> 8) as u8, c[0] as u8]);
                arr.extend([(c[1] >> 8) as u8, c[1] as u8]);
                arr.extend([(c[2] >> 8) as u8, c[2] as u8]);
                if self.alpha {
                    arr.extend([(c[3] >> 8) as u8, c[3] as u8])
                }
                wrote += test.write(&arr).unwrap();
            }
        } else {
            for px in self.buffer.pixels() {
                let c = [
                    (px.0[0] * 256.0) as u8,
                    (px.0[1] * 256.0) as u8,
                    (px.0[2] * 256.0) as u8,
                    (px.0[3] * 256.0) as u8,
                ];
                let arr = if self.alpha {
                    vec![c[0], c[1], c[2], c[3]]
                } else {
                    vec![c[0], c[1], c[2]]
                };
                wrote += test.write(&arr).unwrap();
            }
        }
        println!(
            "Wrote {} bytes in {:4.4} seconds",
            wrote,
            now.elapsed().as_secs_f32()
        );
    }
}

