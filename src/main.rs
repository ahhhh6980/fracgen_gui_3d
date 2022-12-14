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
//! Show a custom window frame instead of the default OS window chrome decorations.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// use gui_testing::MyApp;

use std::{
    f64::consts::PI,
    fs::{self, File},
    path::PathBuf,
};

use colortypes::{CIELab, Color, FromColorType, Image, Rgb, Srgb, Xyz, D65};
use eframe::{
    egui::{
        self, color_picker, menu,
        panel::Side,
        plot::{Value, Values},
        style::Spacing,
        Checkbox, Event, FontDefinitions, Frame, InnerResponse, Key, Modifiers, Order,
        PointerButton, Response, TextEdit,
    },
    emath::Align,
};
use epaint::{
    color::Hsva, text::LayoutJob, Color32, Fonts, Pos2, Rect, Rounding, Shadow, Stroke,
    TextureHandle, Vec2,
};
use gilrs::{GamepadId, Gilrs};
use FracgenGui3D::{rendering::*, FractalArgs, Renderer, Vec3};
use image::EncodableLayout;
use nalgebra::{clamp, Norm};
use num::complex::Complex64;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

pub struct InputMenu {
    pub viewport_size: (String, String, bool),
    pub export_size: (String, String, bool),
    pub samples: String,
    pub width: f32,
    pub page: usize,
    pub bail: (String, bool),
    pub iter_count: String,
    pub origin: (String, String),
    pub julia: (String, String, bool),
    pub julia_speed: f64,
    pub bail_speed: f64,
    pub rot_speed: f64,
    pub mov_speed: f64,
    pub tweak_speed: f64,
    pub cam_speed: f64,
    pub grid_speed: f64,
}

/// Memoized Code highlighting
pub fn highlight(ctx: &egui::Context, theme: &CodeTheme, code: &str, language: &str) -> LayoutJob {
    impl egui::util::cache::ComputerMut<(&CodeTheme, &str, &str), LayoutJob> for Highlighter {
        fn compute(&mut self, (theme, code, lang): (&CodeTheme, &str, &str)) -> LayoutJob {
            self.highlight(theme, code, lang)
        }
    }

    type HighlightCache = egui::util::cache::FrameCache<LayoutJob, Highlighter>;

    let mut memory = ctx.memory();
    let highlight_cache = memory.caches.cache::<HighlightCache>();
    highlight_cache.get((theme, code, language))
}

#[derive(Clone, Copy, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
enum SyntectTheme {
    Base16EightiesDark,
    Base16MochaDark,
    Base16OceanDark,
    Base16OceanLight,
    InspiredGitHub,
    SolarizedDark,
    SolarizedLight,
}

impl SyntectTheme {
    fn syntect_key_name(&self) -> &'static str {
        match self {
            Self::Base16EightiesDark => "base16-eighties.dark",
            Self::Base16MochaDark => "base16-mocha.dark",
            Self::Base16OceanDark => "base16-ocean.dark",
            Self::Base16OceanLight => "base16-ocean.light",
            Self::InspiredGitHub => "InspiredGitHub",
            Self::SolarizedDark => "Solarized (dark)",
            Self::SolarizedLight => "Solarized (light)",
        }
    }

    pub fn is_dark(&self) -> bool {
        match self {
            Self::Base16EightiesDark
            | Self::Base16MochaDark
            | Self::Base16OceanDark
            | Self::SolarizedDark => true,

            Self::Base16OceanLight | Self::InspiredGitHub | Self::SolarizedLight => false,
        }
    }
}

#[derive(Clone, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct CodeTheme {
    dark_mode: bool,

    syntect_theme: SyntectTheme,
}

impl Default for CodeTheme {
    fn default() -> Self {
        Self {
            dark_mode: true,
            syntect_theme: SyntectTheme::SolarizedDark,
        }
    }
}

// store in memory
// ctx.data().insert_persisted(egui::Id::new("dark"), self);

// Pull from memory
/*
ctx.data()
            .get_persisted(egui::Id::new("dark"))
            .unwrap_or_else(CodeTheme::dark)
*/

impl CodeTheme {
    pub fn dark() -> Self {
        Self {
            dark_mode: true,
            syntect_theme: SyntectTheme::SolarizedDark,
        }
    }
}

struct Highlighter {
    ps: syntect::parsing::SyntaxSet,
    ts: syntect::highlighting::ThemeSet,
}

impl Default for Highlighter {
    fn default() -> Self {
        Self {
            ps: syntect::parsing::SyntaxSet::load_defaults_newlines(),
            ts: syntect::highlighting::ThemeSet::load_defaults(),
        }
    }
}

impl Highlighter {
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn highlight(&self, theme: &CodeTheme, code: &str, lang: &str) -> LayoutJob {
        self.highlight_impl(theme, code, lang).unwrap_or_else(|| {
            // Fallback:
            LayoutJob::simple(
                code.into(),
                egui::FontId::monospace(14.0),
                if theme.dark_mode {
                    egui::Color32::LIGHT_GRAY
                } else {
                    egui::Color32::DARK_GRAY
                },
                f32::INFINITY,
            )
        })
    }

    fn highlight_impl(&self, theme: &CodeTheme, text: &str, language: &str) -> Option<LayoutJob> {
        use syntect::easy::HighlightLines;
        use syntect::highlighting::FontStyle;
        use syntect::util::LinesWithEndings;

        let syntax = self
            .ps
            .find_syntax_by_extension("ocl")
            .or_else(|| self.ps.find_syntax_by_name("C"))?;

        let theme = theme.syntect_theme.syntect_key_name();
        let mut h = HighlightLines::new(syntax, &self.ts.themes[theme]);

        use egui::text::{LayoutSection, TextFormat};

        let mut job = LayoutJob {
            text: text.into(),
            ..Default::default()
        };

        for line in LinesWithEndings::from(text) {
            for (style, range) in h.highlight_line(line, &self.ps).ok()? {
                let fg = style.foreground;
                let text_color = egui::Color32::from_rgb(fg.r, fg.g, fg.b);
                let italics = style.font_style.contains(FontStyle::ITALIC);
                let underline = style.font_style.contains(FontStyle::ITALIC);
                let underline = if underline {
                    egui::Stroke::new(1.0, text_color)
                } else {
                    egui::Stroke::none()
                };
                job.sections.push(LayoutSection {
                    leading_space: 0.0,
                    byte_range: as_byte_range(text, range),
                    format: TextFormat {
                        font_id: egui::FontId::monospace(14.0),
                        color: text_color,
                        italics,
                        underline,
                        ..Default::default()
                    },
                });
            }
        }

        Some(job)
    }
}

fn as_byte_range(whole: &str, range: &str) -> std::ops::Range<usize> {
    let whole_start = whole.as_ptr() as usize;
    let range_start = range.as_ptr() as usize;
    assert!(whole_start <= range_start);
    assert!(range_start + range.len() <= whole_start + whole.len());
    let offset = range_start - whole_start;
    offset..(offset + range.len())
}

pub struct LastData {
    pub args: FractalArgs,
}

pub struct Viewport {
    pub image_size: (usize, usize),
    pub image: Image<Rgb, D65>,
    pub texture: Option<TextureHandle>,
    pub image_space: f32,
    pub img_empty_space: (f32, f32),
    pub renderer: Renderer,
    pub last: LastData,
    pub active: bool,
    pub needs_resize: Option<(f32, f32)>,
}

#[allow(unused)]
impl Viewport {
    fn resize_h(&mut self, new_h: f32) {
        self.resize((self.image_size.0 as f32, new_h));
    }
    fn resize_w(&mut self, new_w: f32) {
        self.resize((new_w, self.image_size.1 as f32));
    }
    fn resize(&mut self, size: (f32, f32)) {
        self.image_size = (size.0.round() as usize, size.1.round() as usize);
        self.image = Image::new(self.image_size);
        self.renderer.buffer = image::ImageBuffer::<image::Rgba<f32>, Vec<f32>>::new(
            size.0.round() as u32,
            size.1.round() as u32,
        );
        self.renderer.args.size = self.image_size;
        self.renderer.raw = Image::new(self.image_size);
        self.renderer.rendered_samples = 0;
        self.renderer.update_gpu();
    }
}

pub struct Events {
    // any mouse click or release
    pub mouse_any: Option<bool>,
    // Mouse left, right, middle clicks
    pub m_left: Option<bool>,
    pub m_right: Option<bool>,
    pub m_middle: Option<bool>,
    pub m_pos: Option<Pos2>,
    // Enter key
    pub enter: Option<bool>,
    // Arrow key presses
    pub a_right: Option<bool>,
    pub a_left: Option<bool>,
    pub a_up: Option<bool>,
    pub a_down: Option<bool>,
    // Scroll vector
    pub scroll: Option<Vec2>,

    pub q: Option<bool>,
    pub w: Option<bool>,
    pub a: Option<bool>,
    pub s: Option<bool>,
    pub z: Option<bool>,
    pub x: Option<bool>,
    pub e: Option<bool>,
    pub r: Option<bool>,
}

impl Events {
    pub fn clear(&mut self) {
        self.m_pos = None;
        self.mouse_any = None;
        self.m_left = None;
        self.m_right = None;
        self.m_middle = None;
        self.enter = None;
        self.a_right = None;
        self.a_left = None;
        self.a_up = None;
        self.a_down = None;
        self.scroll = None;
        self.q = None;
        self.w = None;
        self.a = None;
        self.s = None;
        self.z = None;
        self.x = None;
        self.e = None;
        self.r = None;
    }
}

// #[derive(Default)]
pub struct FracgenGUI {
    pub param_inp: String,
    pub camera_rot: (f32, f32),
    pub camera_off: (f32, f32),
    pub face_origin: bool,
    pub l_trig: (bool, bool, bool),
    pub r_trig: (bool, bool, bool),
    pub gilrs: Gilrs,
    pub controller: Option<GamepadId>,
    pub abs_scroll: f64,
    pub viewport: Viewport,
    pub input_menu: InputMenu,
    pub pointer: Option<Pos2>,
    pub node_menu_height: f32,
    pub renderer: Renderer,
    pub events: Events,
    pub temp_julia: Option<Complex64>,
    pub viewport_interact: bool,
    pub samples: usize,
    pub over_options: bool,
    pub layouter: fn(&egui::Ui, &str, f32) -> std::sync::Arc<epaint::Galley>,
}
impl Default for FracgenGUI {
    #[inline]
    fn default() -> FracgenGUI {
        let cpus = num_cpus::get();
        let threads = if cpus <= 4 {
            cpus
        } else {
            (num_cpus::get() as f64).ceil() as usize - 2
        };

        // let threads = 1;

        let df_vp_size = (512, 512);
        let df_op_size = (512, 512);
        println!("SIZE OF ARGS; {}", std::mem::size_of::<FractalArgs>());
        println!("SIZE OF ARGS; {}", std::mem::size_of::<RenderData>());
        let o = FractalArgs::new((0, 0)).origin;
        // let args = FractalArgs::new();
        FracgenGUI {
            param_inp: "".into(),
            face_origin: true,
            camera_rot: (0.0, 0.0),
            camera_off: (0.0, 0.0),
            l_trig: (false, false, false),
            r_trig: (false, false, false),
            gilrs: Gilrs::new().unwrap(),
            controller: None,
            layouter: |ui: &egui::Ui, string: &str, _: f32| {
                ui.fonts().layout_job(egui::text::LayoutJob::simple(
                    string.into(),
                    epaint::FontId::new(32.0, epaint::FontFamily::Monospace),
                    Color32::from_rgb(255, 255, 255),
                    f32::INFINITY,
                ))
            },
            over_options: false,
            samples: 1,
            abs_scroll: 0.0,
            viewport_interact: false,
            temp_julia: None,
            events: Events {
                m_pos: None,
                mouse_any: None,
                m_left: None,
                m_right: None,
                m_middle: None,
                enter: None,
                a_right: None,
                a_left: None,
                a_up: None,
                a_down: None,
                scroll: None,
                q: None,
                w: None,
                a: None,
                s: None,
                z: None,
                x: None,
                e: None,
                r: None,
            },
            renderer: Renderer::new(
                df_op_size,
                FractalArgs::new(df_op_size),
                threads,
                true,
                false,
            ),
            viewport: Viewport {
                active: true,
                image_size: df_vp_size,
                image: Image::new(df_vp_size),
                texture: None,
                image_space: 0.0,
                img_empty_space: (0.0, 0.0),
                last: LastData {
                    args: FractalArgs::new(df_vp_size),
                },
                renderer: Renderer::new(
                    df_vp_size,
                    FractalArgs::new(df_vp_size),
                    threads,
                    false,
                    true,
                ),
                needs_resize: None,
            },
            input_menu: {
                let def_args = FractalArgs::new((0, 0));
                InputMenu {
                    rot_speed: 1.00,
                    mov_speed: 1.00,
                    tweak_speed: 0.01,
                    cam_speed: 1.00,
                    grid_speed: 0.01,
                    bail_speed: 128.0,
                    julia_speed: 0.01,
                    viewport_size: (df_vp_size.0.to_string(), df_vp_size.1.to_string(), false),
                    export_size: (df_op_size.0.to_string(), df_op_size.1.to_string(), false),
                    samples: 144.to_string(),
                    width: 128.0,
                    page: 0,
                    bail: (def_args.bail.to_string(), false),
                    iter_count: def_args.limit.to_string(),
                    origin: (o.x.to_string(), o.y.to_string()),
                    julia: (0.to_string(), 0.to_string(), false),
                }
            },
            pointer: None,
            node_menu_height: 128.0,
        }
    }
}

fn image_to_colorimage(image: &Image<Rgb, D65>) -> epaint::ColorImage {
    epaint::ColorImage {
        size: [image.width(), image.height()],
        pixels: image
            .pixels()
            .map(|x| {
                let nx = x.to_arr8().0;
                epaint::color::Color32::from_rgb(nx[0], nx[1], nx[2])
            })
            .collect::<Vec<_>>(),
    }
}

fn process_key_standard_mode(
    p: bool,
    up: bool,
    ctrl_code: bool,
    render_arg: &mut f64,
    correlate: &mut Option<bool>,
    scalar_mult: f64,
    fix: Option<fn(&mut f64) -> f64>,
) {
    if up {
        if ctrl_code {
            *render_arg += 1.0 * scalar_mult;
        } else {
            *render_arg *= 1.05 * scalar_mult;
        }
    } else if ctrl_code {
        *render_arg -= 1.0 * scalar_mult;
    } else {
        *render_arg /= 1.05 * scalar_mult;
    }
    if let Some(fix_funct) = fix {
        (fix_funct)(render_arg);
    }
    println!("{}", *render_arg);
    *correlate = Some(p)
}

/*
match key {
                                        Key::Num1 => {
                                            self.viewport.renderer.args.color_funct_choice = 1;
                                            self.viewport.renderer.args.color_funct = stripe;
                                        }
                                        Key::Num2 => {
                                            self.viewport.renderer.args.color_funct_choice = 2;
                                            self.viewport.renderer.args.color_funct = image_mapping;
                                        }
                                        Key::Num3 => {
                                            self.viewport.renderer.args.color_funct_choice = 3;
                                            self.viewport.renderer.args.color_funct = fade_original;
                                        }
                                        Key::Num4 => {
                                            self.viewport.renderer.args.color_funct_choice = 4;
                                            self.viewport.renderer.args.color_funct = paper_cut;
                                        }
                                        Key::Num5 => {
                                            self.viewport.renderer.args.color_funct_choice = 5;
                                            self.viewport.renderer.args.color_funct = testingf;
                                        }
                                        Key::Num6 => {
                                            self.viewport.renderer.args.color_funct_choice = 6;
                                            self.viewport.renderer.args.color_funct = trap_avg;
                                        }
                                        Key::Num7 => {
                                            self.viewport.renderer.args.color_funct_choice = 7;
                                            self.viewport.renderer.args.color_funct = normal_map;
                                        }
                                        Key::Num8 => {
                                            self.viewport.renderer.args.color_funct_choice = 8;
                                            self.viewport.renderer.args.color_funct = foral;
                                        }
                                        // Reserved
                                        Key::Num9 => {
                                            self.viewport.renderer.args.color_funct_page += 1;
                                            if self.viewport.renderer.args.color_funct_page > 3 {
                                                self.viewport.renderer.args.color_funct_page -= 1
                                            }
                                        }
                                        Key::Num0 => {
                                            self.viewport.renderer.args.color_funct_page -= 1;
                                            if self.viewport.renderer.args.color_funct_page > 3 {
                                                self.viewport.renderer.args.color_funct_page = 0
                                            }
                                        }
                                        _ => (),
                                    }
*/

pub fn process_color_key(
    key: Key,
    choice: &mut usize,
    page: &mut usize,
    change: &mut fn(&Renderer, RenderData) -> Color<CIELab, D65>,
    f: Vec<(&str, fn(&Renderer, RenderData) -> Color<CIELab, D65>)>,
) {
    match key {
        Key::Num1 => {
            *choice = 1 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        Key::Num2 => {
            *choice = 2 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        Key::Num3 => {
            *choice = 3 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        Key::Num4 => {
            *choice = 4 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        Key::Num5 => {
            *choice = 5 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        Key::Num6 => {
            *choice = 6 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        Key::Num7 => {
            *choice = 7 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        Key::Num8 => {
            *choice = 8 + (8 * *page);
            let chosen = f[(*choice - 1) % f.len()];
            *change = chosen.1;
            println!("PICKED {}:{}", chosen.0, *choice);
        }
        // Reserved
        Key::Num9 => {
            *page += 1;
            if *page > 3 {
                *page -= 1
            }
        }
        Key::Num0 => {
            *page -= 1;
            if *page > 3 {
                *page = 0
            }
        }
        _ => (),
    }
}
fn r_axis(o: f32, u: Vec3) -> nalgebra::Matrix3<f32> {
    let ct = o.cos();
    let omc = 1. - ct;
    let st = o.sin();
    nalgebra::Matrix3::<f32>::new(
        ct + u.x * u.x * omc,
        u.x * u.y * omc - u.z * st,
        u.x * u.z * omc + u.y * st,
        u.y * u.x * omc + u.z * st,
        ct + u.y * u.y * omc,
        u.y * u.z * omc - u.x * st,
        u.z * u.x * omc - u.y * st,
        u.z * u.y * omc + u.x * st,
        ct + u.z * u.z * omc,
    )
}
#[allow(unused)]
impl eframe::App for FracgenGUI {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut color_interact = false;
        self.events.clear();
        let input = ctx.input().clone();
        let files = input.raw.dropped_files;
        if !files.is_empty() {
            self.viewport.renderer.args.image_map_textures.clear();
            let count = files.len();
            let paths = files
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<Option<PathBuf>>>();
            paths.iter().for_each(|p| {
                if let Some(path) = p {
                    let ext = path.extension();
                    if let Some(ftype) = ext {
                        match ftype.to_str() {
                            Some("gif") => {
                                self.viewport.renderer.args.gif_map = true;
                                // Gif input
                                let mut decoder = gif::DecodeOptions::new();
                                decoder.set_color_output(gif::ColorOutput::RGBA);
                                // Decode gif
                                let mut gif = decoder.read_info(File::open(path).unwrap()).unwrap();
                                // Move all frames into media
                                let mut i = 0;
                                while let Some(f) = gif.read_next_frame().unwrap() {
                                    let (w, h) = (f.width, f.height);
                                    self.viewport.renderer.args.image_map_textures.push({
                                        let size = (f.width as usize, f.height as usize);
                                        let mut out: Image<Xyz, D65> = Image::new(size);
                                        for (a, b) in out.pixels_mut().zip(f.buffer.chunks_exact(4))
                                        {
                                            *a = Xyz::from_color(Rgb::new([
                                                b[0] as f64 / 255.0,
                                                b[1] as f64 / 255.0,
                                                b[2] as f64 / 255.0,
                                                b[3] as f64 / 255.0,
                                            ]))
                                        }

                                        out
                                    });
                                    i += 1;
                                }
                                println!(
                                    "\nFRAMES: {}\n",
                                    self.viewport.renderer.args.image_map_textures.len()
                                );
                                println!(
                                    "\n{},{} over {}\n",
                                    self.viewport.renderer.args.image_map_textures[0].width(),
                                    self.viewport.renderer.args.image_map_textures[0].height(),
                                    self.viewport.renderer.args.image_map_textures.len()
                                );
                            }
                            Some("exr") | Some("png") | Some("jpg") | Some("jpeg") => {
                                self.viewport.renderer.args.gif_map = false;
                                println!("{}", path.to_str().unwrap_or("error"));
                                self.viewport.renderer.args.image_map_textures.push({
                                    let a = image::open(path).unwrap().to_rgba32f();
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
                                    let skybox_img = a;

                                    let sbw = size.0;
                                    let sbh = size.1;
                                    self.viewport.renderer.skybox.0 = skybox_img;
                                    self.viewport.renderer.skybox.1 = ocl::Image::<f32>::builder()
                                        .channel_order(ocl::enums::ImageChannelOrder::Rgba)
                                        .channel_data_type(ocl::enums::ImageChannelDataType::Float)
                                        .image_type(ocl::enums::MemObjectType::Image2d)
                                        .dims((sbw, sbh))
                                        .flags(
                                            ocl::flags::MEM_READ_ONLY
                                                | ocl::flags::MEM_HOST_WRITE_ONLY
                                                | ocl::flags::MEM_COPY_HOST_PTR,
                                        )
                                        .copy_host_slice(&self.viewport.renderer.skybox.0)
                                        .queue(self.viewport.renderer.kernel.3.clone())
                                        .build()
                                        .unwrap();
                                    self.viewport.renderer.args_gpu.sb_size =
                                        [size.0 as f32, size.1 as f32].into();
                                    // self.viewport.renderer.skybox.0;
                                    self.viewport
                                        .renderer
                                        .kernel
                                        .4
                                        .set_arg("skybox", &self.viewport.renderer.skybox.1);
                                    self.viewport
                                        .renderer
                                        .kernel
                                        .4
                                        .set_arg("args", &self.viewport.renderer.args_gpu);

                                    // self.renderer.skybox.0 = skybox_img;
                                    // self.renderer.skybox.1 = self.renderer.skybox.1.clone();
                                    self.renderer.args_gpu.sb_size =
                                        [size.0 as f32, size.1 as f32].into();
                                    // self.viewport.renderer.skybox.0;
                                    self.renderer
                                        .kernel
                                        .4
                                        .set_arg("skybox", &self.viewport.renderer.skybox.1);
                                    self.renderer
                                        .kernel
                                        .4
                                        .set_arg("args", &self.viewport.renderer.args_gpu);

                                    //self.renderer.skybox.0 = skybox_img;
                                    //self.renderer.args_gpu.sb_size = [size.0 as f32, size.0 as f32].into();

                                    out
                                });
                            }
                            _ => {}
                        }
                    }
                }
            });
        }

        // let key_presses = input.keys_down;
        // input.pointer
        let mouse_input = input.pointer;
        // let a = input.
        let mut scalar_mult = 1.0;
        let mut axis_bool = false;
        let mut alt_code = false;
        let mut ctrl_code = false;
        let egui::Modifiers {
            alt: a,
            ctrl: c,
            shift: s,
            mac_cmd: m,
            command: c2,
        } = input.modifiers;
        {
            if s {
                scalar_mult = 1.25;
            }
            axis_bool = s;
            alt_code = a;
            ctrl_code = c;
        }
        let mut pressed = false;

        while let Some(gilrs::Event { id, event, time }) = self.gilrs.next_event() {
            println!("{:?} New event from {}: {:?}", time, id, event);
            let zone = (-5000.0 / 32740.0)..(5000.0 / 32740.0);
            self.controller = Some(id);
            // gilrs::Gamepad::uuid(&self)
            let gamepad = self.gilrs.gamepad(id);
            match event {
                gilrs::EventType::AxisChanged(axis, v, _) => match axis {
                    gilrs::Axis::RightStickX => {
                        self.camera_rot.0 = if zone.contains(&v) { 0. } else { v };
                    }
                    gilrs::Axis::RightStickY => {
                        self.camera_rot.1 = if zone.contains(&v) { 0. } else { v };
                    }
                    gilrs::Axis::LeftStickX => {
                        self.camera_off.0 = if zone.contains(&v) { 0. } else { v };
                    }
                    gilrs::Axis::LeftStickY => {
                        self.camera_off.1 = if zone.contains(&v) { 0. } else { v };
                    }

                    _ => (),
                },
                gilrs::EventType::ButtonPressed(button, code) => match button {
                    gilrs::Button::LeftTrigger2 => {
                        self.l_trig.0 = true;
                    }
                    gilrs::Button::LeftTrigger => {
                        self.l_trig.1 = !self.l_trig.1;
                    }
                    gilrs::Button::LeftThumb => {
                        self.l_trig.2 = !self.l_trig.2;
                    }
                    gilrs::Button::RightTrigger2 => {
                        self.r_trig.0 = true;
                    }
                    gilrs::Button::RightTrigger => {
                        self.r_trig.1 = !self.r_trig.1;
                    }
                    gilrs::Button::RightThumb => {
                        self.r_trig.2 = !self.r_trig.2;
                    }
                    gilrs::Button::DPadUp => {
                        self.face_origin = !self.face_origin;
                    }
                    _ => (),
                },
                gilrs::EventType::ButtonReleased(button, code) => match button {
                    gilrs::Button::LeftTrigger2 => {
                        self.l_trig.0 = false;
                    }
                    gilrs::Button::RightTrigger2 => {
                        self.r_trig.0 = false;
                    }
                    _ => (),
                },
                _ => (),
            }
        }
        //Vector(( u(1,0,0),-u(0,1,0),u(0,0,1)))
        let camera_rot_tot =
            (self.camera_rot.0 * self.camera_rot.0 + self.camera_rot.1 * self.camera_rot.1).sqrt();
        let camera_off_tot =
            (self.camera_off.0 * self.camera_off.0 + self.camera_off.1 * self.camera_off.1).sqrt();
        let originnn = self.viewport.renderer.args.origin.clone();
        let z = (nalgebra::Matrix1x3::<f32>::new(0., 0., 1.)
            * self.viewport.renderer.args.rotation_mat)
            .normalize();
        let y = (nalgebra::Matrix1x3::<f32>::new(0., 1., 0.)
            * self.viewport.renderer.args.rotation_mat)
            .normalize();
        let x = (nalgebra::Matrix1x3::<f32>::new(1., 0., 0.)
            * self.viewport.renderer.args.rotation_mat)
            .normalize();
        let ms1 = if self.l_trig.2 { 2.0 } else { 1. };
        let ms2 = if self.r_trig.2 { 2.0 } else { 1. };
        let mut scalar_move = (if self.l_trig.0 {
            0.01 * ms1
        } else {
            0.025 * ms1
        }  * self.input_menu.cam_speed* self.input_menu.mov_speed) as f32;

        let mut scalar_rot = (if self.r_trig.0 {
            0.005 * ms2
        } else {
            0.0125 * ms2
        }  * self.input_menu.cam_speed* self.input_menu.rot_speed) as f32;
        // if self.face_origin && originnn.length() > 0.0 {
        //     //*self.viewport.renderer.args.rotation_mat
        //     let idk = nalgebra::Matrix1x3::<f32>::new(originnn.x,originnn.y,originnn.z);

        //     let vecz = Vec3::from((z.data.0[0][0],z.data.0[1][0],z.data.0[2][0])).normalize();

        //     let origin = Vec3::from((idk.data.0[0][0],idk.data.0[1][0],idk.data.0[2][0])).normalize();

        //     let axis = vecz.cross(origin).normalize();
        //     let theta = -vecz.cross(origin).length().atan2(vecz.dot(origin));

        //     self.viewport.renderer.args.rotation_mat *= r_axis(theta, (axis));
        // }
        if (camera_off_tot > 0.0f32) {
            pressed = true;
            let ox = scalar_move * x * self.camera_off.0;
            let ozy = if !self.l_trig.1 {
                scalar_move * z * self.camera_off.1
            } else {
                scalar_move * y * -self.camera_off.1
            };
            let offset_a = scalar_move
                * (Vec3::from((ox.data.0[0][0], ox.data.0[1][0], ox.data.0[2][0]))
                    + Vec3::from((ozy.data.0[0][0], ozy.data.0[1][0], ozy.data.0[2][0])))
                .normalize();
            self.viewport.renderer.args.camera_pos =
                self.viewport.renderer.args.camera_pos + offset_a;
            //let rel = self.viewport.renderer.args.relative_pos;
            //let rel = nalgebra::Matrix1x3::new(rel.x,rel.y,rel.z) * self.viewport.renderer.args.rotation_mat;
            self.viewport.renderer.args.origin = self.viewport.renderer.args.camera_pos;
            //self.viewport.renderer.args.intrinsic = nalgebra::Matrix1x3::<f32>::new(0.,0.,1.);
        }

        if (camera_rot_tot > 0.0f32) {
            pressed = true;
            let p = self.viewport.renderer.args.relative_dir;
            if self.r_trig.1 {
                let z_theta = (2. * scalar_rot * -self.camera_rot.0) * std::f32::consts::PI;
                let mut rotz = r_axis(
                    z_theta + self.viewport.renderer.args.relative_dir.z,
                    (z.data.0[0][0], z.data.0[1][0], z.data.0[2][0]).into(),
                );
                self.viewport.renderer.args.rotation_mat *= rotz;
            } else {
                let x_theta = (scalar_rot * -self.camera_rot.1) * std::f32::consts::PI;
                let y_theta = (scalar_rot * -self.camera_rot.0) * std::f32::consts::PI;

                let mut rotx = r_axis(
                    x_theta + self.viewport.renderer.args.relative_dir.x,
                    (x.data.0[0][0], x.data.0[1][0], x.data.0[2][0]).into(),
                );
                let mut roty = r_axis(
                    y_theta + self.viewport.renderer.args.relative_dir.y,
                    (y.data.0[0][0], y.data.0[1][0], y.data.0[2][0]).into(),
                );
                self.viewport.renderer.args.relative_dir.x = 0.;
                self.viewport.renderer.args.relative_dir.y = 0.;
                self.viewport.renderer.args.rotation_mat *= (rotx * roty);
            }
            let npt = nalgebra::Matrix1x3::<f32>::new(0., 0., 1.)
                * self.viewport.renderer.args.rotation_mat;
            let npz = nalgebra::Matrix1x3::<f32>::new(0., 1., 0.)
                * self.viewport.renderer.args.rotation_mat;
            self.viewport.renderer.args.lookat =
                Vec3::from((npt.data.0[0][0], npt.data.0[1][0], npt.data.0[2][0])).normalize();
            self.viewport.renderer.args.cam_up =
                Vec3::from((npz.data.0[0][0], npz.data.0[1][0], npz.data.0[2][0])).normalize();
        }
        if !self.viewport.renderer.rendering && self.viewport_interact && !self.over_options {
            for event in input.events {
                match event {
                    Event::Key {
                        key,
                        pressed: p,
                        modifiers,
                    } => {
                        if key != Key::Space {
                            if key != Key::Enter {
                                pressed = p;
                            }
                        } else if !self.viewport.renderer.rendering {
                            self.viewport.renderer.render(false, 8, false)
                        }
                        match modifiers {
                            // This is for iteration functions
                            Modifiers {
                                alt: false,
                                ctrl: false,
                                shift: false,
                                mac_cmd: false,
                                command: false,
                            } => match key {
                                _=>(),
                                /* 
                                Key::Num1 => {
                                    self.viewport.renderer.args.iter_funct_choice = 1;
                                    self.viewport.renderer.args.iter_funct = fractal_mandelbrot;
                                }
                                Key::Num2 => {
                                    self.viewport.renderer.args.iter_funct_choice = 2;
                                    self.viewport.renderer.args.iter_funct = fractal_burning_ship;
                                }
                                Key::Num3 => {
                                    self.viewport.renderer.args.iter_funct_choice = 3;
                                    self.viewport.renderer.args.iter_funct = fractal_hybrid_lober;
                                }
                                Key::Num4 => {
                                    self.viewport.renderer.args.iter_funct_choice = 4;
                                    self.viewport.renderer.args.iter_funct = fractal_testing;
                                }
                                Key::Num5 => {
                                    self.viewport.renderer.args.iter_funct_choice = 5;
                                    self.viewport.renderer.args.iter_funct = fractal_spade;
                                }
                                // Key::Num6 => {
                                //     self.viewport.renderer.args.iter_funct_choice = Some(6);
                                //     self.viewport.renderer.args.iter_funct = fractal_mandelbrot;
                                // }
                                // Key::Num7 => {
                                //     self.viewport.renderer.args.iter_funct_choice = Some(7);
                                //     self.viewport.renderer.args.iter_funct = fractal_mandelbrot;
                                // }
                                // Key::Num8 => {
                                //     self.viewport.renderer.args.iter_funct_choice = Some(8);
                                //     self.viewport.renderer.args.iter_funct = fractal_mandelbrot;
                                // }
                                // Reserved
                                // Key::Num9 => (),
                                // Key::Num0 => (),
                                _ => (),
                                */
                            },
                            // This is for coloring functions
                            Modifiers {
                                alt: true,
                                ctrl: false,
                                shift: false,
                                mac_cmd: false,
                                command: false,
                            } => (), /*process_color_key(
                                key,
                                &mut self.viewport.renderer.args.color_funct_choice,
                                &mut self.viewport.renderer.args.color_funct_page,
                                &mut self.viewport.renderer.args.color_funct,
                                vec![
                                    ("VHS_SMOOTH", VHS_SMOOTH),
                                    ("image_mapping", image_mapping),
                                    ("stripe", stripe),
                                    ("RAINBOW_SMOOTH", RAINBOW_SMOOTH),
                                    ("VHS_DISCRETE", VHS_DISCRETE),
                                    ("trap_avg", trap_avg),
                                    ("RAINBOW_DISCRETE", RAINBOW_DISCRETE),
                                    ("floral", floral),
                                    ("OLD_coloring",OLD_coloring),
                                    ("OLD_normal_map", OLD_normal_map),
                                    ("Refraction", refract)]
                            )*/
                            

                            // This is for conditional functions
                            Modifiers {
                                alt: false,
                                ctrl: false,
                                shift: true,
                                mac_cmd: false,
                                command: false,
                            } => match key {
                                Key::Num1 => (),
                                Key::Num2 => (),
                                Key::Num3 => (),
                                Key::Num4 => (),
                                Key::Num5 => (),
                                Key::Num6 => (),
                                Key::Num7 => (),
                                Key::Num8 => (),
                                // Reserved
                                Key::Num9 => (),
                                Key::Num0 => (),
                                _ => (),
                            },
                            _ => {}
                        }
                        if p {
                            match key {
                                Key::F =>
                                {
                                    #[allow(clippy::collapsible_if)]
                                    if pressed {
                                        if !self.viewport.renderer.init_rendering {
                                            if self.viewport.renderer.recording {
                                                println!(
                                                    "STOPPED RECORDING @ {} FRAMES",
                                                    self.viewport.renderer.recorded_frames
                                                );
                                                self.viewport.renderer.recording = false;
                                                println!("Rendering Now!");
                                                self.renderer.clear();
                                                self.renderer.copy_args(&self.viewport.renderer);
                                                self.renderer.kernel.0 =
                                                    self.viewport.renderer.kernel.0.clone();
                                                self.renderer.update_gpu();

                                                self.renderer.txt =
                                                    self.viewport.renderer.txt.clone();
                                                let tooo = if let Ok(r) =
                                                    self.input_menu.samples.parse()
                                                {
                                                    r
                                                } else {
                                                    self.viewport.renderer.rendered_samples
                                                };
                                                self.renderer.render_animation(
                                                    tooo,
                                                    &mut self.viewport.renderer.animation,
                                                );
                                                self.renderer.clear();
                                                println!();
                                            } else {
                                                println!("STARTED RECORDING");
                                                self.viewport.renderer.init_rendering = true;
                                            }
                                        }
                                    }
                                }
                                Key::J => {
                                    self.viewport.renderer.args.julia.1 =
                                        !self.viewport.renderer.args.julia.1;
                                }
                                Key::Enter => {
                                    if self.events.enter.is_none() {
                                        self.events.enter = Some(p)
                                    }
                                }
                                Key::Q => {
                                    if self.events.q.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            false,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.h,
                                            &mut self.events.q,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }
                                Key::W => {
                                    if self.events.w.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            true,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.h,
                                            &mut self.events.w,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }

                                Key::A => {
                                    if self.events.a.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            false,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.h2,
                                            &mut self.events.a,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }
                                Key::S => {
                                    if self.events.s.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            true,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.h2,
                                            &mut self.events.s,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }

                                Key::Z => {
                                    if self.events.z.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            false,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.cexp,
                                            &mut self.events.z,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }
                                Key::X => {
                                    if self.events.x.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            true,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.cexp,
                                            &mut self.events.x,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }
                                Key::E => {
                                    if self.events.z.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            false,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.cexp2,
                                            &mut self.events.e,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }
                                Key::R => {
                                    if self.events.x.is_none() {
                                        process_key_standard_mode(
                                            pressed,
                                            true,
                                            ctrl_code,
                                            &mut self.viewport.renderer.args.cexp2,
                                            &mut self.events.r,
                                            scalar_mult,
                                            None,
                                        );
                                    }
                                }
                                Key::ArrowLeft => {
                                    if self.events.a_left.is_none() {
                                        if ctrl_code {
                                            self.viewport.renderer.args.limit /= 1.05 * scalar_mult;
                                        } else {
                                            self.viewport.renderer.args.limit -= 1.0 * scalar_mult;
                                        }
                                        if self.viewport.renderer.args.limit < 0.0 {
                                            self.viewport.renderer.args.limit = 1.0;
                                        }
                                        self.events.a_left = Some(p)
                                    }
                                }
                                Key::ArrowRight => {
                                    if self.events.a_right.is_none() {
                                        if ctrl_code {
                                            self.viewport.renderer.args.limit *= 1.05 * scalar_mult;
                                        } else {
                                            self.viewport.renderer.args.limit += 1.0 * scalar_mult;
                                        }
                                        self.events.a_right = Some(p)
                                    }
                                }
                                Key::ArrowUp => {
                                    if self.events.a_up.is_none() {
                                        self.viewport.renderer.args.bail *= 1.05 * scalar_mult;
                                        self.events.a_down = Some(p)
                                    }
                                }
                                Key::ArrowDown => {
                                    if self.events.a_down.is_none() {
                                        self.viewport.renderer.args.bail /= 1.05 * scalar_mult;
                                        self.events.a_up = Some(p)
                                    }
                                }
                                Key::P => {
                                    if p {
                                        self.viewport.active = !self.viewport.active;
                                    }
                                }
                                _ => (),
                            }
                        }
                    }
                    Event::Scroll(v) => {
                        let mut q = v.y;
                        if axis_bool {
                            q = v.x;
                        }
                        if q.signum() < 0.0 {
                            self.viewport.renderer.args.zoom /= 1.1 * scalar_mult;
                        } else {
                            self.viewport.renderer.args.zoom *= 1.1 * scalar_mult;
                        }
                        pressed = true;
                        self.events.scroll = Some(v)
                    }
                    Event::PointerButton {
                        pos,
                        button,
                        pressed: p,
                        modifiers,
                    } => {
                        pressed = p;
                        self.events.m_pos = Some(pos);
                        match button {
                            PointerButton::Primary => self.events.m_left = Some(p),
                            PointerButton::Secondary => self.events.m_right = Some(p),
                            PointerButton::Middle => self.events.m_middle = Some(p),
                        }
                    }
                    _ => (),
                }
            }
        }
        if self.viewport.active {
            if let Some(size) = self.viewport.needs_resize {
                self.viewport.resize(size);

                self.viewport.renderer.clear();
                self.viewport.renderer.render(false, 1, false);
                // self.viewport.renderer
                self.viewport.needs_resize = None;
            }
        }
        let ratio =
            self.viewport.renderer.args.size.0 as f64 / self.viewport.renderer.args.size.1 as f64;
        let title_bar_height = 32.0;
        custom_window_frame(title_bar_height, ctx, frame, "FracgenGUI 3D 0.0.1", |ui| {
            let mut frame_size = ui.available_size();
            frame_size.y -= title_bar_height;
            let mut rect = ui.max_rect();
            // rect.set_width(rect.width() - 128.0);
            let painter = ui.painter();
            // Paint the frame:
            painter.rect(rect, 0.0, Color32::from_rgb(0, 0, 0), Stroke::none());
            egui::Area::new("Window")
                .fixed_pos((0.0, title_bar_height))
                .movable(false)
                .show(ctx, |ui| {
                    egui::SidePanel::new(Side::Right, "OptionsPanel")
                        .resizable(true)
                        .width_range((frame_size.x / 3.0).min(1200.0)..=frame_size.x - (frame_size.x / 3.0))
                        .frame(
                            Frame::none()
                                .fill(Color32::from_rgb(24, 24, 24))
                                .outer_margin(0.0)
                                .rounding(Rounding {
                                    nw: 0.0,
                                    ne: 0.0,
                                    sw: 0.0,
                                    se: 0.0,
                                }),
                        )
                        .show_inside(ui, |ui| {
                            // if  {
                                
                            // }
                            let idk = ui.input().clone();
                            let over_options = ui.rect_contains_pointer(ui.max_rect());
                            // println!("{}", over_options);
                            if (mouse_input.primary_down() && !self.over_options && over_options) || (!over_options && (!mouse_input.primary_down() && self.over_options)) {
                                self.over_options = false;
                            } else if over_options {
                                self.over_options = over_options;
                            }
                            // egui::ScrollArea::new([false, true]).show_rows(ui, 32.0, 8, |ui, row_range| {
                            //     for row in row_range {

                            //     }
                            // });
                            let www = ui.available_width();
                            egui::ScrollArea::new([true, true])
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    // ui.men
                                    // menu::bar()
                                    let visuals = ui.visuals_mut();
                                    visuals.button_frame = false;
                                    // visuals.selection.bg_fill =
                                    //     Color32::from_rgba_unmultiplied(255, 255, 255, 255);
                                    visuals.selection.stroke.color =
                                        Color32::from_rgba_unmultiplied(255, 255, 255, 255);
                                    visuals.selection.stroke.width = 1.0;
                                    // visuals.widgets.style(response)
                                    menu::bar(ui, |ui| {
                                        egui::ScrollArea::horizontal().show(ui, |ui| {
                                            let visuals = ui.visuals_mut();
                                            ui.set_height(24.0);
                                            let mut rect = ui.max_rect();
                                            // rect.set_width(rect.width() - 128.0);
                                            let painter = ui.painter();
                                            // Paint the frame:
                                            painter.rect(
                                                rect,
                                                0.0,
                                                Color32::from_rgb(32, 32, 32),
                                                Stroke::none(),
                                            );
                                            ui.add_space(6.0);
                                            let style = ui.style_mut();
                                            // style.visuals.window_fill()
                                            // ui.add_sized(
                                            //     256.0,
                                            //     menu::
                                            //     menu::menu_button(&mut ui, "Viewport", |ui| {}),
                                            // );
                                            // let mut fill = ui.visuals_mut().widgets.open.bg_fill;
                                            ui.visuals_mut().button_frame = true;
                                            // ui.visuals_mut().
                                            let mut fill = ui.visuals_mut().widgets.hovered.bg_fill =
                                                Color32::from_rgba_unmultiplied(16, 16, 16, 255);
                                            // fill = Color32::from_rgba_unmultiplied(0, 0, 0, 255);
                                            let mut fill = ui.visuals_mut().widgets.inactive.bg_fill =
                                                Color32::from_rgba_unmultiplied(24, 24, 24, 255);
                                            // fill = Color32::from_rgba_unmultiplied(0, 0, 0, 255);
    
                                            // fill = Color32::from_rgba_unmultiplied(0, 0, 0, 255);
    
                                            ui.menu_button("Viewtitleport", |ui| {
                                                ui.set_height(0.0);
                                                ui.set_width(0.0);
                                                self.input_menu.page = 0;
                                                ui.close_menu()
                                            });
                                            ui.menu_button("Coloring", |ui| {
                                                ui.set_height(0.0);
                                                ui.set_width(0.0);
                                                self.input_menu.page = 1;
                                                ui.close_menu()
    
                                                // ui.set_height(0.0);
                                                // ui.set_width(0.0);
                                                // ui.menu_button("title", add_contents)
                                                // self.input_menu.page = 1;
                                                // ui.close_menu()
                                            });
                                            ui.menu_button("Iteration", |ui| {
                                                ui.set_height(0.0);
                                                ui.set_width(0.0);
                                                self.input_menu.page = 2;
                                                ui.close_menu()
                                                // ui.set_height(0.0);
                                                // ui.set_width(0.0);
                                                // ui.menu_button("title", add_contents)
                                                // self.input_menu.page = 1;
                                                // ui.close_menu()
                                            });
    
                                            ui.menu_button("ABS", |ui| {
                                                ui.set_height(0.0);
                                                ui.set_width(0.0);
                                                self.input_menu.page = 3;
                                                ui.close_menu()
                                                // ui.set_height(0.0);
                                                // ui.set_width(0.0);
                                                // ui.menu_button("title", add_contents)
                                                // self.input_menu.page = 1;
                                                // ui.close_menu()
                                            });

                                            ui.menu_button("KERNEL", |ui| {
                                                ui.set_height(0.0);
                                                ui.set_width(0.0);
                                                self.input_menu.page = 4;
                                                ui.close_menu()
                                                // ui.set_height(0.0);
                                                // ui.set_width(0.0);
                                                // ui.menu_button("title", add_contents)
                                                // self.input_menu.page = 1;
                                                // ui.close_menu()
                                            });
                                            ui.menu_button("PLOTS", |ui| {
                                                ui.set_height(0.0);
                                                ui.set_width(0.0);
                                                self.input_menu.page = 5;
                                                ui.close_menu()
                                                // ui.set_height(0.0);
                                                // ui.set_width(0.0);
                                                // ui.menu_button("title", add_contents)
                                                // self.input_menu.page = 1;
                                                // ui.close_menu()
                                            });
                                            ui.menu_button("iGRID", |ui| {
                                                ui.set_height(0.0);
                                                ui.set_width(0.0);
                                                self.input_menu.page =6;
                                                ui.close_menu()
                                                // ui.set_height(0.0);
                                                // ui.set_width(0.0);
                                                // ui.menu_button("title", add_contents)
                                                // self.input_menu.page = 1;
                                                // ui.close_menu()
                                            });
    
                                            // .response;
                                            // if vp_page.clicked() {
                                            //     self.input_menu.page = 0
                                            // }
                                        })
                                        
                                    });
                                    /// ASS
                                    match self.input_menu.page {
                                        // Render settings
                                        0 => {
                                            let mut a = ui.spacing_mut();
                                            // a = Spacing::default();
                                            a.item_spacing.y = 0.0;
                                            ui.set_width(www);
                                            let mut size = ui.available_size();
                                            let mut bsize = ui.available_size();
                                            size.y = 16.0;
                                            bsize.y = 16.0;
                                            bsize.x -= 46.0;
                                            // bsize.x /= 2.0;
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Viewport Dimensions/Keep Ratio: ");
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let width_response = ui.add_sized(
                                                    (bsize.x / 2.0, 16.0),
                                                    TextEdit::singleline(
                                                        &mut self.input_menu.viewport_size.0,
                                                    )
                                                    .hint_text("Viewport Width"),
                                                );
                                                let height_response = ui.add_sized(
                                                    (bsize.x / 2.0, 16.0),
                                                    TextEdit::singleline(
                                                        &mut self.input_menu.viewport_size.1,
                                                    )
                                                    .hint_text("Viewport Height"),
                                                );
                                                ui.add_sized(
                                                    (16.0, 16.0),
                                                    Checkbox::new(
                                                        &mut self.input_menu.viewport_size.2,
                                                        "",
                                                    ),
                                                );
                                                if width_response.lost_focus()
                                                    && ui.input().key_pressed(egui::Key::Enter)
                                                {
                                                    if let Ok(new_w) = self
                                                        .input_menu
                                                        .viewport_size
                                                        .0
                                                        .parse::<f32>()
                                                    {
                                                        let mut new_h = if let Some((_, new_h)) =
                                                            self.viewport.needs_resize
                                                        {
                                                            new_h
                                                        } else {
                                                            self.viewport.image_size.1 as f32
                                                        };
                                                        if self.input_menu.viewport_size.2 {
                                                            new_h = new_h * (new_w as f32 / new_h);
                                                            self.input_menu.viewport_size.1 =
                                                                (new_h.round() as usize)
                                                                    .to_string();
                                                        }
                                                        self.viewport.needs_resize =
                                                            Some((new_w, new_h));
                                                    } else {
                                                        self.input_menu.viewport_size.0 =
                                                            if let Some((new_w, _)) =
                                                                self.viewport.needs_resize
                                                            {
                                                                new_w.to_string()
                                                            } else {
                                                                self.viewport
                                                                    .image_size
                                                                    .0
                                                                    .to_string()
                                                            };
                                                    }
                                                }

                                                if height_response.lost_focus()
                                                    && ui.input().key_pressed(egui::Key::Enter)
                                                {
                                                    if let Ok(new_h) = self
                                                        .input_menu
                                                        .viewport_size
                                                        .1
                                                        .parse::<f32>()
                                                    {
                                                        let mut new_w = if let Some((new_w, _)) =
                                                            self.viewport.needs_resize
                                                        {
                                                            new_w
                                                        } else {
                                                            self.viewport.image_size.0 as f32
                                                        };
                                                        if self.input_menu.viewport_size.2 {
                                                            new_w = new_w * (new_h as f32 / new_w);
                                                            self.input_menu.viewport_size.0 =
                                                                (new_w.round() as usize)
                                                                    .to_string();
                                                        }
                                                        self.viewport.needs_resize =
                                                            Some((new_w, new_h));
                                                    } else {
                                                        self.input_menu.viewport_size.1 =
                                                            if let Some((new_w, _)) =
                                                                self.viewport.needs_resize
                                                            {
                                                                new_w.to_string()
                                                            } else {
                                                                self.viewport
                                                                    .image_size
                                                                    .1
                                                                    .to_string()
                                                            };
                                                    }
                                                }
                                            });
                                            
                                            ui.add_space(4.0);
                                            ui.separator();
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Export Dimensions/Keep Ratio: ");
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let export_width_response = ui.add_sized(
                                                    (bsize.x / 2.0, 16.0),
                                                    TextEdit::singleline(
                                                        &mut self.input_menu.export_size.0,
                                                    )
                                                    .hint_text("Export Width"),
                                                );
                                                let export_height_response = ui.add_sized(
                                                    (bsize.x / 2.0, 16.0),
                                                    TextEdit::singleline(
                                                        &mut self.input_menu.export_size.1,
                                                    )
                                                    .hint_text("Export Height"),
                                                );
                                                ui.add_sized(
                                                    (16.0, 16.0),
                                                    Checkbox::new(
                                                        &mut self.input_menu.export_size.2,
                                                        "",
                                                    ),
                                                );

                                                if export_width_response.lost_focus()
                                                    && ui.input().key_pressed(egui::Key::Enter)
                                                {
                                                    if let Ok(new_w) =
                                                        self.input_menu.export_size.0.parse::<f32>()
                                                    {
                                                        // if self.input_menu.export_size.2 {
                                                        //     let ratio =
                                                        //         new_w / self.renderer.args.size.0 as f32;
                                                        //     new_h *= ratio;
                                                        //     self.input_menu.export_size.1 =
                                                        //         (new_h.round() as usize)
                                                        //             .to_string();
                                                        // }
                                                        let mut new_h = self.renderer.args.size.1 as f32;
                                                        self.renderer.args.size = (
                                                            new_w.round() as usize,
                                                            new_h.round() as usize,
                                                        );
                                                        self.renderer.raw =
                                                            Image::new(self.renderer.args.size);
                                                        self.input_menu.export_size.0 =
                                                            (new_w.round() as usize).to_string();
                                                    } else {
                                                        self.input_menu.export_size.0 =
                                                            self.renderer.args.size.0.to_string();
                                                    }
                                                }

                                                if export_height_response.lost_focus()
                                                    && ui.input().key_pressed(egui::Key::Enter)
                                                {
                                                    if let Ok(new_h) =
                                                        self.input_menu.export_size.1.parse::<f32>()
                                                    {
                                                        let mut new_w = self.renderer.args.size.0 as f32;
                                                        // if self.input_menu.export_size.2 {
                                                        //     let ratio =
                                                        //         new_h / self.renderer.args.size.1 as f32;
                                                        //     new_w *= ratio;
                                                        //     self.input_menu.export_size.0 =
                                                        //         (new_w.round() as usize)
                                                        //             .to_string();
                                                        // }
                                                        self.renderer.args.size = (
                                                            new_w.round() as usize,
                                                            new_h.round() as usize,
                                                        );
                                                        self.renderer.raw =
                                                            Image::new(self.renderer.args.size);
                                                        self.input_menu.export_size.1 =
                                                            (new_h.round() as usize).to_string();
                                                    } else {
                                                        self.input_menu.export_size.1 =
                                                            self.renderer.args.size.1.to_string();
                                                    }
                                                }
                                            });
                                            ui.add_space(4.0);
                                            ui.horizontal(|ui| {
                                                ui.vertical_centered_justified(|ui| {
                                                    let ksjdha = (ui.available_width(), 16.0);
                                                    ui.set_width(ui.available_width());
                                                    ui.set_height(16.0);
                                                    ui.horizontal(|ui|{
                                                        if ui.button("Choose Output Folder").clicked() {
                                                            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                                                self.renderer.output = path.display().to_string();
                                                            }
                                                        }
                                                        if ui.button("Open Output Folder").clicked() {
                                                            // if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                                            //     self.renderer.output = path.display().to_string();
                                                            // }
                                                            if let Ok(p) = fs::canonicalize(PathBuf::from(&self.renderer.output)) {
                                                                print!("{:?}", p);
                                                                std::process::Command::new( "nautilus" )
                                                                .arg(p) // <- Specify the directory you'd like to open.
                                                                .spawn( )
                                                                .unwrap( );
                                                            }
                                                            
                                                        }
                                                    })
                                                    
                                                });
                                            });
                                            ui.add_space(4.0);
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let aw = ui.available_width() / 2.0 - (56.0 + 64.0);
                                                ui.vertical(|ui| {
                                                    let ksjdha = (aw, 16.0);
                                                    ui.add_sized(ksjdha, egui::TextEdit::singleline(&mut self.renderer.output).hint_text("File path"))
                                                });
                                                ui.vertical(|ui| {
                                                    let ksjdha = (aw, 16.0);
                                                    ui.add_sized(ksjdha, egui::TextEdit::singleline(&mut self.renderer.output_name.0).hint_text("File name"))
                                                });
                                                ui.vertical(|ui| {
                                                    let ksjdha = ( 64.0, 16.0);
                                                    ui.add_sized(ksjdha, egui::Checkbox::new(&mut self.renderer.output_name.1, "Named?"));
                                                });
                                                ui.vertical(|ui| {
                                                    let ksjdha = ( 64.0, 16.0);
                                                    ui.add_sized(ksjdha, egui::Checkbox::new(&mut self.renderer.alpha, "Alpha?"));
                                                });
                                                ui.vertical(|ui| {
                                                    let ksjdha = ( 64.0, 16.0);
                                                    ui.add_sized(ksjdha, egui::Checkbox::new(&mut self.renderer.hdr, "Hdr?"));
                                                });

                                            });
                                            ui.add_space(2.0);
                                            ui.separator();
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Viewport/Export Samples: ");
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let w = ui.available_width();
                                                ui.add_sized(
                                                    (bsize.x / 6.0, 16.0),
                                                    TextEdit::singleline(
                                                        &mut self
                                                            .viewport
                                                            .renderer
                                                            .rendered_samples
                                                            .to_string(),
                                                    )
                                                    .hint_text("Viewport Samples")
                                                    .interactive(false),
                                                );
                                                let samples_response = ui.add_sized(
                                                    (bsize.x / 6.0, 16.0),
                                                    TextEdit::singleline(
                                                        &mut self.input_menu.samples,
                                                    )
                                                    .hint_text("Export Samples"),
                                                );
                                                let samples_response = ui.add_sized(
                                                    (bsize.x / 6.0, 16.0),
                                                    egui::DragValue::new(&mut self.renderer.split).prefix("S:"),
                                                );
                                                let samples_response = ui.add_sized(
                                                    (bsize.x / 6.0, 16.0),
                                                    egui::DragValue::new(&mut self.viewport.renderer.args.chunk_size.0).prefix("X:"),
                                                );
                                                let samples_response = ui.add_sized(
                                                    (bsize.x / 6.0, 16.0),
                                                    egui::DragValue::new(&mut self.viewport.renderer.args.chunk_size.1).prefix("Y:"),
                                                );
                                                let samples_response = ui.add_sized(
                                                    (bsize.x / 6.0, 16.0),
                                                    egui::DragValue::new(&mut self.viewport.renderer.args.chunk_stop).prefix("Y%:"),
                                                );
                                            });
                                            ui.add_space(2.0);
                                            
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let w = ui.available_width();
                                                ui.add_sized(
                                                    (bsize.x / 3.0, 16.0),
                                                    egui::DragValue::new(
                                                        &mut self
                                                            .samples
                                                            
                                                    ).prefix("VP_samples/p:"),
                                                );
                                                let mut idfkx = (self.input_menu.export_size.0.parse::<f32>().unwrap_or(0.) / self.viewport.renderer.args.chunk_size.0);
                                                let mut idfky = (self.input_menu.export_size.1.parse::<f32>().unwrap_or(0.) / self.viewport.renderer.args.chunk_size.1);
                                                ui.add_sized(
                                                    (bsize.x / 3.0, 16.0),
                                                    egui::DragValue::new(
                                                        &mut idfkx
                                                            
                                                    ).prefix("X Chunks:"),
                                                );
                                                ui.add_sized(
                                                    (bsize.x / 3.0, 16.0),
                                                    egui::DragValue::new(
                                                        &mut idfky
                                                            
                                                    ).prefix("Y Chunks:"),
                                                );
                                            });
                                            
                                            ui.add_space(4.0);
                                            ui.separator();
                                            ()
                                        }
                                        // Coloring stuff
                                        1 => {
                                            ui.vertical(|ui| {
                                                // let mut responses = Vec::new();
                                                ui.vertical(|ui| {
                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self.viewport.renderer.args.hue_offset_pre_exp,
                                                            0.0..=360.0,
                                                        )
                                                        .logarithmic(true)
                                                        .text("Hue Shift PreExp"),
                                                    );
                                                    pressed |= result.changed();

                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self.viewport.renderer.args.h,
                                                            -16.0..=16.0,
                                                        )
                                                        .text("h"),
                                                    );
                                                    pressed |= result.changed();
                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self.viewport.renderer.args.h2,
                                                            -16.0..=16.0,
                                                        )
                                                        .text("h2"),
                                                    );
                                                    pressed |= result.changed();
                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self.viewport.renderer.args.cexp,
                                                            0.0..=128.0,
                                                        ).logarithmic(true)
                                                        .text("cexp"),
                                                    );
                                                    pressed |= result.changed();
                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self.viewport.renderer.args.cexp2,
                                                            0.0..=128.0,
                                                        ).logarithmic(true)
                                                        .text("cexp2"),
                                                    );
                                                    pressed |= result.changed();
                                                    // if result.changed() {
                                                    //     pressed = true;
                                                    // }
            
                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self
                                                                .viewport
                                                                .renderer
                                                                .args
                                                                .hue_offset_post_exp,
                                                            0.0..=360.0,
                                                        )
                                                        .text("Hue Shift PostExp"),
                                                    );
                                                    if result.changed() {
                                                        pressed = true;
                                                    }
            
                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self.viewport.renderer.args.hue_scale,
                                                            0.1..=10.0,
                                                        )
                                                        .text("Hue Scale"),
                                                    );
                                                    if result.changed() {
                                                        pressed = true;
                                                    }
            
                                                    let result = ui.add(
                                                        egui::Slider::new(
                                                            &mut self.viewport.renderer.args.hue_exponent,
                                                            1.0..=2.0,
                                                        )
                                                        .logarithmic(true)
                                                        .text("Hue Exp"),
                                                    );
                                                    if result.changed() {
                                                        pressed = true;
                                                    }
                                                });
                                               
                                            });
                                        }
                                        // Iteration stuff
                                        2 => {
                                            let mut a = ui.spacing_mut();
                                            // a = Spacing::default();
                                            a.item_spacing.y = 0.0;
                                            ui.set_width(www);
                                            let mut size = ui.available_size();
                                            let mut bsize = ui.available_size();
                                            size.y = 16.0;
                                            bsize.y = 16.0;
                                            bsize.x -= 46.0;
                                            // bsize.x /= 2.0;
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Bounds: ");
                                            });
                                            // ui.horizontal(|ui|{ui.add_space(8.0);
                                            //     ui.label("Change: ");
                                            //     let response = ui.add(
                                            //         egui::Slider::new(
                                            //             &mut self.input_menu.julia_speed,
                                            //             0.0001..=2.0,
                                            //         )
                                            //         .logarithmic(true),
                                            //     );
                                            //     pressed |= response.changed();});
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.bail,
                                                    ).speed(self.input_menu.bail_speed)
                                                    // .logarithmic(true)
                                                    .prefix("Bailout: "),
                                                );
                                                pressed |= response.changed();
                                                
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.derbail,
                                                    ).speed(self.input_menu.bail_speed)
                                                    .prefix("DerBailout: "),
                                                );
                                                pressed |= response.changed();
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                let response = ui.add(
                                                    egui::Slider::new(
                                                        &mut self.viewport.renderer.args.limit,
                                                        1.0..=(2 << 16) as f64,
                                                    )
                                                    .logarithmic(true)
                                                    .prefix("Limit: "),
                                                );
                                                pressed |= response.changed();
                                            });
                                            ui.add_space(8.0);
                                            ui.separator();
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.);
                                                ui.label("Bailout Speed: ");
                                                let rate = ( (self.input_menu.bail_speed) / 100.).clamp(0.00000000001, 0.5);
                                                ui.add(
                                                    egui::DragValue::new(&mut self.input_menu.bail_speed).speed(rate).clamp_range(0.00000000001..=1.).fixed_decimals(16)
                                                );
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.);
                                                ui.label("Julia Speed: ");
                                                let rate = ( (self.input_menu.julia_speed) / 100.).clamp(0.00000000001, 0.5);
                                                ui.add(
                                                    egui::DragValue::new(&mut self.input_menu.julia_speed).speed(rate).clamp_range(0.00000000001..=1.).fixed_decimals(16)
                                                );
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.);
                                                ui.label("Tweak Speed: ");
                                                let rate = ( (self.input_menu.tweak_speed) / 100.).clamp(0.00000000001, 0.5);
                                                ui.add(
                                                    egui::DragValue::new(&mut self.input_menu.tweak_speed).speed(rate).clamp_range(0.00000000001..=1.).fixed_decimals(16)
                                                );
                                            });
                                            ui.add_space(8.0);
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Julia Re: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.julia.0.re,
                                                    )
                                                    .speed(self.input_menu.julia_speed),
                                                );

                                                pressed |= response.changed();
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Julia Im: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.julia.0.im,
                                                    )
                                                    .speed(self.input_menu.julia_speed),
                                                );
                                                pressed |= response.changed();
                                            });
                                            ui.add_space(8.0);
                                            let mut p = false;
                                            ui.horizontal(|ui| {
                                                
                                                ui.add_space(8.0);
                                                ui.label("Origin X: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.camera_pos.x,
                                                    )
                                                    .speed(self.input_menu.tweak_speed),
                                                );

                                                pressed |= response.changed();
                                                
                                                ui.add_space(8.0);
                                                ui.label("Local: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.relative_pos.x,
                                                    )
                                                    .speed(self.input_menu.tweak_speed),
                                                );
                                                pressed |= response.changed();
                                                
                                                
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Origin Y: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.camera_pos.y,
                                                    )
                                                    .speed(self.input_menu.tweak_speed),
                                                );
                                                pressed |= response.changed();
                                                ui.label("Local: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.relative_pos.y,
                                                    )
                                                    .speed(self.input_menu.tweak_speed),
                                                );
                                                pressed |= response.changed();
                                                
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Origin Z: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.camera_pos.z,
                                                    )
                                                    .speed(self.input_menu.tweak_speed),
                                                );
                                                pressed |= response.changed();
                                                ui.label("Local: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.relative_pos.z,
                                                    )
                                                    .speed(self.input_menu.tweak_speed),
                                                );
                                                pressed |= response.changed();
                                                
                                            });
                                            p |= pressed;
                                            if p {
                                                let rel = self.viewport.renderer.args.relative_pos;
                                                let rel = nalgebra::Matrix1x3::new(rel.x,rel.y,rel.z) * self.viewport.renderer.args.rotation_mat;
                                                self.viewport.renderer.args.origin =self.viewport.renderer.args.camera_pos + Vec3::from((rel.data.0[0][0],rel.data.0[1][0],rel.data.0[2][0]));
                                                
                                                let mut rotx = r_axis(self.viewport.renderer.args.relative_dir.x, (x.data.0[0][0],x.data.0[1][0],x.data.0[2][0]).into());
                                                let mut roty = r_axis(self.viewport.renderer.args.relative_dir.y, (y.data.0[0][0],y.data.0[1][0],y.data.0[2][0]).into());
                                                let mut rotz = r_axis(self.viewport.renderer.args.relative_dir.z, (z.data.0[0][0],z.data.0[1][0],z.data.0[2][0]).into());

                                                self.viewport.renderer.args.relative_dir.x=0.;
                                                self.viewport.renderer.args.relative_dir.y=0.;
                                                self.viewport.renderer.args.rotation_mat *= (rotx*roty*rotz) ;
                                                //self.viewport.renderer.args.lookat = self.viewport.renderer.args.rotation_mat *
                                            }
                                            ui.add_space(8.0);
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Zoom: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.zoom,
                                                    )
                                                    .speed(self.input_menu.tweak_speed),
                                                );

                                                pressed |= response.changed();
                                            });
                                            ui.add_space(4.0);

                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Init Z Re: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.z_init.re,
                                                    )
                                                    .speed(self.input_menu.julia_speed),
                                                );

                                                pressed |= response.changed();
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add_space(8.0);
                                                ui.label("Init Z Im: ");
                                                let response = ui.add(
                                                    egui::DragValue::new(
                                                        &mut self.viewport.renderer.args.z_init.im,
                                                    )
                                                    .speed(self.input_menu.julia_speed),
                                                );
                                                pressed |= response.changed();
                                            });
                                            ui.add_space(4.0);
                                            // egui::ScrollArea::new(has_bar)

                                            ui.separator();
                                            

                                            ui.horizontal(|ui| {
                                                let response = ui.add(egui::Checkbox::new(&mut self.viewport.renderer.args.fill_set, "Fill Set?"));
                                                pressed |= response.changed();
                                                let response = ui.add(egui::Checkbox::new(&mut self.viewport.renderer.args.only_set, "Draw ONLY Set?"));
                                                pressed |= response.changed();
                                            });
                                            
                                            ()
                                        }
                                        // Abs folding stuff
                                        3 => {
                                            // let mut scroll_delta = input.scroll_delta;
                                            // // let response = ui.allocate_response(desired_size, sense)
                                            // // if
                                            // self.abs_scroll += scroll_delta.y as f64;
                                            // // if ui.button("Scroll down").clicked() {
                                            // //     scroll_delta.y -= 0.05;
                                            // // }
                                            // // if ui.button("Scroll up").clicked() {
                                            // //     scroll_delta.y += 0.05;
                                            // // }
                                            egui::ScrollArea::vertical().show_rows(
                                                ui,
                                                64.0 + 9.0 + (16.0 * 6.0),
                                                self.viewport.renderer.args.z_steps,
                                                |ui, rows| {
                                                    // ui.scroll_with_delta(Vec2 {
                                                    //     x: 0.0,
                                                    //     y: self.abs_scroll as f32,
                                                    // });
                                                    // ui.scroll_with_delta(scroll_delta);
                                                    let rl = rows.len();
                                                    let mut rows = rows;
                                                    // ui.scroll_to_cursor(Some(Align::BOTTOM));
                                                    ui.vertical(|ui| {
                                                ui.set_height(16.0);
                                                ui.horizontal(|ui| {
                                                    let mut rect = ui.available_rect_before_wrap();
                                                    
                                                    let painter = ui.painter();
                                                    // Paint the frame:
                                                    painter.rect(rect, 0.0, Color32::from_rgb(16, 16, 16), Stroke::none());
                                                    
                                                    if ui.button("Add").clicked() {
                                                        self.viewport
                                                            .renderer
                                                            .args
                                                            .z_offsets
                                                            .push( LinearTransform::default());
    
                                                        self.viewport.renderer.args.z_steps += 1;
                                                        rows = 0..self.viewport.renderer.args.z_steps
                                                    }
                                                    if ui.button("Delete").clicked()
                                                    && self.viewport.renderer.args.z_steps > 1
                                                {
                                                    self.viewport.renderer.args.z_offsets.pop();
                                                    self.viewport.renderer.args.z_steps -= 1;
                                                    rows = 0..self.viewport.renderer.args.z_steps
                                                }
                                                
                                                });
                                                    });
                                                    
                                                
                                                    for mut row in rows {
                                                        let mut changed = false;

                                                        ui.vertical(|ui| {
                                                            // let deleted = 
                                                            ui.horizontal(|ui| {
                                                                ui.add_space(8.0);
                                                                ui.set_height(16.0);
                                                                 let mut rect = ui.available_rect_before_wrap();
                                                                //  rect.min.x += 4.0;
                                                                rect.max.x -= 8.0;
                                                                 rect.max.y += 4.0;
                                                                let painter = ui.painter();
                                                                // Paint the frame:
                                                                painter.rect(rect, 0.0, Color32::from_rgb(12, 12, 12), Stroke::none());
                                                                // ui.scroll_to_cursor(Some(Align::BOTTOM));
                                                                if row > 0 {
                                                                    
                                                                 if ui.button("Move Up").clicked()
                                                                    && self.viewport.renderer.args.z_steps > 1
                                                                    {
                                                                    self.viewport.renderer.args.z_offsets.swap(row, row-1);
                                                                    changed = true;
                                                                    pressed = true;
                                                                    }
                                                                } else {
                                                                    ui.add_space(64.0);

                                                                }
                                                                if row < rl - 1 && ui.button("Move Down").clicked() {
                                                                    self.viewport
                                                                        .renderer
                                                                        .args
                                                                        .z_offsets
                                                                        .swap(row, row+1);

                                                                    changed = true;
                                                                    pressed = true;
                                                                } 
                                                            });
                                                            if !changed {
 // ui.set_max_height(64.0);
                                                            // ui.set_height(64.0);
                                                            // ui.horizontal(|ui|)
                                                            // egui::M
                                                            ui.push_id(
                                                                format!("Row {} Collapse", row),
                                                                |ui| {
                                                                ui.add_space(4.0);

                                                                    ui.horizontal(|ui| {
                                                                ui.add_space(8.0);

                                                                        ui.add_sized(
                                                                            (16.0, 16.0),
                                                                            egui::Checkbox::new(
                                                                                &mut self
                                                                                    .viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .apply_inv,
                                                                                "Apply Inverse?",
                                                                            ),
                                                                        );
                                                                        ui.collapsing(
                                                                            match self
                                                                                .viewport
                                                                                .renderer
                                                                                .args
                                                                                .z_offsets[row]
                                                                                .mode
                                                                            {
                                                                                LinearTransformMode::Offset => "Offset",
                                                                                LinearTransformMode::Rotate => "Rotate",
                                                                                LinearTransformMode::Scale => "Scale",
                                                                                LinearTransformMode::Transform => "Transform",
                                                                                LinearTransformMode::Mobius => "Mobius",
                                                                                LinearTransformMode::Power => "Power",
                                                                            },
                                                                            |ui| {
                                                                                // ui.id
                                                                                ui.set_max_height(
                                                                                    16.0 * 6.0,
                                                                                );
                                                                                if ui
                                                                                    .button(
                                                                                        "Offset",
                                                                                    )
                                                                                    .clicked()
                                                                                {
                                                                                    self.viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .mode = LinearTransformMode::Offset;
                                                                                }
                                                                                if ui
                                                                                    .button(
                                                                                        "Rotate",
                                                                                    )
                                                                                    .clicked()
                                                                                {
                                                                                    self.viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .mode = LinearTransformMode::Rotate;
                                                                                }
                                                                                if ui
                                                                                    .button("Scale")
                                                                                    .clicked()
                                                                                {
                                                                                    self.viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .mode = LinearTransformMode::Scale;
                                                                                }
                                                                                if ui
                                                                                    .button(
                                                                                        "Transform",
                                                                                    )
                                                                                    .clicked()
                                                                                {
                                                                                    self.viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .mode = LinearTransformMode::Transform;
                                                                                }
                                                                                if ui
                                                                                    .button(
                                                                                        "Mobius",
                                                                                    )
                                                                                    .clicked()
                                                                                {
                                                                                    self.viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .mode = LinearTransformMode::Mobius;
                                                                                }
                                                                                if ui
                                                                                    .button(
                                                                                        "Power",
                                                                                    )
                                                                                    .clicked()
                                                                                {
                                                                                    self.viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .mode = LinearTransformMode::Power;
                                                                                }
                                                                            },
                                                                        );
                                                                    });
                                                                },
                                                            );
                                                            ui.horizontal(|ui| {
                                                                ui.set_height(16.0);
                                                                ui.add_space(8.0);
                                                                match self
                                                                    .viewport
                                                                    .renderer
                                                                    .args
                                                                    .z_offsets[row]
                                                                    .mode
                                                                {
                                                                    LinearTransformMode::Offset => {
                                                                        let wx = (ui
                                                                            .available_width()
                                                                            / 2.0)
                                                                            - 80.0;
                                                                        ui.add_sized(
                                                                            (16.0, 16.0),
                                                                            egui::Checkbox::new(
                                                                                &mut self
                                                                                    .viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .active,
                                                                                "",
                                                                            ),
                                                                        );
                                                                        let b = self
                                                                            .viewport
                                                                            .renderer
                                                                            .args
                                                                            .z_offsets[row]
                                                                            .abs;
                                                                        let response = ui
                                                                            .add_sized(
                                                                            (wx, 16.0),
                                                                            egui::DragValue::new(
                                                                                &mut self
                                                                                    .viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .offset
                                                                                    .re,
                                                                            )
                                                                            .speed(
                                                                                self.input_menu
                                                                                    .julia_speed,
                                                                            )
                                                                            .prefix(if b[0] {
                                                                                format!(
                                                                                    "|Re({})|: ",
                                                                                    row
                                                                                )
                                                                            } else {
                                                                                format!(
                                                                                    "Re({}): ",
                                                                                    row
                                                                                )
                                                                            }),
                                                                        );
                                                                        pressed |=
                                                                            response.changed();
                                                                        ui.add_sized(
                                                                            (16.0, 16.0),
                                                                            egui::Checkbox::new(
                                                                                &mut self
                                                                                    .viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .abs[0],
                                                                                "",
                                                                            ),
                                                                        );
                                                                        let response = ui
                                                                            .add_sized(
                                                                            (wx, 16.0),
                                                                            egui::DragValue::new(
                                                                                &mut self
                                                                                    .viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .offset
                                                                                    .im,
                                                                            )
                                                                            .speed(
                                                                                self.input_menu
                                                                                    .julia_speed,
                                                                            )
                                                                            .prefix(if b[1] {
                                                                                format!(
                                                                                    "|Im({})|: ",
                                                                                    row
                                                                                )
                                                                            } else {
                                                                                format!(
                                                                                    "Im({}): ",
                                                                                    row
                                                                                )
                                                                            }),
                                                                        );
                                                                        pressed |=
                                                                            response.changed();
                                                                        ui.add_sized(
                                                                            (16.0, 16.0),
                                                                            egui::Checkbox::new(
                                                                                &mut self
                                                                                    .viewport
                                                                                    .renderer
                                                                                    .args
                                                                                    .z_offsets[row]
                                                                                    .abs[1],
                                                                                "",
                                                                            ),
                                                                        );
                                                                    }
                                                                    LinearTransformMode::Scale => {
                                                                        let offset = &mut self
                                                                            .viewport
                                                                            .renderer
                                                                            .args
                                                                            .z_offsets[row];
                                                                        ui.vertical(|ui| {
                                                                            ui.horizontal(
                                                                                |ui| {
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut offset.active, ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut offset.abs[0], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut offset.abs[1], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                }
                                                                            );
                                                                            let response = ui.add(
                                                                                egui::DragValue::new(
                                                                                    &mut offset.scalar_or_angle
                                                                                ).speed(
                                                                                    self.input_menu.julia_speed
                                                                                )
                                                                            );
                                                                            pressed |= response.changed();
                                                                        });
                                                                    },
                                                                    LinearTransformMode::Rotate => 
                                                                    {
                                                                        let offset = &mut self
                                                                        .viewport
                                                                        .renderer
                                                                        .args
                                                                        .z_offsets[row];
                                                                    ui.vertical(|ui| {
                                                                        ui.horizontal(
                                                                            |ui| {
                                                                                let response =
                                                                                ui.add(
                                                                                    egui::Checkbox::new(
                                                                                        &mut offset.active, ""
                                                                                    )
                                                                                );
                                                                                pressed |= response.changed();
                                                                                let response =
                                                                                ui.add(
                                                                                    egui::Checkbox::new(
                                                                                        &mut offset.abs[0], ""
                                                                                    )
                                                                                );
                                                                                pressed |= response.changed();
                                                                                let response =
                                                                                ui.add(
                                                                                    egui::Checkbox::new(
                                                                                        &mut offset.abs[1], ""
                                                                                    )
                                                                                );
                                                                                pressed |= response.changed();
                                                                            }
                                                                        );
                                                                        let response = ui.add(
                                                                            egui::DragValue::new(
                                                                                &mut offset.scalar_or_angle
                                                                            ).speed(
                                                                                self.input_menu.julia_speed
                                                                            )
                                                                        );
                                                                        pressed |= response.changed();
                                                                    });
                                                                    },
                                                                    LinearTransformMode::Transform => {
                                                                        let offset = &mut self
                                                                            .viewport
                                                                            .renderer
                                                                            .args
                                                                            .z_offsets[row];
                                                                        ui.vertical(|ui| {
                                                                            let mut angle = false;
                                                                            let mut scale = false;
                                                                            ui.horizontal(
                                                                                |ui| {
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut offset.active, ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut offset.abs[0], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut offset.abs[1], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                }
                                                                            );
                                                                            ui.horizontal(|ui| {
                                                                                // let response = ui.drag_angle(&mut offset.angles[1]);
                                                                                // if response.changed()  {
                                                                                //     // offset.angles[0] = offset.matrix.r2.x.atan2(offset.matrix.r1.x) as f32;
                                                                                //     offset.matrix.r1.x = offset.angles[0].cos() as f64;
                                                                                //     offset.matrix.r2.x = -offset.angles[0].sin() as f64;
                                                                                //     pressed = true;
                                                                                //     angle = true;
                                                                                // }
                                                                                ui.add_space(16.0);
                                                                                let response =
                                                                                    ui.add(
                                                                                        egui::DragValue::new(
                                                                                            &mut offset.matrix.r1.x
                                                                                        ).speed(
                                                                                            self.input_menu.julia_speed
                                                                                        )
                                                                                    );
                                                                                pressed |= response.changed();

                                                                                // if response.changed() && !angle {
                                                                                //     offset.angles[0] = offset.matrix.r2.x.atan2(offset.matrix.r1.x) as f32;
                                                                                //     pressed = true;
                                                                                //     scale = true;
                                                                                // }
                                                                                let response =
                                                                                    ui.add(
                                                                                        egui::DragValue::new(
                                                                                            &mut offset.matrix.r1.y
                                                                                        ).speed(
                                                                                            self.input_menu.julia_speed
                                                                                        )
                                                                                    );
                                                                                pressed |= response.changed();

                                                                                // if response.changed() && !angle {
                                                                                //     offset.angles[1] = offset.matrix.r2.y.atan2(offset.matrix.r1.y) as f32;
                                                                                //     pressed = true;
                                                                                //     scale = true;
                                                                                // }
                                                                            });
                                                                            ui.horizontal(|ui| {
                                                                                // let response = ui.drag_angle(&mut offset.angles[0]);
                                                                                // if response.changed() {
                                                                                    // offset.angles[0] = offset.matrix.r2.x.atan2(offset.matrix.r1.x) as f32;
                                                                                    // let m = off/set.matrix.r2.y.hypot(offset.matrix.r1.y);
                                                                                    // offset.matrix.r1.y = offset.angles[1].cos() as f64;
                                                                                    // offset.matrix.r2.y = offset.angles[1].sin() as f64;
                                                                                    // pressed = true;
                                                                                // }
                                                                                ui.add_space(16.0);
                                                                                let response =
                                                                                    ui.add(
                                                                                        egui::DragValue::new(
                                                                                            &mut offset.matrix.r2.x
                                                                                        ).speed(
                                                                                            self.input_menu.julia_speed
                                                                                        )
                                                                                    );
                                                                                pressed |= response.changed();
                                                                                // if response.changed() && !angle {
                                                                                //     offset.angles[0] = offset.matrix.r2.x.atan2(offset.matrix.r1.x) as f32;
                                                                                //     pressed = true;
                                                                                // }
                                                                                let response =
                                                                                    ui.add(
                                                                                        egui::DragValue::new(
                                                                                            &mut offset.matrix.r2.y
                                                                                        ).speed(
                                                                                            self.input_menu.julia_speed
                                                                                        )
                                                                                    );
                                                                                pressed |= response.changed();

                                                                                // if response.changed() && !angle {
                                                                                //     offset.angles[1] = offset.matrix.r2.x.atan2(offset.matrix.r1.x) as f32;
                                                                                //     pressed = true;
                                                                                // }
                                                                            });
                                                                            ui.add_space(4.0);
                                                                            ui.horizontal(|ui| {
                                                                                let response =
                                                                                    ui.add(
                                                                                        egui::DragValue::new(
                                                                                            &mut offset.offset.re
                                                                                        ).prefix("Offset: ").speed(
                                                                                            self.input_menu.julia_speed
                                                                                        )
                                                                                    );
                                                                                pressed |= response.changed();
                                                                                let response =
                                                                                    ui.add(
                                                                                        egui::DragValue::new(
                                                                                            &mut offset.offset.im
                                                                                        ).speed(
                                                                                            self.input_menu.julia_speed
                                                                                        )
                                                                                    );
                                                                                pressed |= response.changed();
                                                                            });
                                                                        });
                                                                    }
                                                                    LinearTransformMode::Mobius => {
                                                                        let transform = &mut self
                                                                            .viewport
                                                                            .renderer
                                                                            .args
                                                                            .z_offsets[row];
                                                                        ui.vertical(|ui| {
                                                                            let mut angle = false;
                                                                            let mut scale = false;
                                                                            ui.horizontal(
                                                                                |ui| {
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut transform.active, ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut transform.abs[0], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut transform.abs[1], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                }
                                                                            );
                                                                            ui.vertical(|ui| {
                                                                                let w = ui.available_width() / 4.0 - 40.0;
                                                                                ui.horizontal(|ui| {
                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[0].re).prefix("Re(a): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();
                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[0].im).prefix("Im(a): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();

                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[1].re).prefix("Re(b): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();
                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[1].im).prefix("Im(b): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();
                                                                                });
                                                                                ui.horizontal(|ui| {
                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[2].re).prefix("Re(c): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();
                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[2].im).prefix("Im(c): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();

                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[3].re).prefix("Re(d): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();
                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.mobius[3].im).prefix("Im(d): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();
                                                                                });
                                                                                
                                                                            })
                                                                            
                                                                        });
                                                                    }
                                                                    LinearTransformMode::Power => {
                                                                        let transform = &mut self
                                                                            .viewport
                                                                            .renderer
                                                                            .args
                                                                            .z_offsets[row];
                                                                        ui.vertical(|ui| {
                                                                            let mut angle = false;
                                                                            let mut scale = false;
                                                                            ui.horizontal(
                                                                                |ui| {
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut transform.active, ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut transform.abs[0], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                    let response =
                                                                                    ui.add(
                                                                                        egui::Checkbox::new(
                                                                                            &mut transform.abs[1], ""
                                                                                        )
                                                                                    );
                                                                                    pressed |= response.changed();
                                                                                }
                                                                            );
                                                                            ui.vertical(|ui| {
                                                                                let w = ui.available_width() / 4.0 - 40.0;
                                                                                ui.horizontal(|ui| {
                                                                                    let response = ui.add_sized((w,16.0), egui::DragValue::new(&mut transform.scalar_or_angle).prefix("Re(a): ").speed(self.input_menu.julia_speed));
                                                                                    pressed |= response.changed();
                                                                                    
                                                                                });
                                                                                
                                                                            })
                                                                            
                                                                        });
                                                                    }
                                                                }
                                                            }); 
                                                            }
                                                           
                                                        });
                                                        ui.separator();
                                                    }

                                                    
                                                },
                                            );
                                            ()
                                        }
                                        // Kernel editing stuff
                                        4 => {
                                            ui.vertical(|ui|{
                                            if ui.button("Apply Kernel").clicked() {
                                                self.viewport.renderer.update_gpu();
                                                self.viewport.renderer.render(false, self.samples, false);
                                            }
                                            if ui.button("Save Kernel To File").clicked() {
                                                let mut path = String::from("Kernels/");
                                                if let Some(new_path) = rfd::FileDialog::new().pick_file() {
                                                    path = new_path.display().to_string();
                                                }
                                                fs::write(path, &self.viewport.renderer.kernel.0);
                                            }
                                            
                                            let l = 4.max(self.viewport.renderer.error.lines().count());
                                            ui.collapsing("Error", |ui| {
                                                ui.set_max_height((l * 8) as f32);
                                                ui.push_id(98123727, |ui| {

                                                    egui::ScrollArea::vertical().always_show_scroll(true).show(ui, |ui| {
                                                    let response = ui.add_sized((ui.available_width(), ui.available_height()), 
                                                    egui::TextEdit::multiline(&mut self.viewport.renderer.error).code_editor().lock_focus(true).desired_rows(l));
                                                    if response.changed() {
                                                        self.viewport.renderer.update_gpu();
    
                                                        self.viewport.renderer.render(false, self.samples, false);
                                                    }
                                                    });
    
                                                });
                                            });
                                            let h = ui.available_height() - (32.0 - 12.0);
                                            ui.push_id(981237897, |ui| {

                                                ui.set_max_height(h);

                                                egui::ScrollArea::both().auto_shrink([true,false]).max_height(h).always_show_scroll(true).show(ui, |ui| {
                                                    // ui.id
                                                    // let rect = ui.max_rect();
                                                    // ui.set_height(h * 0.9);
                                                    // ui.scroll_to_cursor(Some(Align::LEFT));
                                                    // Rich
                                                    // impl egui::util::cache::ComputerMut<(&CodeTheme, &str, &str), LayoutJob> for Highlighter<'_> {
                                                    //     fn compute(&mut self, (theme, code, lang): (&CodeTheme, &str, &str)) -> LayoutJob {
                                                    //         self.highlight(theme, code, lang)
                                                    //     }
                                                    // }

                                                    // ui.code(text)
                                                    // let memory = ctx.memory();
                                                    // type HighlightCache = egui::util::cache::FrameCache<LayoutJob, Highlighter<'_>>;
                                                    // let cache = memory.caches.cache();
                                                    let mut theme = ui.ctx().data()
                                                    .get_persisted(egui::Id::new("dark"))
                                                    .unwrap_or_else(CodeTheme::dark);
                                                    // ui.collapsing("Theme", |ui| {
                                                    //     ui.group(|ui| {
                                                    //         theme.ui(ui);
                                                    //         theme.clone().store_in_memory(ui.ctx());
                                                    //     });
                                                    // });
                                                    let mut layouter = |ui: &egui::Ui, string: &str, wrap_width: f32| {
                                                        let mut layout_job =
                                                            highlight(ui.ctx(), &theme, string, "c");
                                                        layout_job.wrap.max_width = wrap_width;
                                                        ui.fonts().layout_job(layout_job)
                                                    };
                                                    let response = ui.add_sized((ui.available_width(), ui.available_height() - 32.0), 
                                                    egui::TextEdit::multiline(&mut self.viewport.renderer.kernel.0)
                                                    .font(egui::TextStyle::Monospace)
                                                    .code_editor()
                                                    .lock_focus(true)
                                                    .desired_width(f32::INFINITY)
                                                    .layouter(&mut layouter)
                                                
                                                );
                                                    if response.changed() {
                                                        self.viewport.renderer.update_gpu();

                                                        self.viewport.renderer.render(false, self.samples, false);
                                                    }
                                                });
                                            });
                                            ui.add_space(3.0);
                                            ui.separator();
                                            // egui::CollapsingHeader
                                            
                                            
                                        });
                                        }
                                        // Plotting stuff
                                        5 =>  {

                                            
                                            

                                            let w_up = self.viewport.renderer.args.cam_up.normalize();
                                            let cam_f = self.viewport.renderer.args.lookat.normalize();
                                            let cam_r = w_up.cross(cam_f).normalize();
                                            
                                            let endsc1 = Values::from_values(vec![Value::new(cam_f.x, cam_f.y)]);
                                            let endsc2 = Values::from_values(vec![Value::new(cam_f.x, cam_f.z)]);
                                            let endsc3 = Values::from_values(vec![Value::new(cam_f.z, cam_f.y)]);

                                            let endsw1 = Values::from_values(vec![Value::new(w_up.x, w_up.y)]);
                                            let endsw2 = Values::from_values(vec![Value::new(w_up.x, w_up.z)]);
                                            let endsw3 = Values::from_values(vec![Value::new(w_up.z, w_up.y)]);

                                            let endsr1 = Values::from_values(vec![Value::new(cam_r.x, cam_r.y)]);
                                            let endsr2 = Values::from_values(vec![Value::new(cam_r.x, cam_r.z)]);
                                            let endsr3 = Values::from_values(vec![Value::new(cam_r.z, cam_r.y)]);
                                            
                                            let red = Color32::from_rgb(255, 0, 0);
                                            let green = Color32::from_rgb(0, 255, 0);
                                            let blue = Color32::from_rgb(0, 0, 255);
                                            let white = Color32::from_rgb(255, 255, 255);
                                            ui.vertical(|ui| {
                                                ui.add_space(8.);
                                                let asize = ui.available_size();
                                                let w = (asize.x - 32.) / 3.;
                                                let w2 = (asize.x - (32. + 64. + 8. + 64.)) / 6.;
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    egui::plot::Plot::new("plot_view_z").center_x_axis(true).center_y_axis(true).show_axes([true,true]).width(w).height(w).show(ui, |plot_ui|  {
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsc1).color(green));
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsr1).color(red));
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsw1).color(blue)); 
                                                    });
        
                                                    egui::plot::Plot::new("plot_view_y").center_x_axis(true).center_y_axis(true).show_axes([true,true]).width(w).height(w).show(ui, |plot_ui|  {
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsc2).color(green));
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsr2).color(red));
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsw2).color(blue)); 
                                                    });
        
                                                    egui::plot::Plot::new("plot_view_x").center_x_axis(true).center_y_axis(true).show_axes([true,true]).width(w).height(w).show(ui, |plot_ui|  {
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsc3).color(green));
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsr3).color(red));
                                                        plot_ui.arrows(egui::plot::Arrows::new(Values::from_values(vec![Value::new(0., 0.);1]), endsw3).color(blue)); 
                                                    });
                                                });

                                                let mut norm_up = false;
                                                let mut norm_f = false;
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    let mut edit_w_up = &mut self.viewport.renderer.args.cam_up;
                                                    let mut edit_cam_f = &mut self.viewport.renderer.args.lookat;
                                                    if ui.button("|cam_up|").clicked() {
                                                        pressed |= true;
                                                        norm_up = true;
                                                    }
                                                    ui.add_sized([w2, 16.], egui::DragValue::new(&mut edit_w_up.x).speed(self.input_menu.tweak_speed));
                                                    ui.add_sized([w2, 16.], egui::DragValue::new(&mut edit_w_up.y).speed(self.input_menu.tweak_speed));
                                                    ui.add_sized([w2, 16.], egui::DragValue::new(&mut edit_w_up.z).speed(self.input_menu.tweak_speed));
                                                    ui.add_space(8.);
                                                    if ui.button("|cam_f |").clicked() {
                                                        norm_f = true;
                                                        pressed |= true;
                                                    }
                                                    ui.add_sized([w2, 16.], egui::DragValue::new(&mut edit_cam_f.x).speed(self.input_menu.tweak_speed));
                                                    ui.add_sized([w2, 16.], egui::DragValue::new(&mut edit_cam_f.y).speed(self.input_menu.tweak_speed));
                                                    ui.add_sized([w2, 16.], egui::DragValue::new(&mut edit_cam_f.z).speed(self.input_menu.tweak_speed));
                                                });
                                                
                                                if norm_up {
                                                self.viewport.renderer.args.cam_up = self.viewport.renderer.args.cam_up.normalize();
                                                }
                                                if norm_f {
                                                self.viewport.renderer.args.lookat = self.viewport.renderer.args.lookat.normalize();
                                                }
                                                
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    ui.label("Tweak Speed: ");
                                                    let rate = ( (self.input_menu.tweak_speed) / 100.).clamp(0.00000000001, 0.5);
                                                    ui.add(
                                                        egui::DragValue::new(&mut self.input_menu.tweak_speed).speed(rate).clamp_range(0.00000000001..=1.).fixed_decimals(16)
                                                    );
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    ui.label("Camera Speed: ");
                                                    let rate = ( (self.input_menu.cam_speed) / 100.).clamp(0.00000000001, 0.5);
                                                    ui.add(
                                                        egui::DragValue::new(&mut self.input_menu.cam_speed).speed(rate).clamp_range(0.00000000001..=10.).fixed_decimals(16)
                                                    );
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    ui.label("Move Speed: ");
                                                    let rate = ( (self.input_menu.mov_speed) / 100.).clamp(0.00000000001, 0.5);
                                                    ui.add(
                                                        egui::DragValue::new(&mut self.input_menu.mov_speed).speed(rate).clamp_range(0.00000000001..=10.).fixed_decimals(16)
                                                    );
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    ui.label("Rotate Speed: ");
                                                    let rate = ( (self.input_menu.rot_speed) / 100.).clamp(0.00000000001, 0.5);
                                                    ui.add(
                                                        egui::DragValue::new(&mut self.input_menu.rot_speed).speed(rate).clamp_range(0.00000000001..=10.).fixed_decimals(16)
                                                    );
                                                });

                                            });
                                            
                                            
                                              
                                           
                                        }
                                        6 => {
                                            ui.vertical(|ui| {
                                                ui.add_space(8.);
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    ui.label("Tweak Speed: ");
                                                    let rate = ( (self.input_menu.tweak_speed) / 100.).clamp(0.00000000001, 0.5);
                                                    ui.add(
                                                        egui::DragValue::new(&mut self.input_menu.tweak_speed).speed(rate).clamp_range(0.00000000001..=1.).fixed_decimals(16)
                                                    );
                                                });
                                                let size = ui.available_size();
                                                let h = (size.x - (64. + 16.)) / 9.;
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    egui::Grid::new("input_grid").num_columns(9).spacing([8.,8.]).striped(true).show(ui, |ui| {
                                                        
                                                        ui.add_sized([h,16.],egui::Label::new("#"));
                                                        for rn in 0..8 {
                                                            ui.add_sized([h,16.],egui::Label::new(format!("{}", rn)));
                                                        }
                                                        ui.end_row();
                                                        for j in 0..8 {
                                                            

                                                            ui.add_sized([h,16.],egui::Label::new(format!("{}", j)));
                                                            for i in 0..8 {
                                                                
                                                                    let response = ui.add_sized([h,16.],
                                                                        egui::DragValue::new(
                                                                            &mut self.viewport.renderer.args.input_grid[j][i],
                                                                        ).speed(self.input_menu.tweak_speed)
                                                                        // .logarithmic(true)
                                                                        ,
                                                                    );
                                                                    pressed |= response.changed();
                                                                    
                                                                
                                                            }
                                                            ui.end_row();
                                                        }
                                                        
                                                    });
                                                });
                                                let leftovers = ui.available_size();
                                                let w = leftovers.x - 16.;
                                                let h = leftovers.y - 64.;
                                                ui.add_space(8.);
                                                ui.horizontal(|ui| {
                                                    ui.add_space(8.);
                                                    ui.add_sized([w,h],egui::TextEdit::multiline(&mut self.param_inp).code_editor());
                                                });
                                                ui.add_space(8.);
                                                let mut offset = 0;
                                                ui.horizontal(
                                                    |ui| {
                                                        ui.add_space(8.);
                                                        if ui.button("Accept").clicked() {
                                                            let hm = self.param_inp.split('|');
                                                            for (i, value) in hm.enumerate() {
                                                                let (x,y) = ((i-offset) % 8, (i-offset) / 8);
                                                                if value.is_empty() {
                                                                    offset+=1;
                                                                } else if let Ok(new_v) = value.parse::<f32>() {
                                                                    self.viewport.renderer.args.input_grid[y][x] = new_v;
                                                                }
                                                                 
                                                            }
                                                            self.param_inp = "".into();
                                                        }
                                                        if ui.button("Clear").clicked() {
                                                            self.param_inp = "".into();
                                                        }
                                                    }
                                                );
                                                ui.add_space(8.);
                                            });
                                            
                                        }
                                        _ => (),
                                    }
                                });
                        });
                    ui.spacing_mut().window_margin = eframe::egui::style::Margin {
                        left: 0.0,
                        right: 0.0,
                        top: 0.0,
                        bottom: 0.0,
                    };
                    let size = ui.available_size();
                    let mut rect = ui.max_rect();
                    let rect = Rect {
                        min: (0.0, title_bar_height).into(),
                        max: (size.x - 4.0, size.y + title_bar_height).into(),
                    };
                    // rect.set_width(rect.width() - 128.0);
                    let painter = ui.painter();
                    // Paint the frame:
                    painter.rect(rect, 0.0, Color32::from_rgb(16, 16, 16), Stroke::none());

                    // ui.ctx().options().
                    // Get available size
                    let mut area = ui.available_size();
                    area.x -= 4.0;
                    // Get our scalars
                    let scale_x = area.y / self.viewport.renderer.args.size.1 as f32;
                    let scale_y = area.x / self.viewport.renderer.args.size.0 as f32;
                    // Get our scaled viewport dimensions
                    let scaled_x = (self.viewport.renderer.args.size.0 as f32 * scale_x);
                    let scaled_y = (self.viewport.renderer.args.size.1 as f32 * scale_y);
                    // Compute if it will fit within the x or y axis dependant on which side to scale up to
                    let diff_x = area.x - scaled_x;
                    let diff_y = area.y - scaled_y;
                    // Use the above information to figure out some stuff
                    let (viewport_pos, vp_new, scale) = if diff_x > diff_y {
                        (
                            (diff_x / 2.0, 0.0),
                            (scaled_x, self.viewport.renderer.args.size.1 as f32 * scale_x),
                            scale_x,
                        )
                    } else {
                        (
                            (0.0, diff_y / 2.0),
                            (self.viewport.renderer.args.size.0 as f32 * scale_y, scaled_y),
                            scale_y,
                        )
                    };
                    let offset = (viewport_pos.0, viewport_pos.1 + title_bar_height);
                    egui::Area::new("viewport")
                        .fixed_pos(offset)
                        .order(Order::Foreground)
                        .movable(false)
                        .show(ctx, |ui| {
                            ui.set_width(vp_new.0);
                            ui.set_height(vp_new.1);
                            let painter = ui.painter();
                            let rect = ui.max_rect();
                            painter.rect(rect, 0.0, Color32::from_rgb(0, 0, 0), Stroke::none());

                            

                            self.viewport.texture = None;
                            let (wa,ha) = (self.viewport.renderer.buffer.width(),self.viewport.renderer.buffer.height());
                            
                            let mut why = Vec::new();
                            let holy_ = self.viewport.renderer.buffer.to_vec().into_par_iter().map(|x|  (x * (u8::MAX as f32)) as u8).collect_into_vec(&mut why);
                            let img = epaint::ImageData::from(epaint::ColorImage::from_rgba_unmultiplied([wa as usize,ha as usize], why.as_bytes()));
                            
                            // self.viewport.texture.as_mut().unwrap().set(img);
                            let texture: &egui::TextureHandle = &egui::Context::load_texture(ctx, "wha", img);

                            // self.viewport.renderer.render(false, 1);
                            // let (w, h) = vp_new;
                            // texture.
                            let (vw, vh) = self.viewport.renderer.args.size;
                            let image = ui.image(texture, vp_new);
                            let image_ctx = &image.ctx;
                            
                            

                            let v = if let Some(pointer) = image.hover_pos() {
                                Some(pointer)
                            } else if let Some(pointer) = mouse_input.hover_pos() {
                                Some(pointer)
                            } else {
                                None
                            };
                            if let Some(pointer) = v { 
                                if !self.over_options {
                                    
                                
                                self.viewport_interact = true;
                                let icoord = (
                                    (pointer.x - (offset.0 + 1.0)) / scale,
                                    (pointer.y - (offset.1 + 1.0)) / scale,
                                );
                                let (vw, vh) = self.viewport.renderer.args.size;
                                let ratio = vh as f32 / vw as f32;
                                let ccoord = (icoord.0 / vw as f32, (icoord.1 / vh as f32) * ratio);

                                

                                if mouse_input.primary_down() {
                                    ///mouse
                                    self.viewport.renderer.clear();
                                    let mut m: egui::Vec2 = egui::Vec2::new(0.0, 0.0);
                                    if let Some(o) = mouse_input.press_origin() {
                                        let ipxpos_o = Pos2::new(
                                            (o.x - (offset.0 + 1.0)) / scale,
                                            (o.y - (offset.1 + 1.0)) / scale,
                                        );
                                        m = Pos2::new(icoord.0, icoord.1) - ipxpos_o;
                                        m = egui::Vec2::new(m.x, m.y);
                                    }
                                    m.x = m.x / self.viewport.renderer.args.size.0 as f32;
                                    m.y = m.y / self.viewport.renderer.args.size.1 as f32;

                                    //self.viewport.renderer.args.origin = self.viewport.last.origin
                                    //    - (Complex64::new(m.x as f64, (m.y as f64 * ratio as f64))
                                    //        / self.viewport.renderer.args.zoom);
                                    pressed = true;
                                }
                                if mouse_input.secondary_down() {
                                    if self.temp_julia.is_none() {
                                        let mx = (icoord.0) as f64
                                            / self.viewport.renderer.args.size.0 as f64
                                            - 0.5;
                                        let my = (icoord.1) as f64
                                            / self.viewport.renderer.args.size.1 as f64
                                            - 0.5;
                                        //self.viewport.renderer.args.julia.0 =
                                        //    ((Complex64::new(mx as f64, my as f64 * ratio as f64)
                                        //        / self.viewport.renderer.args.zoom)
                                        //        + self.viewport.renderer.args.origin);
                                        pressed = true;
                                    }
                                    if mouse_input.middle_down() {
                                        self.viewport.renderer.args.julia.1 =
                                            !self.viewport.renderer.args.julia.1;
                                        pressed = true;
                                    }
                                }
                                if mouse_input.any_released() {
                                    //self.viewport.last.origin = self.viewport.renderer.args.origin;
                                }
                                if self.viewport.active && !self.viewport.renderer.rendering {
                                    self.viewport.renderer.render(false, self.samples, true);
                                }

                                image_ctx.request_repaint();

                                if self.viewport.active && self.events.enter.is_some() {
                                    println!("Rendering Now!");
                                    self.renderer.clear();
                                    self.renderer.copy_args(&self.viewport.renderer);
                                    self.renderer.kernel.0 = self.viewport.renderer.kernel.0.clone();
                                    self.renderer.update_gpu();
                                    self.renderer.export_image_with_info(
                                        false,
                                        true,
                                        if let Ok(r) = self.input_menu.samples.parse() {
                                            Some(r)
                                        } else {
                                            Some(self.viewport.renderer.rendered_samples)
                                        },
                                        // Some(64),
                                        None,
                                        None
                                    );
                                    self.renderer.clear();
                                    println!();
                                }
                            }
                            } else {
                                self.viewport_interact = false;
                            }
                        });
                });
        });
        if pressed {
            self.viewport.renderer.clear();
            self.viewport.renderer.render(false, self.samples, true);
        }
    }
}

fn custom_window_frame(
    height: f32,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
    title: &str,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    ctx.set_visuals(egui::style::Visuals::dark());
    use egui::*;
    let text_color = ctx.style().visuals.text_color();

    CentralPanel::default()
        .frame(Frame::none())
        .show(ctx, |ui| {
            let rect = ui.max_rect();
            let painter = ui.painter();

            // Paint the frame:
            painter.rect(
                rect,
                0.0,
                ctx.style().visuals.window_fill(),
                Stroke::new(0.0, text_color),
            );

            // Paint the title:
            painter.text(
                rect.center_top() + vec2(0.0, height / 2.0),
                Align2::CENTER_CENTER,
                title,
                FontId::proportional(height - 4.0),
                text_color,
            );

            // Add the close button:
            let close_response = ui.put(
                Rect::from_min_size(rect.left_top(), Vec2::splat(height)),
                Button::new(RichText::new("???").size(height - 4.0)).frame(false),
            );
            if close_response.clicked() {
                frame.quit();
            }

            // Interact with the title bar (drag to move window):
            let title_bar_rect = {
                let mut rect = rect;
                rect.max.y = rect.min.y + height;
                rect
            };
            let title_bar_response =
                ui.interact(title_bar_rect, Id::new("title_bar"), Sense::drag());
            if title_bar_response.drag_started() {
                frame.drag_window();
            }

            // Add the contents:
            let content_rect = {
                let mut rect = rect;
                rect.min.y = title_bar_rect.max.y;
                rect
            };
            // ui.la

            let mut content_ui = ui.child_ui(content_rect, *ui.layout());
            add_contents(&mut content_ui);
        });
}

fn main() {
    let options = eframe::NativeOptions {
        // Hide the OS-specific "chrome" around the window:
        decorated: false,
        // To have rounded corners we need transparency:
        transparent: false,
        min_window_size: Some(egui::vec2(320.0, 100.0)),
        ..Default::default()
    };
    eframe::run_native(
        "FracgenGUI", // unused title
        options,
        Box::new(|_cc| Box::new(FracgenGUI::default())),
    );
}

