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
#![feature(is_some_with)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// use gui_testing::MyApp;

use std::{fs::File, path::PathBuf};

use colortypes::{CIELab, Color, FromColorType, Image, Rgb, Srgb, Xyz, D65};
use eframe::egui::{
    self, color_picker, Checkbox, Event, InnerResponse, Key, Modifiers, PointerButton, Response,
    TextEdit,
};
use epaint::{color::Hsva, Color32, Pos2, TextureHandle, Vec2};
use num::complex::Complex64;
use FracgenGui3D::{rendering::*, FractalArgs, Renderer};

pub struct InputMenu {
    pub viewport_size: (String, String, bool),
    pub export_size: (String, String, bool),
    pub samples: String,
    pub width: f32,
}

pub struct LastData {
    pub origin: Complex64,
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
    fn resize(&mut self, new_size: (f32, f32)) {
        self.image_size = (new_size.0.round() as usize, new_size.1.round() as usize);
        self.image = Image::new(self.image_size);
        self.renderer.size = self.image_size;
        self.renderer.raw = Image::new(self.image_size);
        self.renderer.rendered_samples = 0;
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
    pub viewport: Viewport,
    pub input_menu: InputMenu,
    pub pointer: Option<Pos2>,
    pub node_menu_height: f32,
    pub renderer: Renderer,
    pub events: Events,
    pub temp_julia: Option<Complex64>,
}
impl Default for FracgenGUI {
    #[inline]
    fn default() -> FracgenGUI {
        // let args = FractalArgs::default();
        FracgenGUI {
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
                (1920, 823),
                FractalArgs::default(),
                (num_cpus::get() as f64).ceil() as usize - 2,
                true,
                false,
            ),
            viewport: Viewport {
                active: false,
                image_size: (512, 220),
                image: Image::new((512, 220)),
                texture: None,
                image_space: 0.0,
                img_empty_space: (0.0, 0.0),
                last: LastData {
                    origin: Complex64::new(-0.75, 0.0),
                    args: FractalArgs::default(),
                },
                renderer: Renderer::new(
                    (512, 220),
                    FractalArgs::default(),
                    (num_cpus::get() as f64).ceil() as usize - 2,
                    false,
                    true,
                ),
                needs_resize: None,
            },
            input_menu: InputMenu {
                viewport_size: ((512).to_string(), (220).to_string(), true),
                export_size: (1920.to_string(), 823.to_string(), true),
                samples: 64.to_string(),
                width: 128.0,
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
    f: Vec<fn(&Renderer, RenderData) -> Color<CIELab, D65>>,
) {
    match key {
        Key::Num1 => {
            *choice = 1 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
        }
        Key::Num2 => {
            *choice = 2 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
        }
        Key::Num3 => {
            *choice = 3 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
        }
        Key::Num4 => {
            *choice = 4 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
        }
        Key::Num5 => {
            *choice = 5 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
        }
        Key::Num6 => {
            *choice = 6 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
        }
        Key::Num7 => {
            *choice = 7 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
        }
        Key::Num8 => {
            *choice = 8 + (8 * *page);
            *change = f[(*choice - 1) % f.len()];
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
                            Some("png") | Some("jpg") | Some("jpeg") => {
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

        if !self.viewport.renderer.rendering {
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
                        } else {
                            self.viewport.renderer.render(false, 8)
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
                                // Key::Num4 => {
                                //     self.viewport.renderer.args.iter_funct_choice = Some(4);
                                //     self.viewport.renderer.args.iter_funct = fractal_mandelbrot;
                                // }
                                // Key::Num5 => {
                                //     self.viewport.renderer.args.iter_funct_choice = Some(5);
                                //     self.viewport.renderer.args.iter_funct = fractal_mandelbrot;
                                // }
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
                            },
                            // This is for coloring functions
                            Modifiers {
                                alt: true,
                                ctrl: false,
                                shift: false,
                                mac_cmd: false,
                                command: false,
                            } => process_color_key(
                                key,
                                &mut self.viewport.renderer.args.color_funct_choice,
                                &mut self.viewport.renderer.args.color_funct_page,
                                &mut self.viewport.renderer.args.color_funct,
                                vec![stripe,image_mapping,fade_original,paper_cut,testingf,trap_avg,normal_map,foral]
                            )
                            ,

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
                                Key::J => {
                                    std::mem::swap(
                                        &mut self.temp_julia,
                                        &mut self.viewport.renderer.args.julia,
                                    );
                                }
                                Key::Enter => {
                                    if self.events.enter.is_none() {
                                        println!("OK COOL???? {}", Some(p).is_some());

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
                                        // println!("\n\n LIMIT: {} \n\n", self.)
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
                self.viewport.renderer.render(false, 1);
                self.viewport.needs_resize = None;
            }
        }
        let ratio = self.viewport.renderer.size.0 as f64 / self.viewport.renderer.size.1 as f64;
        custom_window_frame(ctx, frame, "FracgenGUI", |ui| {
            // ui.with_layout(ctx, add_contents)
            // egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                let rect = ctx.available_rect();
                ui.set_height(rect.height() - self.node_menu_height);
                ui.horizontal_centered(|ui| {
                    let mut top_height = 0.0;
                    ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
                        let rect = ui.max_rect();
                        let painter = ui.painter();
                        painter.rect_filled(rect, 0.0, Color32::from_rgb(20, 20, 20));
                        // ui.set_width_range(128, 256.0)
                        // let spacing =;
                        // ui.set_width(self.input_menu.width);
                        // ui.shrink_width_to_current();
                        // top_height = ui.available_height();
                        // ui.style_mut().spacing.item_spacing.x = 0.0;

                        // let (mut cr, mut wr, mut hr) = (Response);
                        egui::Window::new("Options (WIP)").show(ctx, |ui| {
                            ui.horizontal_top(|ui| {
                                // ui.set_width(ui.available_width());

                                // ui.add_sized(
                                //     Vec2 {
                                //         x: ui.available_width(),
                                //         y: 16.0,
                                //     },

                                // ui.indent(id_source, add_contents)
                                ui.vertical(|ui| {
                                    egui::Grid::new("Input_grid")
                                        .num_columns(10)
                                        .show(ui, |ui| {
                                            ui.set_height(16.0);
                                            ui.set_width(ui.available_width());

                                            ui.label("Viewport: ");
                                            ui.label("Size: ");
                                            let width_response = ui.add(
                                                TextEdit::singleline(
                                                    &mut self.input_menu.viewport_size.0,
                                                )
                                                .hint_text("Viewport Width"),
                                            );
                                            let height_response = ui.add(
                                                TextEdit::singleline(
                                                    &mut self.input_menu.viewport_size.1,
                                                )
                                                .hint_text("Viewport Height"),
                                            );
                                            ui.add(Checkbox::new(
                                                &mut self.input_menu.viewport_size.2,
                                                "Keep Ratio?",
                                            ));
                                            ui.label("Samples: ");
                                            ui.add(
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

                                            if width_response.lost_focus()
                                                && ui.input().key_pressed(egui::Key::Enter)
                                            {
                                                if let Ok(new_w) =
                                                    self.input_menu.viewport_size.0.parse::<f32>()
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
                                                            (new_h.round() as usize).to_string();
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
                                                            self.viewport.image_size.0.to_string()
                                                        };
                                                }
                                            }

                                            if height_response.lost_focus()
                                                && ui.input().key_pressed(egui::Key::Enter)
                                            {
                                                if let Ok(new_h) =
                                                    self.input_menu.viewport_size.1.parse::<f32>()
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
                                                            (new_w.round() as usize).to_string();
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
                                                            self.viewport.image_size.1.to_string()
                                                        };
                                                }
                                            }
                                            ui.end_row();
                                            ui.label("Export: ");
                                            ui.label("Size: ");
                                            let export_width_response = ui.add(
                                                TextEdit::singleline(
                                                    &mut self.input_menu.export_size.0,
                                                )
                                                .hint_text("Export Width"),
                                            );
                                            let export_height_response = ui.add(
                                                TextEdit::singleline(
                                                    &mut self.input_menu.export_size.1,
                                                )
                                                .hint_text("Export Height"),
                                            );
                                            ui.add(Checkbox::new(
                                                &mut self.input_menu.export_size.2,
                                                "Keep Ratio?",
                                            ));
                                            ui.label("Samples: ");
                                            let samples_response = ui.add(
                                                TextEdit::singleline(&mut self.input_menu.samples)
                                                    .hint_text("Export Samples"),
                                            );
                                            ui.end_row();

                                            if export_width_response.lost_focus()
                                                && ui.input().key_pressed(egui::Key::Enter)
                                            {
                                                if let Ok(new_w) =
                                                    self.input_menu.export_size.0.parse::<f32>()
                                                {
                                                    let mut new_h = self.renderer.size.1 as f32;
                                                    if self.input_menu.export_size.2 {
                                                        let ratio =
                                                            new_w / self.renderer.size.0 as f32;
                                                        new_h *= ratio;
                                                        println!("NEW : {}", new_h);
                                                        self.input_menu.export_size.1 =
                                                            (new_h.round() as usize).to_string();
                                                    }
                                                    self.renderer.size = (
                                                        new_w.round() as usize,
                                                        new_h.round() as usize,
                                                    );
                                                    self.renderer.raw =
                                                        Image::new(self.renderer.size);
                                                    self.input_menu.export_size.0 =
                                                        (new_w.round() as usize).to_string();
                                                } else {
                                                    self.input_menu.export_size.0 =
                                                        self.renderer.size.0.to_string();
                                                }
                                            }

                                            if export_height_response.lost_focus()
                                                && ui.input().key_pressed(egui::Key::Enter)
                                            {
                                                if let Ok(new_h) =
                                                    self.input_menu.export_size.1.parse::<f32>()
                                                {
                                                    let mut new_w = self.renderer.size.0 as f32;
                                                    if self.input_menu.export_size.2 {
                                                        let ratio =
                                                            new_h / self.renderer.size.1 as f32;
                                                        new_w *= ratio;
                                                        println!("NEW : {}", new_w);
                                                        self.input_menu.export_size.0 =
                                                            (new_w.round() as usize).to_string();
                                                    }
                                                    self.renderer.size = (
                                                        new_w.round() as usize,
                                                        new_h.round() as usize,
                                                    );
                                                    self.renderer.raw =
                                                        Image::new(self.renderer.size);
                                                    self.input_menu.export_size.1 =
                                                        (new_h.round() as usize).to_string();
                                                } else {
                                                    self.input_menu.export_size.1 =
                                                        self.renderer.size.1.to_string();
                                                }
                                            }

                                            // ui.radio_value(
                                            //     &mut self.viewport.renderer.args.hue_offset,
                                            //     1.0,
                                            //     "?",
                                            // )
                                            // ui.sli
                                        })
                                });

                                // let half = Vec2 {
                                //     x: (ui.available_width() / 3.0) - 8.0,
                                //     y: 16.0,
                                // };
                                // (
                                //     ui.add_sized(
                                //         Vec2 { x: 16.0, y: 16.0 },
                                //         ,
                                //     ),
                                //     ui.add_sized(
                                //         half,
                                //         ,
                                //     ),
                                //     ui.add_sized(
                                //         half,
                                //         ,
                                //     ),
                                //     ui.add_sized(
                                //         half,
                                //         TextEdit::singleline(&mut self.input_menu.size_input.1)
                                //             .hint_text("Viewport Height"),
                                //     ),
                                // )
                            });
                            ui.separator();
                            egui::Grid::new("ass2").show(ui, |ui| {
                                ui.label("Iter Method:");
                                ui.add(
                                    TextEdit::singleline(
                                        &mut self
                                            .viewport
                                            .renderer
                                            .args
                                            .iter_funct_choice
                                            .to_string(),
                                    )
                                    .hint_text("Iter Method")
                                    .interactive(false),
                                );
                                ui.label("Bail Method:");
                                ui.add(
                                    TextEdit::singleline(
                                        &mut self
                                            .viewport
                                            .renderer
                                            .args
                                            .conditional_choice
                                            .to_string(),
                                    )
                                    .hint_text("Cond Method")
                                    .interactive(false),
                                );
                                ui.label("Coloring Method:");
                                ui.add(
                                    TextEdit::singleline(
                                        &mut self
                                            .viewport
                                            .renderer
                                            .args
                                            .color_funct_choice
                                            .to_string(),
                                    )
                                    .hint_text("Color Method")
                                    .interactive(false),
                                );
                            });
                            // let idfk =
                            // egui::Grid::new("lashdajshdkljashd").show(ui, |ui| {
                            //     ui.label("Origin:");
                            //     ui.label("Julia:");
                            //     ui.end_row();
                            // });
                            let huh = ui.available_width() - 32.0;
                            egui::Grid::new("Positioning").show(ui, |ui| {
                                let alt = Complex64::new(f64::NAN, f64::NAN);
                                ui.add_sized(
                                    (huh / 6.0 - 1.0, 8.0),
                                    egui::Label::new("Origin: ").wrap(true),
                                );
                                // ui.label("Origin: ");
                                ui.add_sized(
                                    (huh / 6.0, 0.0),
                                    TextEdit::singleline(
                                        &mut self.viewport.renderer.origin.re.to_string(),
                                    )
                                    .hint_text("Origin Re")
                                    .interactive(false),
                                );
                                ui.add_sized(
                                    (huh / 6.0, 0.0),
                                    TextEdit::singleline(
                                        &mut self.viewport.renderer.origin.im.to_string(),
                                    )
                                    .hint_text("Origin Im")
                                    .interactive(false),
                                );
                                ui.add_sized(
                                    (huh / 6.0 - 8.0, 16.0),
                                    egui::Label::new("Julia: ").wrap(true),
                                );
                                // ui.label("Julia: ");
                                ui.add_sized(
                                    (huh / 6.0, 0.0),
                                    TextEdit::singleline(
                                        &mut self
                                            .viewport
                                            .renderer
                                            .args
                                            .julia
                                            .unwrap_or(alt)
                                            .re
                                            .to_string(),
                                    )
                                    .hint_text("Julia Re")
                                    .interactive(false),
                                );
                                ui.add_sized(
                                    (huh / 6.0, 0.0),
                                    TextEdit::singleline(
                                        &mut self
                                            .viewport
                                            .renderer
                                            .args
                                            .julia
                                            .unwrap_or(alt)
                                            .im
                                            .to_string(),
                                    )
                                    .hint_text("Julia Im")
                                    .interactive(false),
                                );
                                ui.end_row();
                            });
                            ui.separator();
                            ui.horizontal_top(|ui| {
                                egui::Grid::new("ass").show(ui, |ui| {
                                    ui.vertical(|ui| {
                                        let color =
                                            Rgb::from_color(self.viewport.renderer.args.set_color)
                                                .to_arr()
                                                .0;
                                        // let mut color = epaint::Rgba::new() {
                                        //     0: [
                                        //         color[0] as f32,
                                        //         color[1] as f32,
                                        //         color[2] as f32,
                                        //         color[3] as f32,
                                        //     ],
                                        // };

                                        let mut color = Hsva::from_rgba_unmultiplied(
                                            color[0] as f32,
                                            color[1] as f32,
                                            color[2] as f32,
                                            color[3] as f32,
                                        );
                                        // let response = ui.color_edit_button_rgba_unmultiplied(&mut color);
                                        // color_picker::show_color(ui, color, desired_size)
                                        let InnerResponse::<()> { inner, response } =
                                            ui.vertical(|ui| {
                                                color_picker::color_picker_hsva_2d(
                                                    ui,
                                                    &mut color,
                                                    color_picker::Alpha::OnlyBlend,
                                                );
                                            });

                                        let color = color.to_rgba_unmultiplied();
                                        let new_arr = [
                                            color[0] as f64,
                                            color[1] as f64,
                                            color[2] as f64,
                                            color[3] as f64,
                                        ];
                                        self.viewport.renderer.args.set_color =
                                            Srgb::from_color(Rgb::new(new_arr));
                                    });

                                    ui.vertical(|ui| {
                                        let result = ui.add(
                                            egui::Slider::new(
                                                &mut self.viewport.renderer.args.hue_offset_pre_exp,
                                                0.0..=360.0,
                                            )
                                            .logarithmic(true)
                                            .text("Hue Shift PreExp"),
                                        );
                                        if result.changed() {
                                            pressed = true;
                                        }

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
                            });
                            ui.separator();
                        })
                    });
                    ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
                        // Load our rendered image into viewport
                        self.viewport.texture = None;
                        let texture: &egui::TextureHandle =
                            self.viewport.texture.get_or_insert_with(|| {
                                ctx.load_texture(
                                    "my-image",
                                    image_to_colorimage(&self.viewport.renderer.process_image()),
                                )
                            });
                        ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
                            let (vw, vh) = (ui.available_width(), ui.available_height());
                            let (rw, rh) =
                                (self.viewport.image.width(), self.viewport.image.height());
                            let l_space = ui.spacing().scroll_bar_width + 2.0;
                            let t_space = 30.0;
                            let (w, h);
                            // Compute new bounds for our viewport
                            if vh > vw {
                                w = ui.available_width();
                                h = rh as f32 * (w as f32 / rw as f32);
                            } else {
                                h = ui.available_height();
                                w = rw as f32 * (h as f32 / rh as f32);
                            }
                            self.viewport.img_empty_space =
                                (ui.available_width() - w, ui.available_height() - h);
                            // self.input_menu.width += self.viewport.img_empty_space.0;
                            top_height += self.viewport.img_empty_space.1;
                            // if h != top_height {
                            //     self.input_menu.width -= ui.available_width()
                            //         * ((ui.available_height() - h) / ui.available_height())
                            // }

                            // Apply the viewport
                            let ictx = ui.image(texture, egui::Vec2::new(w as f32, h as f32)).ctx;
                            // Figure out where our pointer is on the viewport

                            self.pointer = if !color_interact {
                                ictx.pointer_hover_pos()
                            } else {
                                None
                            };
                            self.renderer.origin = self.viewport.last.origin;

                            if let Some(pointer_pos) = self.pointer {
                                let scalar = self.viewport.image_size.0 as f32 / w as f32;
                                let ipxpos = (
                                    (pointer_pos.x - l_space) * scalar,
                                    (pointer_pos.y - t_space) * scalar,
                                );
                                let old_origin = self.viewport.renderer.origin;
                                // let old_origin = self.viewport.renderer.origin;
                                if (0..self.viewport.image_size.0 as i32)
                                    .contains(&(ipxpos.0.floor() as i32))
                                    && (0..self.viewport.image_size.1 as i32)
                                        .contains(&(ipxpos.1.ceil() as i32))
                                    && self.viewport.active
                                    && !self.viewport.renderer.rendering
                                {
                                    if let Some(v) = self.events.scroll {
                                        let mut q = v.y;
                                        if axis_bool {
                                            q = v.x;
                                        }
                                        if q.signum() < 0.0 {
                                            self.viewport.renderer.zoom /= 1.1 * scalar_mult;
                                        } else {
                                            self.viewport.renderer.zoom *= 1.1 * scalar_mult;
                                        }
                                    }
                                    if self.viewport.active && self.events.enter.is_some() {
                                        println!("Rendering Now!");

                                        self.renderer.copy_args(&self.viewport.renderer);
                                        self.renderer.export_image_with_info(
                                            false,
                                            false,
                                            Some(self.viewport.renderer.rendered_samples),
                                            None,
                                        );
                                        self.renderer.clear();
                                        println!();
                                    }
                                    if mouse_input.primary_down() {
                                        ///mouse
                                        self.viewport.renderer.clear();
                                        let mut m: egui::Vec2 = egui::Vec2::new(0.0, 0.0);
                                        if let Some(o) = mouse_input.press_origin() {
                                            let ipxpos_o = Pos2::new(
                                                (o.x - l_space) * scalar,
                                                (o.y - t_space) * scalar,
                                            );
                                            m = Pos2::new(ipxpos.0, ipxpos.1) - ipxpos_o;
                                            m = egui::Vec2::new(m.x, m.y);
                                        }
                                        m.x = (m.x + 0.5) / self.viewport.renderer.size.0 as f32;
                                        m.y = (m.y + 0.5) / self.viewport.renderer.size.1 as f32;

                                        self.viewport.renderer.origin = self.viewport.last.origin
                                            - (Complex64::new(m.x as f64, (m.y as f64 / ratio))
                                                / self.viewport.renderer.zoom);
                                        pressed = true;
                                    }
                                    if mouse_input.secondary_down() {
                                        if self.temp_julia.is_none() {
                                            let mx = (ipxpos.0 + 0.5) as f64
                                                / self.viewport.renderer.size.0 as f64
                                                - 0.5;
                                            let my = (ipxpos.1 + 0.5) as f64
                                                / self.viewport.renderer.size.1 as f64
                                                - 0.5;
                                            self.viewport.renderer.args.julia = Some(
                                                (Complex64::new(mx as f64, my as f64 / ratio)
                                                    / self.viewport.renderer.zoom)
                                                    + self.viewport.renderer.origin,
                                            );
                                            pressed = true;
                                        }
                                        if mouse_input.middle_down()
                                            && self.temp_julia != self.viewport.renderer.args.julia
                                            && self.viewport.renderer.args.julia.is_some()
                                        {
                                            std::mem::swap(
                                                &mut self.temp_julia,
                                                &mut self.viewport.renderer.args.julia,
                                            );
                                            pressed = true;
                                        }
                                        if !self.events.m_middle.unwrap_or(true)
                                            && self.temp_julia != self.viewport.renderer.args.julia
                                            && self.temp_julia.is_some()
                                        {
                                            std::mem::swap(
                                                &mut self.temp_julia,
                                                &mut self.viewport.renderer.args.julia,
                                            );
                                            pressed = true;
                                        }
                                    }
                                    if mouse_input.any_released() {
                                        self.viewport.last.origin = self.viewport.renderer.origin;
                                    }
                                    if self.viewport.active {
                                        self.viewport.renderer.render(false, 1);
                                        ictx.request_repaint();
                                    }
                                }
                            }
                        });
                        if self.viewport.active && pressed {
                            self.viewport.renderer.clear();

                            self.viewport.renderer.render(false, 1);
                        }
                    });
                });
            });
            ui.vertical_centered(|ui| {
                let rect = ui.max_rect();
                let painter = ui.painter();
                // painter.rect_filled(rect, 0.0, Color32::from_rgb(16, 16, 16));
                // ui.button("ok");
                // ui.text_edit_multiline(&mut self.text);
                // let response = ui.add_sized(
                //     ui.available_size(),
                //     egui::TextEdit::multiline(TextEdit::multiline(text)),
                // );
                // ui.text_edit_multiline(&mut self.text).
                // ui.text_edit_multiline(&mut self.text)
                // ui.add(
                //     egui::TextEdit::multiline(&mut self.text)
                //         .font(egui::TextStyle::Monospace) // for cursor height
                //         .code_editor()
                //         .desired_rows(10)
                //         .lock_focus(true)
                //         .desired_width(f32::INFINITY),
                // );
            });
        });
    }
}

fn custom_window_frame(
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
    title: &str,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    ctx.set_visuals(egui::style::Visuals::dark());
    use egui::*;
    let text_color = ctx.style().visuals.text_color();

    // Height of the title bar
    let height = 28.0;

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
                Button::new(RichText::new("❌").size(height - 4.0)).frame(false),
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
            let mut content_ui = ui.child_ui(content_rect, *ui.layout());
            add_contents(&mut content_ui);
        });
}

fn main() {
    let options = eframe::NativeOptions {
        // Hide the OS-specific "chrome" around the window:
        decorated: false,
        // To have rounded corners we need transparency:
        transparent: true,
        min_window_size: Some(egui::vec2(320.0, 100.0)),
        ..Default::default()
    };
    eframe::run_native(
        "FracgenGUI", // unused title
        options,
        Box::new(|_cc| Box::new(FracgenGUI::default())),
    );
}
