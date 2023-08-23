use image::DynamicImage;
use libc;
use pyo3::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    os::unix::prelude::OpenOptionsExt,
};

#[pyclass]
pub struct Gamepad {
    writer: BufWriter<File>,
    cmd_buffer: String,
}

#[pymethods]
impl Gamepad {
    #[new]
    pub fn new(pipe_path: &str) -> Self {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(pipe_path)
            .unwrap();
        let writer = BufWriter::new(f);
        Self {
            writer,
            cmd_buffer: String::new(),
        }
    }

    /// Sets the state of a button.
    pub fn set_btn_state(&mut self, btn: Button, pressed: bool) {
        let cmd = if pressed { "PRESS" } else { "RELEASE" };
        self.cmd_buffer
            .push_str(&format!("{} {}\n", cmd, btn.as_string()));
    }

    /// Sets the value of a stick.
    pub fn set_stick_value(&mut self, stick: Stick, x: f32, y: f32) {
        self.cmd_buffer
            .push_str(&format!("SET {} {} {}\n", stick.as_string(), x, y));
    }

    /// Takes a screenshot.
    pub fn take_screenshot(&mut self) {
        self.cmd_buffer.push_str("PRESS D_UP\nRELEASE D_UP\n");
    }

    /// Reloads the game state.
    pub fn reload_state(&mut self) {
        self.cmd_buffer.push_str("PRESS D_DOWN\nRELEASE D_DOWN\n");
    }

    /// Toggles pause.
    pub fn toggle_pause(&mut self) {
        self.cmd_buffer.push_str("PRESS D_LEFT\nRELEASE D_LEFT\n");
    }

    /// Flushes the command buffer.
    pub fn flush(&mut self) {
        self.writer.write_all(self.cmd_buffer.as_bytes()).unwrap();
        self.cmd_buffer.clear();
    }
}

#[pyclass]
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub enum Button {
    A,
    B,
    X,
    Z,
    L,
}

impl Button {
    pub fn as_string(&self) -> String {
        match self {
            Button::A => "A",
            Button::B => "B",
            Button::X => "X",
            Button::Z => "Z",
            Button::L => "L",
        }
        .to_string()
    }
}

#[pyclass]
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub enum Stick {
    Main,
    C,
}

impl Stick {
    pub fn as_string(&self) -> String {
        match self {
            Stick::Main => "MAIN",
            Stick::C => "C",
        }
        .to_string()
    }
}

/// Interface to a single Dolphin instance.
pub struct Console {
    gamepad: Gamepad,
    screenshot_path: String,
}

impl Console {
    pub fn new(dolphin_path: &str) -> Self {
        let pipe_path = format!("{dolphin_path}/Pipes/pipe1");
        let screenshot_path = format!("{dolphin_path}/ScreenShots/GALE01");
        Self {
            gamepad: Gamepad::new(&pipe_path),
            screenshot_path,
        }
    }

    /// Returns the latest screenshot.
    pub fn get_screen(&self) -> DynamicImage {
        image::io::Reader::open(format!("{}/screenshot-1.png", self.screenshot_path))
            .unwrap()
            .decode()
            .unwrap()
    }

    /// Sets the current input.
    pub fn set_input(&mut self, btn_map: HashMap<Button, bool>, stick_map: HashMap<Stick, (f32, f32)>) {
        for btn in btn_map.keys() {
            self.gamepad.set_btn_state(*btn, *btn_map.get(btn).unwrap());
        }

        for stick in stick_map.keys() {
            let val = *stick_map.get(stick).unwrap();
            self.gamepad.set_stick_value(*stick, val.0, val.1);
        }
    }
}
