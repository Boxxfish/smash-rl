use pyo3::prelude::*;
use std::{
    fs::File,
    io::{BufWriter, Write}, os::unix::prelude::OpenOptionsExt,
};
use libc;

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

    /// Flushes the command buffer.
    pub fn flush(&mut self) {
        self.writer.write_all(self.cmd_buffer.as_bytes()).unwrap();
        self.cmd_buffer.clear();
    }
}

#[pyclass]
#[derive(Copy, Clone)]
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
#[derive(Copy, Clone)]
pub enum Stick {
    Main,
    Control,
}

impl Stick {
    pub fn as_string(&self) -> String {
        match self {
            Stick::Main => "MAIN",
            Stick::Control => "CONTROL",
        }
        .to_string()
    }
}
