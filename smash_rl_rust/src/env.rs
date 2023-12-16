use std::sync::{Arc, RwLock};

use crate::{micro_fighter::{MicroFighter, StepOutput}, training::RetrievalContext};
use buffer_graphics_lib::prelude::*;
use minifb::{Window, WindowOptions};
use tch::Tensor;

pub const IMG_SIZE: u32 = 32;
const SCALE: usize = 8;

fn insert_obs(obs: Tensor, frame_stack: &mut [Tensor]) {
    for i in (1..frame_stack.len()).rev() {
        frame_stack[i] = frame_stack[i - 1].copy();
    }
    frame_stack[0] = obs;
}

struct RenderState {
    buffer: Vec<u32>,
    window: Window,
}

/// Rust version of the Micro Fighter environment.
pub struct MFEnv {
    pub game: MicroFighter,
    pub observation_space: Vec<Vec<i64>>,
    pub action_space: i32,
    pub player_stats: tch::Tensor,
    pub bot_stats: tch::Tensor,
    pub dmg_reward_amount: f32,
    pub last_dist: f32,
    pub player_frame_stack: Vec<Tensor>,
    pub bot_frame_stack: Vec<Tensor>,
    pub max_skip_frames: u32,
    pub num_frames: u32,
    pub num_channels: u32,
    // pub render_state: Option<RenderState>,
    pub view_channels: (u32, u32, u32),
    pub time_limit: u32,
    pub current_time: u32,
    pub bot_action: u32,
}

pub struct MFEnvInfo {
    pub player_won: bool,
}

impl MFEnv {
    pub fn new(
        max_skip_frames: u32,
        num_frames: u32,
        rendering: bool,
        view_channels: (u32, u32, u32),
        time_limit: u32,
    ) -> Self {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let game = MicroFighter::new(false);
        let num_channels = 4;
        let observation_space = vec![
            vec![
                num_frames as i64,
                num_channels as i64,
                IMG_SIZE as i64,
                IMG_SIZE as i64,
            ],
            vec![36],
        ];
        let action_space = 9;
        let player_stats = Tensor::zeros([18], options);
        let bot_stats = Tensor::zeros([18], options);
        let dmg_reward_amount = 1.0;
        let last_dist = 0.0;
        let player_frame_stack = (0..num_frames)
            .map(|_| {
                Tensor::zeros(
                    [num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
        let bot_frame_stack = (0..num_frames)
            .map(|_| {
                Tensor::zeros(
                    [num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
        let render_state = if rendering {
            Some(RenderState {
                buffer: vec![0; (IMG_SIZE * IMG_SIZE) as usize],
                window: Window::new(
                    "MFEnv - AI View",
                    IMG_SIZE as usize * SCALE,
                    IMG_SIZE as usize * SCALE,
                    WindowOptions::default(),
                )
                .expect("Couldn't create window"),
            })
        } else {
            None
        };
        let current_time = 0;
        Self {
            game,
            observation_space,
            action_space,
            player_stats,
            bot_stats,
            dmg_reward_amount,
            last_dist,
            player_frame_stack,
            bot_frame_stack,
            max_skip_frames,
            num_frames,
            num_channels,
            // render_state,
            view_channels,
            time_limit,
            current_time,
            bot_action: 0,
        }
    }

    pub fn step(&mut self, action: u32) -> (Vec<Tensor>, f32, bool, bool, MFEnvInfo) {
        let _guard = tch::no_grad_guard();
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let skip_frames = self.max_skip_frames;
        let step_output = self.game.step(action);
        let mut dmg_reward = step_output.net_damage;
        for _ in 0..(skip_frames) {
            self.game.bot_step(self.bot_action);
            let step_output = self.game.step(action);
            dmg_reward += step_output.net_damage;
            if step_output.round_over {
                break;
            }
        }

        // Spatial observation
        let channels = self.gen_channels(&step_output, true);
        insert_obs(Tensor::stack(&channels, 0), &mut self.player_frame_stack);
        let bot_channels = self.gen_channels(&step_output, false);
        insert_obs(Tensor::stack(&bot_channels, 0), &mut self.bot_frame_stack);

        // Stats observation
        self.player_stats = Tensor::zeros([18], options);
        let one = Tensor::ones([], options);
        self.player_stats
            .get(step_output.player_state as i64)
            .copy_(&one);
        self.player_stats
            .get(16)
            .copy_(&(step_output.player_damage as f32 / 100.0 * &one));
        self.player_stats
            .get(17)
            .copy_(&(step_output.player_dir * &one));
        self.bot_stats = Tensor::zeros([18], options);
        self.bot_stats
            .get(step_output.opponent_state as i64)
            .copy_(&one);
        self.bot_stats
            .get(16)
            .copy_(&(step_output.opponent_damage as f32 / 100.0 * &one));
        self.bot_stats
            .get(17)
            .copy_(&(step_output.opponent_dir * one));
        let stats_obs = Tensor::concatenate(&[self.player_stats.copy(), self.bot_stats.copy()], 0);

        let terminated = step_output.round_over;
        let mut round_reward = 0.0;
        if terminated {
            round_reward = if step_output.player_won { 1.0 } else { -1.0 };
        }

        let dmg_reward = dmg_reward as f32 / 100.0;

        let reward = dmg_reward * (self.dmg_reward_amount) + round_reward;

        // Account for time limit
        self.current_time += 1;
        let truncated = self.current_time >= self.time_limit;

        let spatial = Tensor::stack(&self.player_frame_stack, 0);
        let stats = stats_obs;

        (
            vec![spatial, stats],
            reward,
            terminated,
            truncated,
            MFEnvInfo {
                player_won: step_output.player_won,
            },
        )
    }

    pub fn reset(&mut self) -> (Vec<Tensor>, MFEnvInfo) {
        let _guard = tch::no_grad_guard();
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let step_output = self.game.reset();
        self.player_frame_stack = (0..self.num_frames)
            .map(|_| {
                Tensor::zeros(
                    [self.num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
        self.bot_frame_stack = (0..self.num_frames)
            .map(|_| {
                Tensor::zeros(
                    [self.num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
        let channels = self.gen_channels(&step_output, true);
        insert_obs(Tensor::stack(&channels, 0), &mut self.player_frame_stack);
        let bot_channels = self.gen_channels(&step_output, false);
        insert_obs(Tensor::stack(&bot_channels, 0), &mut self.bot_frame_stack);

        self.player_stats = Tensor::zeros([18], options);
        let one = Tensor::ones([1], options);
        self.player_stats
            .get(step_output.player_state as i64)
            .set_data(&one);
        self.player_stats
            .get(16)
            .set_data(&(step_output.player_damage as f32 / 100.0 * &one));
        self.player_stats
            .get(17)
            .set_data(&(step_output.player_dir * &one));
        self.bot_stats = Tensor::zeros([18], options);
        self.bot_stats
            .get(step_output.opponent_state as i64)
            .set_data(&one);
        self.bot_stats
            .get(16)
            .set_data(&(step_output.opponent_damage as f32 / 100.0 * &one));
        self.bot_stats
            .get(17)
            .set_data(&(step_output.opponent_dir * &one));
        let stats_obs = Tensor::concatenate(&[self.player_stats.copy(), self.bot_stats.copy()], 0);

        self.last_dist = step_output.player_pos.0.pow(2) as f32 / 200_u32.pow(2) as f32;
        self.current_time = 0;
        
        let spatial = Tensor::stack(&self.player_frame_stack, 0);
        let stats = stats_obs;

        (
            vec![spatial, stats],
            MFEnvInfo { player_won: false },
        )
    }

    fn gen_channels(&self, step_output: &StepOutput, is_player: bool) -> Vec<Tensor> {
        let _guard = tch::no_grad_guard();
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let hboxes = &step_output.hboxes;
        let mut hit_channel = Tensor::zeros([IMG_SIZE as i64, IMG_SIZE as i64], options);
        let mut dmg_channel = Tensor::zeros([IMG_SIZE as i64, IMG_SIZE as i64], options);
        let mut player_channel = Tensor::zeros([IMG_SIZE as i64, IMG_SIZE as i64], options);
        let mut box_channel = Tensor::zeros([IMG_SIZE as i64, IMG_SIZE as i64], options);
        for hbox in hboxes {
            let mut buffer: [u8; (IMG_SIZE * IMG_SIZE * 4) as usize] =
                [0; (IMG_SIZE * IMG_SIZE * 4) as usize];
            let mut graphics = Graphics::new(&mut buffer, IMG_SIZE as usize, IMG_SIZE as usize).unwrap();
            let rot = hbox.angle;
            // Rotate points around center and offset, then convert to integral image space
            let points = [
                [-(hbox.w as i32) / 2, hbox.h as i32 / 2],
                [hbox.w as i32 / 2, hbox.h as i32 / 2],
                [hbox.w as i32 / 2, -(hbox.h as i32) / 2],
                [-(hbox.w as i32) / 2, -(hbox.h as i32) / 2],
            ];
            let points: Vec<Coord> = points
                .iter()
                .map(|point| {
                    Coord::new(
                        ((point[0] as f32 * rot.cos() - point[1] as f32 * rot.sin()
                            + (hbox.x + hbox.w as i32 / 2) as f32
                            + (self.game.get_screen_size() / 2) as f32)
                            * (IMG_SIZE as f32 / self.game.get_screen_size() as f32))
                            as isize,
                        ((point[0] as f32 * rot.sin()
                            + point[1] as f32 * rot.cos()
                            + (hbox.y + hbox.h as i32 / 2) as f32
                            + (self.game.get_screen_size() / 2) as f32)
                            * (IMG_SIZE as f32 / self.game.get_screen_size() as f32))
                            as isize,
                    )
                })
                .collect();
            let polygon = Polygon::from_points(&points);
            graphics.draw_polygon(polygon, DrawType::Fill(Color::rgb(1, 1, 1)));
            let box_arr =
                Tensor::from_slice(&buffer.iter().step_by(4).copied().collect::<Vec<u8>>())
                    .reshape([IMG_SIZE as i64, IMG_SIZE as i64]);
            let inv_box_arr = 1.0 - &box_arr;
            hit_channel = hit_channel * &inv_box_arr + hbox.is_hit as u8 as f32 * &box_arr;
            dmg_channel = dmg_channel * &inv_box_arr + (hbox.damage as f32 / 10.0) * &box_arr;
            player_channel = player_channel * &inv_box_arr
                + (hbox.is_player == is_player) as u8 as f32 * &box_arr;
            box_channel = box_channel * &inv_box_arr + &box_arr;
        }

        vec![hit_channel, dmg_channel, player_channel, box_channel]
    }

    pub fn render(&mut self) {
        // if let Some(render_state) = &mut self.render_state {
        //     let channels = &self.player_frame_stack[0];
        //     let r = channels.get(self.view_channels.0 as i64).flip([0]) * 255.0;
        //     let g = channels.get(self.view_channels.1 as i64).flip([0]) * 255.0;
        //     let b = channels.get(self.view_channels.2 as i64).flip([0]) * 255.0;
        //     let window = &mut render_state.window;
        //     let mut r_buf = vec![0.0_f32; (IMG_SIZE * IMG_SIZE) as usize];
        //     let mut g_buf = vec![0.0_f32; (IMG_SIZE * IMG_SIZE) as usize];
        //     let mut b_buf = vec![0.0_f32; (IMG_SIZE * IMG_SIZE) as usize];
        //     r.copy_data(&mut r_buf, (IMG_SIZE * IMG_SIZE) as usize);
        //     g.copy_data(&mut g_buf, (IMG_SIZE * IMG_SIZE) as usize);
        //     b.copy_data(&mut b_buf, (IMG_SIZE * IMG_SIZE) as usize);
        //     for y in 0..IMG_SIZE {
        //         for x in 0..IMG_SIZE {
        //             let index = (y * IMG_SIZE + x) as usize;
        //             let r = r_buf[index] as u32;
        //             let g = g_buf[index] as u32;
        //             let b = b_buf[index] as u32;
        //             render_state.buffer[index] = (r << 16) + (g << 8) + b;
        //         }    
        //     }
        //     window
        //         .update_with_buffer(&render_state.buffer, IMG_SIZE as usize, IMG_SIZE as usize)
        //         .expect("Couldn't render buffer");
        // }
    }

    pub fn bot_obs(&self) -> Vec<Tensor> {
        let spatial = Tensor::stack(&self.bot_frame_stack, 0);
        let stats = Tensor::concatenate(&[self.bot_stats.copy(), self.player_stats.copy()], 0);

        vec![spatial, stats]
    }

    pub fn bot_step(&mut self, action: u32) {
        self.bot_action = action;
    }

    pub fn set_dmg_reward_amount(&mut self, amount: f32) {
        self.dmg_reward_amount = amount;
    }
}
