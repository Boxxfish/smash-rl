use crate::micro_fighter::{MicroFighter, StepOutput};
use buffer_graphics_lib::prelude::*;
use tch::Tensor;

const IMG_SIZE: u32 = 64;

fn insert_obs(obs: Tensor, frame_stack: &mut [Tensor]) {
    for i in (1..frame_stack.len()).rev() {
        frame_stack[i] = frame_stack[i - 1].copy();
    }
    frame_stack[0] = obs;
}

/// Rust version of the Micro Fighter environment.
pub struct MFEnv {
    game: MicroFighter,
    observation_space: ([i64; 4], [i64; 1]),
    action_space: i32,
    player_stats: tch::Tensor,
    bot_stats: tch::Tensor,
    dmg_reward_amount: f32,
    last_dist: f32,
    player_frame_stack: Vec<Tensor>,
    bot_frame_stack: Vec<Tensor>,
    max_skip_frames: u32,
    num_frames: u32,
    num_channels: u32,
}

pub struct MFEnvInfo {
    pub player_won: bool,
}

impl MFEnv {
    pub fn new(max_skip_frames: u32, num_frames: u32) -> Self {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let game = MicroFighter::new(false);
        let num_channels = 4;
        let observation_space = (
            [
                num_frames as i64,
                num_channels as i64,
                IMG_SIZE as i64,
                IMG_SIZE as i64,
            ],
            [36],
        );
        let action_space = 9;
        let player_stats = Tensor::zeros([18], options);
        let bot_stats = Tensor::zeros([18], options);
        let dmg_reward_amount = 1.0;
        let last_dist = 0.0;
        let player_frame_stack = (0..num_frames)
            .map(|_| {
                Tensor::zeros(
                    &[num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
        let bot_frame_stack = (0..num_frames)
            .map(|_| {
                Tensor::zeros(
                    &[num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
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
        }
    }

    pub fn step(&mut self, action: u32) -> ((Tensor, Tensor), f32, bool, bool, MFEnvInfo) {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let skip_frames = self.max_skip_frames;
        let step_output = self.game.step(action);
        let mut dmg_reward = step_output.net_damage;
        for _ in 0..(skip_frames) {
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
        let one = Tensor::ones(&[1], options);
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
            .set_data(&(step_output.opponent_dir * one));
        let stats_obs = Tensor::concatenate(&[self.player_stats.copy(), self.bot_stats.copy()], 0);

        let terminated = step_output.round_over;
        let mut round_reward = 0.0;
        if terminated {
            round_reward = if step_output.player_won { 1.0 } else { -1.0 };
        }

        let curr_dist = step_output.player_pos.0.pow(2) as f32 / 200_u32.pow(2) as f32;
        let delta_dist = curr_dist - self.last_dist;
        self.last_dist = curr_dist;

        let dmg_reward = dmg_reward as f32 / 10.0 - delta_dist;

        let reward = dmg_reward * (self.dmg_reward_amount) + round_reward;

        (
            (Tensor::stack(&self.player_frame_stack, 0), stats_obs),
            reward,
            terminated,
            false,
            MFEnvInfo {
                player_won: step_output.player_won,
            },
        )
    }

    pub fn reset(&mut self) -> ((Tensor, Tensor), MFEnvInfo) {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let step_output = self.game.reset();
        self.player_frame_stack = (0..self.num_frames)
            .map(|_| {
                Tensor::zeros(
                    &[self.num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
        self.bot_frame_stack = (0..self.num_frames)
            .map(|_| {
                Tensor::zeros(
                    &[self.num_channels as i64, IMG_SIZE as i64, IMG_SIZE as i64],
                    options,
                )
            })
            .collect();
        let channels = self.gen_channels(&step_output, true);
        insert_obs(Tensor::stack(&channels, 0), &mut self.player_frame_stack);
        let bot_channels = self.gen_channels(&step_output, false);
        insert_obs(Tensor::stack(&bot_channels, 0), &mut self.bot_frame_stack);

        self.player_stats = Tensor::zeros([18], options);
        let one = Tensor::ones(&[1], options);
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

        (
            (Tensor::stack(&self.player_frame_stack, 0), stats_obs),
            MFEnvInfo { player_won: false },
        )
    }

    fn gen_channels(&self, step_output: &StepOutput, is_player: bool) -> Vec<Tensor> {
        let options = (tch::Kind::Float, tch::Device::Cpu);
        let hboxes = &step_output.hboxes;
        let mut hit_channel = Tensor::zeros(&[IMG_SIZE as i64, IMG_SIZE as i64], options);
        let mut dmg_channel = Tensor::zeros(&[IMG_SIZE as i64, IMG_SIZE as i64], options);
        let mut player_channel = Tensor::zeros(&[IMG_SIZE as i64, IMG_SIZE as i64], options);
        let mut box_channel = Tensor::zeros(&[IMG_SIZE as i64, IMG_SIZE as i64], options);
        for hbox in hboxes {
            let mut buffer: [u8; (IMG_SIZE * IMG_SIZE * 4) as usize] =
                [0; (IMG_SIZE * IMG_SIZE * 4) as usize];
            let mut graphics = Graphics::new(&mut buffer, 800, 600).unwrap();
            let rot = -hbox.angle;
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
            let box_arr = Tensor::from_slice(&buffer.iter().step_by(4).copied().collect::<Vec<u8>>()).reshape([IMG_SIZE as i64, IMG_SIZE as i64]);
            let inv_box_arr = 1.0 - &box_arr;
            hit_channel = hit_channel * &inv_box_arr + hbox.is_hit as u8 as f32 * &box_arr;
            dmg_channel = dmg_channel * &inv_box_arr + (hbox.damage as f32 / 10.0) * &box_arr;
            player_channel = player_channel * &inv_box_arr + (hbox.is_player == is_player) as u8 as f32 * &box_arr;
            box_channel = box_channel * &inv_box_arr + &box_arr;
        }

        vec![hit_channel, dmg_channel, player_channel, box_channel]
    }

    pub fn render(&self) {
        // if self.render_mode == "human":
        //     channels = self.player_frame_stack[0]
        //     r = channels[self.view_channels[0]]
        //     g = channels[self.view_channels[1]]
        //     b = channels[self.view_channels[2]]
        //     view = np.flip(np.stack([r, g, b]).transpose(2, 1, 0), 1).clip(0, 1) * 255.0
        //     view_surf = pygame.Surface([IMG_SIZE, IMG_SIZE])
        //     pygame.surfarray.blit_array(view_surf, view)
        //     pygame.transform.scale(
        //         view_surf, [IMG_SIZE * IMG_SCALE, IMG_SIZE * IMG_SCALE], self.screen
        //     )
        //     pygame.display.flip()
        //     self.clock.tick(60)
    }

    pub fn bot_obs(&self) -> (Tensor, Tensor) {
        (
            Tensor::stack(&self.bot_frame_stack, 0),
            Tensor::concatenate(&[self.bot_stats.copy(), self.player_stats.copy()], 0),
        )
    }

    pub fn bot_step(&mut self, action: u32) {
        self.game.bot_step(action);
    }

    pub fn set_dmg_reward_amount(&mut self, amount: f32) {
        self.dmg_reward_amount = amount;
    }
}
