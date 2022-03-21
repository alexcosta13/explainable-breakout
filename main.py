import time

import wandb
import yaml

import numpy as np

from agent import Agent
from dqn import build_dqn
from gamewrapper import GameWrapper
from replaybuffer import ReplayBuffer


def main(args):
    game_wrapper = GameWrapper(args["ENV_NAME"], args["MAX_NOOP_STEPS"])
    if args["WRITE_TERMINAL"]:
        print(
            "The environment has the following {} actions: {}".format(
                game_wrapper.env.action_space.n,
                game_wrapper.env.unwrapped.get_action_meanings(),
            )
        )

    wandb.init(config=args)

    # Build main and target networks
    main_dqn = build_dqn(
        game_wrapper.env.action_space.n,
        args["LEARNING_RATE"],
        input_shape=args["INPUT_SHAPE"],
    )
    target_dqn = build_dqn(
        game_wrapper.env.action_space.n, input_shape=args["INPUT_SHAPE"]
    )

    replay_buffer = ReplayBuffer(
        size=args["MEM_SIZE"], input_shape=args["INPUT_SHAPE"], use_per=args["USE_PER"]
    )
    agent = Agent(
        main_dqn,
        target_dqn,
        replay_buffer,
        game_wrapper.env.action_space.n,
        input_shape=args["INPUT_SHAPE"],
        batch_size=args["BATCH_SIZE"],
        use_per=args["USE_PER"],
    )

    # Training and evaluation
    if args["LOAD_FROM"] is None:
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        if args["WRITE_TERMINAL"]:
            print("Loading from", args["LOAD_FROM"])
        meta = agent.load(args["LOAD_FROM"], args["LOAD_REPLAY_BUFFER"])

        # Apply information loaded from meta
        frame_number = meta["frame_number"]
        rewards = meta["rewards"]
        loss_list = meta["loss_list"]

    # Main loop
    try:
        while frame_number < args["TOTAL_FRAMES"]:
            # Training
            epoch_frame = 0
            while epoch_frame < args["FRAMES_BETWEEN_EVAL"]:
                start_time = time.time()
                game_wrapper.reset()
                # life_lost = True
                episode_reward_sum = 0
                for _ in range(args["MAX_EPISODE_LENGTH"]):
                    # Get action
                    action = agent.get_action(frame_number, game_wrapper.state)

                    # Take step
                    (
                        processed_frame,
                        reward,
                        terminal,
                        life_lost,
                    ) = game_wrapper.step(action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    # Add experience to replay memory
                    agent.add_experience(
                        action=action,
                        frame=processed_frame[:, :, 0],
                        reward=reward,
                        clip_reward=args["CLIP_REWARD"],
                        terminal=life_lost,
                    )

                    # Update agent
                    if (
                        frame_number % args["UPDATE_FREQ"] == 0
                        and agent.replay_buffer.count > args["MIN_REPLAY_BUFFER_SIZE"]
                    ):
                        loss, _ = agent.learn(
                            args["BATCH_SIZE"],
                            gamma=args["DISCOUNT_FACTOR"],
                            frame_number=frame_number,
                            priority_scale=args["PRIORITY_SCALE"],
                        )
                        loss_list.append(loss)

                    # Update target network
                    if (
                        frame_number % args["UPDATE_FREQ_TARGET_NETWORK"] == 0
                        and frame_number > args["MIN_REPLAY_BUFFER_SIZE"]
                    ):
                        agent.update_target_network()

                    # Break the loop when the game is over
                    if terminal:
                        break

                rewards.append(episode_reward_sum)

                # Output the progress every 10 games
                if len(rewards) % 10 == 0:
                    # Write to TensorBoard
                    if args["WRITE_WANDB"]:
                        wandb.log(
                            {
                                "frame_number": frame_number,
                                "reward": np.mean(rewards[-10:]),
                                "loss": np.mean(loss_list[-10:]),
                                "smooth_loss": np.mean(loss_list[-100:]),
                                "time": time.time() - start_time,
                            }
                        )

                    if args["WRITE_TERMINAL"]:
                        print(
                            f"Game number: {str(len(rewards)).zfill(6)}\t"
                            f"Frame number: {str(frame_number).zfill(8)}\t"
                            f"Average reward: {np.mean(rewards[-10:]):0.1f}\t"
                            f"Time taken: {(time.time() - start_time):.1f}s"
                        )

            # Evaluation every `FRAMES_BETWEEN_EVAL` frames
            terminal = True
            eval_rewards = []
            evaluate_frame_number = 0

            for _ in range(args["EVAL_LENGTH"]):
                if terminal:
                    game_wrapper.reset(evaluation=True)
                    life_lost = True
                    episode_reward_sum = 0
                    terminal = False

                # Breakout requires a "fire" action (action #1) to start the
                # game each time a life is lost.
                # Otherwise, the agent would sit around doing nothing.
                action = (
                    1
                    if life_lost
                    else agent.get_action(
                        frame_number, game_wrapper.state, evaluation=True
                    )
                )

                # Step action
                _, reward, terminal, life_lost = game_wrapper.step(action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                # On game-over
                if terminal:
                    eval_rewards.append(episode_reward_sum)

            if len(eval_rewards) > 0:
                final_score = np.mean(eval_rewards)
            else:
                # In case the game is longer than the number of frames allowed
                final_score = episode_reward_sum
            # Print score and write to tensorboard

            if args["WRITE_WANDB"]:
                wandb.log(
                    {"frame_number": frame_number, "evaluation_score": final_score}
                )

            if args["WRITE_TERMINAL"]:
                print("Evaluation score:", final_score)

            # Save model
            if len(rewards) > 300 and args["SAVE_TO"] is not None:
                agent.save(
                    f"{args['SAVE_TO']}/save-{str(frame_number).zfill(8)}",
                    frame_number=frame_number,
                    rewards=rewards,
                    loss_list=loss_list,
                    save_buffer=args["SAVE_REPLAY_BUFFER"]
                )
    except KeyboardInterrupt:
        print("\nTraining exited early.")

        if args["SAVE_TO"] is None:
            try:
                args["SAVE_TO"] = input(
                    "Would you like to save the trained model?"
                    "If so, type in a save path, otherwise, interrupt with ctrl+c. "
                )
            except KeyboardInterrupt:
                print("\nExiting...")

        if args["SAVE_TO"] is not None:
            print("Saving...")
            agent.save(
                f"{args['SAVE_TO']}/save-{str(frame_number).zfill(8)}",
                frame_number=frame_number,
                rewards=rewards,
                loss_list=loss_list,
                save_buffer=args["SAVE_REPLAY_BUFFER"]
            )
            print("Saved.")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
