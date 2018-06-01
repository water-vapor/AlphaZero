# AlphaZero

## About

(The project and its developers are not affiliated with Google)

This project is a replication of AlphaGo Zero described in [Mastering the game of Go without human knowledge](https://www.nature.com/nature/journal/v550/n7676/pdf/nature24270.pdf). To match the function of Alpha Zero described in [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), the algorithms are implemented in a general framework which supports multiple board games (currently Go, mnk game, reversi).

## Training

The game to train is specified in `AlphaZero/config/game.yaml`. The training can be started with 
```bash
python -m AlphaZero.train.parallel.reinforcement
```

You may want to set the parameters of the system. You can do so by modifying the configuration files in `AlphaZero/config`.

- `<type_of_game>.yaml`: options of game environment and the modules to be imported.
- `reinforce.yaml`: learning parameters for reinforcement learning.
- `supervised.yaml`: learning parameters for supervised learning.
- `rl_sys_config.yaml`: the system settings of the trainer. The detailed explanation of each item is in [Github Wiki](https://github.com/vaporized/AlphaZero/wiki/Items-in-RL-Config-File).

## Standalone Self Play Module

You can run the self play module on multiple computers.
```bash
python -m AlphaZero.train.parallel.selfplay <IP of master session>
```

You also need to set the system parameters including the port for connection in `AlplaZero/config/rl_sys_config.yaml`.

## Go GUI Playing Interface

We use [GoGUI](https://sourceforge.net/projects/gogui/) for graphic UI. Although this project
is not actively maintained now, our program theoretically supports all the Go visualization
softwares using GTP.

Go to `Program -> New Program` to connect our program. Put `python -m AlphaZero.gtp` for command
and the root directory of this project for working directory.

You can set the parameters of the player in `AlphaZero/config/gtp.yaml`. Only the first 4 items are
important. You can also use command line arguments to override the settings in this file, which is
useful when you want two players with different configuration.

You can hold matches between different programs.
```bash
gogui-twogtp -black "<command of black>" -white "<command of white>" -games <num of games> -size 19 -alternate -sgffile <dir for result> -auto
```

You can visualize the match.
```bash
gogui -program "gogui-twogtp -black \"<command of black>\" -white \"<command of white>\"" -computer-both
```

Remember to add backslash if necessary.

## Command Line Playing Interface

The game to play is specified in `AlphaZero/config/play_cmd.yaml`. The game can be started with 
```bash
python -m AlphaZero.play_cmd
```

