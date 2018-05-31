# AlphaZero

## About

(The project and its developers are not affiliated with Google)

This project is a replication of AlphaGo Zero described in [Mastering the game of Go without human knowledge](https://www.nature.com/nature/journal/v550/n7676/pdf/nature24270.pdf). To match the function of Alpha Zero described in [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), the algorithms are implemented in a general framework which supports multiple board games (currently Go, mnk game, reversi).

## Training

The game to train is specified in `AlphaZero/config/game.yaml`. The training can be started with 
```bash
python -m AlphaZero.train.parallel.reinforcement
```

## Go GUI Playing Interface

## Command Line Playing Interface

The game to play is specified in `AlphaZero/config/play_cmd.yaml`. The game can be started with 
```bash
python -m AlphaZero.play_cmd
```

