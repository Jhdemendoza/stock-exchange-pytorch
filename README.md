# Stock-Exchange-PyTorch

<img src="https://discuss.pytorch.org/uploads/default/original/2X/3/35226d9fbc661ced1c5d17e374638389178c3176.png" width="200" />

### Originally, I started this project to learn some basics of **reinforcement learning** and have taken a detour. The code-base has become a **mess**. `git stash`ing for now.
### Before you say you **want** reinforcement learning, please make sure you really do.

### Examples
Some of the ideas used for this project are illustrated in the [examples](examples/) folder.
Things do get interesting once you start digging into some of the log file analysis.


# Original
`stock-exchange-pytorch` applies deep learning algorithms
to financial market data in `pytorch`.
Namely, `reinforcement learning` and  `supervised learning`
are applied to a portfolio.

Reinforcement Learning
- [x] Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](https://www.nature.com/articles/nature14236)
- [x] Double DQN [[3]](http://arxiv.org/abs/1509.06461)
- [x] Dueling network DQN (Dueling DQN) [[4]](https://arxiv.org/abs/1511.06581)
- [x] Deep Deterministic Policy Gradient (DDPG) [[5]](http://arxiv.org/abs/1509.02971)

Supervised Learning
- [x] ~~Gated Recurrent Unit (GRU) approach to fit distributions [[7]](https://arxiv.org/abs/1406.1078)~~
- [x] Try as few parameters as possible to actually make some viable products [and it works better than some work by pros](examples/log_file_analyze_with_confusion_matrix_minute_data.ipynb)

Data
- [x] ~~`market data` from [Investor's Exchange (IEX)](https://iextrading.com/)~~
- [x] Use Investor's Exchange to automate [downloading and saving](download_daily_data.py)
You can download data from IEXtrading [directly](download_daily_data.py). If you want something serious (by millisecond level), look at [here](download_deep_data.py).


### Requirements
By default, it assumes you have installed `PyTorch-0.4+` as the name suggests. 
It also assumes you have a decent `NVIDIA GPU` with `Python 3.5+`, and `pip install gym`
if necessary.

You can run the following to train a demo `ddpg` with the 
provided data.
```buildoutcfg
python3 train_reinforce_ddpg.py
```

You can run the following to train a demo `dueling DQN` with the 
provided data.
```buildoutcfg
python3 train_reinforce_dqn.py
```

If you want to test the result, simply run
```buildoutcfg
python3 test_reinforce.py
```
If all went well, you might see something like this:
![screen shot](img/dueling_dqn.gif)


**Supervised learning** is done with `GRU` network, and can be found in 
`train_supervised.py`

### gym_stock_exchange [link](https://github.com/wbaik/gym-stock-exchange)
This is an `environment` which depends on `open-ai`'s [gym](https://github.com/openai/gym).
It supports single stock or a portfolio. For `action spaces` , both `continuous` and
`discrete` are supported.
Obviously, `DDPG` uses continuous action-space, and `DQN` uses discrete settings.
Code for [continuous](gym_exchange/envs/stock_exchange_continuous.py) and
[discrete](gym_exchange/envs/stock_exchange.py) are found here.

### Reminder
Things to think about before trying any models:
- Data Size
- Param Size for a model
- Are you sure `reinforcement learning` adds much value?

### Future work
- [x] Provide constraints in holdings for `gym_stock_exchange`
- [x] Provide `portfolio` by default in the `gym_stock_exchange`
- [x] Provide examples
- [x] Provide more thorough support for `supervised learnings`
- [ ] ~~Implement `PPO` and other `policy gradient` methods~~
- [ ] Implement `options` and `other derivatives` valuations
