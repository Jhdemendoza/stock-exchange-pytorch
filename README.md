# Stock-Exchange-Pytorch
`stock-exchange-pytorch` implements some of the `deep learning` algorithms
to be applied in financial markets with `pytorch`.
Namely, `reinforcement learning` and  `supervised learning`
are exploited.

Reinforcement Learning
- [x] Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](https://www.nature.com/articles/nature14236)
- [x] Double DQN [[3]](http://arxiv.org/abs/1509.06461)
- [x] Dueling network DQN (Dueling DQN) [[4]](https://arxiv.org/abs/1511.06581)
- [ ] Asynchronous Advantage Actor-Critic (A3C) [[5]](http://arxiv.org/abs/1602.01783)
- [ ] Proximal Policy Optimization Algorithms (PPO) [[6]](https://arxiv.org/abs/1707.06347)
- [ ] Deep Deterministic Policy Gradient (DDPG) [[7]](http://arxiv.org/abs/1509.02971)

Supervised Learning
- [x] Gated Recurrent Unit (GRU) approach to fit distribution of returns in 
 a probabilistic sense [[8]](https://arxiv.org/abs/1406.1078)
- [ ] Use `uber/pyro` or `pymc` to test other approaches of probabilistic programming

### Requirements
By default, it assumes you have installed `pytorch` as the name suggest. 
It also assumes you have a decent `NVIDIA GPU`, and using `Python 3.5+`, and `pip install gym`
as well if necessary.

Once `pytorch` is installed, you can run the following to train a demo `dueling DQN` with the 
provided data.
```buildoutcfg
python3 train_reinforce.py
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
While it supports multiple holdings of securities, on differing amounts between `-1 and 1` which is 
discretelely divisible by `any number`, it does not provide constraints options or some of the 
rules of the game.


### Future work
* Provide constraints in holdings for `gym_stock_exchange`
* Provide `policy gradient` approaches for the `agents` 
* Provide `portfolio` type of support by default in the `gym_stock_exchange`
* Provide `options` and `other derivatives` valuations through agents learning the payoff from those products

