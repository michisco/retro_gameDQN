# Playing retro games with Dueling DQN
This project was realized for the Intelligent Systems For Pattern Recognition course and the purpose was to implement a Dueling DQN agent to play three games from Atari and NES environment: **Pong**, **Boxing** and **Super Mario Bros.** using OpenAI Gym.

More information about regarding the project can be found in the [report](https://github.com/michisco/retro_gameDQN/blob/main/report.pdf) and in the [presentation](https://github.com/michisco/retro_gameDQN/blob/main/presentation.pdf).

## Usage
All the agents (DQN, Dueling DQN, Double Dueling DQN) can be run on Kaggle with the notebooks in the _Notebooks_ folder. Each notebook is divided into several enviroments (games) and types of agents.

## Results
Below we have some plots about average reward per episode for each enviroments using the best agent, Double Dueling DQN. <br />
<img src="img/pongReward.png" width="350"> <img src="img/boxingReward.png" width="350"> <br />

<img src="img/SMBReward_rightonly.png" width="350"> <img src="img/SMBReward.png" width="350"> <br />

In addition, we can see the results after the training phase: <br />
<img src="img/pong.gif" width="200"> <img src="img/boxing.gif" width="200">  <br />

<img src="img/smb_rightonly.gif" width="200"> <img src="img/smb_simplemove.gif" width="200"> <br />

## References
1. Kauten, C. (2018). [Super Mario Bros for OpenAI Gym](https://github.com/Kautenja/gym-super-mario-bros). GitHub 
2. Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2017). [Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents](https://arxiv.org/abs/1709.06009). arXiv
3. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). [Playing atari with deep reinforcement learning](https://arxiv.org/abs/1312.5602 ). arXiv
4. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., . . . Hassabis, D. (2015, February). [Human-level control through deep reinforcement learning](http://dx.doi.org/10.1038/nature14236). Nature 
5. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2015). [Dueling network architectures for deep reinforcement learning](https://arxiv.org/abs/1511.06581). arXiv
