# Dual V-Learning (DVL) 

### [**[Project Page](https://hari-sikchi.github.io/dual-rl/)**] 

### [**[Paper](https://arxiv.org/abs/2302.08560)**] 



Official code base for **[Dual RL: Unification and New Methods for Reinforcement and Imitation Learning](https://arxiv.org/abs/2302.08560)** by [Harshit Sikchi](https://hari-sikchi.github.io/)\*, [Qinqing Zheng](https://enosair.github.io/)\*, [Amy Zhang](https://www.ece.utexas.edu/people/faculty/amy-zhang), and [Scott Niekum](https://people.cs.umass.edu/~sniekum/).




This repository contains code for **Dual V-Learning (DVL)** framework for Reinforcement Learning proposed in our paper.


Please refer to instructions inside the **offline** folder to get started with installation and running the code.


## Benefits of DVL over other offline RL methods
✅  Fixes the instability of Extreme Q Learning \(XQL\)   \
✅  Directly models V* in continuous action spaces  \
✅  Implict, no OOD Sampling or actor-critic formulation \
✅  Conservative with respect to the induced behavior policy distribution \
✅  Improves performance on the D4RL benchmark versus similar approaches

### Citation
```
@article{sikchi2023imitation,
  title={Imitation from Arbitrary Experience: A Dual Unification of Reinforcement and Imitation Learning Methods},
  author={Sikchi, Harshit and Zhang, Amy and Niekum, Scott},
  journal={arXiv preprint arXiv:2302.08560},
  year={2023}
}
```




## Questions
Please feel free to email us if you have any questions. 

Harshit Sikchi ([hsikchi@utexas.edu](mailto:hsikchi@utexas.edu?subject=[GitHub]%DVL))


## Acknowledgement

This repository builds heavily on the XQL(https://github.com/Div99/xql) and IQL(https://github.com/ikostrikov/implicit_q_learning) codebases. Please make sure to cite them as well when using this code.

