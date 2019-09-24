# Deep-Reinforcement-Learning-for-Solving-Job-Shop-Scheduling-Problems

*In the past decades, many optimization methods have been devised and applied to
job shop scheduling problem (JSSP) to find the optimal solution. Most methods assume
that the scheduling results are applied to static environments. However, the whole
environments in the real word are always dynamic and many unexpected events make
original solutions to fail.*

*In this essay, we view JSSP as a sequential decision making
problem and propose to use deep reinforcement learning model to tackle this problem.
Th e combination of deep learning and reinforcement learning avoids to handcraft
features as used in traditional reinforcement learning, and it is expected that the
combination will make the whole learning phase more efficient. Our proposed model
consists of actor network and critic network, and both networks include convolution
layers and fully connected layer. Ac tor network let agent learn how to behave in
different situations, while critic network help agent evaluate the value of statement t hen
return to ac tor network. The whole network is trained with parallel training on a multi
agent environment a nd different simple dispatching rules are considered as actions.* 

*We evaluate our proposed model on more than ten instances that are present in a famous
benchmark problem library OR library. The evaluation results indicate that no matter
in static JSSP benchmark problem or in stochastic JSSP, our method can compete with
other alternatives.*
