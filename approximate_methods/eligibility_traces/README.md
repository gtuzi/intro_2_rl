[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf
[Seijen 2016]: https://arxiv.org/pdf/1512.04087 

# Eligibility Traces

## Table of Contents
- [Intro](#Intro)
- [Implemented Algorithms - Estimation](#implemented-algorithms-for-estimation)
- [Implemented Algorithms - Control](#implemented-algorithms-for-control)
- [Explanations, Development, and Experimental Details](#explanations-development-and-experimental-details)
- [Offline lambda-return vs. TD(lambda)](#offline-lambda-return-vs-tdlambda-algorithm)
- [Truncated TD(lambda)](#truncated-tdlambda)
- [True Online TD(lambda)](#true-online-tdlambda)
- [Sarsa(lambda)](#sarsalambda)
- [Generalized Off-Policy TD Algorithms](#generalized-off-policy-td-algorithms)

## Intro
Eligibility traces (ET)s unify and generalize TD and Monte Carlo methods. When TD
methods are augmented with ETs, they produce a family of methods spanning
a spectrum that has Monte Carlo methods at one end ($\lambda = 1$) and one-step TD methods
at the other ($\lambda = 0$). In between are intermediate methods that are often better than
either extreme method. ETs also provide a way of implementing Monte Carlo
methods online and on continuing problems without episodes. 
What ETs offer is a short-term memory vector mechanism, 
a memory vector, the _eligibility trace_ $\mathbf{z}_t \in \mathbb{R}^d$, 
that parallels the $n$-step TD method's long-term weight vector 
$\mathbf{w}_t \in \mathbb{R}^d$.

Some of the advantages of ETs are:
* Only a single trace vector is required rather than a store of the last $n$ feature vectors
* Learning also occurs continually and uniformly in time rather than being delayed and
then catching up at the end of the episode.
* Learning can occur and affect behavior immediately after a state is 
encountered rather than being delayed $n$ steps.

## Implemented Algorithms for Estimation
- [x] TD($\lambda$): `algorithms/TDLambda`
- [x] Truncated TD($\lambda$) - TTD($\lambda$): `algorithms/TTDLambda`
- [x] Offline $\lambda$ Return: `algorithms/OfflineLambdaReturn`
- [x] Online $\lambda$ Return: `algorithms/OnlineLambdaReturn`
- [x] Online TD($\lambda$): `algorithms/OnlineTDLambda`

## Implemented Algorithms for Control
- [x] Sarsa($\lambda$): `agents/SarsaLambda`
- [x] True Online Sarsa($\lambda$): `agents/TrueOnlineSarsaLambda`
- [x] Off-Policy Expected Sarsa($\lambda$): `agents/OffPolicyExpectedSarsaLambda`
- [x] Off-Policy Tree-Backup($\lambda$) - TB($\lambda$): `agents/TBLambda`
- [x] Off-Policy Gradient-TD($\lambda$) for control - GQ($\lambda$): `agents/GQLambda`
- [x] Off-Policy Hybrid-TD($\lambda$) for control - HQ($\lambda$): `agents/HQLambda`


## Explanations, Development, and Experimental Details
Full development and discussion visit the notebook [here](summary.ipynb)


## Offline $\lambda$-return vs. TD($\lambda$) Algorithm

| TD($\lambda$)                                                               | Offline $\lambda$-return Algorithm                                                      |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| <img src="images/experiment_fig12.6_td_lambda.png" alt="Grid" width="350"/> | <img src="images/experiment_fig12.6_offline_lambda_return.png" alt="Grid" width="350"/> |


## Truncated TD($\lambda$)


| <img src="images/ttd_lambda_n_1.png" alt="Grid" width="250"/>  | <img src="images/ttd_lambda_n_5.png" alt="Grid" width="250"/>  | <img src="images/ttd_lambda_n_10.png" alt="Grid" width="250"/> |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| <img src="images/ttd_lambda_n_20.png" alt="Grid" width="250"/> | <img src="images/ttd_lambda_n_40.png" alt="Grid" width="250"/> |                                                                |

## True Online TD($\lambda$)

<img src="images/experiment_fig12.6_online_td_lambda.png" alt="Grid" width="450"/>

## Sarsa($\lambda$)

| Traces       | Sarsa($\lambda$)                                                                          | True Online Sarsa($\lambda$)                                                                        |
|--------------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Replacing    | <img src="images/SarsaLambda_MountainCar_ReplacingTraces.png" alt="Grid" width="450"/>    | --                                                                                                  |
| Accumulating | <img src="images/SarsaLambda_MountainCar_AccumulatingTraces.png" alt="Grid" width="450"/> | <img src="images/TrueOnlineSarsaLambda_MountainCar_AccumulatingTraces.png" alt="Grid" width="450"/> |


## Generalized Off-Policy TD Algorithms

| <img src="images/OffPolicy_Expected_Sarsa_Lambda.png" alt="Grid" width="450"/> | <img src="images/OffPolicy_TB_Lambda.png" alt="Grid" width="450"/> |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------|
| <img src="images/OffPolicy_GQLambda.png" alt="Grid" width="450"/>              | <img src="images/OffPolicy_HQLambda.png" alt="Grid" width="450"/>  |
