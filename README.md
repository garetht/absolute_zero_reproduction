# Reinforced Self-play Reasoning with Zero Data

This is a reproduction of the method of reinforcement learning (self-play with task-relative REINFORCE++) used in the paper [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335?context=cs.LG). Our setup differs in that instead of generating Python programs as the paper does, we instead adapt the setup to the prime inversion problem.

## Problem

Our project attempts to teach the model the equation

$$xy \equiv 1 \pmod{p} \text{ , } p \text{ prime}$$

The induction task from the paper becomes a task to solve for $x$, e.g. 

$$x \times 4 \equiv 1 \pmod{7}$$

and we can determine that, uniquely, $x \equiv 2 \pmod{7}$ in this example.

The abduction and induction tasks are determined in the same way, where the solver has to solve for $y$ and $p$ respectively. Like the induction task in the paper, multiple primes are possible, and we will accept any of those that fulfill the task for a given $x$ and $y$.

We can control the computational difficulty of the problem by increasing the size of possible $p$ candidates. This makes this problem more amenable to RL since we are able to tune the problem difficulty to match the initial capability of the model.

## Setup

We followed the Absolute Zero paper in training on a model in the `Qwen/Qwen2.5-3B` series, but we started with `Qwen/Qwen2.5-3B-Instruct` instead of a base model, because we found it unproductive to attempt to instruct the base model to respond in a parseable format for us to correctly reward it and bootstrap its learning.

We ran this training on Nvidia A100s in a cloud-provisioned instance on RunPod.

## Repository Layout


