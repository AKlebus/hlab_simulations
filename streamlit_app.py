import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.coin_flipping import CoinFlipSimulator
from src.bandit import BanditSimulator
from src.bayesian_optimization import *

st.title("🗼 HLAB 2024 Seminar Simulations")
st.write(
    "### Hi!"
)
st.write(
    "### This app is a collection of simulations for the \"Choosing with Chance\" seminar in HLAB 2024. I hope you enjoy!"
)

# Coin Flip Simulator - Day 1
st.write(
    "## Coin Flip Simulator"
)
P = 0.65 # Probability of heads
if 'coin_sim' not in st.session_state:
    st.session_state.coin_sim = CoinFlipSimulator(p = P)
coin_sim = st.session_state.coin_sim

if st.button("Flip Coin"):
    coin_sim.flip()
    st.pyplot(coin_sim.visualize())

if st.button("Reset Coin Simulation"):
    st.session_state.coin_sim = CoinFlipSimulator(p = P)
    st.write("Simulator reset!")


# Multi Armed Bandit Simulator - Day 2

st.write(
    "## Multi-Armed Bandit Simulator"
)
P = [0.7, 0.3, 0.5] # Probabilities of success for each arm
if 'bandit_sim' not in st.session_state:
    st.session_state.bandit_sim = BanditSimulator(p = P)
bandit_sim = st.session_state.bandit_sim

col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Machine 1'):
        bandit_sim.pull(0)

with col2:
    if st.button('Machine 2'):
        bandit_sim.pull(1)

with col3:
    if st.button('Machine 3'):
        bandit_sim.pull(2)

st.pyplot(bandit_sim.visualize())

if st.button("Reset Bandit Simulation"):
    st.session_state.bandit_sim = BanditSimulator(p = P)
    st.write("Simulator reset!")

# Bayesian Optimization Simulator (without Posterior)

st.write("## Bayesian Optimization Simulator")

st.write('### Function 1')

if 'bo_sim_1' not in st.session_state:
    obj = ObjectiveFunction(function_1, bounds = [0.0, 5.0], noise = 0.25)
    st.session_state.bo_sim_1 = BayesianOptimizationSimulator(obj)
bo_sim_1 = st.session_state.bo_sim_1

bo_input_1 = st.text_input(label = 'Query Point 1')

if bo_input_1 is not '':
    x = float(bo_input_1)
    bo_sim_1.add_data_point(x)
    st.write(f"{bo_sim_1.X}, {bo_sim_1.y}")

st.pyplot(bo_sim_1.visualize_samples())

if st.button('Reset BO 1'):
    obj = ObjectiveFunction(function_1, bounds = [0.0, 5.0], noise = 0.25)
    st.session_state.bo_sim_1 = BayesianOptimizationSimulator(obj)

st.write('### Function 2')

if 'bo_sim_2' not in st.session_state:
    obj = ObjectiveFunction(function_2, bounds = [0.0, 5.0], noise = 0.25)
    st.session_state.bo_sim_2 = BayesianOptimizationSimulator(obj)
bo_sim_2 = st.session_state.bo_sim_2

bo_input_2 = st.text_input(label = 'Query Point 2')

if bo_input_2 is not '':
    x = float(bo_input_2)
    bo_sim_2.add_data_point(x)
    st.write(f"{bo_sim_2.X}, {bo_sim_2.y}")

st.pyplot(bo_sim_2.visualize_samples())

if st.button('Reset BO 2'):
    obj = ObjectiveFunction(function_2, bounds = [0.0, 5.0], noise = 0.25)
    st.session_state.bo_sim_2 = BayesianOptimizationSimulator(obj)

st.write('### Function 3')

if 'bo_sim_3' not in st.session_state:
    obj = ObjectiveFunction(function_3, bounds = [0.0, 5.0], noise = 1)
    st.session_state.bo_sim_3 = BayesianOptimizationSimulator(obj)
bo_sim_3 = st.session_state.bo_sim_3

bo_input_3 = st.text_input(label = 'Query Point 3')

if bo_input_3 is not '':
    x = float(bo_input_3)
    bo_sim_3.add_data_point(x)
    st.write(f"{bo_sim_3.X}, {bo_sim_3.y}")

st.pyplot(bo_sim_3.visualize_samples())

if st.button('Reset BO 3'):
    obj = ObjectiveFunction(function_3, bounds = [0.0, 5.0], noise = 1)
    st.session_state.bo_sim_3 = BayesianOptimizationSimulator(obj)