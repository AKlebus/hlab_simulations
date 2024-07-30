import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.coin_flipping import CoinFlipSimulator
from src.bandit import BanditSimulator

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