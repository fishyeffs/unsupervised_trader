# unsupervised_mashallah
god's willed neural network to make one million gorillion dollar

100 epochs with close as the target feature:\
![image](https://github.com/fishyeffs/unsupervised_mashallah/assets/57490638/8ac2f8d8-0c0c-4269-931b-d681bde0dde7)

# Stages 
Stages of implementation based on priority and difficulty:
## Level 1
Create AI model that can classify trading patterns. Computes % match to the pattern (compared against real historical data) \n
## Level 2
Create a widget which connects to binance account etc. so that software can be used on binance / create website which takes displays market data and indicators etc \n
## Level 3
Create extension which uses computer vision to analyse graph on user’s screen \n

# What does it do?
High level requirements are ordered by priority
## High Level Definition
- Prediction
  - Bearish/bullish
    - Bottom/top out value
  - Buy/sell signal (percentage)
  - Range of prices/forecast
- Trade Pattern Classification and Ranking
- Notifications
- Ability to annotate live charts

## Specifics
Technical-ish definitions
- Trade Pattern Classification and Ranking
- Use AI to classify pattern developing in the market
- Rank based of percentage of the match
- Hovering over pattern will illustrate pattern on graph (UI)
- Across different time periods, 1h , 4h, 8h, 1d etc.
- Predicition
  - Need to collect data on how bullish/bearish pattern is
  - Collection of patterns indicates Buy/Sell
  - Need to collect data on bearish/bullish target of pattern
  - Show a projected range of prices that could unfold
  - Do this to a range of patterns?
- Notifications
  - Ability to turn on email notifications when new patterns emerge
  - Only notify if pattern matches over a threshold percentage
  - In-App notifications
- Ability to annotate charts
  - Research existing annotation
  - Can you implement a preexisting tool that can already do this (free :))
