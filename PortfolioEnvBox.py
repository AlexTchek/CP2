import numpy as np
import pandas as pd
import random
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

class PortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        not_security: list
            a list of non securities (index, currency)
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                reset_to_zero = False,
                cov_xtra_names = [],
                day = 0):
        
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list
        self.reset_to_zero = reset_to_zero
        self.total_commission = 0
        
        self.actions = np.array([0] * stock_dim)
        self.actions_norm = np.array([0] * stock_dim)

        self.action_space = spaces.Box(low=0, high=1, shape = (stock_dim,))
        self.current_state = np.array([0] * stock_dim)
        self.cash = initial_amount #деньги
        #self.state_space + 
        # current_state + covariance matrix + technical indicators + xtra_cov
        obs_shape = ((1 + len(self.tech_indicator_list) + len(cov_xtra_names)) * stock_dim,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = obs_shape)
        
        self.reset()
        self.day = day

        
    def step(self, actions):
        if len(actions) == 2:
            actions = actions[0]        
        
        self.commission = 0
        self.actions_norm = actions
        if sum(actions) > 0:
            self.actions_norm = actions / sum(actions) * self.stock_dim
            actions = actions / sum(actions) #нормированы к единице

        self.actions = actions
        self.actions_memory.append(actions)
        self.terminal = self.day >= len(self.df.index.unique())-1
        close_prices = self.df.loc[self.day]['close'].to_numpy()
        balance = np.sum(self.current_state * close_prices) + self.cash #баланс счета
        if balance < 800_000:
            self.terminal = True
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal, {}

        else:
            h = 0
            
            add_reward = 0
            if h <= self.hmax:
                sell_money = np.sum(self.current_state[actions == 0] * close_prices[actions == 0])
                sell_commission = abs(sell_money * self.transaction_cost_pct)

                self.commission += sell_commission
                self.cash += sell_money #выручили денег от продажи (за вычетом комиссии)
                self.cash -= sell_commission

                self.current_state[actions == 0] = 0 #обнулили позиции (те что продали)

                securities_to_be = np.logical_or(self.current_state > 0, actions > 0) #сколько бумаг будет (те что были + те что купить)
                if np.sum(securities_to_be) > 0:
                    money_per_security = balance * actions #целевая сумма денег под каждую бумагу
                else:
                    money_per_security = balance

                money_per_security *= (1 - self.transaction_cost_pct)
                target_quantity = np.int32(money_per_security / close_prices * securities_to_be) #сколько должно быть бумаг

                buy_sell = target_quantity - self.current_state #сколько требуется докупить-допродать
                buy_money = np.sum(buy_sell * close_prices) #денег от купить-продать
                buy_commission = np.sum(np.abs(buy_sell) * close_prices * self.transaction_cost_pct)
                self.commission += buy_commission

                self.cash -= buy_money
                self.cash -= buy_commission #покупки-продажи (плюс комиссии)
                self.current_state = target_quantity
            else:
                add_reward = -1
                self.terminal = True

            self.last_day_memory = self.data
            #load next state
            self.day += 1
            self.total_commission += self.commission
            self.set_state()

            close_prices2 = self.df.loc[self.day]['close'].to_numpy()
            balance2 = np.sum(self.current_state * close_prices2) + self.cash

            portfolio_return = (balance2 - balance) / balance 
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = portfolio_return + add_reward#new_portfolio_value 
            #print("Step reward: ", self.reward)
            #self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}
    
    def set_state(self):
        self.data = self.df.loc[self.day,:]
        self.covs = []
        self.xtra = self.data['cov_xtra'].values[0]

        ind = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list]).flatten()
        self.state = np.append(self.xtra, ind, axis=0)
        self.state = np.append(self.actions_norm, self.state, axis=0)
        self.state = self.state / 100
  
    def reset(self):
        self.asset_memory = [self.initial_amount]
        if self.reset_to_zero:
            self.day = 0
        else:
            self.day = random.randint(0, int((max(self.df.index) - 1) / 2)) #reset перекидывает в первую половину
        self.set_state()
        
        self.portfolio_value = self.initial_amount
        self.total_commission = 0
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[0] * self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]] 

        self.current_state = np.array([0] * self.stock_dim)
        self.cash = self.initial_amount #деньги
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
  
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs