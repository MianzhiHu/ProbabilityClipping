from DelayedDiscountingModels import DelayedDiscounting
import numpy as np
import pandas as pd

exponential_e02 = DelayedDiscounting('dd_exponential', softmax_function='e02')
hyperbolic_e02 = DelayedDiscounting('dd_hyperbolic', softmax_function='e64')
hyperboloid_e02 = DelayedDiscounting('dd_hyperboloid', softmax_function='e02')
expo_simulated_data = hyperbolic_e02.simulate(n_participants=100, n_trials=120, later_delay_range=(19, 180),
                                              small_amount_range=(20, 20), large_amount_range=(22, 85),
                                              manual_bounds=[(0.0, 1.0), (0.0, 0.05)])
summary = expo_simulated_data.groupby('participant_id')['choice'].value_counts(normalize=True).unstack()