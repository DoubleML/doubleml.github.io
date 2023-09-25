import pandas as pd
import numpy as np
import scipy as sp
import argparse
from sklearn.metrics import mean_squared_error


def calc_beta_shapes(g, alpha):
    alpha_shape = g*((1/alpha)-1)
    beta_shape = (1-g)*((1/alpha)-1)
    return alpha_shape, beta_shape


def calc_delta(Rsqhat, alphahat, g, Q, t, y):
    alpha_shape, beta_shape = calc_beta_shapes(g, alphahat)
    term1 = np.mean(sp.special.polygamma(1, alpha_shape + t) +
                    sp.special.polygamma(1, beta_shape+(1-t)))
    term2 = mean_squared_error(y, Q)
    den = term1/term2
    delta = np.sqrt(Rsqhat/den)
    return delta


def calc_bias(alphahat, g, delta):
    alpha_shape, beta_shape = calc_beta_shapes(g, alphahat)
    bias_term = sp.special.digamma(alpha_shape+1) - sp.special.digamma(
        beta_shape) - sp.special.digamma(alpha_shape) + sp.special.digamma(beta_shape+1)
    bias = np.mean(bias_term)*delta
    return bias


def main():
    myparser = argparse.ArgumentParser()
    myparser.add_argument('-input_csv', required=True)
    myparser.add_argument('-variable_importances', required=True)
    myparser.add_argument('-variable_name', required=True)
    myparser.add_argument('-do_att', required=False,
                          default=False, type=bool)
    args = myparser.parse_args()
    input_df = pd.read_csv(args.input_csv, header=0)
    if args.do_att:
        input_df = input_df[input_df['t'] == 1]
    variable_importances = pd.read_csv(args.variable_importances, header=1)
    alphahat = variable_importances.loc[variable_importances['covariate_name']
                                        == args.variable_name, 'ahat'].values
    Rsqhat = variable_importances.loc[variable_importances['covariate_name']
                                      == args.variable_name, 'Rsqhat'].values
    delta = calc_delta(
        Rsqhat, alphahat, input_df['g'], input_df['Q'], input_df['t'], input_df['y'])
    bias = calc_bias(alphahat, input_df['g'], delta)
    print(bias)
    return None


if __name__ == "__main__":
    main()
