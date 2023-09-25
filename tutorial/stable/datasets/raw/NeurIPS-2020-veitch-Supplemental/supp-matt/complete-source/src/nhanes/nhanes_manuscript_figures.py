import pandas as pd
import os
from plotnine import *

os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
all_cov = pd.read_csv(
    '../out/nhanes_dbp_bart_logreg_kfold_grouped/sensitivity_output/austen_plot_coordinates.csv', header=1)
min_noage = pd.read_csv(
    '../out/nhanes_dbp_bart_logreg_kfold_grouped_noage/sensitivity_output/austen_plot_coordinates.csv', header=1)

bias = 2

p = (ggplot()
     + geom_line(data=all_cov, mapping=aes(x='alpha', y='Rsq'),
                 color='#585858', size=1, na_rm=True)
     + geom_line(data=min_noage, mapping=aes(x='alpha', y='Rsq'),
                 color='#FF1A00', size=1, na_rm=True)
     + theme_light()
     + theme(figure_size=(3.5, 3.5),
             legend_key=element_blank(),
             axis_title=element_text(size=12),
             axis_text=element_text(color='black', size=10),
             plot_title=element_text(size=12),
             legend_text=element_text(size=12))
     + labs(x='Influence on treatment ' + r'($\mathregular{\alpha}$)',
            fill='',
            y='Influence on outcome ' +
            r'(partial $R^2$)',
            title=f"Bias = {bias}")
     + scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
     + scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
     )
p
