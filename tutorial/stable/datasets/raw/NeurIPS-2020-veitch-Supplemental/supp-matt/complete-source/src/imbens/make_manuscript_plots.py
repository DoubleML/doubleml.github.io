import pandas as pd
import os
from plotnine import *
os.chdir("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
names = ['imbens1', 'imbens2', 'imbens3', 'imbens4']
for name in names:
    plot_coords = pd.read_csv(
        f'../out/{name}_rf_no_smote/sensitivity_output/austen_plot_coordinates.csv', header=1)
    variable_importances_plot = pd.read_csv(
        f'../out/{name}_rf_no_smote/sensitivity_output/variable_importances.csv', header=1)
    variable_importances_plot = variable_importances_plot[
        variable_importances_plot['covariate_name'] != 'treatment']
    variable_importances_plot['labels'] = 'Individual covariates'
    variable_importances_plot.loc[variable_importances_plot['covariate_name']
                                  == 'pre_program_earnings', 'labels'] = 'Pre-program earnings'
    variable_importances_plot.loc[variable_importances_plot['covariate_name']
                                  == 'recent_earnings', 'labels'] = 'Recent earnings'
    p = (ggplot(data=plot_coords, mapping=aes(x='alpha', y='Rsq'))
         + geom_line(color='#585858', size=1, na_rm=True)
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
                title=f"Bias = 1000")
         + scale_x_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         + scale_y_continuous(expand=[0, 0, 0, 0], limits=(-0.03, 1))
         )
    p = p + geom_point(data=variable_importances_plot,
                       mapping=aes(x='ahat',
                                   y='Rsqhat',
                                   fill='labels'),
                       color='black',
                       alpha=0.8,
                       size=4) + scale_fill_manual(['#D55E00', '#0072B2', '#E69F00'])
    # scale_fill_manual(['#FF1A00', '#4BDA10', '#FFC300', '#3288BD', '#FDAE61', '#900C3F', '#D53E4F', '#66C2A5', '#FF5733', '#FFFFBF', '#ABDDA4', '#9E0142'])
    p.save(f'../out/{name}_rf_no_smote/sensitivity_output/austen_plot_recolored_cb.png',
           dpi=500, verbose=False)
