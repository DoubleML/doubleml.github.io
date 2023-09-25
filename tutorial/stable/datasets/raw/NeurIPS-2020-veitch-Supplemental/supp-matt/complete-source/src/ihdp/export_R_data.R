setwd("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/data/ihdp-raw/ihdp-hill-2011/data")

sim.data=load('sim.data')

example_data=load('example.data')
write.csv(ihdp, 'example_data.csv', row.names = F)

