library(ltm)

dat <- read.csv("result.csv", header=F)
mod  <- rasch(dat, IRT.param=TRUE)
print(mod)
plot(mod)
print(summary(mod))
