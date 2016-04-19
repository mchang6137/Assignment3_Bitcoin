library(ggplot2)
library(data.table)

test.data = read.table("../data/testTriplets.txt")
test.data[,1] = test.data[,1] + 1
test.data[,2] = test.data[,2] + 1
test.data = data.table(test.data)
setnames(test.data, c("giver", "receiver", "bool"))


probs.data = read.table("probs.txt")
probs.data = data.table(probs.data)
prob = probs.data

dat = data.frame(prob, as.factor(test.data[,bool]))
colnames(dat) = c("prob", "value")
ggplot(dat, aes(value, prob)) +
        geom_violin() +
        geom_jitter(alpha=0.1) +
    scale_y_log10() +
    labs(x="test value", y="prob")
