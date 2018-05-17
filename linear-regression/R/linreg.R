library(ggplot2)
set.seed(42)

NUM_SAMPLES <- 200
GRID_LEN <- 200
SDEV = 0.6

f <- function(x) {
  return(1.5*x + 0.5)
}
df = data.frame( x = seq(0,1,length = GRID_LEN))
df$y = f(df$x)
df$sampx = runif(NUM_SAMPLES)
df$sampy = f(df$sampx)
df$noisy = df$sampy + rnorm(NUM_SAMPLES, sd = SDEV)

p <- ggplot(df, aes(x=sampx, y=noisy))+geom_point(mapping = aes(labels = "data"), color = "red", show.legend=TRUE) +
  geom_line(aes(x = x, y = y)) +
    scale_color_discrete(labels =c("data", "line")p
