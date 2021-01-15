library(solvergater.solvers)

exact <- c(0.1, 1.5, 2.9)

lshape <- heat_solver()
run(lshape, exact, ignore.stderr = FALSE, ignore.stdout = FALSE)

