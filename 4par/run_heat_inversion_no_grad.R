library(solvergater.solvers)
library(pushoverr)

main <- function() {
  tryCatch({
    result <- execute()
    saveRDS(result, "heat_inversion_result_no_grad.rda")
    pushover_normal("Heat inversion completed")
    },
    error = function(e) {
      m <- conditionMessage(e)
      pushover_emergency(m)
      m
    }
  )
}

execute <- function() {
  heat <- heat_solver(ignore.stdout = FALSE, ignore.stderr = FALSE)
  heat_funs <- objective_functions(heat, heat_data$exact_qoi)
  x0 <- c(2, 2, 2)
  opt_method <- "BFGS"
  max_iter <- 1000
  nruns_before <- run_count(heat)
  proc_time_before <- proc.time()
  result <- optim(x0, fn = heat_funs$value, gr = NULL,
    method = opt_method)
  proc_time_after <- proc.time()
  nruns_after <- run_count(heat)
  list(x0 = x0, nruns = nruns_after - nruns_before, 
    proc_time = proc_time_after - proc_time_before, result = result,
    method = opt_method, maxit = max_iter)
}

main()

