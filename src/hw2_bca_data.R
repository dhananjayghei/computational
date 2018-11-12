library(zoo)
# Reading the BCA data from FRED
dat <- read.csv("../data/bca_fredUS.csv")
# Converting the dates into quarters
dat$quarter <- as.yearqtr(as.Date(as.character(dat$quarter),
                                  format="%Y-%m-%d"))
# Constructing the variables
# Sales tax
dat$sales_tax_bn <- dat$fed_exc_tax_bn+dat$state_sales_tax_bn+dat$state_other_tax_bn
dat$real_sales_tax <- dat$sales_tax_bn/(dat$defl_pce/100)
# Real services from consumer durables
dat$real_serv_dur <- dat$pce_dur_bn*1.04/(dat$defl_pce_dur/100)
# Depreciation from consumer durables
dat$real_dep_dur <- dat$pce_dur_bn*.75/(dat$defl_pce_dur/100)
# Real Output
# Real GDP - Sales tax deflated by PCE deflator + Services from consumer durables (return =4%) deflated by PCE + Depreciation from consumer durables deflated by PCE
dat$real_output <- dat$gdp_bn-dat$real_sales_tax+dat$real_serv_dur+dat$real_dep_dur
# Real Output per capita (Adjusting for billions)
dat$real_output_pc <- dat$real_output*1e9/dat$non_inst_pop

# Investment
# Real GPDI + Real Government investment + Real PCE on durables - Sales tax deflated by PCE x Share of durables in PCE 
dat$inv <- dat$gpdi_bn+(dat$gvt_ginv_nom_bn*100/dat$defl_gov)+dat$pce_dur_bn-(dat$real_sales_tax*dat$pce_dur_bn/dat$pce_bn)
# Rean investment per capital (Adjusting for billions)
dat$real_inv_pc <- dat$inv*1e9/dat$non_inst_pop

# Real Government per capita (Adjusting for billions) 
# Real government consumption expenditure + Real net exports
dat$real_gov_pc <- ((dat$gvt_cons_exp_nom_bn*100/dat$defl_gov)+dat$exports_bn-dat$imports_bn)*1e9/dat$non_inst_pop

# Real labor input
# Total hours/Non-institutional population
dat$real_labor_inp <- dat$hours_worked/dat$non_inst_pop

# Saving the data set
bca <- dat[, c("quarter", "real_output_pc", "real_inv_pc", "real_gov_pc", "real_labor_inp")]
# Converting to logarithms
bca[, 2:5] <- apply(bca[,2:5], 2, log)
bca <- bca[complete.cases(bca), ]

# Storing the data
write.csv(bca, "../data/bcaFinal.csv")
