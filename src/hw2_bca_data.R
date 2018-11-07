# Reading population data
pop <- read.csv("../data/bca_popUS.csv")
pop$total_pop <- pop$total_pop*1000
pop$pop_labor_force <- pop$pop_labor_force*1000
cubicpop <- splinefun(x=pop$year, y=pop$pop_labor_force, method="fmm",
                      ties="mean")
pop.quarterly <- cubicpop(seq(1980, 2014.75, .25))

# Average population growth rate for US
# should match .98 as in Table 2 from the Technical Appendix
pop.Avg <- ((211545900/150227400)^(1/35)-1)*100

# Reading OECD data for US
bcaUS <- read.csv("../data/bca_oecdUS.csv")
bcaUS$pop_qtrly <- pop.quarterly

# Per-capita hours
bcaUS$hours_pc <- bcaUS$tot_emp*bcaUS$tot_hours/(bcaUS$pop_qtrly)
bcaUS$hours_pc_log <- log(bcaUS$hours_pc)
# Per-capita investment
## # Converting GCF to billions USD
## bcaUS$ct_gcfBN <- bcaUS$ct_gcf/1e9
# Converting Durable goods expenditure to billions USD
bcaUS$ct_dur_goodsBN <- bcaUS$ct_dur_goods*1e9
# Calculating per-capita investment
# (GCF+Durables)/(Deflator*Population)
bcaUS$inv_pc <- (bcaUS$ct_gcf+bcaUS$ct_dur_goodsBN)/(bcaUS$gdp_deflator*bcaUS$pop_qtrly)
bcaUS$inv_pc_log <- log(bcaUS$inv_pc)
# Per-capita government expenditure
bcaUS$gov_pc <- (bcaUS$gov_fnl_cons_exp+bcaUS$exports-bcaUS$imports)/(bcaUS$gdp_deflator*bcaUS$pop_qtrly)
# Converting it to log
bcaUS$gov_pc_log <- log(bcaUS$gov_pc)

# Not sure how to get sales tax data









