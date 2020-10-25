library(ggplot2)
library(readr)
library(dplyr)
library(CausalGAM)
library(missForest)
set.seed(2020)

##SECTION 1: Databending. First we get our data in the right shape to run our algorithm

# First we load in our dataset (Dataset 4, County Level Information about COVID-19 Outcomes and County Demographics)
county_covid = read_csv("county_covid.csv")
county_cat = read_csv("cat_dates.csv") # We also read in a cleaned version of this data that puts interventions into buckets instead of specific dates
county_cat = county_cat %>% select(-c("X1", "dem_to_rep_ratio"))
colnames(county_cat)[3:8] = c("stay_at_home", "fifty_gatherings", "five_hundred_gatherings", "public_schools", "restaurant_dine-in", "entertainment_gym")
county_covid = county_covid %>% left_join(county_cat, by = c("StateName", "CountyName"))
county_covid = county_covid %>% mutate(deaths_per_case = tot_deaths/tot_cases) #Get COVID mortality rates for each county
county_covid = county_covid %>% mutate(percent_elderly = (`PopMale60-642010` + `PopFmle60-642010` + `PopMale65-742010` + `PopFmle65-742010` + `PopMale75-842010` + `PopFmle75-842010` +`PopMale>842010` + `PopFmle>842010`)/CensusPopulation2010) # Get the percentage of people above 60 for each county
county_covid = county_covid %>% mutate(urban = 0) 
county_covid$urban = if_else(county_covid$`Rural-UrbanContinuumCode2013` <= 3, 1, 0) # Dichotomize Urban/Rural for our algorithm
county_covid$prop_medicare_eligible  = county_covid$`#EligibleforMedicare2018`/county_covid$`PopulationEstimate2018` # Get per capita instead of total information so that it confounds less with our treatment
county_covid$prop_medicare_enrolled  = county_covid$`MedicareEnrollment,AgedTot2017`/(county_covid$PopTotalMale2017 + county_covid$PopTotalFemale2017)
county_covid = county_covid %>% select("CountyName", # Select the variables that we are interested in moving forward with
                                       "StateName",
                                       "CensusRegionName",
                                       "CensusDivisionName",
                                       "FracMale2017",
                                       "MedianAge2010",
                                       "DiabetesPercentage",
                                       "HeartDiseaseMortality",
                                       "StrokeMortality",
                                       "Smokers_Percentage",
                                       "RespMortalityRate2014",
                                       "dem_to_rep_ratio",
                                       "SVIPercentile",
                                       "deaths_per_case",
                                       "percent_elderly",
                                       "urban",
                                       "prop_medicare_eligible",
                                       "prop_medicare_enrolled",
                                       "stay_at_home",
                                       "fifty_gatherings",
                                       "five_hundred_gatherings",
                                       "public_schools",
                                       "restaurant_dine-in",
                                       "entertainment_gym")

## SECTION 2: Imputation. Some of our data is missing, so we impute it using the missForest package in R.

county_covid$CensusRegionName = as.factor(county_covid$CensusRegionName) # We make sure to label factor variables so that they aren't read as character vectors
county_covid$CensusDivisionName = as.factor(county_covid$CensusDivisionName)
county_covid$StateName = as.factor(county_covid$StateName)
county_covid$CountyName = as.factor(county_covid$CountyName)

non_covars = county_covid %>% select(c("CountyName", "StateName", "deaths_per_case", "urban"))

to_impute = as.data.frame(county_covid %>% select(-c("CountyName", "StateName", "deaths_per_case", "urban")))

county_covid_imputation = missForest(to_impute, ntree = 200)$ximp

county_covid = cbind(non_covars, county_covid_imputation) # Combine the dataset

## SECTION 3: Fitting our model. We fit an Augmented Inverse Propensity Weighted model to estimate average treatment effects

propensity_model = gam(urban ~ FracMale2017 + MedianAge2010 + DiabetesPercentage + HeartDiseaseMortality + StrokeMortality + Smokers_Percentage + RespMortalityRate2014 + dem_to_rep_ratio + percent_elderly + prop_medicare_eligible + prop_medicare_enrolled + stay_at_home + fifty_gatherings + public_schools, family=binomial(link = "logit"), data = county_covid)

outcome_model = gam(deaths_per_case ~ FracMale2017 + MedianAge2010 + DiabetesPercentage + HeartDiseaseMortality + StrokeMortality + Smokers_Percentage + RespMortalityRate2014 + dem_to_rep_ratio + SVIPercentile + percent_elderly + prop_medicare_eligible + prop_medicare_enrolled + stay_at_home + fifty_gatherings + five_hundred_gatherings + public_schools + `restaurant_dine-in` + entertainment_gym, family = "gaussian", data = county_covid, subset = county_covid$urban == 1)

output = estimate.ATE(pscore.formula = urban ~ FracMale2017 + MedianAge2010 + DiabetesPercentage + HeartDiseaseMortality + StrokeMortality + Smokers_Percentage + RespMortalityRate2014 + dem_to_rep_ratio + percent_elderly + prop_medicare_eligible + prop_medicare_enrolled + stay_at_home + fifty_gatherings + public_schools,
                      pscore.family = binomial(link = logit), 
                      outcome.formula.t = deaths_per_case ~ FracMale2017 + MedianAge2010 + DiabetesPercentage + HeartDiseaseMortality + StrokeMortality + Smokers_Percentage + RespMortalityRate2014 + dem_to_rep_ratio + SVIPercentile  + percent_elderly + prop_medicare_eligible + prop_medicare_enrolled + stay_at_home + fifty_gatherings + five_hundred_gatherings + public_schools + `restaurant_dine-in` + entertainment_gym,
                      outcome.formula.c = deaths_per_case ~ FracMale2017 + MedianAge2010 + DiabetesPercentage + HeartDiseaseMortality + StrokeMortality + Smokers_Percentage + RespMortalityRate2014 + dem_to_rep_ratio + SVIPercentile + percent_elderly + prop_medicare_eligible + prop_medicare_enrolled + stay_at_home + fifty_gatherings + five_hundred_gatherings + public_schools + `restaurant_dine-in` + entertainment_gym,
                      outcome.family = "gaussian",
                      treatment.var = "urban",
                      outcome.var = "deaths_per_case",
                      data = county_covid)
# Our output is (1) our estimated average treatment effect, its aymptotic standard error, and then its upper and lower 95% bounds
output_vec = c(output$ATE.AIPW.hat, output$ATE.AIPW.asymp.SE, output$ATE.AIPW.hat -1.96*output$ATE.AIPW.asymp.SE, output$ATE.AIPW.hat +1.96*output$ATE.AIPW.asymp.SE)
output_vec = as.data.frame(output_vec)
write_csv(output_vec, "output_df")