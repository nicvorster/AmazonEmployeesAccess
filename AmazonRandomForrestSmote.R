library(tidymodels)
library(embed) # for target encoding
library(vroom)
library(ranger)
library(themis)

amazondata <- vroom("Amazontrain.csv")
amazontestData  <- vroom("Amazontest.csv")

amazondata$ACTION = as.factor(amazondata$ACTION)

my_recipe <- recipe(ACTION ~ . , data=amazondata) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))## %>%  #target encoding (must
  #step_normalize(all_predictors())  %>%
#  step_pca(all_predictors(), threshold=0.9) %>% 
 # step_smote(all_outcomes(), neighbors=4) %>% 
#  step_downsample()

# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazondata)

#### Random Forrest ###
## RF model
RF_mod <- rand_forest(mtry = 1,
                      min_n=16,
                      trees=500) %>%
set_engine("ranger") %>%
set_mode("classification")

## Create a workflow with model & recipe
RF_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(RF_mod)
## Set up grid of tuning values
#tuning_grid <- grid_regular(mtry(range = c(1,10)),
    #                         min_n(),
     #                        levels = 3) ## L^2 total tuning possibilities

## Split data for CV
#folds <- vfold_cv(amazondata, v = 3, repeats=1)

## Run the CV
#CV_results <- RF_wf %>%
 # tune_grid(resamples=folds,
  #          grid=tuning_grid,
   #         metrics=metric_set(roc_auc))

# Find Best Tuning Parameters
#bestTune <- CV_results %>%
 # select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  RF_wf %>%
#  finalize_workflow(bestTune) %>%
  fit(data=amazondata)

amazon_predictions <- predict(final_wf, new_data=amazontestData, type="prob")

## Kaggle Submission
kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazontestData) %>% 
  rename(ACTION= .pred_1) %>% 
  select(id, ACTION) 


## Write out the file
vroom_write(x=kaggle_submission, file="./RFSmotePreds.csv", delim=",")
