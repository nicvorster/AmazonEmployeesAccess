library(tidymodels)
library(embed) # for target encoding
library(vroom)

amazondata <- vroom("Amazontrain.csv")
amazontestData  <- vroom("Amazontest.csv")

amazondata$ACTION = as.factor(amazondata$ACTION)

my_recipe <- recipe(ACTION ~ . , data=amazondata) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors()) %>%  # dummy variable encoding
step_normalize(all_predictors()) %>%
step_pca(all_predictors(), threshold=0.8) 
  ###step_lencode_mixed(vars_I_want_to_target_encode, outcome = vars(target_var)) #target encoding (must
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazondata)

### LOGISTIC REGRESSION ###

logRegModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

## Put into a workflow here
logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data= amazondata)
## Make predictions
amazon_predictions <- predict(logReg_workflow,
                              new_data=amazontestData,
                              type= "prob") # "class" or "prob"

## Kaggle Submission 1
kaggle_submission1 <- amazon_predictions %>%
  bind_cols(., amazontestData) %>% 
  rename(ACTION= .pred_1) %>% 
  select(id, ACTION) 
 

## Write out the file
vroom_write(x=kaggle_submission1, file="./LogRegPreds.csv", delim=",")


### PENALIZED LINEAR REGRESSION ###

pen_mod <- logistic_reg(mixture= 0.8, penalty= 0.001) %>% #Type of model
  set_engine("glmnet")

penalized_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(pen_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(amazondata, v = 3, repeats=1)

## Run the CV
CV_results <- penalized_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc))#, f_meas, sens, recall, spec,
                            # precision, accuracy)) #Or leave metrics NULL

# Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best()

## Finalize the Workflow & fit it
final_wf <-
penalized_workflow %>%
finalize_workflow(bestTune) %>%
fit(data=amazondata)


## Make predictions
amazon2_predictions <- predict(final_wf,
                              new_data=amazontestData,
                              type= "prob") # "class" or "prob"

## Kaggle Submission 2
kaggle_submission2 <- amazon2_predictions %>%
  bind_cols(., amazontestData) %>% 
  rename(ACTION= .pred_1) %>% 
  select(id, ACTION) 


## Write out the file
vroom_write(x=kaggle_submission2, file="./PenRegPreds.csv", delim=",")

