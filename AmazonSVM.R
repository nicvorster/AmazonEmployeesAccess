library(tidymodels)
library(embed) # for target encoding
library(vroom)
library(kernlab)


amazondata <- vroom("Amazontrain.csv")
amazontestData  <- vroom("Amazontest.csv")

amazondata$ACTION = as.factor(amazondata$ACTION)

my_recipe <- recipe(ACTION ~ . , data=amazondata) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
 # step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
 ### step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%  #target encoding (must
# also 
 # step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
 # step_lencode_bayes(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.85) 


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
#prep <- prep(my_recipe)
#baked <- bake(prep, new_data = amazondata)

### SVM models
svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

svm_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(svmRadial)

## Fit or Tune Model HERE
## Tune smoothness and Laplace here
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(amazondata, v = 5, repeats=1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazondata)

amazon_predictions <- predict(final_wf, new_data=amazontestData, type="prob")

## Kaggle Submission
kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazontestData) %>% 
  rename(ACTION= .pred_1) %>% 
  select(id, ACTION) 

## Write out the file
vroom_write(x=kaggle_submission, file="./SVMPreds.csv", delim=",")

