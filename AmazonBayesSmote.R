library(tidymodels)
library(embed) # for target encoding
library(vroom)
library(discrim)
library(themis)

amazondata <- vroom("Amazontrain.csv")
amazontestData  <- vroom("Amazontest.csv")

amazondata$ACTION = as.factor(amazondata$ACTION)

my_recipe <- recipe(ACTION ~ . , data=amazondata) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%  #target encoding (must
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.85) %>% 
  step_smote(all_outcomes(), neighbors=4) %>% 
  step_downsample()

# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazondata)


## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(smoothness(),
                            Laplace(),
                            levels = 2) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(amazondata, v = 3, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazondata)

amazon_predictions <- predict(final_wf, new_data=amazontestData, type="prob")

## Kaggle Submission
kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazontestData) %>% 
  rename(ACTION= .pred_1) %>% 
  select(id, ACTION) 

## Write out the file
vroom_write(x=kaggle_submission, file="./BayesSmotePreds.csv", delim=",")

