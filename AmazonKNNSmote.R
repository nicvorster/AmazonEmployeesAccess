library(tidymodels)
library(embed) # for target encoding
library(vroom)
library(kknn)
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

#### KNN ###
## knn model
knn_model <- nearest_neighbor(neighbors=20) %>% # set or tune
  set_mode("classification") %>%
set_engine("kknn")

knn_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(knn_model)

## Finalize the Workflow & fit it
final_wf <-
  knn_wf %>%
  fit(data=amazondata)

amazon_predictions <- predict(final_wf, new_data=amazontestData, type="prob")

## Kaggle Submission
kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazontestData) %>% 
  rename(ACTION= .pred_1) %>% 
  select(id, ACTION) 


## Write out the file
vroom_write(x=kaggle_submission, file="./KNNSmotePreds.csv", delim=",")
