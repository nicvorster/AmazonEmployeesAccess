library(tidymodels)
library(embed) # for target encoding
library(vroom)

amazondata <- vroom("Amazontrain.csv")
amazontestData  <- vroom("Amazontest.csv")

my_recipe <- recipe(ACTION ~ . , data=amazondata) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors()) # dummy variable encoding
  ###step_lencode_mixed(vars_I_want_to_target_encode, outcome = vars(target_var)) #target encoding (must
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazondata)


