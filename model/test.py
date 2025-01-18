import json

import pandas as pd

predictions = pd.read_json('../preprocess_data/dev/mono/monolingual_predictions.json', orient='index')
with open('../preprocess_data/dev/mono/tasks.json', 'r') as file:
    tasks = json.load(file)['monolingual']

a = 0
b = 0
for language, language_tasks in tasks.items():
    post_dev_ids = language_tasks['posts_dev']
    fact_checks_ids = language_tasks['fact_checks']
    for post_dev_id in post_dev_ids:
        for fact_check_id in fact_checks_ids:
            a = max(a, predictions.index.get_loc(post_dev_id))
            b = max(b, fact_check_id)
print(a)
print(a, b)
