import pandas as pd


post = pd.read_csv('../sample_data/trial_posts.csv')
fact_check = pd.read_csv('../sample_data/trial_fact_checks.csv')

post['post_text'] = post['verdicts'].apply(lambda x: x[0] if x else '') + ' ' + post['ocr'].apply(lambda x: x[1]) + ' ' + post['text'].apply(lambda x: x[1] if x else '')
post = post[['post_id','post_text']]
post.to_csv('../sample_data/trial_posts_pre.csv', index=False)

fact_check['fact_check_text'] = fact_check['claim'].apply(lambda x: x[1]) + ' ' + post['text'].apply(lambda x: x[1] if x else '')
fact_check = fact_check[['fact_check_id','fact_check_text']]
fact_check.to_csv('../sample_data/trial_fact_checks_pre.csv', index=False)
