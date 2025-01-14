import ast
import pandas as pd

# origin=0,translate=1,trial=2
translation = 1
if translation == 0:
    path = "origin"
elif translation == 1:
    path = "test"
elif translation == 2:
    path = "trial"

# 解析列数据
parse_col = lambda s: ast.literal_eval(s.replace('\n', '')) if s else s

# 读取数据
fact_checks = pd.read_csv('../test_data/fact_checks.csv')
posts = pd.read_csv('../test_data/posts.csv')
#pairs = pd.read_csv('../test_data/trial_pairs.csv')


# fact_check预处理
fact_checks = fact_checks.fillna('').set_index('fact_check_id')
for col in ['claim', 'title']:
    fact_checks[col] = fact_checks[col].apply(parse_col)
fact_checks['fact_check_text'] = (fact_checks['claim'].apply(lambda x: str(x[0]) if x else '') + ' '
                                  + fact_checks['title'].apply(lambda x: str(x[0]) if x else ''))
fact_checks_text = fact_checks['fact_check_text'].to_dict()
fact_checks = fact_checks.reset_index()

# post预处理
posts = posts.fillna('').set_index('post_id')
for col in ['ocr', 'text']:
    posts[col] = posts[col].apply(parse_col)
posts['post_text'] = posts['ocr'].apply(lambda x: str(x[0][0]) if x else '') + ' ' + posts['text'].apply(
    lambda x: str(x[0]) if x else '')
posts_text = posts['post_text'].to_dict()
posts = posts.reset_index()

# # pairs预处理
# pairs['fact_check'] = pairs['fact_check_id'].map(fact_checks_text)
# pairs['post'] = pairs['post_id'].map(posts_text)

# 保存处理后的数据
fact_checks.to_csv(f'../preprocess_data/{path}/fact_checks.csv', columns=['fact_check_id', 'fact_check_text'],
                   index=False)
posts.to_csv(f'../preprocess_data/{path}/posts.csv', columns=['post_id', 'post_text'], index=False)
#pairs.to_csv(f'../preprocess_data/{path}/pairs.csv', columns=['post', 'fact_check'], index=False)
