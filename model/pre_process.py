import ast
import pandas as pd

parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

fact_checks = pd.read_csv('../test_data/fact_checks.csv')
posts = pd.read_csv('../test_data/posts.csv')

#fact check预处理
fact_checks = fact_checks.fillna('').set_index('fact_check_id')
for col in ['claim', 'instances', 'title']:
    fact_checks[col] = fact_checks[col].apply(parse_col)
fact_checks['fact_check_text'] = fact_checks['claim'].apply(lambda x: str(x[1]) if x else '') + ' ' + fact_checks[
    'title'].apply(lambda x: str(x[1]) if x else '')
fact_checks = fact_checks['fact_check_text']
fact_checks.to_csv('../preprocess_test/fact_checks_pre.csv', index=True)

# post预处理
posts = posts.fillna('').set_index('post_id')
for col in ['instances', 'ocr', 'verdicts', 'text']:
    posts[col] = posts[col].apply(parse_col)
posts['post_text'] = posts['verdicts'].apply(lambda x: str(x[0]) if x else '') + ' ' + posts['ocr'].apply(
    lambda x: str(x[0][1]) if x else '') + ' ' + posts['text'].apply(lambda x: str(x[1]) if x else '')
posts = posts['post_text']
posts.to_csv('../preprocess_test/posts_pre.csv', index=True)
