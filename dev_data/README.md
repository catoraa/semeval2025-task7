### Public files

fact_checks.csv - contains a subset of 153743 fact-checks in 8 languages ('ara', 'deu', 'eng', 'fra', 'msa', 'por', 'spa', 'tha') covering all subtasks.

posts.csv - contains all monolingual train/dev posts and crosslingual train/dev posts (there is no overlap between the two subsets). It contains posts in 14 languages.

pairs.csv - contains all train pairs (monolingual and crosslingual)

tasks.json - JSON file containing a list of fact-check IDs, train posts IDs and dev post IDs for each of the subtasks (monolingual - for 8 languages, crosslingual)

monolingual_predictions.json - a submission file for monolingual task containing dev post IDs and an empty column expecting a list of retrieved fact-checks for each row (post ID)

crosslingual_predictions.json - a submission file for crosslingual task containing dev post IDs and an empty column expecting a list of retrieved fact-checks for each row (post ID)

For more information, please refer to our [CodaBench Competition Page](https://www.codabench.org/competitions/3737/).

