import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


# Load the latest version
sw = kagglehub.load_dataset(
KaggleDatasetAdapter.PANDAS,
"nikhileswarkomati/suicide-watch",
"Suicide_Detection.csv",
)

suicide_text_map = sw.set_index('Unnamed: 0')['text'].to_dict()


def reconstruct_prompt(row):
    # Check if the prompt is redacted and we have a valid reconstruction ID
    if row['prompt'] == "REDACTED" and pd.notnull(row['reconstruction_id_if_redacted']):
        # Pull the original text from our map using the ID
        return suicide_text_map.get(int(row['reconstruction_id_if_redacted']))
    return row['prompt']


# Finalised code for cleaning and feature engineering
def downsize_by_min(df):
    min_class_size = df['label'].value_counts().min()

    # Select all columns except the grouping one manually
    balanced_df = df.groupby('label', group_keys=False)[['prompt', 'label']].apply(
        lambda x: x.sample(min_class_size, random_state=42)
    ).reset_index(drop=True)

    return balanced_df


def length_aware_sample(df, target_size):
    balanced_chunks = []
    
    for cat in df['label'].unique():
        cat_group = df[df['label'] == cat].copy()
        cat_group['word_count'] = cat_group['prompt'].str.split().str.len()
        
        # Sort logic: 
        # For vanilla: Keep the longest (to bridge the gap toward adversarial)
        # For adversarial: Keep the shortest (to bridge the gap toward vanilla)
        if 'vanilla' in cat:
            cat_group = cat_group.sort_values('word_count', ascending=False)
        else:
            cat_group = cat_group.sort_values('word_count', ascending=True)
            
        # Take the top N samples based on this priority
        balanced_chunks.append(cat_group.head(target_size))
        
    return pd.concat(balanced_chunks).sample(frac=1, random_state=42).reset_index(drop=True)



