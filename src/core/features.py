import pandas as pd

from src.utils.helper import reconstruct_prompt, downsize_by_min, length_aware_sample


def clean_jailbreak(df):
    df = df.copy().rename(columns = {'data_type' : 'label'})
    
    mask = (df['label'] == 'adversarial_benign') & df['adversarial'].isna()

    # Remove rows from adversarial_benign that have NaN in adversarial
    df.drop(index=df[mask].index, inplace=True)

    df['prompt'] = df['adversarial'].fillna(df['vanilla'])

    # Take necessary columns
    columns = ['prompt', 'label']
    df = df[columns]

    df.drop_duplicates()

    df = downsize_by_min(df)
    df = length_aware_sample(df, 40000)

    df = df.copy().sample(frac=1).reset_index(drop=True)
    
    print("Jailbreak dataset has been cleaned by removing duplicates and long prompts")
    return df

def clean_aegis(test, train):
    data = pd.concat([test, train])
    
    # Apply the reconstruction
    data['prompt'] = data.apply(reconstruct_prompt, axis=1)
    
    columns = ['prompt', 'prompt_label']
    data = data[columns].copy().rename(columns={'prompt_label': 'label'})

    # those labelled unsafe are kept
    data = data.sort_values(by='label', ascending=False)
    # Drop duplicates based on the text column 
    data = data.drop_duplicates(subset=['prompt'], keep='first')

    data = length_aware_sample(data, 11000)

    data = data.copy().sample(frac=1).reset_index(drop=True)
    print("Aegis dataset has been cleaned by removing duplicates and long prompts")
    return data

def merge_data(aegis, jailbreak):
    combined = pd.concat([aegis, jailbreak])
    min_class_size = combined['label'].value_counts().min()

    balanced_df = combined.groupby('label', group_keys=False)[['prompt', 'label']].apply(
        lambda x: x.sample(5000, random_state=42)
    ).reset_index(drop=True)


    print("Datasets have been combined.")
    return balanced_df.copy().sample(frac=1).reset_index(drop=True)

