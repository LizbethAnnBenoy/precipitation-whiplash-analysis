from mlxtend.frequent_patterns import apriori, association_rules

def run_pattern_mining(df):
    df_binary = df.copy()
    df_binary['high_precip'] = (df['precipitation_mm'] > df['precipitation_mm'].quantile(0.75))
    df_binary['high_ivt'] = (df['ivt'] > df['ivt'].quantile(0.75))
    df_binary['low_soil_moisture'] = (df['dry_land_memory'] < df['dry_land_memory'].quantile(0.25))
    df_binary['in_drought'] = (df['drought_severity_score'] > 0)

    pattern_features = ['is_ar', 'extreme_precip', 'is_whiplash', 'high_precip',
                        'high_ivt', 'low_soil_moisture', 'in_drought']

    transactions = df_binary[pattern_features].astype(bool)

    frequent_itemsets = apriori(transactions, min_support=0.01, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules = rules.sort_values('lift', ascending=False)

    for idx, row in rules.head(10).iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        print(f"   Rule: {antecedent} → {consequent}")
        print(f"   Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.2f}")
        print("   " + "-"*70)

    return rules
