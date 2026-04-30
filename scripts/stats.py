from scipy.stats import chi2_contingency, mannwhitneyu

def run_stats(df):
    contingency_table = pd.crosstab(df['is_ar'], df['is_whiplash'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    whiplash_precip = df[df['is_whiplash'] == True]['precipitation_mm']
    no_whiplash_precip = df[df['is_whiplash'] == False]['precipitation_mm']

    u_stat, p_value2 = mannwhitneyu(whiplash_precip, no_whiplash_precip, alternative='greater')

    return chi2, p_value, u_stat, p_value2
