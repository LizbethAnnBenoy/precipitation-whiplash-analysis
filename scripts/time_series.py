from scipy.stats import linregress

def run_time_series(df):
    df['year'] = (df['date_id'] // 365) + 2000
    df['day_of_year'] = df['date_id'] % 365
    df['month'] = ((df['day_of_year'] // 30) % 12) + 1

    yearly_stats = df.groupby('year').agg({
        'is_whiplash': 'sum',
        'extreme_precip': 'sum',
        'is_ar': 'sum',
        'precipitation_mm': 'mean',
        'drought_severity_score': 'mean'
    }).reset_index()

    yearly_stats.columns = ['year', 'whiplash_events', 'extreme_precip_days',
                            'ar_days', 'avg_precipitation', 'avg_drought_severity']

    slope, intercept, r_value, p_value, std_err = linregress(
        yearly_stats['year'],
        yearly_stats['whiplash_events']
    )

    yearly_stats.to_csv('yearly_trends.csv', index=False)

    return yearly_stats, slope, p_value
