import pandas as pd

def run_eda(df):
    total_days = len(df)
    whiplash_days = df['is_whiplash'].sum()
    ar_days = df['is_ar'].sum()
    extreme_precip_days = df['extreme_precip'].sum()

    print(f"\n Result STATISTICS:")
    print(f"   Total Days: {total_days:,}")
    print(f"   Whiplash Events: {whiplash_days:,} ({whiplash_days/total_days*100:.2f}%)")
    print(f"   AR Days: {ar_days:,} ({ar_days/total_days*100:.2f}%)")
    print(f"   Extreme Precipitation Days: {extreme_precip_days:,} ({extreme_precip_days/total_days*100:.2f}%)")

    return total_days, whiplash_days, ar_days, extreme_precip_days
