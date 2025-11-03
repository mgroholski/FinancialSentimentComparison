import os
import numpy as np
import pandas as pd
import chardet  # kept if you use elsewhere


def convert_to_utc(
    df: pd.DataFrame, date_column: str, keep_tz: bool = False
) -> pd.DataFrame:
    """
    Parse dates and convert to UTC. By default returns naive UTC (tz removed).
    """
    df = df.copy()
    s = pd.to_datetime(df[date_column], errors="coerce", utc=True)
    if keep_tz:
        df[date_column] = s
    else:
        df[date_column] = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def _has_any_valid_dates(df: pd.DataFrame, date_column: str) -> bool:
    return df[date_column].notna().any() if date_column in df.columns else False


def fill_missing_dates(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Ensure daily coverage from min(Date) to max(Date). Requires at least one valid date.
    """
    df = df.copy()
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    else:
        df[date_column] = pd.to_datetime(df.index, errors="coerce")

    df = df[df[date_column].notna()]
    if df.empty:
        raise ValueError(f"No parsable dates found in '{date_column}'.")

    df = df.sort_values(date_column).drop_duplicates(subset=[date_column], keep="first")
    df[date_column] = df[date_column].dt.normalize()

    start = df[date_column].min().normalize()
    end = df[date_column].max().normalize()
    full_range = pd.date_range(start=start, end=end, freq="D")

    full_df = pd.DataFrame({date_column: full_range}).merge(
        df, on=date_column, how="left"
    )

    if "Lexrank_summary" in full_df.columns:
        full_df["News_flag"] = full_df["Lexrank_summary"].notna().astype(int)
    else:
        full_df["News_flag"] = 0

    return full_df


def integrate_data(
    stock_price_df: pd.DataFrame, news_df: pd.DataFrame, stock_price_csv_file: str
):
    stock_price_df_copy = stock_price_df.copy()
    news_df_copy = news_df.copy()

    stock_price_df_copy.columns = stock_price_df_copy.columns.str.capitalize()
    news_df_copy.columns = news_df_copy.columns.str.capitalize()

    for col in ["Unnamed: 0", "Unnamed: 0.1"]:
        stock_price_df_copy = stock_price_df_copy.drop(columns=[col], errors="ignore")
        news_df_copy = news_df_copy.drop(columns=[col], errors="ignore")

    # ---- Dates to naive UTC ----
    stock_price_df_copy = convert_to_utc(stock_price_df_copy, "Date", keep_tz=False)
    news_df_copy = convert_to_utc(news_df_copy, "Date", keep_tz=False)

    # If either side has no valid dates, skip this stock entirely
    if not _has_any_valid_dates(stock_price_df_copy, "Date"):
        print(
            f"Skipping {stock_price_csv_file}: no parsable dates in stock price file."
        )
        return -1, pd.DataFrame()
    if not _has_any_valid_dates(news_df_copy, "Date"):
        print(
            f"Skipping {stock_price_csv_file}: no parsable dates in news for this stock."
        )
        return -1, pd.DataFrame()

    # normalize to midnight
    stock_price_df_copy["Date"] = pd.to_datetime(
        stock_price_df_copy["Date"]
    ).dt.normalize()
    news_df_copy["Date"] = pd.to_datetime(news_df_copy["Date"]).dt.normalize()

    stock_price_df_copy = stock_price_df_copy.sort_values("Date").set_index("Date")
    news_df_copy = news_df_copy.sort_values("Date").reset_index(drop=True)

    # Fill missing news dates; if this raises, skip the stock
    try:
        news_df_copy = fill_missing_dates(news_df_copy, "Date")
    except ValueError as e:
        print(f"Skipping {stock_price_csv_file}: {e}")
        return -1, pd.DataFrame()

    merged_df = stock_price_df_copy.merge(
        news_df_copy, left_index=True, right_on="Date", how="left"
    )
    df_cleaned = merged_df.dropna(subset=["News_flag"])

    if "Close" in df_cleaned.columns and len(df_cleaned["Close"]) < 333:
        print(stock_price_csv_file)
        print("Lower than 333")
        return 0, df_cleaned

    return 1, df_cleaned


def start_inte(stock_price_folder_path: str, news_folder_path: str, saving_path: str):
    stock_price_csv_files = [
        f for f in os.listdir(stock_price_folder_path) if f.endswith(".csv")
    ]

    news_file_path = os.path.join(news_folder_path, "nasdaq_exteral_data.csv")
    news_df = pd.read_csv(news_file_path)
    news_df.columns = news_df.columns.str.capitalize()
    news_df = news_df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")

    for stock_price_csv_file in stock_price_csv_files:
        print(stock_price_csv_file)
        stock_file_path = os.path.join(stock_price_folder_path, stock_price_csv_file)
        stock_name = os.path.splitext(stock_price_csv_file)[0]

        stock_price_df = pd.read_csv(stock_file_path)
        stock_price_df.columns = stock_price_df.columns.str.capitalize()
        stock_price_df = stock_price_df.drop(
            columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore"
        )

        stock_news_df = news_df[news_df["Stock_symbol"] == stock_name]

        flag, merged_data = integrate_data(
            stock_price_df, stock_news_df, stock_price_csv_file
        )

        # Only save if we didn't skip due to unparsable dates
        if flag in (0, 1):
            out_path = os.path.join(saving_path, stock_price_csv_file)
            merged_data.to_csv(out_path, index=False)
        else:
            # flag == -1 → skipped
            continue


if __name__ == "__main__":
    stock_price_folder_path = "stock_price_data_preprocessed"
    news_folder_path = "news_data_preprocessed"
    saving_path = "price_news_integrate"

    start_inte(stock_price_folder_path, news_folder_path, saving_path)
