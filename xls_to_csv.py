import pandas as pd
import os
from glob import glob

# Define folder where "Excel" files are stored
data_folder = "Data/NFLPlayers"  # Change this to your actual folder path

# Get all XLS files
xls_files = glob(os.path.join(data_folder, "*.xls"))

# List to store processed player data
player_data = []

def process_player(file_path):
    """Reads and processes a single player's 'Excel' file (which is actually HTML)."""
    # Extract player name from filename
    player_name = os.path.basename(file_path).replace("_", " ").replace(".xls", "").title()

    try:
        # Read the HTML table inside the "xls" file
        df = pd.read_html(file_path)[0]  # Read first table from HTML
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

    # Check if required columns exist
    required_columns = {"Team", "Conf", "Rushing Yards", "Total TD", "Carries", "Receptions", "Year"}
    if not required_columns.issubset(df.columns):
        print(f"❌ Skipping {player_name}: Missing required columns.")
        return None

    # Add Player Name
    df["Player"] = player_name  

    # Get unique colleges and conferences in the order played
    colleges = df["Team"].unique().tolist()
    conferences = df["Conference"].unique().tolist()
    max_schools = max(len(colleges), len(conferences))  # Max schools/conferences for any player

    # Dictionary to store player stats
    player_dict = {"Player": player_name}

    # Add each college and conference dynamically
    for i in range(max_schools):
        player_dict[f"College {i+1}"] = colleges[i] if i < len(colleges) else "None"
        player_dict[f"Conference {i+1}"] = conferences[i] if i < len(conferences) else "None"

    # Aggregate stats per college
    agg_stats = df.groupby("Team").agg({
        "Rushing Yards": "sum",
        "Total TD": "sum",
        "Carries": "sum",
        "Receptions": "sum",
        "Year": "nunique"  # Count unique years played at each school
    }).reset_index()

    # Add stats for each college dynamically
    for i, row in agg_stats.iterrows():
        college_number = i + 1
        player_dict[f"Total Yards (College {college_number})"] = row["Rushing Yards"]
        player_dict[f"Total TDs (College {college_number})"] = row["Total TD"]
        player_dict[f"Total Carries (College {college_number})"] = row["Carries"]
        player_dict[f"Total Receptions (College {college_number})"] = row["Receptions"]
        player_dict[f"Years at College {college_number}"] = row["Year"]

    return player_dict  # Return processed player data

# Process each player file
for file in xls_files:
    player_df = process_player(file)
    if player_df is not None:
        player_data.append(player_df)

# Convert list of dicts to a DataFrame
final_df = pd.DataFrame(player_data)

# Save to CSV
output_path = os.path.join(data_folder, "college_player_stats.csv")
final_df.to_csv(output_path, index=False)

print(f"✅ Player data saved to {output_path}")
